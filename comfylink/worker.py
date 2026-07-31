"""Worker: claim relay jobs, run them on local ComfyUI, ship results to R2.

A job's lifecycle here mirrors the app's GenerationController, but headless and
driven by the relay instead of a user:

  claim → (stage input images) → submit to local ComfyUI → POLL ComfyUI's
  stable REST state (/history + /queue) until it finishes/errors/vanishes →
  /view bytes → presigned PUT to R2 → POST result. Cancellation is learned from
  the relay's progress response — on a single background beat that the relay
  HOLDS until a cancel arrives (see _job_heartbeat), so it lands within seconds
  instead of at the next scheduled beat.

Why poll REST instead of the websocket: the WS message format (which event
signals "done") changes across ComfyUI releases that users routinely install,
so a parser tuned to one version silently stops recognising completion on the
next and the job hangs. /history and /queue have been stable for years, so
reading completion from them keeps the worker version-independent. It also lets
us cancel PRECISELY (see _cancel_comfy) instead of issuing a global interrupt
that could kill a user's local generation.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from typing import Optional
from uuid import uuid4

import aiohttp

from . import loras
from .auth import TokenAuth
from .comfy import ComfyClient
from .config import RELAY_URL, STATE, detect_comfy_url
from .jobs import (
    apply_inputs,
    encode_output,
    extract_outputs,
    within_cap,
)
from .log import log
from .relay import RelayClient, RelayError
from .status import STATUS

HEARTBEAT_INTERVAL = 25  # seconds
IDLE_RECHECK = 2  # seconds between "am I paired yet?" checks while unpaired

# JOB_HEARTBEAT_INTERVAL is the FLOOR on how often a claimed-but-not-yet-finished
# job re-pokes the relay's progress endpoint. It MUST stay well under the relay's
# reaper staleThreshold (5 min): the reaper marks any claimed/running job whose
# updated_at hasn't advanced for that long as failed, so a steady beat keeps a
# legitimately long generation alive forever.
#
# Since the "带取消等待的长轮询" change there is exactly ONE beat for a running job
# (_job_heartbeat; _run_prompt no longer beats inline), and it uses
# progress(wait=True): the relay bumps updated_at the moment the request ARRIVES
# and then holds it ~25s waiting for a cancel. So the beat's real cadence is set
# by the relay's hold, and this constant only serves as the floor for the case
# where the request comes back immediately — an OLD relay that doesn't know the
# `wait` field. Without that floor a rolled-back relay would turn the heartbeat
# into a tight request loop.
JOB_HEARTBEAT_INTERVAL = 20  # seconds (floor between beats)

# JOB_HEARTBEAT_RETRY is the backoff after a FAILED beat. A failure is logged and
# ignored (the result path reports real errors), but we must not retry instantly:
# a relay redeploy / network blip would otherwise become a hot loop against a
# struggling relay. Small enough that a healthy job's updated_at stays far from
# the reaper's 5-minute threshold across a short outage.
JOB_HEARTBEAT_RETRY = 5  # seconds to wait after a failed beat

# POLL_INTERVAL is how often _run_prompt re-reads ComfyUI's *scoped*
# /history/{prompt_id} while waiting for a job to finish. It is the dominant
# term in "ComfyUI finished" → "the app sees the image": whatever we pick here
# is the average detection lag we add on top of the render itself. The request
# is a localhost GET for a single prompt id (cheap, and cheap regardless of how
# much history the user has accumulated), so 0.25s is affordable and cuts ~0.4s
# of average lag versus the old 1s.
#
# ⚠️ This constant deliberately drives NOTHING but that one local GET:
#   * the relay beat / cancel pickup lives in its own task (_job_heartbeat) and
#     is paced by the relay's own hold — polling ComfyUI 4× faster must NOT mean
#     4× the traffic to the relay;
#   * /queue has its own, deliberately slower cadence (QUEUE_POLL_INTERVAL);
#   * every timeout below (OUTPUTS_GRACE, EXECUTION_BACKSTOP_TIMEOUT) is judged
#     against time.monotonic(), not against "how many times we looped", so
#     changing this value cannot silently change how long they last.
POLL_INTERVAL = 0.25  # seconds between local /history polls

# QUEUE_POLL_INTERVAL is how often _run_prompt additionally reads ComfyUI's
# /queue. That read answers only one question — "is our prompt still
# queued/running, or did it vanish (user hit interrupt / cleared the queue)?" —
# and unlike /history/{id} it is NOT scoped: ComfyUI serialises every queued
# entry, prompt graphs included, so a user sitting on a deep local queue makes
# it a multi-MB response on the same event loop that is driving their render.
# Detecting a vanished prompt a fraction of a second sooner buys nothing (it is
# already an error path), so this stays at the historical 1s cadence while the
# cheap /history poll speeds up. Every /queue read is skipped, never batched:
# an un-read window simply means "assume still running", which is what the
# previous iteration concluded anyway.
QUEUE_POLL_INTERVAL = 1.0  # seconds between local /queue polls

# OUTPUTS_GRACE handles ComfyUI Issue #11540: status can flip to
# completed/success a beat BEFORE the outputs are written into /history. If we
# grabbed outputs the instant we saw "completed" we'd sometimes get an empty
# set, so once completion is observed we keep polling (up to this long) for the
# outputs to materialise before handing off to _collect_outputs.
OUTPUTS_GRACE = 5.0  # seconds to wait for lagging outputs after "completed"

# UPLOAD_CONCURRENCY caps how many collected outputs upload to R2 at once. The
# local→R2 uplink is the bottleneck for multi-image batches, so a small fan-out
# (sign_upload + put_object per output) meaningfully cuts wall time; the cap
# keeps it a steady parallelism that a slow/thin connection can sustain without
# exhausting its connection slots.
UPLOAD_CONCURRENCY = 4  # max outputs uploading to R2 concurrently

# EXECUTION_BACKSTOP_TIMEOUT is a conservative last-resort cap: if a prompt sits
# in ComfyUI's running queue this long without ever completing, we give up on it
# and report failed(error_code='execution_stalled') so the worker is freed for
# the next claim. Crucially we do NOT issue a global /interrupt here — a stuck
# poll must never kill a user's local generation; the relay's reaper plus the
# next process restart's orphan sweep reconcile the abandoned job. The window is
# intentionally generous (30 min) so it only ever trips on a genuinely wedged
# ComfyUI, never on a slow-but-healthy render. Measured against time.monotonic()
# from the moment we submit, so it is 30 minutes of real time whatever
# POLL_INTERVAL is set to.
EXECUTION_BACKSTOP_TIMEOUT = 30 * 60  # seconds before abandoning a wedged prompt

# REVOKED_CONFIRM_STRIKES is the depth-in-defence guard against a transient
# 401/403 mis-unpairing a still-valid device. A revoke is *local and permanent*
# (STATE.remove_pairing drops the pairing for good), so we must be sure before we
# pull the trigger. The relay's root fix already maps a transient DB error to
# 503 and only returns 401 when the device is GENUINELY gone (ErrNotFound), so a
# single 401 *should* be a real revoke — but a relay redeploy/restart can still
# briefly surface a stray 401/403. This is the second line of defence: we require
# this many CONSECUTIVE auth rejections (a successful register in between resets
# the count) before believing the device was truly unpaired. A real revoke still
# lands after N tries (~N×5s ≈ 15s here), which is fine for a rare, user-driven
# action; a lone blip is absorbed.
REVOKED_CONFIRM_STRIKES = 3

# PLUGIN_TOO_OLD_RETRY is the backoff after the relay refuses to serve this
# plugin version (403 + error_code "plugin_too_old"). Deliberately much longer
# than the 5s generic-error retry: a blocked plugin can do nothing useful, so it
# must not hammer the relay. We DO keep retrying (rather than giving up) so that
# rolling the relay's `min` back — or the user updating the files in place —
# recovers on its own without a ComfyUI restart.
PLUGIN_TOO_OLD_RETRY = 60  # seconds between re-register attempts while blocked

# The relay's fixed contract for the "your plugin is too old" refusal.
PLUGIN_TOO_OLD_CODE = "plugin_too_old"

# LORA_REFRESH_INTERVAL is how often the background inventory loop re-checks the
# loras directory. Unlike object_info (collected once per register, i.e. only on
# a ComfyUI restart/reconnect), a LoRA is something users add *while ComfyUI is
# running* — download a file, drop it in the folder, hit refresh in the web UI.
# Tying the inventory to register alone would mean "install a LoRA, then restart
# ComfyUI before your phone can show its trigger words", which defeats the point
# of a remote-control app.
#
# A warm refresh is cheap enough to afford on a timer: folder_paths caches its
# own listing behind a directory-mtime check, every digest is served from the
# path+size+mtime cache, and an unchanged manifest is dropped by the content-hash
# compare before any network call. So the steady-state cost is one os.stat per
# LoRA every 10 minutes and zero requests. Only genuinely new/changed files are
# ever hashed.
LORA_REFRESH_INTERVAL = 10 * 60  # seconds between inventory re-checks

# LORA_UNSUPPORTED_RETRY backs the loop right off once the relay has told us it
# has no inventory endpoint (404 — i.e. an older relay). We keep retrying rather
# than giving up so a relay deploy is picked up without a ComfyUI restart, but at
# a cadence that costs nothing.
LORA_UNSUPPORTED_RETRY = 60 * 60  # seconds between retries against an old relay


class _Revoked(Exception):
    """Device token no longer valid (unpaired from the app)."""


class _PluginTooOld(Exception):
    """The relay refuses to serve this plugin version (403 plugin_too_old).

    DELIBERATELY a different exception from _Revoked: a revoke is destructive
    (STATE.remove_pairing, permanent), while being too old is a *temporary,
    recoverable* server-side policy that says nothing about whether this device
    is still paired. Conflating the two would self-destruct a perfectly valid
    pairing within seconds of the relay raising its minimum version.
    """

    def __init__(self, min_version: str = "", update_url: str = ""):
        self.min_version = min_version
        self.update_url = update_url
        super().__init__(
            "relay refuses this plugin version"
            + (f" (requires v{min_version} or newer)" if min_version else "")
        )


class _TooOldLatch:
    """Per-pairing signal: "the heartbeat learned we are blocked".

    Why this exists: the relay's version gate lives on register + heartbeat, NOT
    on the hot /v1/jobs/claim long-poll (gating claim would cost a DB read per
    poll and add a fail-open surface). So an already-registered, always-on
    ComfyUI would otherwise keep claiming jobs forever after the operator raises
    the minimum version — the block would only bite on the next process restart.
    The heartbeat loop arms this latch and stops beating; the claim loop reads it
    at the top of each iteration and unwinds into _serve_pairing's existing
    _PluginTooOld handler (pairing kept, STATUS set, long backoff, re-register).

    One instance per _serve_pairing call, so pairings never interfere; cleared on
    every successful register so recovery (operator rolls `min` back) works.
    Not thread-safe by design — a single pairing's loops are one event loop's
    tasks, and every access is a plain attribute read/write with no await in
    between.
    """

    def __init__(self) -> None:
        self._blocked: Optional[_PluginTooOld] = None

    def arm(self, e: _PluginTooOld) -> None:
        self._blocked = e

    def clear(self) -> None:
        self._blocked = None

    @property
    def armed(self) -> bool:
        return self._blocked is not None

    def check(self) -> None:
        """Raise _PluginTooOld if armed; no-op otherwise."""
        if self._blocked is not None:
            raise _PluginTooOld(self._blocked.min_version, self._blocked.update_url)


def _as_plugin_too_old(e: RelayError) -> Optional[_PluginTooOld]:
    """_PluginTooOld built from a 403 + error_code "plugin_too_old"; else None.

    Pure + separately testable. Every place that maps a 403 onto _Revoked MUST
    consult this first, so a version block can never be counted as an auth
    rejection (i.e. never becomes a revoke strike).
    """
    if e.status != 403 or getattr(e, "code", "") != PLUGIN_TOO_OLD_CODE:
        return None
    p = getattr(e, "payload", None) or {}
    return _PluginTooOld(
        str(p.get("min_version") or ""),
        str(p.get("update_url") or ""),
    )


class JobCanceled(Exception):
    pass


class JobFailed(Exception):
    def __init__(self, message: str, error_code: str = ""):
        self.message = message
        # Optional machine-readable code propagated to the relay result so the
        # app can branch on it (e.g. 'execution_stalled' for the no-activity
        # watchdog). Empty for plain ComfyUI execution errors.
        self.error_code = error_code
        super().__init__(message)


class Worker:
    """Runs a single relay job against local ComfyUI. One job at a time."""

    def __init__(self, relay: RelayClient, comfy: ComfyClient):
        self.relay = relay
        self.comfy = comfy

    async def handle_job(self, job: dict) -> str:
        """Run a job to a terminal state, reporting the result to the relay.

        Returns the final status string (for tests/logging).
        """
        job_id = job["id"]
        # 取消信号:心跳任务是唯一的观察者,_run_prompt 是唯一的消费者。一旦置位就
        # 永远置位(粘性)—— 后续心跳失败也绝不清掉它。
        canceled = asyncio.Event()
        # Background heartbeat: the job's ONLY channel to the relay while it runs.
        # It re-pokes progress with a fixed (0, 0) "running" beat so the job's
        # updated_at advances through the long collect/upload phase too and the
        # relay reaper never mistakes a healthy plugin for a dead one — and, since
        # it goes out with wait=True, the SAME request is what learns about a
        # cancel (the relay holds it until one arrives). Cancelled in finally —
        # the task is bounded to this job's lifetime and never leaks.
        hb = asyncio.create_task(self._job_heartbeat(job_id, canceled))
        try:
            prompt = dict(job.get("api_prompt") or {})
            inputs = job.get("inputs") or []
            await self._stage_inputs(prompt, inputs)
            await self.relay.progress(job_id, "running", 0, 0)
            prompt_id = await self._run_prompt(job_id, prompt, canceled)
            max_bytes = int(job.get("max_output_bytes") or 0)
            output_format = job.get("output_format") or "png"
            images, total = await self._collect_outputs(prompt_id, output_format)
            if not within_cap(total, max_bytes):
                await self.relay.result(
                    job_id, "failed", [], "output exceeds your plan",
                    error_code="output_too_large", total_bytes=total,
                )
                log.warning(
                    "job %s output %.1f MB exceeds cap %.1f MB — skipped upload",
                    job_id, total / 1048576, max_bytes / 1048576,
                )
                return "failed"
            uploaded = await self._upload_outputs(job_id, images)
            await self.relay.result(job_id, "done", uploaded, total_bytes=total)
            log.info("job %s done (%d image(s), %.1f MB)", job_id, len(uploaded), total / 1048576)
            return "done"
        except JobCanceled:
            await self.relay.result(job_id, "canceled", [], "canceled")
            log.info("job %s canceled", job_id)
            return "canceled"
        except JobFailed as e:
            await self.relay.result(job_id, "failed", [], e.message,
                                    error_code=e.error_code)
            log.warning("job %s failed: %s", job_id, e.message)
            return "failed"
        except Exception as e:  # noqa: BLE001 - never let one job kill the loop
            await _safe_fail(self.relay, job_id, str(e))
            log.exception("job %s errored", job_id)
            return "failed"
        finally:
            # Stop the heartbeat before this job's result lands; await the cancel
            # so no orphaned task survives into the next claim.
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass

    async def _job_heartbeat(self, job_id: str, canceled: asyncio.Event) -> None:
        """The running job's single relay channel: keep-alive AND cancel pickup.

        Each beat is progress(wait=True) with a fixed (0, 0) "running" payload —
        we no longer report real progress numbers (the app renders a spinner +
        elapsed time, not a bar). The relay bumps updated_at the moment the
        request ARRIVES (so the reaper keeps seeing a live job through a long
        generation and the whole collect/upload phase) and then HOLDS the request
        ~25s waiting for a cancel. That makes the beat its own rhythm — no fixed
        sleep needed — and drops the cancel latency from "up to one beat
        interval" to seconds.

        On a cancel we set the (sticky) event and STOP beating: the relay would
        answer every subsequent wait immediately (cancel is already set), which
        would turn this loop into a hot request loop. The remaining work is short
        — _run_prompt sees the event within one local poll, cancels our prompt,
        and handle_job posts the terminal result.

        A failed beat is logged and ignored (the result path reports real
        errors), but with a JOB_HEARTBEAT_RETRY backoff so a relay/network blip
        can't become a tight retry loop. And when a beat returns EARLY without a
        cancel — an old relay that ignores the `wait` field — we top the wait up
        to JOB_HEARTBEAT_INTERVAL for the same reason.
        """
        while True:
            started = time.monotonic()
            try:
                r = await self.relay.progress(job_id, "running", 0, 0, wait=True)
            except Exception as e:  # noqa: BLE001 - keep beating; result path reports real errors
                log.debug("job %s heartbeat error: %s", job_id, e)
                await asyncio.sleep(JOB_HEARTBEAT_RETRY)
                continue
            if r.get("cancel"):
                canceled.set()
                log.info("job %s: relay requested cancel", job_id)
                return
            # Floor between beats (see JOB_HEARTBEAT_INTERVAL): with a relay that
            # honours `wait` this sleep is skipped entirely — the hold already
            # paced us.
            elapsed = time.monotonic() - started
            if elapsed < JOB_HEARTBEAT_INTERVAL:
                await asyncio.sleep(JOB_HEARTBEAT_INTERVAL - elapsed)

    async def _stage_inputs(self, prompt: dict, inputs: list[dict]) -> None:
        if not inputs:
            return
        key_to_name: dict[str, str] = {}
        for inp in inputs:
            url = inp.get("url")
            if not url:
                continue
            data = await self.relay.get_object(url)
            up = await self.comfy.upload_image(inp.get("name") or "input.png", data)
            key_to_name[inp.get("r2_key", "")] = up.get("name") or inp.get("name", "")
        apply_inputs(prompt, inputs, key_to_name)

    async def _run_prompt(self, job_id: str, prompt: dict,
                          canceled: Optional[asyncio.Event] = None) -> str:
        """Submit the prompt and watch ComfyUI's REST state until it terminates.

        Version-independent: instead of parsing websocket frames (whose format
        drifts across ComfyUI releases) we poll the stable /history and /queue
        endpoints. Each iteration decides the job's fate:

          * /history has our prompt with status_str == "error"  -> JobFailed
          * /history has our prompt with outputs                 -> return pid
          * /history says "completed" but outputs are still empty (#11540) ->
            keep polling up to OUTPUTS_GRACE, then return pid anyway (and let
            _collect_outputs surface "no valid output")
          * not in /history but present in /queue (running/pending) -> keep going
          * in NEITHER /history NOR /queue -> the prompt was interrupted/deleted
            on ComfyUI (a user hit interrupt in their browser, or an external
            queue clear) -> JobFailed(error_code="interrupted"), reported within
            seconds instead of waiting out any long timeout
          * still running past EXECUTION_BACKSTOP_TIMEOUT -> JobFailed(
            error_code="execution_stalled") WITHOUT a global interrupt

        This loop touches ONLY the local ComfyUI — it makes no relay request at
        all. Two independent cadences, deliberately NOT one loop counter:

          * POLL_INTERVAL (0.25s) — the scoped, cheap /history/{id} read that
            actually decides "is it done"; the thing worth making fast.
          * QUEUE_POLL_INTERVAL (1s) — the unscoped /queue read, only needed for
            the "did it vanish" error path, kept at its historical rate so a
            faster loop cannot multiply the load on the user's ComfyUI.

        The relay side lives entirely in _job_heartbeat, which runs as its own
        task: it keeps the job's updated_at fresh AND, because its beat is a
        wait-for-cancel long-poll, learns about a cancel within seconds. All this
        loop does is read the `canceled` event it sets — so relay traffic per
        running job is one held request, whatever the local poll rate is.

        Every deadline below is judged against time.monotonic(), never against
        an "elapsed += POLL_INTERVAL" accumulator, so the poll rate is free to
        change without moving any timeout (an accumulator also silently drifts
        long, since it ignores the per-request latency it never counts).

        A cancel is honoured by cancelling ONLY our prompt (see _cancel_comfy) —
        never a blanket interrupt that could kill a user's local generation.
        """
        if canceled is None:
            # No cancel channel wired (no heartbeat running) → an event nobody
            # ever sets, so the check below is a cheap no-op.
            canceled = asyncio.Event()
        client_id = str(uuid4())
        prompt_id = await self.comfy.submit(prompt, client_id)

        started = time.monotonic()      # submit time; the backstop counts from here
        next_queue = started            # ≤ now → the first "missing" poll reads /queue
        completed_at: Optional[float] = None  # when ComfyUI said done (outputs lagging)

        while True:
            now = time.monotonic()
            if canceled.is_set():
                # The heartbeat picked up the relay's cancel. Cancel PRECISELY
                # (pending → targeted queue delete, running → interrupt; see
                # _cancel_comfy) and unwind — never a blanket interrupt.
                await self._cancel_comfy(prompt_id)
                raise JobCanceled()
            hist = await self.comfy.history(prompt_id)
            entry = hist.get(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise JobFailed(_history_error(status))
                if entry.get("outputs"):
                    # Outputs are present — _collect_outputs takes it from here.
                    return prompt_id
                if status.get("completed") and completed_at is None:
                    # #11540: completed flipped true before outputs landed. Wait
                    # them out, then give up the grace and hand off regardless.
                    completed_at = now
            elif now >= next_queue:
                # Not in history. Only *now and then* (QUEUE_POLL_INTERVAL) do we
                # pay for the unscoped /queue read; in between we assume it is
                # still queued/running, exactly as the last read concluded.
                next_queue = now + QUEUE_POLL_INTERVAL
                q = await self.comfy.queue()
                if not _in_queue(q, prompt_id):
                    # Neither in history nor queued/running: the prompt was
                    # interrupted or deleted on ComfyUI out from under us.
                    raise JobFailed("job was interrupted on ComfyUI",
                                    error_code="interrupted")

            if completed_at is not None and now - completed_at >= OUTPUTS_GRACE:
                # Outputs never materialised; return and let _collect_outputs
                # report the empty result as "no valid output".
                return prompt_id

            if now - started >= EXECUTION_BACKSTOP_TIMEOUT:
                # Wedged for too long. Abandon WITHOUT a global interrupt so we
                # never kill a user's local generation; the relay reaper / next
                # orphan sweep reconcile the row.
                log.warning(
                    "job %s: still running after %ds — abandoning (no interrupt)",
                    job_id, int(now - started),
                )
                raise JobFailed(
                    "ComfyUI execution exceeded the safety timeout",
                    error_code="execution_stalled",
                )

            await asyncio.sleep(POLL_INTERVAL)

    async def _cancel_comfy(self, prompt_id: str) -> None:
        """Cancel OUR prompt precisely, never killing a user's local generation.

        A global /interrupt stops whatever ComfyUI is *currently running*, so we
        only issue it once we've confirmed OUR prompt is the running one. A
        still-pending prompt is removed with the targeted POST /queue delete,
        which leaves a running job untouched. If the prompt is in neither list it
        has already finished/vanished — nothing to do. Best-effort: a failure to
        read the queue is logged, not raised (the caller is already aborting).
        """
        try:
            q = await self.comfy.queue()
        except Exception as e:  # noqa: BLE001 - cancellation is best-effort
            log.warning("could not read ComfyUI queue to cancel %s: %s", prompt_id, e)
            return
        if _queue_has(q, "queue_pending", prompt_id):
            await self.comfy.queue_delete([prompt_id])
        elif _queue_has(q, "queue_running", prompt_id):
            await self.comfy.interrupt()
        # else: not pending or running anymore — nothing to cancel.

    async def _collect_outputs(
        self, prompt_id: str, output_format: str
    ) -> tuple[list[dict], int]:
        """Fetch each output's bytes, optionally convert to WebP, and measure.

        Returns ``(items, total_bytes)`` where each item carries the (possibly
        converted) ``data``/``filename``/``content_type``/``media_type`` plus
        the original subfolder/type. WebP conversion happens BEFORE measuring so
        it actually helps the user fit under their cap; videos are never
        converted (encode_output skips them). No upload happens here.
        """
        history = await self.comfy.history(prompt_id)
        items: list[dict] = []
        total = 0
        for it in extract_outputs(history, prompt_id):
            raw = await self.comfy.view(it["filename"], it["subfolder"], it["type"])
            data, filename, ct = encode_output(
                raw, it["filename"], output_format, it["media_type"]
            )
            total += len(data)
            items.append({"data": data, "filename": filename, "content_type": ct,
                          "subfolder": it["subfolder"], "type": it["type"],
                          "media_type": it["media_type"]})
        return items, total

    async def _upload_outputs(self, job_id: str, items: list[dict]) -> list[dict]:
        """Upload already-collected output bytes to R2; return relay payloads.

        Each payload carries ``media_type`` ("image"|"video") and the resolved
        ``content_type`` so the relay/app can render images vs videos correctly.

        Uploads run concurrently (bounded by UPLOAD_CONCURRENCY) since the R2
        uplink dominates a multi-image batch's wall time. The returned list is
        gathered in ``items`` order — the app renders outputs in sequence, so it
        must never be reordered. Failure semantics are unchanged: any single
        sign_upload/put_object exception propagates out (gather's default), and
        handle_job's except turns it into a failed result.
        """
        sem = asyncio.Semaphore(UPLOAD_CONCURRENCY)

        async def upload_one(it: dict) -> dict:
            async with sem:
                ct = it["content_type"]
                key, url = await self.relay.sign_upload(
                    job_id, "output", it["filename"], ct)
                await self.relay.put_object(url, it["data"], ct)
                return {"r2_key": key, "filename": it["filename"],
                        "subfolder": it["subfolder"], "type": it["type"],
                        "media_type": it["media_type"], "content_type": ct}

        # Tasks are built in items order, and gather preserves that order, so the
        # returned payloads stay aligned with the input regardless of completion
        # order. return_exceptions defaults to False → first error propagates.
        return await asyncio.gather(*(upload_one(it) for it in items))


def _queue_has(q: dict, key: str, prompt_id: str) -> bool:
    """True if prompt_id appears in the given /queue section.

    ComfyUI returns each section as a list of entries whose prompt id sits at
    index 1: ``"queue_running": [[number, "<pid>", ...], ...]``. Robust to
    short/odd rows. Pure + separately testable.
    """
    for entry in q.get(key) or []:
        if isinstance(entry, (list, tuple)) and len(entry) > 1 and entry[1] == prompt_id:
            return True
    return False


def _in_queue(q: dict, prompt_id: str) -> bool:
    """True if prompt_id is anywhere in ComfyUI's running OR pending queue."""
    return (_queue_has(q, "queue_running", prompt_id)
            or _queue_has(q, "queue_pending", prompt_id))


def _history_error(status: dict) -> str:
    """Pull a human error message out of a /history status block.

    ComfyUI records ``status.messages`` as a list of ``[event_name, payload]``
    pairs; the failure detail lives in the ``execution_error`` payload's
    ``exception_message``. Falls back to a generic string when absent. Pure.
    """
    for msg in status.get("messages") or []:
        if isinstance(msg, (list, tuple)) and len(msg) >= 2 and msg[0] == "execution_error":
            payload = msg[1] or {}
            detail = payload.get("exception_message") or payload.get("exception_type")
            if detail:
                return str(detail)
    return "ComfyUI reported an execution error"


async def _safe_fail(relay: RelayClient, job_id: str, message: str) -> None:
    try:
        await relay.result(job_id, "failed", [], message)
    except Exception:  # noqa: BLE001
        pass


async def serve(stop: asyncio.Event | None = None) -> None:
    """Supervise one service task per pairing; idle (unpaired) when there are none.

    A single machine can be paired to several accounts at once — each pairing has
    its own backend_id and runs its own register/heartbeat/claim loop. All jobs
    share ONE global lock (one GPU → one generation at a time); whoever claims a
    job first gets the lock first (asyncio.Lock is FIFO-fair).

    Every IDLE_RECHECK seconds the supervisor reconciles: pairings in STATE with
    no running task get one; tasks whose pairing was removed (unpaired) get
    cancelled. Relay address is baked in; pairing is done from the ComfyUI panel.
    """
    comfy_url = detect_comfy_url()
    log.info("ComfyLink worker started (relay %s, comfy %s)", RELAY_URL, comfy_url)
    # One generation at a time on the single local GPU, shared fairly across all
    # paired accounts.
    job_lock = asyncio.Lock()
    # Per-process "swept orphans once" set, keyed by backend_id (see
    # _abandon_orphans). Shared across reconnects so each backend is swept exactly
    # once per process — never again on a later reconnect (which would kill a job
    # that is genuinely running by then). The plugin owns this rather than the
    # relay's reaper because we re-register + heartbeat immediately on restart, so
    # the reaper's "job stale AND backend offline" double-check never holds.
    swept: set[str] = set()
    tasks: dict[str, asyncio.Task] = {}
    async with aiohttp.ClientSession() as session:
        try:
            while not _stopped(stop):
                _reconcile(tasks, job_lock, session, comfy_url, stop, swept)
                if not tasks:
                    # No pairings → the machine is idle/unpaired.
                    STATUS.set(state="unpaired", active=False, error="")
                await asyncio.sleep(IDLE_RECHECK)
        finally:
            for t in tasks.values():
                t.cancel()
            if tasks:
                await asyncio.gather(*tasks.values(), return_exceptions=True)


def _reconcile(tasks: dict, job_lock: asyncio.Lock, session, comfy_url,
               stop: asyncio.Event | None, swept: set) -> None:
    """Bring the running task set in line with STATE.pairings.

    Reaps finished tasks, cancels tasks whose pairing was removed, and starts a
    task for any pairing that lacks one. A pairing that revoked itself has already
    removed itself from STATE (so it is not restarted); a task that crashed
    unexpectedly while its pairing still exists IS restarted (resilience).
    """
    wanted = {pr.backend_id: pr for pr in STATE.pairings}
    # Reap finished tasks (surfacing unexpected crashes); they may be restarted
    # below if their pairing still exists.
    for bid in list(tasks):
        t = tasks[bid]
        if t.done():
            if not t.cancelled():
                exc = t.exception()
                if exc is not None:
                    log.warning("pairing %s task exited unexpectedly: %s", bid, exc)
            del tasks[bid]
    # Cancel tasks whose pairing is gone (unpaired).
    for bid in list(tasks):
        if bid not in wanted:
            tasks[bid].cancel()
            del tasks[bid]
    # Start a task for every pairing that lacks one.
    for bid, pr in wanted.items():
        if bid not in tasks:
            tasks[bid] = asyncio.create_task(
                _serve_pairing(pr, job_lock, session, comfy_url, stop, swept)
            )


async def _serve_pairing(pairing, job_lock: asyncio.Lock, session, comfy_url,
                         stop: asyncio.Event | None, swept: set) -> None:
    """Serve one pairing (one backend_id) until it is unpaired or stop is set.

    Registers, sweeps this backend's orphans once, then claims+runs jobs under
    the shared job_lock. A 401/403 from the relay usually means the account
    unpaired this device — but because unpairing is local + permanent, we don't
    act on a single rejection: we require REVOKED_CONFIRM_STRIKES *consecutive*
    401/403s (a successful register resets the count) before removing the pairing
    and returning (the supervisor won't restart it). This absorbs any stray
    transient 401/403 (e.g. during a relay redeploy) while a genuine revoke still
    lands after N tries. Transient errors are logged and retried; one pairing's
    failure never tears down the others.

    A 403 carrying error_code "plugin_too_old" is handled SEPARATELY and never
    counts as an auth rejection: the pairing is kept, no strike is recorded, the
    panel shows a loud "update the plugin" bar, and we retry on a long backoff.
    """
    relay = RelayClient(session, RELAY_URL, TokenAuth(pairing))
    comfy = ComfyClient(session, comfy_url)
    worker = Worker(relay, comfy)
    bid = pairing.backend_id
    # Consecutive 401/403 auth-rejection count for the strike-based unpair
    # confirmation (see REVOKED_CONFIRM_STRIKES). Reset to 0 on every successful
    # register below so unrelated blips never compound into a false unpair.
    revoked_strikes = 0
    # Per-pairing "the heartbeat saw a version block" latch (see _TooOldLatch):
    # the relay gates register + heartbeat but not claim, so this is how a block
    # armed AFTER we registered reaches the claim loop.
    blocked = _TooOldLatch()
    while not _stopped(stop) and STATE.get_pairing(bid) is not None:
        hb = None
        lo = None
        try:
            STATUS.set(state="connecting", error="")
            await _register(relay, comfy, pairing)
            # A successful register proves auth is still valid → clear any strikes
            # accumulated from earlier transient 401/403 so they can't add up
            # across unrelated blips.
            revoked_strikes = 0
            # ...and proves the relay is serving this version again, so drop any
            # latched block (operator rolled `min` back / user updated in place)
            # and let the fresh heartbeat + claim loops below run normally.
            blocked.clear()
            # One-shot orphan sweep per backend, AFTER a successful register and
            # BEFORE claiming. The shared `swept` set guards against re-sweeping
            # on a later reconnect (which would kill a job that is live by then).
            if bid not in swept:
                await _abandon_orphans(relay, pairing, swept)
            # Reached ONLY after _register returned normally. A blocked plugin
            # raises _PluginTooOld out of _register above, so control never gets
            # here and NO job is ever claimed while blocked — the heartbeat loop
            # isn't started either. (_claim_loop additionally re-raises
            # _PluginTooOld if the relay starts blocking mid-loop — either from a
            # claim 403 or from the latch the heartbeat arms — which lands in the
            # handler below and stops claiming.)
            hb = asyncio.create_task(_heartbeat_loop(relay, pairing, stop, blocked))
            # LoRA inventory: its own task, so a cold run that spends minutes
            # hashing hundreds of GB delays neither the heartbeat nor the claim
            # loop below. Torn down with hb in the finally — never leaked.
            lo = asyncio.create_task(_loras_loop(relay, pairing, stop))
            await _claim_loop(relay, worker, pairing, job_lock, stop, blocked)
        except asyncio.CancelledError:
            raise
        except _PluginTooOld as e:
            # NON-DESTRUCTIVE, on purpose. The relay refuses to serve this plugin
            # version; that says nothing about whether the device is still paired,
            # and the block may be a misjudgement or may be lifted (operator rolls
            # `min` back / user updates). So we:
            #   * NEVER call STATE.remove_pairing — the pairing survives forever,
            #     no matter how many times this fires;
            #   * NEVER touch revoked_strikes (neither += nor reset): a version
            #     block is orthogonal to auth, so it must not push the device
            #     toward an unpair, nor mask a genuine revoke by clearing strikes;
            #   * surface it loudly (red bar in the panel) and keep retrying on a
            #     long backoff so recovery is automatic, no restart needed.
            STATUS.set(
                state="error",
                error="Plugin too old — the relay stopped serving it. "
                      + (f"Update to v{e.min_version} or newer and restart ComfyUI."
                         if e.min_version else "Update the plugin and restart ComfyUI."),
                plugin_too_old=True,
                plugin_min_version=e.min_version,
                plugin_update_url=e.update_url,
            )
            log.warning(
                "relay refuses this plugin version (min %s) for pairing %s — "
                "pairing KEPT, no jobs will be claimed; retrying in %ds. "
                "Update the plugin (%s) and restart ComfyUI.",
                e.min_version or "?", bid, PLUGIN_TOO_OLD_RETRY,
                e.update_url or "https://github.com/huijiutian/ComfyUI-ComfyLink",
            )
            await asyncio.sleep(PLUGIN_TOO_OLD_RETRY)
        except _Revoked:
            # Second line of defence (relay already returns 503 for transient
            # errors, reserving 401 for a genuinely-missing device): don't unpair
            # on a single 401/403. Only after REVOKED_CONFIRM_STRIKES *consecutive*
            # rejections (any successful register above zeroes the count) do we
            # treat it as a real revoke and drop the pairing for good.
            revoked_strikes += 1
            if revoked_strikes >= REVOKED_CONFIRM_STRIKES:
                STATE.remove_pairing(bid)
                log.info(
                    "device unpaired — removing pairing %s (%s) after %d "
                    "consecutive auth rejections",
                    bid, pairing.account or "?", revoked_strikes,
                )
                return
            STATUS.set(error="auth rejected; retrying")
            log.warning(
                "pairing %s auth rejected (strike %d/%d), retrying in 5s",
                bid, revoked_strikes, REVOKED_CONFIRM_STRIKES,
            )
            await asyncio.sleep(5)
        except TypeError:
            # A TypeError here is a PROGRAMMING error (wrong signature, wrong
            # argument type), never a network blip — and the broad `except
            # Exception` below would treat it as one and retry it forever. That
            # is exactly how a signature drift between _claim_loop and a test's
            # stand-in turned into a silently hanging test suite instead of a
            # failing one; in production it is just as bad, a permanent 5s retry
            # loop that never makes progress and never says why.
            #
            # So: log it with a full traceback and let it out. The supervisor
            # (_reconcile) reaps the task, logs "task exited unexpectedly", and
            # restarts the pairing on the next IDLE_RECHECK tick — so this is
            # still self-healing, it is just LOUD instead of silent, and a test
            # driving _serve_pairing directly now fails immediately.
            log.exception(
                "pairing %s hit a programming error (not a transient fault) — "
                "letting it surface instead of retrying it forever", bid,
            )
            raise
        except Exception as e:  # noqa: BLE001 - isolate this pairing's failures
            STATUS.set(error=str(e))
            log.warning("pairing %s connection error: %s; retrying in 5s", bid, e)
            await asyncio.sleep(5)
        finally:
            for t in (hb, lo):
                if t is None:
                    continue
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
                except Exception:  # noqa: BLE001
                    pass


async def _run_locked(job_lock: asyncio.Lock, worker: "Worker", job: dict) -> str:
    """Run one job while holding the global generation lock.

    The single local GPU runs one generation at a time; multiple accounts queue
    fairly on this lock (asyncio.Lock is FIFO). ``active`` is machine-level — set
    only while a job actually runs (lock held), so the panel shows "generating"
    whenever any account's job is on the GPU.
    """
    async with job_lock:
        STATUS.set(active=True)
        try:
            return await worker.handle_job(job)
        finally:
            STATUS.set(active=False)


async def _abandon_orphans(relay: RelayClient, pairing, swept: set) -> None:
    """Fail any zombie jobs left claimed/running on this pairing's backend.

    Called once per backend per process: a just-started worker is running zero
    jobs (ComfyUI's queue is in-memory and empty on launch), so anything the
    relay still has as claimed/running on this backend is a leftover from a run
    that was killed mid-job. We ask the relay to mark those failed and record the
    backend in ``swept`` so we never sweep it again — a later reconnect must not
    clear a job that is genuinely running by then. Failures (network blip, relay
    hiccup) leave the backend unswept so the NEXT reconnect retries; we swallow
    the exception so a failed sweep never tears down the serve loop.
    """
    try:
        n = await relay.abandon_jobs(pairing.backend_id)
    except Exception as e:  # noqa: BLE001 - best-effort; retry on next reconnect
        log.warning("orphan sweep failed (will retry on reconnect): %s", e)
        return
    swept.add(pairing.backend_id)
    if n > 0:
        log.info("cleared %d orphaned job(s) from a previous run (backend %s)",
                 n, pairing.backend_id)


def object_info_hash(oi: dict) -> str:
    """Stable content hash of an object_info snapshot.

    sort_keys makes it deterministic regardless of dict ordering, so an
    unchanged ComfyUI node set always hashes identically across restarts. Pure
    and side-effect-free => unit-testable on its own.
    """
    return hashlib.md5(
        json.dumps(oi, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def _register(relay: RelayClient, comfy: ComfyClient, pairing) -> None:
    try:
        resp = await relay.register(pairing.backend_id, STATE.backend_name)
    except RelayError as e:
        # FIRST: a 403 that carries error_code "plugin_too_old" is a VERSION
        # block, not an auth failure. It must never fall through to _Revoked
        # below (that path unpairs the device for good after a few strikes).
        too_old = _as_plugin_too_old(e)
        if too_old is not None:
            raise too_old from e
        # 401/403 = this device was unpaired from the app (or its token is no
        # longer valid) → treat as revoked so the supervisor drops the pairing.
        # Each pairing owns a unique backend_id, so the old "owned by another
        # account" self-heal is gone — a fresh pairing never collides.
        if e.status in (401, 403):
            raise _Revoked() from e
        raise
    # Register succeeded → this plugin is being served again; clear any "too old"
    # banner left over from an earlier block so a recovered plugin stops showing
    # the red bar in the panel.
    STATUS.set(plugin_too_old=False, plugin_min_version="", plugin_update_url="")
    # Account email for the panel ("paired to <email>"); best-effort, may be "".
    pairing.account = (resp or {}).get("account", "") if isinstance(resp, dict) else ""
    try:
        oi = await comfy.object_info()
        new_hash = object_info_hash(oi)
        if pairing.object_info_hash and pairing.object_info_hash == new_hash:
            # object_info bucket is non-expiring: a remembered hash means the
            # snapshot is still in R2, so skip the (multi-MB) re-upload.
            log.info(
                "object_info unchanged (hash %s), skipping upload", new_hash[:12]
            )
        else:
            await relay.upload_object_info(pairing.backend_id, oi)
            # Only remember the hash after a successful upload — on failure the
            # except below leaves it untouched so the next start retries.
            pairing.object_info_hash = new_hash
            STATE.save()
            log.info(
                "uploaded object_info (hash %s)", new_hash[:12]
            )
        STATUS.set(state="online", node_count=len(oi), error="")
        log.info("registered backend %s (%d node types)", pairing.backend_id, len(oi))
    except Exception as e:  # noqa: BLE001 - online even if object_info upload failed
        STATUS.set(state="online", error=f"object_info: {e}")
        log.warning("object_info not reported (ComfyUI reachable?): %s", e)


async def _heartbeat_loop(relay: RelayClient, pairing,
                          stop: asyncio.Event | None,
                          blocked: Optional[_TooOldLatch] = None) -> None:
    """Keep this backend marked online. Errors are swallowed — with ONE exception.

    Every failure stays a debug-level no-op (the claim loop owns revoke handling
    and the result path owns job errors) EXCEPT a "plugin_too_old" 403: the relay
    gates on register + heartbeat but NOT on claim, so for an always-on ComfyUI
    that already registered before the block was armed, this is the only place
    the block is ever observed. We arm the latch and stop beating; the claim loop
    picks it up within one poll and unwinds into the non-destructive handler.
    """
    while not _stopped(stop) and STATE.get_pairing(pairing.backend_id) is not None:
        try:
            await relay.heartbeat(pairing.backend_id)
        except RelayError as e:
            too_old = _as_plugin_too_old(e)
            if too_old is not None:
                if blocked is not None:
                    blocked.arm(too_old)
                log.warning(
                    "heartbeat refused: plugin too old (min %s) — stopping the "
                    "heartbeat; this pairing will stop claiming new jobs",
                    too_old.min_version or "?",
                )
                return
            log.debug("heartbeat error: %s", e)
        except Exception as e:  # noqa: BLE001 - claim loop handles revoke
            log.debug("heartbeat error: %s", e)
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def _report_loras(relay: RelayClient, pairing) -> bool:
    """Ship this machine's LoRA inventory to R2 if it changed. Never raises.

    Returns True while the relay supports the endpoint, False once it has
    answered 404 (an older relay) so the caller can back right off.

    Failure is ALWAYS non-fatal here, deliberately unlike _register: the
    inventory is a convenience (trigger words in the app), so a missing endpoint,
    an unconfigured R2, or a network blip must leave the backend online and
    claiming jobs exactly as before. Nothing in this function touches STATUS —
    the panel has no business showing an error because a nice-to-have did not
    upload.
    """
    try:
        manifest = await loras.collect()
    except Exception as e:  # noqa: BLE001 - inventory must never break the worker
        log.warning("lora inventory: collection failed: %s", e)
        return True
    if not manifest.get("loras") and not manifest.get("checkpoints"):
        # No models installed (or none readable). Nothing to say; an empty upload
        # would only overwrite a previously-good manifest with nothing. Note both
        # arrays are checked: a machine with checkpoints but no LoRAs still has
        # something the app can use (the family check).
        return True
    new_hash = loras.manifest_hash(manifest)
    if pairing.loras_hash and pairing.loras_hash == new_hash:
        log.debug("lora inventory unchanged (hash %s), skipping upload", new_hash[:12])
        return True
    try:
        await relay.upload_loras(pairing.backend_id, manifest)
    except RelayError as e:
        if e.status == 404:
            # Old relay, new plugin: the route simply does not exist. Expected
            # during a staged rollout — a warning, not an error, and everything
            # else carries on untouched.
            log.warning(
                "relay has no LoRA inventory endpoint (404) — skipping upload; "
                "trigger words will appear once the relay is updated"
            )
            return False
        log.warning("lora inventory upload failed: %s", e)
        return True
    except Exception as e:  # noqa: BLE001 - best-effort
        log.warning("lora inventory upload failed: %s", e)
        return True
    # Only remember the hash after a successful upload, so a failure re-uploads
    # on the next pass (same rule as object_info_hash).
    pairing.loras_hash = new_hash
    STATE.save()
    log.info("uploaded model inventory (%d LoRA(s), %d checkpoint(s), hash %s)",
             len(manifest.get("loras") or []),
             len(manifest.get("checkpoints") or []), new_hash[:12])
    return True


async def _loras_loop(relay: RelayClient, pairing,
                      stop: asyncio.Event | None) -> None:
    """Background LoRA inventory reporter for one pairing.

    Runs as its OWN task next to the heartbeat, never inline in _register: the
    first pass may spend minutes hashing tens of GB (in a worker thread — see
    loras.collect), and registration, heartbeats and job claims must not wait a
    single millisecond for it.

    Exits immediately outside ComfyUI: without folder_paths there is no
    inventory to build, and that will not change later in this process, so
    looping would just burn a wakeup every interval forever.
    """
    if not loras.folder_paths_available():
        log.debug("lora inventory: folder_paths unavailable, not collecting")
        return
    while not _stopped(stop) and STATE.get_pairing(pairing.backend_id) is not None:
        try:
            supported = await _report_loras(relay, pairing)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 - the loop itself must never die
            log.warning("lora inventory: %s", e)
            supported = True
        await asyncio.sleep(
            LORA_REFRESH_INTERVAL if supported else LORA_UNSUPPORTED_RETRY
        )


async def _claim_loop(relay: RelayClient, worker: Worker, pairing,
                      job_lock: asyncio.Lock, stop: asyncio.Event | None,
                      blocked: Optional[_TooOldLatch] = None) -> None:
    bid = pairing.backend_id
    log.info("listening for jobs on backend %s (idle until one arrives)", bid)
    while not _stopped(stop) and STATE.get_pairing(bid) is not None:
        # Checked at the TOP of every iteration, which is deliberately the only
        # place the version block can stop us:
        #   * we do NOT interrupt an in-flight claim long-poll (~28s) — the
        #     worst case is one beat + one poll before claiming stops, which is
        #     fine for a policy change and keeps the poll path simple;
        #   * we do NOT abort a job that is already generating — _run_locked
        #     completes at the bottom of the loop body, so a running job finishes
        #     and reports its result normally. Only NEW work is stopped.
        if blocked is not None:
            blocked.check()  # raises _PluginTooOld → _serve_pairing's handler
        try:
            job = await relay.claim(bid)
        except RelayError as e:
            # Same discrimination as _register: a "plugin_too_old" 403 is a
            # version block, NOT an auth rejection, so it must never become a
            # revoke strike. It propagates to _serve_pairing's non-destructive
            # handler, which stops this claim loop for the backoff window.
            too_old = _as_plugin_too_old(e)
            if too_old is not None:
                raise too_old from e
            if e.status in (401, 403):
                raise _Revoked() from e
            log.warning("claim error: %s; retrying in 3s", e)
            await asyncio.sleep(3)
            continue
        except Exception as e:  # noqa: BLE001
            log.warning("claim error: %s; retrying in 3s", e)
            await asyncio.sleep(3)
            continue
        if job:
            log.info("claimed job %s", job.get("id"))
            # Serialize on the single GPU; STATUS.active is set inside _run_locked
            # only while the job is actually on the GPU (not while it queues).
            await _run_locked(job_lock, worker, job)


def _stopped(stop: asyncio.Event | None) -> bool:
    return stop is not None and stop.is_set()
