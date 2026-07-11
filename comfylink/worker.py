"""Worker: claim relay jobs, run them on local ComfyUI, ship results to R2.

A job's lifecycle here mirrors the app's GenerationController, but headless and
driven by the relay instead of a user:

  claim → (stage input images) → submit to local ComfyUI → POLL ComfyUI's
  stable REST state (/history + /queue) until it finishes/errors/vanishes →
  /view bytes → presigned PUT to R2 → POST result. Cancellation is learned from
  the relay's progress response.

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
from uuid import uuid4

import aiohttp

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

# JOB_HEARTBEAT_INTERVAL is how often a claimed-but-not-yet-finished job re-pokes
# the relay's progress endpoint (and checks for a cancel request). It MUST stay
# well under the relay's reaper staleThreshold (5 min): the reaper marks any
# claimed/running job whose updated_at hasn't advanced for that long as failed, so
# a steady ~20s heartbeat keeps a legitimately long generation alive forever.
JOB_HEARTBEAT_INTERVAL = 20  # seconds

# POLL_INTERVAL is how often _run_prompt re-reads ComfyUI's /history and /queue
# while waiting for a job to finish. ~1s is responsive without hammering the
# local server.
POLL_INTERVAL = 1.0  # seconds between REST status polls

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
# ComfyUI, never on a slow-but-healthy render.
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


class _Revoked(Exception):
    """Device token no longer valid (unpaired from the app)."""


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
        # Background heartbeat: keep poking the relay's progress endpoint (~every
        # JOB_HEARTBEAT_INTERVAL) with a fixed (0, 0) "running" beat so the job's
        # updated_at advances through the long collect/upload phase too and the
        # relay reaper never mistakes a healthy plugin for a dead one. Cancelled
        # in finally — the task is bounded to this job's lifetime and never leaks.
        hb = asyncio.create_task(self._job_heartbeat(job_id))
        try:
            prompt = dict(job.get("api_prompt") or {})
            inputs = job.get("inputs") or []
            await self._stage_inputs(prompt, inputs)
            await self.relay.progress(job_id, "running", 0, 0)
            prompt_id = await self._run_prompt(job_id, prompt)
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

    async def _job_heartbeat(self, job_id: str) -> None:
        """Re-poke the relay with a fixed (0, 0) keep-alive every interval.

        We no longer report real progress numbers (the app renders a spinner +
        elapsed time, not a bar), so this just re-sends a "running" beat so the
        relay's updated_at keeps advancing while a long generation — or the
        collect/upload phase — runs. A failed beat is logged and ignored; the
        result path reports real errors and _run_prompt's own loop carries the
        cancel signal.
        """
        while True:
            await asyncio.sleep(JOB_HEARTBEAT_INTERVAL)
            try:
                await self.relay.progress(job_id, "running", 0, 0)
            except Exception as e:  # noqa: BLE001 - keep beating; result path reports real errors
                log.debug("job %s heartbeat error: %s", job_id, e)

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

    async def _run_prompt(self, job_id: str, prompt: dict) -> str:
        """Submit the prompt and watch ComfyUI's REST state until it terminates.

        Version-independent: instead of parsing websocket frames (whose format
        drifts across ComfyUI releases) we poll the stable /history and /queue
        endpoints every POLL_INTERVAL. Each iteration decides the job's fate:

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

        Every JOB_HEARTBEAT_INTERVAL it also re-pokes the relay (fixed 0, 0) and
        honours a cancel request by cancelling ONLY our prompt (see
        _cancel_comfy) — never a blanket interrupt that could kill a user's
        local generation.
        """
        client_id = str(uuid4())
        prompt_id = await self.comfy.submit(prompt, client_id)

        elapsed = 0.0       # total time since submit (drives the backstop)
        since_beat = 0.0    # time since the last relay heartbeat / cancel check
        grace = 0.0         # time spent waiting for lagging outputs (#11540)
        completed = False   # ComfyUI reported completion; outputs not yet present

        while True:
            hist = await self.comfy.history(prompt_id)
            entry = hist.get(prompt_id)
            if entry is not None:
                status = entry.get("status") or {}
                if status.get("status_str") == "error":
                    raise JobFailed(_history_error(status))
                if entry.get("outputs"):
                    # Outputs are present — _collect_outputs takes it from here.
                    return prompt_id
                if status.get("completed"):
                    # #11540: completed flipped true before outputs landed. Wait
                    # them out, then give up the grace and hand off regardless.
                    completed = True
            else:
                q = await self.comfy.queue()
                if not _in_queue(q, prompt_id):
                    # Neither in history nor queued/running: the prompt was
                    # interrupted or deleted on ComfyUI out from under us.
                    raise JobFailed("job was interrupted on ComfyUI",
                                    error_code="interrupted")

            if completed:
                grace += POLL_INTERVAL
                if grace >= OUTPUTS_GRACE:
                    # Outputs never materialised; return and let _collect_outputs
                    # report the empty result as "no valid output".
                    return prompt_id

            if elapsed >= EXECUTION_BACKSTOP_TIMEOUT:
                # Wedged for too long. Abandon WITHOUT a global interrupt so we
                # never kill a user's local generation; the relay reaper / next
                # orphan sweep reconcile the row.
                log.warning(
                    "job %s: still running after %ds — abandoning (no interrupt)",
                    job_id, int(elapsed),
                )
                raise JobFailed(
                    "ComfyUI execution exceeded the safety timeout",
                    error_code="execution_stalled",
                )

            if since_beat >= JOB_HEARTBEAT_INTERVAL:
                since_beat = 0.0
                r = await self.relay.progress(job_id, "running", 0, 0)
                if r.get("cancel"):
                    await self._cancel_comfy(prompt_id)
                    raise JobCanceled()

            await asyncio.sleep(POLL_INTERVAL)
            elapsed += POLL_INTERVAL
            since_beat += POLL_INTERVAL

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
    """
    relay = RelayClient(session, RELAY_URL, TokenAuth(pairing))
    comfy = ComfyClient(session, comfy_url)
    worker = Worker(relay, comfy)
    bid = pairing.backend_id
    # Consecutive 401/403 auth-rejection count for the strike-based unpair
    # confirmation (see REVOKED_CONFIRM_STRIKES). Reset to 0 on every successful
    # register below so unrelated blips never compound into a false unpair.
    revoked_strikes = 0
    while not _stopped(stop) and STATE.get_pairing(bid) is not None:
        hb = None
        try:
            STATUS.set(state="connecting", error="")
            await _register(relay, comfy, pairing)
            # A successful register proves auth is still valid → clear any strikes
            # accumulated from earlier transient 401/403 so they can't add up
            # across unrelated blips.
            revoked_strikes = 0
            # One-shot orphan sweep per backend, AFTER a successful register and
            # BEFORE claiming. The shared `swept` set guards against re-sweeping
            # on a later reconnect (which would kill a job that is live by then).
            if bid not in swept:
                await _abandon_orphans(relay, pairing, swept)
            hb = asyncio.create_task(_heartbeat_loop(relay, pairing, stop))
            await _claim_loop(relay, worker, pairing, job_lock, stop)
        except asyncio.CancelledError:
            raise
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
        except Exception as e:  # noqa: BLE001 - isolate this pairing's failures
            STATUS.set(error=str(e))
            log.warning("pairing %s connection error: %s; retrying in 5s", bid, e)
            await asyncio.sleep(5)
        finally:
            if hb is not None:
                hb.cancel()
                try:
                    await hb
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
        # 401/403 = this device was unpaired from the app (or its token is no
        # longer valid) → treat as revoked so the supervisor drops the pairing.
        # Each pairing owns a unique backend_id, so the old "owned by another
        # account" self-heal is gone — a fresh pairing never collides.
        if e.status in (401, 403):
            raise _Revoked() from e
        raise
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
                          stop: asyncio.Event | None) -> None:
    while not _stopped(stop) and STATE.get_pairing(pairing.backend_id) is not None:
        try:
            await relay.heartbeat(pairing.backend_id)
        except Exception as e:  # noqa: BLE001 - claim loop handles revoke
            log.debug("heartbeat error: %s", e)
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def _claim_loop(relay: RelayClient, worker: Worker, pairing,
                      job_lock: asyncio.Lock, stop: asyncio.Event | None) -> None:
    bid = pairing.backend_id
    log.info("listening for jobs on backend %s (idle until one arrives)", bid)
    while not _stopped(stop) and STATE.get_pairing(bid) is not None:
        try:
            job = await relay.claim(bid)
        except RelayError as e:
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
