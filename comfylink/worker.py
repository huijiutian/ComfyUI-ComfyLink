"""Worker: claim relay jobs, run them on local ComfyUI, ship results to R2.

A job's lifecycle here mirrors the app's GenerationController, but headless and
driven by the relay instead of a user:

  claim → (stage input images) → submit to local ComfyUI → watch local WS for
  completion (executing && node==null) → /view bytes → presigned PUT to R2 →
  POST result. Cancellation is learned from the relay's progress response.
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
    progress_event,
    within_cap,
)
from .log import log
from .relay import RelayClient, RelayError
from .status import STATUS

HEARTBEAT_INTERVAL = 25  # seconds
IDLE_RECHECK = 2  # seconds between "am I paired yet?" checks while unpaired

# JOB_HEARTBEAT_INTERVAL is how often a claimed-but-not-yet-finished job re-pokes
# the relay's progress endpoint even when ComfyUI emits no new progress. It MUST
# stay well under the relay's reaper staleThreshold (5 min): the reaper marks any
# claimed/running job whose updated_at hasn't advanced for that long as failed, so
# a steady ~20s heartbeat keeps a legitimately long generation alive forever.
JOB_HEARTBEAT_INTERVAL = 20  # seconds

# EXECUTION_STALL_TIMEOUT is a *no-activity* fallback for the one failure the
# relay's reaper deliberately won't catch: the backend process is alive (its
# heartbeat keeps flowing) but a single job's worker is wedged waiting on a
# ComfyUI that has gone completely silent — no executing/progress/executed, not
# even a websocket keep-alive frame. The relay's "double confirmation" reaper
# (job stale AND backend stale) leaves such a job running forever, so the plugin
# self-heals: if NOTHING arrives on the job's websocket for this long we declare
# the execution stalled, report failed(error_code='execution_stalled') and free
# the worker for the next claim.
#
# This measures time since the LAST activity, NOT total run time — a legitimately
# long generation keeps emitting progress (and the ws keep-alive emits frames
# every ~30s too), so the window resets constantly and a healthy long job NEVER
# trips it. Only a truly dead/hung ComfyUI stays silent long enough. The default
# is intentionally generous (10 min of total silence); bump it if you run nodes
# that can genuinely go quiet for minutes (e.g. a slow external API call) — it
# only guards against real hangs.
EXECUTION_STALL_TIMEOUT = 600  # seconds of complete websocket silence


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
        # Latest progress (value, max) the running job has seen, so the heartbeat
        # loop can re-send the CURRENT known figures (not stale 0s) when ComfyUI
        # has gone quiet. Reset at the start of each job.
        self._progress = (0, 0)

    async def handle_job(self, job: dict) -> str:
        """Run a job to a terminal state, reporting the result to the relay.

        Returns the final status string (for tests/logging).
        """
        job_id = job["id"]
        self._progress = (0, 0)
        # Background heartbeat: even when ComfyUI emits no progress, keep poking
        # the relay's progress endpoint (~every JOB_HEARTBEAT_INTERVAL) so the
        # job's updated_at advances and the relay reaper never mistakes a long-but-
        # healthy generation for a dead plugin. Cancelled in finally — the task is
        # bounded to this job's lifetime and never leaks.
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
        """Re-report the current known progress every JOB_HEARTBEAT_INTERVAL.

        This is a keep-alive, not a real progress source: it just re-sends the
        latest (value, max) the job has seen so the relay's updated_at keeps
        advancing while a long generation runs silently. A failed beat is logged
        and ignored — the real progress reports and the claim loop's revoke
        handling carry the actual signal.
        """
        while True:
            await asyncio.sleep(JOB_HEARTBEAT_INTERVAL)
            value, maximum = self._progress
            try:
                await self.relay.progress(job_id, "running", value, maximum)
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
        client_id = str(uuid4())
        async with self.comfy.ws_connect(client_id) as ws:
            prompt_id = await self.comfy.submit(prompt, client_id)
            # No-activity watchdog: receive with a short poll timeout and track
            # how long we've gone WITHOUT any websocket message (any frame counts
            # as activity and resets the clock). When accumulated silence crosses
            # EXECUTION_STALL_TIMEOUT we treat ComfyUI as hung and fail the job so
            # the worker is freed — see the constant's note for the rationale.
            poll = min(JOB_HEARTBEAT_INTERVAL, EXECUTION_STALL_TIMEOUT)
            silent = 0.0
            while True:
                try:
                    # aiohttp's own receive timeout: raises TimeoutError on a
                    # silent poll window WITHOUT cancelling the underlying read,
                    # so the next receive() resumes cleanly (unlike wrapping it
                    # in asyncio.wait_for, which cancels ws.receive() and can
                    # corrupt aiohttp's internal _waiting state).
                    msg = await ws.receive(timeout=poll)
                except asyncio.TimeoutError:
                    # Nothing arrived this poll window — accrue the silence and
                    # bail only once the *cumulative* gap exceeds the threshold.
                    silent += poll
                    if silent >= EXECUTION_STALL_TIMEOUT:
                        await self.comfy.interrupt()
                        log.warning(
                            "job %s: no ComfyUI activity for %ds — assuming "
                            "execution stalled", job_id, int(silent),
                        )
                        raise JobFailed(
                            "ComfyUI execution stalled (no activity)",
                            error_code="execution_stalled",
                        )
                    continue
                # Any received frame is activity — reset the silence window.
                silent = 0.0
                if msg.type == aiohttp.WSMsgType.TEXT:
                    ev = progress_event(json.loads(msg.data))
                    if ev is None:
                        continue
                    pid = ev.get("prompt_id")
                    if ev["kind"] == "progress":
                        # Remember it so the heartbeat re-sends the CURRENT figure.
                        self._progress = (ev["value"], ev["max"])
                        r = await self.relay.progress(job_id, "running", ev["value"], ev["max"])
                        if r.get("cancel"):
                            await self.comfy.interrupt()
                            raise JobCanceled()
                    elif ev["kind"] == "done" and pid in (None, prompt_id):
                        return prompt_id
                    elif ev["kind"] == "error" and pid in (None, prompt_id):
                        raise JobFailed(ev["message"])
                elif msg.type in (aiohttp.WSMsgType.CLOSED, aiohttp.WSMsgType.ERROR):
                    break
        raise JobFailed("websocket closed before completion")

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
        """
        out: list[dict] = []
        for it in items:
            ct = it["content_type"]
            key, url = await self.relay.sign_upload(job_id, "output", it["filename"], ct)
            await self.relay.put_object(url, it["data"], ct)
            out.append({"r2_key": key, "filename": it["filename"],
                        "subfolder": it["subfolder"], "type": it["type"],
                        "media_type": it["media_type"], "content_type": ct})
        return out


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
    the shared job_lock. A 401/403 from the relay means the account unpaired this
    device → remove the pairing and return (the supervisor won't restart it).
    Transient errors are logged and retried; one pairing's failure never tears
    down the others.
    """
    relay = RelayClient(session, RELAY_URL, TokenAuth(pairing))
    comfy = ComfyClient(session, comfy_url)
    worker = Worker(relay, comfy)
    bid = pairing.backend_id
    while not _stopped(stop) and STATE.get_pairing(bid) is not None:
        hb = None
        try:
            STATUS.set(state="connecting", error="")
            await _register(relay, comfy, pairing)
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
            STATE.remove_pairing(bid)
            log.info("device unpaired — removing pairing %s (%s)",
                     bid, pairing.account or "?")
            return
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
