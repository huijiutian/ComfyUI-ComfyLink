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
    """Run the worker forever. Idles until paired (via the panel); connects and
    serves jobs while paired; returns to idle if the device is unpaired.

    Relay address is baked in; pairing is done from the ComfyUI panel.
    """
    comfy_url = detect_comfy_url()
    log.info("ComfyLink worker started (relay %s, comfy %s)", RELAY_URL, comfy_url)
    async with aiohttp.ClientSession() as session:
        while not _stopped(stop):
            if not STATE.paired:
                STATUS.set(state="unpaired")
                await asyncio.sleep(IDLE_RECHECK)
                continue
            try:
                await _serve_paired(session, comfy_url, stop)
            except _Revoked:
                STATE.clear_pairing()
                STATE.save()
                STATUS.set(state="unpaired", error="unpaired from the app")
                log.info("device unpaired — back to idle")
            except Exception as e:  # noqa: BLE001 - keep the loop alive
                STATUS.set(state="error", error=str(e))
                log.warning("connection error: %s; retrying in 5s", e)
                await asyncio.sleep(5)


async def _serve_paired(session, comfy_url, stop) -> None:
    relay = RelayClient(session, RELAY_URL, TokenAuth(STATE))
    comfy = ComfyClient(session, comfy_url)
    worker = Worker(relay, comfy)
    STATUS.set(state="connecting", error="")
    await _register(relay, comfy)
    hb = asyncio.create_task(_heartbeat_loop(relay, stop))
    try:
        await _claim_loop(relay, worker, stop)
    finally:
        hb.cancel()


def object_info_hash(oi: dict) -> str:
    """Stable content hash of an object_info snapshot.

    sort_keys makes it deterministic regardless of dict ordering, so an
    unchanged ComfyUI node set always hashes identically across restarts. Pure
    and side-effect-free => unit-testable on its own.
    """
    return hashlib.md5(
        json.dumps(oi, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


async def _register(relay: RelayClient, comfy: ComfyClient) -> None:
    try:
        await relay.register(STATE.backend_id, STATE.backend_name)
    except RelayError as e:
        if e.status in (401, 403):
            raise _Revoked() from e
        raise
    try:
        oi = await comfy.object_info()
        new_hash = object_info_hash(oi)
        if STATE.object_info_hash and STATE.object_info_hash == new_hash:
            # object_info bucket is non-expiring: a remembered hash means the
            # snapshot is still in R2, so skip the (multi-MB) re-upload.
            log.info(
                "object_info unchanged (hash %s), skipping upload", new_hash[:12]
            )
        else:
            await relay.upload_object_info(STATE.backend_id, oi)
            # Only remember the hash after a successful upload — on failure the
            # except below leaves it untouched so the next start retries.
            STATE.object_info_hash = new_hash
            STATE.save()
            log.info(
                "uploaded object_info (hash %s)", new_hash[:12]
            )
        STATUS.set(state="online", node_count=len(oi), error="")
        log.info("registered backend %s (%d node types)", STATE.backend_id, len(oi))
    except Exception as e:  # noqa: BLE001 - online even if object_info upload failed
        STATUS.set(state="online", error=f"object_info: {e}")
        log.warning("object_info not reported (ComfyUI reachable?): %s", e)


async def _heartbeat_loop(relay: RelayClient, stop: asyncio.Event | None) -> None:
    while not _stopped(stop) and STATE.paired:
        try:
            await relay.heartbeat(STATE.backend_id)
        except Exception as e:  # noqa: BLE001 - claim loop handles revoke
            log.debug("heartbeat error: %s", e)
        await asyncio.sleep(HEARTBEAT_INTERVAL)


async def _claim_loop(relay: RelayClient, worker: Worker,
                      stop: asyncio.Event | None) -> None:
    log.info("listening for jobs (idle until one arrives)")
    while not _stopped(stop) and STATE.paired:
        try:
            job = await relay.claim(STATE.backend_id)
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
            STATUS.set(active=True)
            try:
                await worker.handle_job(job)
            finally:
                STATUS.set(active=False)


def _stopped(stop: asyncio.Event | None) -> bool:
    return stop is not None and stop.is_set()
