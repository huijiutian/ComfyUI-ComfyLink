"""Worker: claim relay jobs, run them on local ComfyUI, ship results to R2.

A job's lifecycle here mirrors the app's GenerationController, but headless and
driven by the relay instead of a user:

  claim → (stage input images) → submit to local ComfyUI → watch local WS for
  completion (executing && node==null) → /view bytes → presigned PUT to R2 →
  POST result. Cancellation is learned from the relay's progress response.
"""

from __future__ import annotations

import asyncio
import json
from uuid import uuid4

import aiohttp

from .auth import TokenAuth
from .comfy import ComfyClient
from .config import RELAY_URL, STATE, detect_comfy_url
from .jobs import apply_inputs, content_type_for, extract_output_images, progress_event
from .log import log
from .relay import RelayClient, RelayError
from .status import STATUS

HEARTBEAT_INTERVAL = 25  # seconds
IDLE_RECHECK = 2  # seconds between "am I paired yet?" checks while unpaired


class _Revoked(Exception):
    """Device token no longer valid (unpaired from the app)."""


class JobCanceled(Exception):
    pass


class JobFailed(Exception):
    def __init__(self, message: str):
        self.message = message
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
        try:
            prompt = dict(job.get("api_prompt") or {})
            inputs = job.get("inputs") or []
            await self._stage_inputs(prompt, inputs)
            await self.relay.progress(job_id, "running", 0, 0)
            prompt_id = await self._run_prompt(job_id, prompt)
            images = await self._upload_outputs(job_id, prompt_id)
            await self.relay.result(job_id, "done", images)
            log.info("job %s done (%d image(s))", job_id, len(images))
            return "done"
        except JobCanceled:
            await self.relay.result(job_id, "canceled", [], "canceled")
            log.info("job %s canceled", job_id)
            return "canceled"
        except JobFailed as e:
            await self.relay.result(job_id, "failed", [], e.message)
            log.warning("job %s failed: %s", job_id, e.message)
            return "failed"
        except Exception as e:  # noqa: BLE001 - never let one job kill the loop
            await _safe_fail(self.relay, job_id, str(e))
            log.exception("job %s errored", job_id)
            return "failed"

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
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    ev = progress_event(json.loads(msg.data))
                    if ev is None:
                        continue
                    pid = ev.get("prompt_id")
                    if ev["kind"] == "progress":
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

    async def _upload_outputs(self, job_id: str, prompt_id: str) -> list[dict]:
        history = await self.comfy.history(prompt_id)
        out: list[dict] = []
        for im in extract_output_images(history, prompt_id):
            data = await self.comfy.view(im["filename"], im["subfolder"], im["type"])
            ct = content_type_for(im["filename"])
            key, url = await self.relay.sign_upload(job_id, "output", im["filename"], ct)
            await self.relay.put_object(url, data, ct)
            out.append({"r2_key": key, "filename": im["filename"],
                        "subfolder": im["subfolder"], "type": im["type"]})
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


async def _register(relay: RelayClient, comfy: ComfyClient) -> None:
    try:
        await relay.register(STATE.backend_id, STATE.backend_name)
    except RelayError as e:
        if e.status in (401, 403):
            raise _Revoked() from e
        raise
    try:
        oi = await comfy.object_info()
        await relay.put_object_info(STATE.backend_id, oi)
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
