"""HTTP routes for the ComfyUI panel (registered on ComfyUI's server).

The web/ frontend extension calls these:
  GET  /comfylink/status  → connection status
  POST /comfylink/pair    → redeem a one-time pairing code
  POST /comfylink/unpair  → drop the pairing (and revoke server-side)
  POST /comfylink/sync    → push a workflow catalog (manifest + blobs) to R2
"""

from __future__ import annotations

import json

import aiohttp
from aiohttp import web

from .auth import TokenAuth
from .config import RELAY_URL, STATE
from .log import log
from .relay import RelayClient, redeem_pair_code
from .status import STATUS
from .version import __version__


async def _do_sync(relay, manifest: dict, blobs: dict, backend_id: str) -> int:
    """Upload workflow blobs then the manifest. Manifest LAST so it never
    references a blob that isn't uploaded yet. Returns the number of blobs."""
    for wf_id, blob in blobs.items():
        _key, url = await relay.sign_put_workflow(backend_id, "blob", wf_id)
        await relay.put_object(url, json.dumps(blob).encode(), "application/json")
    _key, url = await relay.sign_put_workflow(backend_id, "manifest")
    await relay.put_object(url, json.dumps(manifest).encode(), "application/json")
    return len(blobs)


def register() -> None:
    """Register routes on ComfyUI's PromptServer. No-op if unavailable."""
    from server import PromptServer  # type: ignore

    routes = PromptServer.instance.routes

    _NO_CACHE = {"Cache-Control": "no-store, no-cache, must-revalidate"}

    @routes.get("/comfylink/status")
    async def _status(_request):
        snap = STATUS.snapshot()
        snap["paired"] = STATE.paired
        snap["backend_name"] = STATE.backend_name
        snap["relay_url"] = RELAY_URL
        snap["version"] = __version__
        # no-store: the panel polls this; browsers must not serve a stale value.
        return web.json_response(snap, headers=_NO_CACHE)

    @routes.post("/comfylink/pair")
    async def _pair(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        code = str(data.get("code") or "").strip()
        name = str(data.get("name") or "").strip() or STATE.backend_name
        if not code:
            return web.json_response({"ok": False, "error": "code required"}, status=400)
        try:
            async with aiohttp.ClientSession() as s:
                token, dev_id = await redeem_pair_code(s, RELAY_URL, code, name)
        except Exception as e:  # noqa: BLE001
            return web.json_response({"ok": False, "error": str(e)}, status=400)
        STATE.device_token = token
        STATE.device_id = dev_id
        STATE.backend_name = name
        STATE.save()
        STATUS.set(state="connecting", error="")
        log.info("paired via panel (device %s)", dev_id)
        return web.json_response({"ok": True})

    @routes.post("/comfylink/unpair")
    async def _unpair(_request):
        token, dev_id = STATE.device_token, STATE.device_id
        STATE.clear_pairing()
        STATE.save()
        STATUS.set(state="unpaired", error="")
        # Best-effort server-side revoke (local unpair already done).
        if token and dev_id:
            try:
                async with aiohttp.ClientSession() as s:
                    await s.delete(
                        RELAY_URL.rstrip("/") + f"/v1/devices/{dev_id}",
                        headers={"Authorization": f"Bearer {token}"},
                        timeout=aiohttp.ClientTimeout(total=10),
                    )
            except Exception:  # noqa: BLE001
                pass
        log.info("unpaired via panel")
        return web.json_response({"ok": True})

    @routes.post("/comfylink/sync")
    async def _sync(request):
        if not STATE.paired:
            return web.json_response({"ok": False, "error": "not paired"}, status=401)
        try:
            data = await request.json()
        except Exception:
            data = {}
        manifest = data.get("manifest")
        blobs = data.get("blobs") or {}
        if not isinstance(manifest, dict):
            return web.json_response({"ok": False, "error": "manifest required"}, status=400)
        if not isinstance(blobs, dict):
            blobs = {}
        try:
            async with aiohttp.ClientSession() as session:
                relay = RelayClient(session, RELAY_URL, TokenAuth(STATE))
                # device token stays server-side; only ok/uploaded/error go back.
                uploaded = await _do_sync(relay, manifest, blobs, STATE.backend_id)
        except Exception as e:  # noqa: BLE001
            log.warning("workflow sync failed: %s", e)
            return web.json_response({"ok": False, "error": str(e)}, status=502)
        log.info("workflow sync uploaded %d blob(s) + manifest", uploaded)
        return web.json_response({"ok": True, "uploaded": uploaded})

    log.info("ComfyLink panel routes registered")
