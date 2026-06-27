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
from .version import __commit__, __version__


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
        snap = STATUS.snapshot()  # machine-level: state/active/node_count/error
        snap["paired"] = STATE.paired
        snap["backend_name"] = STATE.backend_name
        # One row per paired account; account email may be "" until that pairing
        # registers (then the worker fills it in from the relay response).
        snap["pairings"] = [
            {"backend_id": pr.backend_id, "account": pr.account}
            for pr in STATE.pairings
        ]
        snap["relay_url"] = RELAY_URL
        snap["version"] = __version__
        snap["commit"] = __commit__  # git short commit (panel display; "dev" if unknown)
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
        # Append a NEW pairing (one ComfyUI → many accounts); never replace the
        # existing ones. The machine name is shared, so update it for all.
        STATE.backend_name = name
        STATE.add_pairing(token, dev_id)  # generates a fresh backend_id + saves
        STATUS.set(state="connecting", error="")
        log.info("paired via panel (device %s)", dev_id)
        return web.json_response({"ok": True})

    @routes.post("/comfylink/unpair")
    async def _unpair(request):
        try:
            data = await request.json()
        except Exception:
            data = {}
        backend_id = str(data.get("backend_id") or "").strip()
        if backend_id:
            pr = STATE.remove_pairing(backend_id)
            removed = [pr] if pr is not None else []
        else:
            # No backend_id → unpair ALL accounts on this machine.
            removed = list(STATE.pairings)
            STATE.clear_pairing()
        # Best-effort server-side revoke per removed device (local unpair done).
        for pr in removed:
            if pr.device_token and pr.device_id:
                try:
                    async with aiohttp.ClientSession() as s:
                        await s.delete(
                            RELAY_URL.rstrip("/") + f"/v1/devices/{pr.device_id}",
                            headers={"Authorization": f"Bearer {pr.device_token}"},
                            timeout=aiohttp.ClientTimeout(total=10),
                        )
                except Exception:  # noqa: BLE001
                    pass
        if not STATE.pairings:
            STATUS.set(state="unpaired", error="")
        log.info("unpaired %d pairing(s) via panel", len(removed))
        return web.json_response({"ok": True, "removed": len(removed)})

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
        # Push the SAME catalog to every paired account so each account's app sees
        # this machine's workflows (incl. any review/audit account). Per-pairing
        # results are collected; one account's failure doesn't abort the others.
        pairings = list(STATE.pairings)
        results: list[dict] = []
        errors: list[str] = []
        async with aiohttp.ClientSession() as session:
            for pr in pairings:
                relay = RelayClient(session, RELAY_URL, TokenAuth(pr))
                try:
                    # device token stays server-side; only ok/uploaded/error go back.
                    await _do_sync(relay, manifest, blobs, pr.backend_id)
                    results.append({"account": pr.account, "ok": True})
                except Exception as e:  # noqa: BLE001
                    log.warning("workflow sync failed for backend %s: %s",
                                pr.backend_id, e)
                    errors.append(str(e))
                    results.append({"account": pr.account, "ok": False, "error": str(e)})
        if errors and len(errors) == len(pairings):
            # Every account failed → surface an error so the panel shows it.
            return web.json_response(
                {"ok": False, "error": errors[0], "results": results}, status=502
            )
        log.info("workflow sync: %d blob(s) + manifest pushed to %d account(s)",
                 len(blobs), len(pairings))
        # `uploaded` mirrors the per-account blob count (the panel counts blobs
        # itself); `results` carries the per-account breakdown.
        return web.json_response(
            {"ok": True, "uploaded": len(blobs), "results": results}
        )

    log.info("ComfyLink panel routes registered")
