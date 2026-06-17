"""ComfyLink — ComfyUI custom node.

Defines no graph nodes. Instead it:
  - registers a small HTTP API + a sidebar panel (web/) to pair this PC and
    show connection status, and
  - runs a background worker that connects to the ComfyLink relay so the
    ComfyLink mobile app can drive this ComfyUI remotely.

The relay address is baked in; pairing is done from the panel. With no pairing
yet, the worker simply idles.
"""

import asyncio
import threading

from .comfylink.log import log
from .comfylink.worker import serve

# ComfyUI looks for these. We contribute no nodes, but we ship a web extension.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]

# Register the panel's HTTP routes (best-effort; never break ComfyUI startup).
try:
    from .comfylink import routes

    routes.register()
except Exception as e:  # noqa: BLE001
    log.warning("could not register panel routes: %s", e)


def _run_worker() -> None:
    try:
        asyncio.run(serve())
    except Exception as e:  # noqa: BLE001 - must never take ComfyUI down
        log.exception("worker thread crashed: %s", e)


_started = False


def _start_once() -> None:
    global _started
    if _started:
        return
    _started = True
    threading.Thread(target=_run_worker, name="comfylink-worker", daemon=True).start()


_start_once()
