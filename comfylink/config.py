"""Configuration + persisted state for the ComfyLink plugin.

The relay address is baked in (no user config file). Pairing is done from the
ComfyUI panel, not by editing JSON. The only on-disk file is an auto-managed
state file (device token + ids) that the user never edits.
"""

from __future__ import annotations

import json
import os
import socket
import uuid
from pathlib import Path

# Baked-in relay address. Override with COMFYLINK_RELAY_URL for local dev.
_DEFAULT_RELAY_URL = "https://comfylink-relay.onrender.com"
RELAY_URL = os.environ.get("COMFYLINK_RELAY_URL", "").strip() or _DEFAULT_RELAY_URL


def detect_comfy_url() -> str:
    """Local ComfyUI HTTP address. Auto-detected from ComfyUI's launch port."""
    env = os.environ.get("COMFYLINK_COMFY_URL")
    if env:
        return env
    try:
        from comfy.cli_args import args  # type: ignore

        port = getattr(args, "port", None) or 8188
        return f"http://127.0.0.1:{port}"
    except Exception:
        return "http://127.0.0.1:8188"


def _default_name() -> str:
    try:
        return socket.gethostname() or "ComfyUI"
    except Exception:
        return "ComfyUI"


def _state_path() -> Path:
    env = os.environ.get("COMFYLINK_STATE")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "comfylink_state.json"


class State:
    """Auto-managed local state (not user-edited). Module singleton: [STATE]."""

    def __init__(self) -> None:
        self.backend_id: str = ""
        self.device_token: str = ""
        self.device_id: str = ""
        self.backend_name: str = _default_name()

    @classmethod
    def load(cls) -> "State":
        st = cls()
        p = _state_path()
        if p.is_file():
            try:
                d = json.loads(p.read_text("utf-8"))
                st.backend_id = d.get("backend_id", "")
                st.device_token = d.get("device_token", "")
                st.device_id = d.get("device_id", "")
                st.backend_name = d.get("backend_name") or st.backend_name
            except Exception:
                pass
        if not st.backend_id:
            st.backend_id = str(uuid.uuid4())
            st.save()
        return st

    @property
    def paired(self) -> bool:
        return bool(self.device_token)

    def save(self) -> None:
        try:
            p = _state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(
                json.dumps({
                    "backend_id": self.backend_id,
                    "device_token": self.device_token,
                    "device_id": self.device_id,
                    "backend_name": self.backend_name,
                }),
                "utf-8",
            )
        except Exception:
            pass

    def clear_pairing(self) -> None:
        self.device_token = ""
        self.device_id = ""


STATE = State.load()
