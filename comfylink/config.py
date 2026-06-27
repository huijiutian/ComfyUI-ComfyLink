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
        # Paired account email (for the panel; "paired to <email>"). In-memory only
        # — refreshed from the relay register response each start, cleared on unpair.
        # Never persisted (no account PII at rest in the state file).
        self.account: str = ""
        # Hash of the last object_info snapshot we successfully uploaded to R2.
        # Lets _register skip re-uploading an unchanged (multi-MB) snapshot. The
        # object_info bucket is non-expiring, so a remembered hash guarantees the
        # object is still there. Empty == "never uploaded" => always uploads.
        self.object_info_hash: str = ""

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
                # Back-compat: missing in old state files loads as "" (re-upload).
                st.object_info_hash = d.get("object_info_hash", "")
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
        # The state file holds the device bearer token (clr_..., full capability
        # to drive this backend, never expires). On a shared/multi-user host a
        # default-umask 0644 file would let any local user read that token, so we
        # create it 0600 (owner read/write only) from the start — never even
        # briefly world-readable. On Windows os.chmod with these mode bits is a
        # best-effort no-op and does not throw.
        try:
            p = _state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps({
                "backend_id": self.backend_id,
                "device_token": self.device_token,
                "device_id": self.device_id,
                "backend_name": self.backend_name,
                "object_info_hash": self.object_info_hash,
            })
            # Create with 0600 atomically (O_CREAT honors the mode only on
            # creation), then chmod to also tighten a pre-existing file that may
            # have been written world-readable before this fix landed.
            fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
            try:
                os.chmod(p, 0o600)
            except OSError:
                pass
        except Exception:
            pass

    def clear_pairing(self) -> None:
        self.device_token = ""
        self.device_id = ""
        self.account = ""


STATE = State.load()
