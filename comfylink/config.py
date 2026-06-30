"""Configuration + persisted state for the ComfyLink plugin.

The relay address defaults to the baked-in production relay, but can be pointed
at a test relay via a local config file (comfylink.json) or env var — see
_load_relay_url. Pairing is done from the ComfyUI panel, not by editing JSON.
The auto-managed state file (device tokens + ids) is keyed per relay so prod and
test pairings never collide.

One ComfyUI can be paired to MULTIPLE accounts at once. Each pairing carries its
own backend_id + device token (the relay's backends table is one backend_id ↔
one account, but a single machine simply registers several backend_ids — one per
account). All pairings share one machine name; jobs run serially on the single
local GPU.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import uuid
from dataclasses import dataclass
from pathlib import Path

# Relay address. Resolution order (first wins):
#   1. COMFYLINK_RELAY_URL env var          (highest; CI / dev)
#   2. "relay_url" in local config comfylink.json  (point at a test relay —
#      drop the file in to switch, remove it to go back to prod)
#   3. baked-in production default
_DEFAULT_RELAY_URL = "https://comfylink-relay.onrender.com"


def _config_path() -> Path:
    """Local config file (gitignored). Override path with COMFYLINK_CONFIG."""
    env = os.environ.get("COMFYLINK_CONFIG")
    if env:
        return Path(env)
    return Path(__file__).resolve().parent.parent / "comfylink.json"


def _load_relay_url() -> str:
    env = os.environ.get("COMFYLINK_RELAY_URL", "").strip()
    if env:
        return env
    try:
        p = _config_path()
        if p.is_file():
            data = json.loads(p.read_text("utf-8")) or {}
            url = str(data.get("relay_url") or "").strip()
            if url:
                return url
    except Exception:  # noqa: BLE001 - bad/missing config falls back to prod
        pass
    return _DEFAULT_RELAY_URL


RELAY_URL = _load_relay_url()
# True when on the baked-in production relay; the panel warns when on a
# non-default (test) relay so it's obvious which relay you're driving.
RELAY_IS_DEFAULT = RELAY_URL == _DEFAULT_RELAY_URL


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
    base = Path(__file__).resolve().parent.parent
    # Pairing state (device tokens / backend_ids) is relay-specific: a token the
    # prod relay issued is invalid on a test relay. Key the state file by relay
    # so prod and test pairings never collide and you can switch back and forth.
    # The default relay keeps the original filename (back-compat with existing
    # installs).
    if RELAY_IS_DEFAULT:
        return base / "comfylink_state.json"
    tag = hashlib.md5(RELAY_URL.encode()).hexdigest()[:8]
    return base / f"comfylink_state.{tag}.json"


@dataclass
class Pairing:
    """One account's pairing to this ComfyUI.

    Each pairing gets its own backend_id so the relay's one-backend-per-account
    model holds while a single machine serves several accounts concurrently.
    """

    backend_id: str
    device_token: str
    device_id: str = ""
    # Paired account email (for the panel; "paired to <email>"). In-memory only —
    # refreshed from the relay register response each start. NEVER persisted (no
    # account PII at rest in the state file).
    account: str = ""
    # Hash of the last object_info snapshot successfully uploaded to R2 for THIS
    # backend_id. Lets _register skip re-uploading an unchanged (multi-MB)
    # snapshot. The object_info bucket is non-expiring, so a remembered hash
    # guarantees the object is still there. Empty == "never uploaded" => uploads.
    object_info_hash: str = ""


class State:
    """Auto-managed local state (not user-edited). Module singleton: [STATE].

    Holds a list of pairings (one per paired account) plus a single machine name
    shared by all of them.
    """

    def __init__(self) -> None:
        self.backend_name: str = _default_name()
        self.pairings: list[Pairing] = []

    @classmethod
    def load(cls) -> "State":
        st = cls()
        p = _state_path()
        if not p.is_file():
            return st
        try:
            d = json.loads(p.read_text("utf-8"))
        except Exception:
            return st
        if not isinstance(d, dict):
            return st
        st.backend_name = d.get("backend_name") or st.backend_name
        raw = d.get("pairings")
        if isinstance(raw, list):
            # New multi-pairing format.
            for item in raw:
                if not isinstance(item, dict):
                    continue
                token = item.get("device_token") or ""
                if not token:
                    continue
                st.pairings.append(Pairing(
                    backend_id=item.get("backend_id") or str(uuid.uuid4()),
                    device_token=token,
                    device_id=item.get("device_id", ""),
                    object_info_hash=item.get("object_info_hash", ""),
                ))
        else:
            # Back-compat: old single-pairing top-level format. Convert to a
            # 1-element pairings list IFF a device token was actually present
            # (an unpaired old machine just loads as zero pairings).
            token = d.get("device_token") or ""
            if token:
                st.pairings.append(Pairing(
                    backend_id=d.get("backend_id") or str(uuid.uuid4()),
                    device_token=token,
                    device_id=d.get("device_id", ""),
                    object_info_hash=d.get("object_info_hash", ""),
                ))
        return st

    @property
    def paired(self) -> bool:
        return len(self.pairings) > 0

    def get_pairing(self, backend_id: str) -> Pairing | None:
        for pr in self.pairings:
            if pr.backend_id == backend_id:
                return pr
        return None

    def add_pairing(self, device_token: str, device_id: str) -> Pairing:
        """Append a new pairing (fresh backend_id), persist, and return it."""
        pr = Pairing(
            backend_id=str(uuid.uuid4()),
            device_token=device_token,
            device_id=device_id,
        )
        self.pairings.append(pr)
        self.save()
        return pr

    def remove_pairing(self, backend_id: str) -> Pairing | None:
        """Drop the pairing with this backend_id, persist, return it (or None)."""
        pr = self.get_pairing(backend_id)
        if pr is not None:
            self.pairings = [p for p in self.pairings if p.backend_id != backend_id]
            self.save()
        return pr

    def clear_pairing(self) -> None:
        """Remove ALL pairings (full local unpair) and persist."""
        self.pairings = []
        self.save()

    def save(self) -> None:
        # The state file holds the device bearer tokens (clr_..., full capability
        # to drive this backend, never expires). On a shared/multi-user host a
        # default-umask 0644 file would let any local user read those tokens, so
        # we create it 0600 (owner read/write only) from the start — never even
        # briefly world-readable. On Windows os.chmod with these mode bits is a
        # best-effort no-op and does not throw.
        try:
            p = _state_path()
            p.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps({
                "backend_name": self.backend_name,
                # account is intentionally NOT persisted (no PII at rest).
                "pairings": [
                    {
                        "backend_id": pr.backend_id,
                        "device_token": pr.device_token,
                        "device_id": pr.device_id,
                        "object_info_hash": pr.object_info_hash,
                    }
                    for pr in self.pairings
                ],
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


STATE = State.load()
