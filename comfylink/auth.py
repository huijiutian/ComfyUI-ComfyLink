"""Token provider for the relay client.

Pairing (redeeming a one-time code for a device token) is triggered from the
ComfyUI panel — see routes.py. This just hands the stored device token to the
relay client. The PC never holds the user's account password.
"""

from __future__ import annotations

from .config import State


class NotPairedError(RuntimeError):
    pass


class TokenAuth:
    def __init__(self, state: State):
        self._state = state

    async def token(self) -> str:
        if not self._state.device_token:
            raise NotPairedError("not paired")
        return self._state.device_token
