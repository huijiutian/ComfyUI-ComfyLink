"""Token provider for the relay client.

Pairing (redeeming a one-time code for a device token) is triggered from the
ComfyUI panel — see routes.py. This just hands ONE pairing's device token to the
relay client, so each pairing authenticates as its own backend/account. The PC
never holds the user's account password.
"""

from __future__ import annotations

from .config import Pairing


class NotPairedError(RuntimeError):
    pass


class TokenAuth:
    """Hands a specific pairing's device token to its RelayClient.

    Bound to one Pairing (or any object exposing a ``device_token`` attribute)
    so a single machine can run several RelayClients — one per paired account —
    each authenticating with its own token.
    """

    def __init__(self, pairing: Pairing):
        self._pairing = pairing

    async def token(self) -> str:
        token = getattr(self._pairing, "device_token", "")
        if not token:
            raise NotPairedError("not paired")
        return token
