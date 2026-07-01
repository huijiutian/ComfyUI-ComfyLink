"""Tests for the strike-based unpair confirmation (transient-401 defence).

A relay 401/403 from register/claim means "this device was unpaired" — but
unpairing is LOCAL and PERMANENT (STATE.remove_pairing drops the pairing for
good, and the supervisor never restarts it). The relay's root fix maps transient
DB errors to 503 and reserves 401 for a genuinely-missing device, but a relay
redeploy/restart can still briefly surface a stray 401/403. So the plugin adds a
second line of defence in _serve_pairing: it counts CONSECUTIVE auth rejections
and only removes the pairing after REVOKED_CONFIRM_STRIKES of them; any single
successful register in between resets the count. These tests pin:

  * consecutive 401s that REACH the threshold → remove_pairing + return;
  * 401s BELOW the threshold → the pairing is kept and register is retried;
  * a successful register between 401s resets the count, so a later isolated
    401 does NOT unpair (a transient blip is absorbed).

Run:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.config import Pairing  # noqa: E402
from comfylink.relay import RelayError  # noqa: E402


class _SupervisorState:
    """Stand-in for config.STATE with the lookups _serve_pairing uses."""

    def __init__(self, pairings):
        self.pairings = list(pairings)
        self.backend_name = "dev"
        self.remove_calls = 0

    def get_pairing(self, backend_id):
        return next((p for p in self.pairings if p.backend_id == backend_id), None)

    def remove_pairing(self, backend_id):
        self.remove_calls += 1
        pr = self.get_pairing(backend_id)
        self.pairings = [p for p in self.pairings if p.backend_id != backend_id]
        return pr

    def save(self):
        pass


def _mock_relay():
    """A relay mock whose object_info upload / abandon calls are harmless."""
    relay = mock.AsyncMock()
    relay.upload_object_info.return_value = None
    return relay


def _mock_comfy():
    comfy = mock.AsyncMock()
    comfy.object_info.return_value = {}  # hashes fine; upload is mocked
    return comfy


class TestRevokeStrikes(unittest.IsolatedAsyncioTestCase):
    async def _serve(self, relay, state, pairing, stop=None, strikes=3):
        """Drive the real _serve_pairing with mocked collaborators.

        Uses the REAL _register / _claim_loop paths so the 401/403 -> _Revoked
        mapping is exercised end-to-end; only construction, heartbeat, sleep and
        STATUS are stubbed. swept already contains bid so the orphan sweep is
        skipped. Bounded by wait_for so a buggy (non-terminating) loop fails
        loudly instead of hanging.
        """
        comfy = _mock_comfy()
        with mock.patch.object(worker, "REVOKED_CONFIRM_STRIKES", strikes), \
                mock.patch.object(worker, "RelayClient", return_value=relay), \
                mock.patch.object(worker, "ComfyClient", return_value=comfy), \
                mock.patch.object(worker, "Worker"), \
                mock.patch.object(worker, "STATUS"), \
                mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_heartbeat_loop", mock.AsyncMock()), \
                mock.patch.object(worker.asyncio, "sleep", mock.AsyncMock()):
            await asyncio.wait_for(
                worker._serve_pairing(
                    pairing, asyncio.Lock(), None, "http://comfy", stop,
                    {pairing.backend_id},  # swept → skip orphan sweep
                ),
                timeout=2,
            )

    async def test_consecutive_401s_reach_threshold_then_unpair(self):
        # register always 401s → _Revoked every loop → after exactly STRIKES
        # rejections the pairing is removed and _serve_pairing returns.
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        relay = _mock_relay()
        relay.register.side_effect = RelayError("device gone", 401)

        await self._serve(relay, state, pairing, strikes=3)

        self.assertEqual(state.remove_calls, 1, "should unpair exactly once")
        self.assertIsNone(state.get_pairing("b1"), "pairing must be dropped")
        # Removed the instant the 3rd consecutive rejection landed — not before,
        # not after (a 4th register would mean it kept looping past threshold).
        self.assertEqual(relay.register.await_count, 3)

    async def test_401s_below_threshold_keep_pairing_and_retry(self):
        # Two 401s with a threshold of 3: the device must NOT be unpaired, and
        # register must be retried (still listening). We set `stop` after the 2nd
        # attempt so the loop exits below the threshold instead of running on.
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        stop = asyncio.Event()
        relay = _mock_relay()

        calls = {"n": 0}

        def register_401(backend_id, name):
            # Sync side_effect on an AsyncMock: raising/returning here drives the
            # awaited call deterministically (no coroutine-await ambiguity).
            calls["n"] += 1
            if calls["n"] >= 2:
                stop.set()  # break the loop after the 2nd (still < 3) strike
            raise RelayError("blip 401", 401)

        relay.register.side_effect = register_401

        await self._serve(relay, state, pairing, stop=stop, strikes=3)

        self.assertEqual(state.remove_calls, 0, "must NOT unpair below threshold")
        self.assertIsNotNone(state.get_pairing("b1"), "pairing must be kept")
        self.assertEqual(relay.register.await_count, 2, "register should be retried")

    async def test_success_between_401s_resets_strikes(self):
        # 401, 401 (strikes → 1, 2), then a SUCCESSFUL register resets to 0.
        # After the reset the claim loop hits a lone 401 → strike 1 (not 3), so
        # the pairing survives. With a threshold of 3, only the reset explains why
        # remove_pairing is never called: without it the counts would be 1,2 then
        # the claim 401 would be strike 3 and unpair.
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        stop = asyncio.Event()
        relay = _mock_relay()

        # register: fail, fail, succeed, succeed.
        relay.register.side_effect = [
            RelayError("blip 401", 401),
            RelayError("blip 401", 401),
            {"account": "a@example.com"},
            {"account": "a@example.com"},
        ]

        # claim (real _claim_loop calls this): after the 1st success a lone 401
        # (→ _Revoked, strike 1); after the 2nd success, stop + return None so the
        # claim loop and then the serve loop exit cleanly.
        claim_calls = {"n": 0}

        def claim(backend_id):
            # Sync side_effect (see register_401): first call → lone 401, second →
            # stop + no job so the claim loop then the serve loop exit cleanly.
            claim_calls["n"] += 1
            if claim_calls["n"] == 1:
                raise RelayError("isolated 401", 401)
            stop.set()
            return None  # 204-style: no job; claim loop re-checks stop and exits

        relay.claim.side_effect = claim

        await self._serve(relay, state, pairing, stop=stop, strikes=3)

        self.assertEqual(
            state.remove_calls, 0,
            "a successful register must reset strikes so an isolated later 401 "
            "does not reach the threshold",
        )
        self.assertIsNotNone(state.get_pairing("b1"))
        self.assertEqual(relay.register.await_count, 4)


if __name__ == "__main__":
    unittest.main()
