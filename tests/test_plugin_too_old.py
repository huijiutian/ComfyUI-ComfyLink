"""Tests for the relay's "your plugin is too old" hard block.

The relay may answer register/heartbeat with 403 + this fixed body:

    {"error": "plugin version too old", "error_code": "plugin_too_old",
     "min_version": "0.3.0", "update_url": "https://github.com/..."}

The DANGEROUS default behaviour would be for that 403 to flow into the existing
"401/403 == the device was unpaired" mapping: _serve_pairing removes the pairing
for good after REVOKED_CONFIRM_STRIKES consecutive rejections, so a version block
would permanently self-destruct a perfectly valid pairing in ~15 seconds. These
tests pin the opposite:

  * _check lifts error_code out of a JSON error body into RelayError.code, and
    degrades to "" (never raising) on a non-JSON / empty / non-dict body;
  * _register raises _PluginTooOld (NOT _Revoked) on 403 + plugin_too_old, and
    still raises _Revoked on a plain/other-coded 403 and on 401;
  * MANY consecutive plugin_too_old 403s leave the pairing intact — remove_pairing
    is never called, no strike is ever recorded;
  * no job is claimed while blocked;
  * a block armed AFTER we registered (so only the heartbeat ever sees it — the
    relay does NOT gate the hot claim long-poll) stops the claim loop instead of
    letting an always-on ComfyUI keep claiming until it restarts;
  * a plain heartbeat error is still swallowed and claiming continues;
  * STATUS.snapshot() carries plugin_too_old / plugin_min_version /
    plugin_update_url, and they are cleared by a successful register.

Run:  python -m unittest discover -s tests
"""

import asyncio
import json
import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.config import Pairing  # noqa: E402
from comfylink.relay import RelayError, _check  # noqa: E402
from comfylink.status import Status  # noqa: E402

# The REAL asyncio.sleep, captured before any test patches worker.asyncio.sleep
# (that patch hits the asyncio module itself, so it is global for its duration).
# Mock relay calls use it to yield control, otherwise a loop whose every await is
# a non-yielding AsyncMock would starve the event loop and hang the test.
_real_sleep = asyncio.sleep

MIN_VERSION = "0.3.0"
UPDATE_URL = "https://github.com/huijiutian/ComfyUI-ComfyLink"

TOO_OLD_BODY = json.dumps({
    "error": "plugin version too old",
    "error_code": "plugin_too_old",
    "min_version": MIN_VERSION,
    "update_url": UPDATE_URL,
})


def _too_old_error() -> RelayError:
    """The RelayError the real _check would build from a plugin_too_old 403."""
    return RelayError(
        f"relay POST /v1/backends/register -> 403: {TOO_OLD_BODY}",
        403,
        code="plugin_too_old",
        payload=json.loads(TOO_OLD_BODY),
    )


class _FakeResponse:
    """Minimal stand-in for aiohttp.ClientResponse as _check consumes it."""

    def __init__(self, status: int, body: str, method: str = "POST",
                 path: str = "/v1/backends/register"):
        self.status = status
        self._body = body
        self.reads = 0
        self.request_info = types.SimpleNamespace(method=method)
        self.url = types.SimpleNamespace(path=path)

    async def text(self) -> str:
        self.reads += 1
        return self._body


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
    relay = mock.AsyncMock()
    relay.upload_object_info.return_value = None
    return relay


def _mock_comfy():
    comfy = mock.AsyncMock()
    comfy.object_info.return_value = {}
    return comfy


class TestCheckErrorCode(unittest.IsolatedAsyncioTestCase):
    async def test_json_error_body_populates_code_and_payload(self):
        r = _FakeResponse(403, TOO_OLD_BODY)
        with self.assertRaises(RelayError) as cm:
            await _check(r)
        e = cm.exception
        self.assertEqual(e.status, 403)
        self.assertEqual(e.code, "plugin_too_old")
        self.assertEqual(e.payload.get("min_version"), MIN_VERSION)
        self.assertEqual(e.payload.get("update_url"), UPDATE_URL)
        # The human message is unchanged: prefix + the raw body.
        self.assertEqual(
            str(e), f"relay POST /v1/backends/register -> 403: {TOO_OLD_BODY}")
        # Body read exactly once (aiohttp cannot serve it twice).
        self.assertEqual(r.reads, 1)

    async def test_non_json_empty_and_non_dict_bodies_degrade_to_empty_code(self):
        for body in ("<html>502 Bad Gateway</html>", "", "   ",
                     "[1, 2, 3]", '"just a string"', "null", "12"):
            with self.subTest(body=body):
                with self.assertRaises(RelayError) as cm:
                    await _check(_FakeResponse(500, body))
                e = cm.exception
                self.assertEqual(e.code, "", "unparseable body must yield code ''")
                self.assertEqual(e.payload, {})
                self.assertIn(body, str(e))

    async def test_json_object_without_error_code_yields_empty_code(self):
        with self.assertRaises(RelayError) as cm:
            await _check(_FakeResponse(403, '{"error": "forbidden"}'))
        self.assertEqual(cm.exception.code, "")

    async def test_non_string_error_code_yields_empty_code(self):
        with self.assertRaises(RelayError) as cm:
            await _check(_FakeResponse(403, '{"error_code": 42}'))
        self.assertEqual(cm.exception.code, "")

    async def test_success_status_does_not_raise(self):
        await _check(_FakeResponse(204, ""))


class TestRegisterDiscrimination(unittest.IsolatedAsyncioTestCase):
    async def _register(self, err):
        relay = _mock_relay()
        relay.register.side_effect = err
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "STATUS", Status()):
            await worker._register(relay, _mock_comfy(), pairing)

    async def test_plugin_too_old_403_raises_plugin_too_old(self):
        with self.assertRaises(worker._PluginTooOld) as cm:
            await self._register(_too_old_error())
        self.assertEqual(cm.exception.min_version, MIN_VERSION)
        self.assertEqual(cm.exception.update_url, UPDATE_URL)

    async def test_plain_403_still_raises_revoked(self):
        with self.assertRaises(worker._Revoked):
            await self._register(RelayError("forbidden", 403))

    async def test_403_with_other_code_still_raises_revoked(self):
        with self.assertRaises(worker._Revoked):
            await self._register(
                RelayError("nope", 403, code="device_revoked",
                           payload={"error_code": "device_revoked"}))

    async def test_401_still_raises_revoked(self):
        with self.assertRaises(worker._Revoked):
            await self._register(RelayError("device gone", 401))

    async def test_401_with_plugin_too_old_code_is_still_revoked(self):
        # The contract is 403; a 401 is an auth answer whatever it claims.
        with self.assertRaises(worker._Revoked):
            await self._register(
                RelayError("weird", 401, code="plugin_too_old",
                           payload={"error_code": "plugin_too_old"}))

    async def test_other_status_propagates_unchanged(self):
        with self.assertRaises(RelayError):
            await self._register(RelayError("boom", 503))


class TestClaimLoopDiscrimination(unittest.IsolatedAsyncioTestCase):
    async def _claim_loop(self, err):
        relay = _mock_relay()
        relay.claim.side_effect = err
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker.asyncio, "sleep", mock.AsyncMock()):
            await asyncio.wait_for(
                worker._claim_loop(relay, mock.MagicMock(), pairing,
                                   asyncio.Lock(), None),
                timeout=2,
            )

    async def test_plugin_too_old_403_raises_plugin_too_old(self):
        with self.assertRaises(worker._PluginTooOld):
            await self._claim_loop(_too_old_error())

    async def test_plain_403_still_raises_revoked(self):
        with self.assertRaises(worker._Revoked):
            await self._claim_loop(RelayError("forbidden", 403))


class _ServeHarness:
    """Shared driver for the real _serve_pairing with mocked collaborators."""

    async def _serve(self, relay, state, pairing, stop=None, status=None,
                     strikes=3, real_heartbeat=False):
        """Drive the real _serve_pairing with mocked collaborators.

        real_heartbeat=True leaves _heartbeat_loop unpatched so the heartbeat →
        latch → claim-loop wiring is exercised end-to-end.
        """
        comfy = _mock_comfy()
        hb_patch = (
            mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0)
            if real_heartbeat
            else mock.patch.object(worker, "_heartbeat_loop", mock.AsyncMock())
        )
        with mock.patch.object(worker, "REVOKED_CONFIRM_STRIKES", strikes), \
                mock.patch.object(worker, "RelayClient", return_value=relay), \
                mock.patch.object(worker, "ComfyClient", return_value=comfy), \
                mock.patch.object(worker, "Worker"), \
                mock.patch.object(worker, "STATUS", status or Status()), \
                mock.patch.object(worker, "STATE", state), \
                hb_patch, \
                mock.patch.object(worker.asyncio, "sleep", mock.AsyncMock()):
            await asyncio.wait_for(
                worker._serve_pairing(
                    pairing, asyncio.Lock(), None, "http://comfy", stop,
                    {pairing.backend_id},  # swept → skip orphan sweep
                ),
                timeout=5,
            )


class TestServePairingNeverUnpairs(_ServeHarness, unittest.IsolatedAsyncioTestCase):
    """The critical regression: a version block must NEVER cost the pairing."""

    async def test_many_consecutive_too_old_403s_keep_the_pairing(self):
        strikes = 3
        rounds = strikes + 2  # well past the unpair threshold
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        stop = asyncio.Event()
        relay = _mock_relay()
        status = Status()

        calls = {"n": 0}

        def register(backend_id, name):
            calls["n"] += 1
            if calls["n"] >= rounds:
                stop.set()
            raise _too_old_error()

        relay.register.side_effect = register

        await self._serve(relay, state, pairing, stop=stop, status=status,
                          strikes=strikes)

        self.assertEqual(state.remove_calls, 0,
                         "a plugin_too_old 403 must NEVER unpair the device")
        self.assertIsNotNone(state.get_pairing("b1"), "pairing must survive")
        self.assertEqual(relay.register.await_count, rounds,
                         "should keep retrying while blocked")
        # No job is ever claimed while blocked (register raised first).
        relay.claim.assert_not_awaited()
        snap = status.snapshot()
        self.assertTrue(snap["plugin_too_old"])
        self.assertEqual(snap["plugin_min_version"], MIN_VERSION)
        self.assertEqual(snap["plugin_update_url"], UPDATE_URL)
        self.assertEqual(snap["state"], "error")
        self.assertTrue(snap["error"], "a human message must be shown")

    async def test_too_old_does_not_consume_or_reset_revoke_strikes(self):
        # 401, then two too-old 403s, then a 401. If the too-old blocks counted
        # as strikes the 3rd rejection would unpair; if they RESET the counter the
        # final 401 would be strike 1. Neither may happen: strikes stay at 1 then
        # 2, so the pairing survives and a genuine revoke is still on track.
        strikes = 3
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        stop = asyncio.Event()
        relay = _mock_relay()

        seq = [
            RelayError("blip 401", 401),
            _too_old_error(),
            _too_old_error(),
            RelayError("blip 401", 401),
        ]
        calls = {"n": 0}

        def register(backend_id, name):
            i = calls["n"]
            calls["n"] += 1
            if calls["n"] >= len(seq):
                stop.set()
            raise seq[i]

        relay.register.side_effect = register

        await self._serve(relay, state, pairing, stop=stop, strikes=strikes)

        self.assertEqual(state.remove_calls, 0)
        self.assertIsNotNone(state.get_pairing("b1"))
        self.assertEqual(relay.register.await_count, len(seq))

    async def test_successful_register_clears_the_blocked_status(self):
        # Blocked twice, then the relay serves us again → the red-bar fields must
        # be cleared so the panel stops shouting.
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        stop = asyncio.Event()
        relay = _mock_relay()
        status = Status()

        calls = {"n": 0}

        def register(backend_id, name):
            calls["n"] += 1
            if calls["n"] <= 2:
                raise _too_old_error()
            return {"account": "a@example.com"}

        relay.register.side_effect = register

        def claim(backend_id):
            stop.set()
            return None  # 204-style: no job; both loops then exit cleanly

        relay.claim.side_effect = claim

        await self._serve(relay, state, pairing, stop=stop, status=status)

        self.assertEqual(state.remove_calls, 0)
        snap = status.snapshot()
        self.assertFalse(snap["plugin_too_old"])
        self.assertEqual(snap["plugin_min_version"], "")
        self.assertEqual(snap["plugin_update_url"], "")


class TestHeartbeatBlockStopsClaiming(_ServeHarness,
                                      unittest.IsolatedAsyncioTestCase):
    """A block armed AFTER we registered must still stop us claiming.

    The relay gates register + heartbeat but NOT the hot /v1/jobs/claim long-poll,
    so for an always-on ComfyUI that registered before the fuse was lit, the
    heartbeat is the ONLY place the block is ever observed. Without the latch it
    was swallowed at debug level and the machine kept claiming jobs forever.
    """

    async def test_heartbeat_403_stops_the_claim_loop_and_keeps_the_pairing(self):
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        stop = asyncio.Event()
        relay = _mock_relay()
        status = Status()

        beat = asyncio.Event()

        def heartbeat(backend_id, object_info_hash="", object_info_synced_at=0.0,
                      foreign_queue_depth=-1, loaded_checkpoint=""):
            # Sync side_effect on an AsyncMock: raising here drives the awaited
            # call deterministically. The loop's except arms the latch and
            # returns with no await in between, so once the claim task is
            # resumed below the latch is guaranteed to be set.
            beat.set()
            raise _too_old_error()

        relay.heartbeat.side_effect = heartbeat

        # register: succeed first (we are serving normally, the fuse is lit
        # later), then block on the retry so the test terminates in a blocked
        # state rather than looping.
        reg = {"n": 0}

        def register(backend_id, name):
            reg["n"] += 1
            if reg["n"] == 1:
                return {"account": "a@example.com"}
            stop.set()
            raise _too_old_error()

        relay.register.side_effect = register

        async def claim(backend_id):
            # Stand in for the ~28s long-poll: return only once the heartbeat has
            # been refused, i.e. this is the LAST claim before the latch bites.
            await beat.wait()
            return None

        relay.claim.side_effect = claim

        await self._serve(relay, state, pairing, stop=stop, status=status,
                          real_heartbeat=True)

        self.assertEqual(state.remove_calls, 0,
                         "a heartbeat version block must NEVER unpair")
        self.assertIsNotNone(state.get_pairing("b1"), "pairing must survive")
        self.assertEqual(
            relay.claim.await_count, 1,
            "claiming must stop at the top of the next iteration, not continue",
        )
        snap = status.snapshot()
        self.assertTrue(snap["plugin_too_old"])
        self.assertEqual(snap["plugin_min_version"], MIN_VERSION)
        self.assertEqual(snap["state"], "error")

    async def test_plain_heartbeat_error_is_still_swallowed(self):
        # Regression guard: making heartbeat failures fatal would be a serious
        # regression (a blip would stop the machine claiming). A non-too_old
        # heartbeat error must stay a debug no-op while claiming continues.
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        stop = asyncio.Event()
        relay = _mock_relay()
        status = Status()

        beats = {"n": 0}

        async def heartbeat(backend_id, object_info_hash="", object_info_synced_at=0.0,
                      foreign_queue_depth=-1, loaded_checkpoint=""):
            beats["n"] += 1
            await _real_sleep(0)  # yield so the claim loop makes progress
            raise RelayError("relay hiccup", 500)

        relay.heartbeat.side_effect = heartbeat
        relay.register.side_effect = lambda b, n: {"account": "a@example.com"}

        claims = {"n": 0}

        async def claim(backend_id):
            claims["n"] += 1
            await _real_sleep(0)
            if claims["n"] >= 3:
                stop.set()
            return None

        relay.claim.side_effect = claim

        await self._serve(relay, state, pairing, stop=stop, status=status,
                          real_heartbeat=True)

        self.assertEqual(relay.claim.await_count, 3,
                         "claiming must continue through heartbeat errors")
        self.assertGreaterEqual(beats["n"], 1, "the heartbeat must keep beating")
        self.assertEqual(state.remove_calls, 0)
        self.assertFalse(status.snapshot()["plugin_too_old"])

    async def test_claiming_resumes_after_the_block_clears(self):
        # Blocked via the heartbeat, then the operator rolls `min` back: the next
        # register succeeds, the latch is cleared, and claiming resumes.
        pairing = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([pairing])
        stop = asyncio.Event()
        relay = _mock_relay()
        status = Status()

        beat = asyncio.Event()
        beats = {"n": 0}

        async def heartbeat(backend_id, object_info_hash="", object_info_synced_at=0.0,
                      foreign_queue_depth=-1, loaded_checkpoint=""):
            beats["n"] += 1
            if beats["n"] == 1:
                beat.set()
                raise _too_old_error()
            await _real_sleep(0)  # unblocked now: healthy beats, yield each time

        relay.heartbeat.side_effect = heartbeat
        relay.register.side_effect = lambda b, n: {"account": "a@example.com"}

        claims = {"n": 0}

        async def claim(backend_id):
            claims["n"] += 1
            if claims["n"] == 1:
                await beat.wait()  # last claim before the latch bites
                return None
            await _real_sleep(0)
            stop.set()  # claiming resumed after the block cleared → done
            return None

        relay.claim.side_effect = claim

        await self._serve(relay, state, pairing, stop=stop, status=status,
                          real_heartbeat=True)

        self.assertEqual(relay.claim.await_count, 2,
                         "claiming must resume once the block clears")
        self.assertEqual(relay.register.await_count, 2)
        self.assertEqual(state.remove_calls, 0)
        snap = status.snapshot()
        self.assertFalse(snap["plugin_too_old"], "the red bar must be cleared")
        self.assertEqual(snap["plugin_min_version"], "")


class TestTooOldLatch(unittest.TestCase):
    def test_check_raises_only_when_armed_and_carries_the_details(self):
        latch = worker._TooOldLatch()
        self.assertFalse(latch.armed)
        latch.check()  # no-op
        latch.arm(worker._PluginTooOld(MIN_VERSION, UPDATE_URL))
        self.assertTrue(latch.armed)
        with self.assertRaises(worker._PluginTooOld) as cm:
            latch.check()
        self.assertEqual(cm.exception.min_version, MIN_VERSION)
        self.assertEqual(cm.exception.update_url, UPDATE_URL)
        latch.clear()
        self.assertFalse(latch.armed)
        latch.check()  # no-op again


class TestStatusSnapshot(unittest.TestCase):
    def test_keys_always_present_with_defaults(self):
        snap = Status().snapshot()
        for key in ("plugin_too_old", "plugin_min_version", "plugin_update_url"):
            self.assertIn(key, snap)
        self.assertIs(snap["plugin_too_old"], False)
        self.assertEqual(snap["plugin_min_version"], "")
        self.assertEqual(snap["plugin_update_url"], "")

    def test_set_then_snapshot_round_trips(self):
        st = Status()
        st.set(plugin_too_old=True, plugin_min_version=MIN_VERSION,
               plugin_update_url=UPDATE_URL)
        snap = st.snapshot()
        self.assertTrue(snap["plugin_too_old"])
        self.assertEqual(snap["plugin_min_version"], MIN_VERSION)
        self.assertEqual(snap["plugin_update_url"], UPDATE_URL)


if __name__ == "__main__":
    unittest.main()
