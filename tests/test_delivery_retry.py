"""Tests for RelayClient._deliver — the bounded retry that rides a short relay/
network outage (e.g. a relay redeploy) so a finished job's result/upload isn't
lost to the deploy window.

Contract pinned here:
  * transient blips (connection error, timeout, 502/503/504) are retried with
    backoff, then the call succeeds and returns its value;
  * non-transient RelayErrors (4xx auth/revoke, 400, SSRF/validation status 0)
    propagate IMMEDIATELY with no retry — a real failure/unpair is never masked;
  * an outage longer than the budget gives up and re-raises the last error;
  * result()/sign_upload() route through _deliver (delivery is guarded).

asyncio.sleep is patched out so the backoff doesn't actually wait.

Run:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import relay as relay_mod  # noqa: E402
from comfylink.relay import RelayClient, RelayError  # noqa: E402


def _bare_client():
    """A RelayClient with no session/auth — _deliver only drives the thunk."""
    return RelayClient.__new__(RelayClient)


class TestDeliverRetry(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Patch the module's asyncio.sleep so backoff is instant; record waits.
        self.sleeps = []

        async def fake_sleep(d):
            self.sleeps.append(d)

        self._patch = mock.patch.object(relay_mod.asyncio, "sleep", fake_sleep)
        self._patch.start()

    async def asyncTearDown(self):
        self._patch.stop()

    async def test_retries_transient_then_succeeds(self):
        relay = _bare_client()
        calls = {"n": 0}

        async def thunk():
            calls["n"] += 1
            if calls["n"] < 3:
                raise aiohttp.ClientConnectionError("relay restarting")
            return "ok"

        out = await relay._deliver("t", thunk)
        self.assertEqual(out, "ok")
        self.assertEqual(calls["n"], 3)          # 2 failures + 1 success
        self.assertEqual(len(self.sleeps), 2)    # slept before each retry
        # Exponential backoff: 1s then 2s.
        self.assertEqual(self.sleeps, [1.0, 2.0])

    async def test_retries_timeout_and_5xx(self):
        relay = _bare_client()
        seq = [asyncio.TimeoutError(),
               RelayError("boom", 503),
               RelayError("boom", 502),
               "done"]

        async def thunk():
            x = seq.pop(0)
            if isinstance(x, BaseException):
                raise x
            return x

        self.assertEqual(await relay._deliver("t", thunk), "done")

    async def test_4xx_propagates_immediately(self):
        relay = _bare_client()
        calls = {"n": 0}

        async def thunk():
            calls["n"] += 1
            raise RelayError("unauthorized", 401)

        with self.assertRaises(RelayError) as cm:
            await relay._deliver("t", thunk)
        self.assertEqual(cm.exception.status, 401)
        self.assertEqual(calls["n"], 1)          # no retry
        self.assertEqual(self.sleeps, [])

    async def test_validation_status_zero_not_retried(self):
        # RelayError(status=0) is a permanent SSRF/validation reject, not a blip.
        relay = _bare_client()
        calls = {"n": 0}

        async def thunk():
            calls["n"] += 1
            raise RelayError("refusing non-https URL")

        with self.assertRaises(RelayError):
            await relay._deliver("t", thunk)
        self.assertEqual(calls["n"], 1)

    async def test_gives_up_after_budget_and_raises_last(self):
        relay = _bare_client()
        attempts = {"n": 0}

        async def thunk():
            attempts["n"] += 1
            raise RelayError("still down", 503)

        with self.assertRaises(RelayError) as cm:
            await relay._deliver("t", thunk)
        self.assertEqual(cm.exception.status, 503)
        # Cumulative backoff is bounded by _RETRY_MAX_ELAPSED.
        self.assertLessEqual(sum(self.sleeps), relay_mod._RETRY_MAX_ELAPSED)
        # It retried several times (1+2+4+8+15*n ~ up to 90s), not just once.
        self.assertGreater(attempts["n"], 3)

    async def test_result_routes_through_deliver(self):
        relay = _bare_client()
        calls = {"n": 0}

        async def flaky_json(method, path, body):
            calls["n"] += 1
            if calls["n"] == 1:
                raise aiohttp.ClientConnectionError("mid-deploy")
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/v1/jobs/j1/result")
            self.assertEqual(body["status"], "done")
            return {}

        relay._json = flaky_json
        await relay.result("j1", "done", [{"r2_key": "k"}], total_bytes=10)
        self.assertEqual(calls["n"], 2)          # retried once, then delivered

    async def test_sign_upload_routes_through_deliver(self):
        relay = _bare_client()
        calls = {"n": 0}

        async def flaky_json(method, path, body):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RelayError("bad gateway", 502)
            return {"r2_key": "k1", "url": "https://r2.example.com/u"}

        relay._json = flaky_json
        key, url = await relay.sign_upload("j1", "image", "a.png", "image/png")
        self.assertEqual((key, url), ("k1", "https://r2.example.com/u"))
        self.assertEqual(calls["n"], 2)


if __name__ == "__main__":
    unittest.main()
