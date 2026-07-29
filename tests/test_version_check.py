"""Tests for the plugin self-update check (comfylink/version_check.py) and its
surfacing in the /comfylink/status route.

Three concerns are pinned:
  * _is_older — the semver comparison used to decide "am I out of date"
    (mirrors app/lib/core/version/semver.dart's isOlder). Unparseable => False.
  * get_update_info — throttling + negative caching: within the TTL the network
    is NOT hit again, and a failing fetch is swallowed (empty result, no raise)
    while STILL advancing the timestamp so it isn't retried every poll.
  * _status route — the `update` field is transparently passed through, and a
    failing update check never breaks the status response.

Run:  python -m unittest discover -s tests
"""

import json
import os
import sys
import types
import unittest
from unittest import mock

from aiohttp import web

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import version_check  # noqa: E402


# --- _is_older ---------------------------------------------------------------


class TestIsOlder(unittest.TestCase):
    def test_strictly_older_is_true(self):
        self.assertTrue(version_check._is_older("0.2.0", "0.3.0"))
        self.assertTrue(version_check._is_older("0.2.0", "1.0.0"))
        self.assertTrue(version_check._is_older("0.2.0", "0.2.1"))

    def test_equal_is_not_older(self):
        self.assertFalse(version_check._is_older("0.2.0", "0.2.0"))

    def test_newer_is_not_older(self):
        self.assertFalse(version_check._is_older("0.3.0", "0.2.0"))
        self.assertFalse(version_check._is_older("1.0.0", "0.9.9"))

    def test_tolerant_of_v_prefix_and_short_forms(self):
        # Leading v and missing trailing segments ("1.0" -> 1.0.0).
        self.assertTrue(version_check._is_older("v1.0", "v1.0.1"))
        self.assertFalse(version_check._is_older("1.0.1", "1.0"))
        self.assertFalse(version_check._is_older("1.0", "1.0.0"))
        # pre-release / build metadata is stripped before comparing.
        self.assertTrue(version_check._is_older("1.0.0-rc1", "1.0.1+build9"))

    def test_unparseable_returns_false(self):
        # Conservative: never nag on a version string we can't understand.
        self.assertFalse(version_check._is_older("abc", "1.0.0"))
        self.assertFalse(version_check._is_older("1.0.0", "not-a-version"))
        self.assertFalse(version_check._is_older("", "1.0.0"))
        self.assertFalse(version_check._is_older("1.0.0", ""))
        self.assertFalse(version_check._is_older("1.0.0.0", "2.0.0"))  # >3 segments


# --- get_update_info: throttle + negative cache ------------------------------


class TestGetUpdateInfo(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Reset the module-level negative cache before each test.
        version_check._last_check = None
        version_check._cached = dict(version_check._EMPTY)

    async def test_success_is_cached_within_ttl(self):
        result = {"available": True, "latest": "0.3.0",
                  "below_min": False, "url": "https://x"}
        fetch = mock.AsyncMock(return_value=result)
        with mock.patch.object(version_check, "_fetch", fetch):
            first = await version_check.get_update_info("https://relay")
            second = await version_check.get_update_info("https://relay")
        self.assertEqual(first, result)
        self.assertEqual(second, result)
        # Second call served from cache — network hit exactly once.
        self.assertEqual(fetch.await_count, 1)
        # Returns a fresh dict each call (mutating it can't corrupt the cache).
        first["available"] = False
        self.assertTrue(version_check._cached["available"])

    async def test_expired_ttl_refetches(self):
        fetch = mock.AsyncMock(return_value=dict(version_check._EMPTY))
        with mock.patch.object(version_check, "_fetch", fetch):
            await version_check.get_update_info("https://relay")
            # Force the cached stamp to look old.
            version_check._last_check -= version_check._TTL + 1
            await version_check.get_update_info("https://relay")
        self.assertEqual(fetch.await_count, 2)

    async def test_failure_is_swallowed_and_negatively_cached(self):
        fetch = mock.AsyncMock(side_effect=RuntimeError("offline"))
        with mock.patch.object(version_check, "_fetch", fetch):
            first = await version_check.get_update_info("https://relay")
            await version_check.get_update_info("https://relay")
        # No raise; degraded to the empty "no update info" result.
        self.assertEqual(first, dict(version_check._EMPTY))
        self.assertFalse(first["available"])
        # Negative cache: the failing endpoint is NOT retried on the next poll.
        self.assertEqual(fetch.await_count, 1)
        self.assertIsNotNone(version_check._last_check)


# --- _status route surfaces the update field ---------------------------------


def _register_and_get_handler(path, method="GET"):
    """Register the panel routes onto a throwaway RouteTableDef and hand back the
    matching handler closure (no ComfyUI PromptServer needed)."""
    table = web.RouteTableDef()
    fake_server = types.ModuleType("server")
    fake_server.PromptServer = types.SimpleNamespace(
        instance=types.SimpleNamespace(routes=table)
    )
    sys.modules["server"] = fake_server
    from comfylink import routes as routes_mod

    routes_mod.register()
    for rd in table:
        if rd.path == path and rd.method == method:
            return rd.handler
    raise AssertionError(f"handler for {method} {path} not registered")


class _State:
    paired = False
    backend_name = "box"
    pairings: list = []


class TestStatusUpdateField(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = staticmethod(_register_and_get_handler("/comfylink/status"))

    def setUp(self):
        from comfylink import routes as routes_mod

        self.routes_mod = routes_mod

    async def _call(self, get_update_info):
        with mock.patch.object(self.routes_mod, "STATE", _State()), \
             mock.patch.object(self.routes_mod, "get_update_info", get_update_info):
            return await self.handler(object())

    async def test_update_field_passed_through(self):
        upd = {"available": True, "latest": "0.3.0",
               "below_min": True, "url": "https://get"}
        resp = await self._call(mock.AsyncMock(return_value=upd))
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.text)
        self.assertEqual(body["update"], upd)

    async def test_update_check_failure_keeps_status_ok(self):
        # Even if the update check raises, status must still return 200 with a
        # safe empty update block (never a 500).
        resp = await self._call(mock.AsyncMock(side_effect=RuntimeError("boom")))
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.text)
        self.assertFalse(body["update"]["available"])
        self.assertEqual(body["update"]["latest"], "")
        # Core status fields still present.
        self.assertIn("version", body)
        self.assertIn("paired", body)


if __name__ == "__main__":
    unittest.main()
