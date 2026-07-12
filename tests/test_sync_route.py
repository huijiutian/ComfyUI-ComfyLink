"""Tests for the /comfylink/sync route handler's multi-account fan-out.

The sync route targets a SET of accounts (the ones checked in the panel) instead
of a single one. The body carries `backend_ids` (a list); the handler pushes the
same manifest + blobs once per named pairing and reports a per-account result.
These tests pin the request-validation and fan-out contract:

  * missing / empty / blank-only `backend_ids` → 400 {"error": "backend_ids required"}
  * a mix of one valid id + one unknown id → 200, ok:true, per-account results
    (valid synced, unknown flagged) — the valid one's _do_sync runs
  * all-unknown ids → 502 {"ok": false, "results": [...]}

routes.register() imports `from server import PromptServer` INSIDE the function,
so we hand it a fake `server` module whose PromptServer.instance.routes is a real
aiohttp RouteTableDef; register() decorates the closures onto it and we pull the
`_sync` handler back out to invoke it directly (no ComfyUI PromptServer needed).

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

from comfylink.config import Pairing  # noqa: E402


def _register_and_get_handler(path, method="POST"):
    """Register the panel routes against a throwaway RouteTableDef and return the
    matching handler closure."""
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


class _Req:
    """Minimal stand-in: the handler only awaits request.json()."""

    def __init__(self, data):
        self._data = data

    async def json(self):
        return self._data


class _State:
    def __init__(self, pairings):
        self.pairings = list(pairings)

    @property
    def paired(self):
        return len(self.pairings) > 0


class TestSyncRouteMultiAccount(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.handler = staticmethod(_register_and_get_handler("/comfylink/sync"))

    def setUp(self):
        from comfylink import routes as routes_mod

        self.routes_mod = routes_mod
        self.state = _State([
            Pairing(backend_id="b1", device_token="t1", account="a@example.com"),
        ])

    async def _call(self, body, do_sync=None):
        """Invoke the handler with STATE patched. If `do_sync` is given, patch
        routes._do_sync with it so no real relay/network is touched."""
        ctxs = [mock.patch.object(self.routes_mod, "STATE", self.state)]
        if do_sync is not None:
            ctxs.append(mock.patch.object(self.routes_mod, "_do_sync", do_sync))
        with ctxs[0]:
            if len(ctxs) > 1:
                with ctxs[1]:
                    return await self.handler(_Req(body))
            return await self.handler(_Req(body))

    # --- validation branches -------------------------------------------------

    async def test_missing_backend_ids_returns_400(self):
        resp = await self._call({"manifest": {}, "blobs": {}})
        self.assertEqual(resp.status, 400)
        self.assertEqual(
            json.loads(resp.text),
            {"ok": False, "error": "backend_ids required"},
        )

    async def test_empty_backend_ids_returns_400(self):
        resp = await self._call({"manifest": {}, "backend_ids": []})
        self.assertEqual(resp.status, 400)
        self.assertEqual(json.loads(resp.text)["error"], "backend_ids required")

    async def test_non_list_backend_ids_returns_400(self):
        resp = await self._call({"manifest": {}, "backend_ids": "b1"})
        self.assertEqual(resp.status, 400)
        self.assertEqual(json.loads(resp.text)["error"], "backend_ids required")

    async def test_blank_only_backend_ids_returns_400(self):
        resp = await self._call({"manifest": {}, "backend_ids": ["  ", ""]})
        self.assertEqual(resp.status, 400)
        self.assertEqual(json.loads(resp.text)["error"], "backend_ids required")

    # --- fan-out branches ----------------------------------------------------

    async def test_mixed_valid_and_unknown_syncs_valid(self):
        do_sync = mock.AsyncMock(return_value=1)
        resp = await self._call(
            {"manifest": {}, "blobs": {"w": {}}, "backend_ids": ["b1", "nope"]},
            do_sync=do_sync,
        )
        self.assertEqual(resp.status, 200)
        body = json.loads(resp.text)
        self.assertTrue(body["ok"])
        self.assertEqual(body["uploaded"], 1)
        # The valid pairing's _do_sync ran exactly once (the unknown id doesn't).
        self.assertEqual(do_sync.await_count, 1)
        results = {r["backend_id"]: r for r in body["results"]}
        self.assertEqual(
            results["b1"],
            {"backend_id": "b1", "account": "a@example.com", "ok": True},
        )
        self.assertFalse(results["nope"]["ok"])
        self.assertEqual(results["nope"]["error"], "unknown account")

    async def test_all_unknown_returns_502(self):
        do_sync = mock.AsyncMock(return_value=1)
        resp = await self._call(
            {"manifest": {}, "backend_ids": ["x", "y"]}, do_sync=do_sync
        )
        self.assertEqual(resp.status, 502)
        body = json.loads(resp.text)
        self.assertFalse(body["ok"])
        self.assertEqual(body["error"], "sync failed")
        self.assertEqual({r["backend_id"] for r in body["results"]}, {"x", "y"})
        self.assertTrue(all(not r["ok"] for r in body["results"]))
        # No valid pairing → _do_sync never called.
        do_sync.assert_not_awaited()

    async def test_do_sync_failure_reported_and_502_when_all_fail(self):
        do_sync = mock.AsyncMock(side_effect=RuntimeError("boom"))
        resp = await self._call(
            {"manifest": {}, "blobs": {"w": {}}, "backend_ids": ["b1"]},
            do_sync=do_sync,
        )
        self.assertEqual(resp.status, 502)
        body = json.loads(resp.text)
        self.assertFalse(body["ok"])
        results = body["results"]
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0]["ok"])
        self.assertIn("boom", results[0]["error"])


if __name__ == "__main__":
    unittest.main()
