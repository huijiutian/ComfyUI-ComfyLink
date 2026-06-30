"""Tests for the workflow catalog sync flow (manifest + blobs → R2).

Mirrors the object_info push: the plugin signs a presigned PUT per artifact and
ships the JSON straight to R2 (blobs never go through the relay). These tests pin
the two unit-testable layers:

  * RelayClient.sign_put_workflow: POSTs to /v1/backends/workflows/sign-put with
    the cross-repo body shape and returns (key, url). workflow_id is omitted
    from the body when None (manifest case).
  * routes._do_sync: the upload orchestration — blobs first, manifest LAST, one
    sign+PUT each, application/json, correct bytes, return == #blobs.

What is NOT covered here (integration-tested instead): the /comfylink/sync
handler's pairing-gate (401 when unpaired) and JSON-parse fallback. Those live
inside the route closure registered against ComfyUI's PromptServer, which we
don't spin up at unit level. routes.py imports `from server import PromptServer`
INSIDE register() (not at module top), so importing _do_sync below is safe.

Run:  python -m unittest discover -s tests
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.relay import RelayClient  # noqa: E402
from comfylink.routes import _do_sync  # noqa: E402


class TestSignPutWorkflow(unittest.IsolatedAsyncioTestCase):
    async def test_blob_posts_contract_body_and_returns_key_url(self):
        relay = RelayClient.__new__(RelayClient)

        async def fake_json(method, path, body):
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/v1/backends/workflows/sign-put")
            self.assertEqual(
                body, {"backend_id": "b1", "artifact": "blob", "workflow_id": "wf9"}
            )
            return {"key": "k", "url": "https://r2.example.com/u"}

        relay._json = fake_json
        key, url = await relay.sign_put_workflow("b1", "blob", "wf9")
        self.assertEqual((key, url), ("k", "https://r2.example.com/u"))

    async def test_manifest_omits_workflow_id(self):
        relay = RelayClient.__new__(RelayClient)

        async def fake_json(method, path, body):
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/v1/backends/workflows/sign-put")
            # No workflow_id key for the manifest artifact.
            self.assertEqual(body, {"backend_id": "b1", "artifact": "manifest"})
            self.assertNotIn("workflow_id", body)
            return {"key": "mk", "url": "https://r2.example.com/m"}

        relay._json = fake_json
        key, url = await relay.sign_put_workflow("b1", "manifest")
        self.assertEqual((key, url), ("mk", "https://r2.example.com/m"))


class TestDoSync(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_blobs_first_then_manifest_last(self):
        manifest = {"version": 1, "workflows": [{"id": "wf1"}, {"id": "wf2"}]}
        blobs = {
            "wf1": {"1": {"class_type": "KSampler", "inputs": {}}},
            "wf2": {"2": {"class_type": "LoadImage", "inputs": {}}},
        }

        relay = mock.AsyncMock()
        # Each sign returns a distinct, identifiable URL so we can map PUT → call.
        relay.sign_put_workflow.side_effect = lambda *a: ("key", f"https://r2/{a[1]}-{a[2] if len(a) > 2 else 'm'}")

        n = await _do_sync(relay, manifest, blobs, "backend")

        # Return value == number of blobs.
        self.assertEqual(n, len(blobs))

        # sign_put_workflow: one per blob (in order) then manifest LAST.
        sign_calls = [c.args for c in relay.sign_put_workflow.await_args_list]
        self.assertEqual(
            sign_calls,
            [
                ("backend", "blob", "wf1"),
                ("backend", "blob", "wf2"),
                ("backend", "manifest"),
            ],
        )
        self.assertEqual(sign_calls[-1], ("backend", "manifest"))

        # put_object: N+1 calls, application/json each.
        put_calls = relay.put_object.await_args_list
        self.assertEqual(len(put_calls), len(blobs) + 1)
        for c in put_calls:
            self.assertEqual(c.args[2], "application/json")

        # Each blob PUT carries that blob's JSON bytes...
        first = put_calls[0].args  # (url, data, content_type)
        second = put_calls[1].args
        self.assertEqual(json.loads(first[1].decode()), blobs["wf1"])
        self.assertEqual(json.loads(second[1].decode()), blobs["wf2"])

        # ...and the LAST put_object is the manifest.
        last = put_calls[-1].args
        self.assertEqual(json.loads(last[1].decode()), manifest)

    async def test_no_blobs_still_uploads_manifest(self):
        manifest = {"version": 1, "workflows": []}
        relay = mock.AsyncMock()
        relay.sign_put_workflow.return_value = ("key", "https://r2/m")

        n = await _do_sync(relay, manifest, {}, "backend")

        self.assertEqual(n, 0)
        # Only the manifest sign + PUT happen.
        relay.sign_put_workflow.assert_awaited_once_with("backend", "manifest")
        relay.put_object.assert_awaited_once()
        url, data, content_type = relay.put_object.await_args.args
        self.assertEqual(content_type, "application/json")
        self.assertEqual(json.loads(data.decode()), manifest)


if __name__ == "__main__":
    unittest.main()
