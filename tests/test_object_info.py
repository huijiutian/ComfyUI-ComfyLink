"""Tests for the object_info R2 upload flow.

object_info is no longer POSTed to the relay as a blob (egress fix). Instead the
plugin signs a presigned PUT and ships the JSON straight to R2. These tests pin
that wiring on both layers:

  * RelayClient.upload_object_info: sign first, then PUT the JSON with
    Content-Type application/json.
  * worker._register: on upload failure the backend still goes online (params
    are non-critical) and never crashes.

Run:  python -m unittest discover -s tests
"""

import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.relay import RelayClient  # noqa: E402


class TestUploadObjectInfo(unittest.IsolatedAsyncioTestCase):
    async def test_signs_then_puts_json(self):
        # Bare RelayClient; we stub the two collaborators upload_object_info uses.
        relay = RelayClient.__new__(RelayClient)
        calls = []

        async def fake_sign(backend_id):
            calls.append(("sign", backend_id))
            return "object-info/b1.json", "https://r2.example.com/signed-put"

        async def fake_put(url, data, content_type):
            calls.append(("put", url, data, content_type))

        relay.sign_object_info = fake_sign
        relay.put_object = fake_put

        oi = {"KSampler": {"input": {}}}
        await relay.upload_object_info("b1", oi)

        # sign happened first, then the PUT.
        self.assertEqual(calls[0], ("sign", "b1"))
        _, url, data, content_type = calls[1]
        self.assertEqual(url, "https://r2.example.com/signed-put")
        self.assertEqual(content_type, "application/json")
        # The body is the JSON-encoded object_info as bytes.
        self.assertEqual(json.loads(data.decode()), oi)

    async def test_sign_uses_json_post_and_returns_key_url(self):
        relay = RelayClient.__new__(RelayClient)

        async def fake_json(method, path, body):
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/v1/backends/object-info/sign")
            self.assertEqual(body, {"backend_id": "b9"})
            return {"key": "k", "url": "https://r2.example.com/u"}

        relay._json = fake_json
        key, url = await relay.sign_object_info("b9")
        self.assertEqual((key, url), ("k", "https://r2.example.com/u"))


class TestRegisterToleratesUploadFailure(unittest.IsolatedAsyncioTestCase):
    async def test_online_even_when_upload_fails(self):
        from comfylink import worker

        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = {"A": {}, "B": {}}
        # Upload blows up (e.g. relay 503 because R2 unconfigured).
        relay.upload_object_info.side_effect = RuntimeError("503 R2 not configured")

        with mock.patch.object(worker, "STATUS") as status:
            # Should NOT raise — failure is tolerated.
            await worker._register(relay, comfy)

        relay.register.assert_awaited_once()
        relay.upload_object_info.assert_awaited_once()
        # Last STATUS.set still flips us online.
        states = [c.kwargs.get("state") for c in status.set.call_args_list]
        self.assertEqual(states[-1], "online")


if __name__ == "__main__":
    unittest.main()
