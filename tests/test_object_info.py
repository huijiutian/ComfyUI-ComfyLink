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
from comfylink.worker import object_info_hash  # noqa: E402


class TestObjectInfoHash(unittest.TestCase):
    def test_stable_for_equal_dicts(self):
        a = {"KSampler": {"input": {"x": 1}}, "LoadImage": {}}
        # Same content, different insertion order — sort_keys makes it identical.
        b = {"LoadImage": {}, "KSampler": {"input": {"x": 1}}}
        self.assertEqual(object_info_hash(a), object_info_hash(b))
        # And idempotent on the exact same object.
        self.assertEqual(object_info_hash(a), object_info_hash(a))

    def test_differs_when_changed(self):
        base = {"KSampler": {"input": {}}}
        added = {"KSampler": {"input": {}}, "NewCustomNode": {"input": {}}}
        self.assertNotEqual(object_info_hash(base), object_info_hash(added))

    def test_returns_md5_hexdigest(self):
        h = object_info_hash({"A": {}})
        self.assertEqual(len(h), 32)
        int(h, 16)  # hex-decodable


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


class _FakeState:
    """DI-friendly stand-in for config.STATE — only backend_name + save() count.

    object_info_hash now lives on the per-account Pairing (see config.Pairing),
    so _register reads/writes the pairing; STATE only contributes the machine
    name and a save() that this fake counts.
    """

    def __init__(self, backend_name="dev"):
        self.backend_name = backend_name
        self.save_calls = 0

    def save(self):
        self.save_calls += 1


class TestRegisterToleratesUploadFailure(unittest.IsolatedAsyncioTestCase):
    async def test_online_even_when_upload_fails(self):
        from comfylink import worker
        from comfylink.config import Pairing

        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = {"A": {}, "B": {}}
        # Upload blows up (e.g. relay 503 because R2 unconfigured).
        relay.upload_object_info.side_effect = RuntimeError("503 R2 not configured")

        state = _FakeState()
        pairing = Pairing(backend_id="b1", device_token="t", object_info_hash="")
        with mock.patch.object(worker, "STATUS") as status, \
                mock.patch.object(worker, "STATE", state):
            # Should NOT raise — failure is tolerated.
            await worker._register(relay, comfy, pairing)

        relay.register.assert_awaited_once()
        relay.upload_object_info.assert_awaited_once()
        # Last STATUS.set still flips us online.
        states = [c.kwargs.get("state") for c in status.set.call_args_list]
        self.assertEqual(states[-1], "online")
        # Failure path must NOT persist the hash — next start retries.
        self.assertEqual(pairing.object_info_hash, "")
        self.assertEqual(state.save_calls, 0)


class TestRegisterSkipLogic(unittest.IsolatedAsyncioTestCase):
    async def test_uploads_and_saves_hash_when_changed(self):
        from comfylink import worker
        from comfylink.config import Pairing

        oi = {"A": {}, "B": {}}
        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = oi

        state = _FakeState()
        pairing = Pairing(backend_id="b1", device_token="t", object_info_hash="")
        with mock.patch.object(worker, "STATUS"), \
                mock.patch.object(worker, "STATE", state):
            await worker._register(relay, comfy, pairing)

        # Hash differs from "" => upload happens and the new hash is persisted.
        relay.upload_object_info.assert_awaited_once_with(pairing.backend_id, oi)
        self.assertEqual(pairing.object_info_hash, object_info_hash(oi))
        self.assertEqual(state.save_calls, 1)

    async def test_skips_upload_when_unchanged(self):
        from comfylink import worker
        from comfylink.config import Pairing

        oi = {"A": {}, "B": {}}
        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = oi

        # Stored hash already matches the current snapshot.
        state = _FakeState()
        pairing = Pairing(backend_id="b1", device_token="t",
                          object_info_hash=object_info_hash(oi))
        with mock.patch.object(worker, "STATUS") as status, \
                mock.patch.object(worker, "STATE", state):
            await worker._register(relay, comfy, pairing)

        # Upload is skipped entirely; hash untouched; no save needed.
        relay.upload_object_info.assert_not_awaited()
        self.assertEqual(state.save_calls, 0)
        # Still goes online with node_count set, as before.
        states = [c.kwargs.get("state") for c in status.set.call_args_list]
        self.assertEqual(states[-1], "online")
        node_counts = [c.kwargs.get("node_count") for c in status.set.call_args_list]
        self.assertIn(len(oi), node_counts)

    async def test_failure_does_not_update_hash(self):
        from comfylink import worker
        from comfylink.config import Pairing

        oi = {"A": {}, "B": {}}
        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = oi
        relay.upload_object_info.side_effect = RuntimeError("boom")

        state = _FakeState()
        pairing = Pairing(backend_id="b1", device_token="t", object_info_hash="")
        with mock.patch.object(worker, "STATUS"), \
                mock.patch.object(worker, "STATE", state):
            await worker._register(relay, comfy, pairing)  # tolerated, no raise

        relay.upload_object_info.assert_awaited_once()
        # Hash stays unset so the next start re-attempts the upload.
        self.assertEqual(pairing.object_info_hash, "")
        self.assertEqual(state.save_calls, 0)


if __name__ == "__main__":
    unittest.main()
