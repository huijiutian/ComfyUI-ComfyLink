"""Tests for the object_info R2 upload flow.

object_info is no longer POSTed to the relay as a blob (egress fix). Instead the
plugin signs a presigned PUT and ships the JSON straight to R2. These tests pin
that wiring on both layers:

  * RelayClient.upload_object_info: sign first, then PUT the JSON with
    Content-Type application/json.
  * worker._register: on upload failure the backend still goes online (params
    are non-critical) and never crashes.
  * the APP-INITIATED refresh (heartbeat field ``object_info_requested_at``):
    the snapshot was historically captured once, at register, so a user who
    installs or deletes a model afterwards kept seeing a stale model list in the
    app. The refresh re-captures on request — and ONLY on request.

Run:  python -m unittest discover -s tests
"""

import asyncio
import json
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
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


# ── the app-initiated refresh (heartbeat body → exactly one re-capture) ──────

class _RefreshPairing:
    """Minimal Pairing stand-in for the refresh path."""

    def __init__(self, object_info_hash="", object_info_synced_at=0.0):
        self.backend_id = "b1"
        self.object_info_hash = object_info_hash
        self.object_info_synced_at = object_info_synced_at
        # Present so a stray LoRA-path read would be visible, never used here.
        self.loras_hash = ""
        self.loras_synced_at = 0.0


class TestObjectInfoRequestedAt(unittest.TestCase):
    """The signal is read defensively, and the two signals never cross wires."""

    def test_old_relay_body_reads_as_no_request(self):
        # "New plugin, old relay": no such field => 0.0 => nothing ever happens.
        for body in ({}, {"other": 1}, None, "nope"):
            with self.subTest(body=body):
                self.assertEqual(
                    worker._requested_at(body, worker.OBJECT_INFO_REQUESTED_KEY), 0.0)

    def test_a_timestamp_is_read(self):
        self.assertEqual(
            worker._requested_at({"object_info_requested_at": 1754000000},
                                 worker.OBJECT_INFO_REQUESTED_KEY), 1754000000.0)
        self.assertEqual(
            worker._requested_at({"object_info_requested_at": "1754000000"},
                                 worker.OBJECT_INFO_REQUESTED_KEY), 1754000000.0)

    def test_junk_degrades_to_zero(self):
        for junk in (None, "", "soon", {}, [], object()):
            with self.subTest(junk=junk):
                self.assertEqual(
                    worker._requested_at({"object_info_requested_at": junk},
                                         worker.OBJECT_INFO_REQUESTED_KEY), 0.0)

    def test_the_two_signals_are_independent(self):
        # A LoRA refresh request must not read as an object_info request (they
        # cost wildly different amounts and the user pressed a different button).
        body = {"loras_requested_at": 1754000000}
        self.assertEqual(worker._requested_at(body, worker.OBJECT_INFO_REQUESTED_KEY),
                         0.0)
        self.assertEqual(worker._requested_at({"object_info_requested_at": 5},
                                              worker.LORAS_REQUESTED_KEY), 0.0)


class TestMaybeRefreshObjectInfo(unittest.IsolatedAsyncioTestCase):
    """⛔ The gate: no signal ⇒ no re-capture. And one signal ⇒ one re-capture."""

    def setUp(self):
        self.calls = []

    def _fake_refresh(self):
        async def _refresh(relay, comfy, pairing, requested):
            self.calls.append(requested)
            pairing.object_info_synced_at = max(
                pairing.object_info_synced_at or 0, requested)
        return _refresh

    def _patched(self):
        return mock.patch.object(worker, "_refresh_object_info", self._fake_refresh())

    async def _drain(self, task):
        if task is not None:
            await task

    async def test_no_signal_means_no_refresh(self):
        pairing = _RefreshPairing()
        with self._patched():
            for body in ({},
                         {"object_info_requested_at": 0},
                         {"object_info_requested_at": None},
                         {"other": 5},
                         # The LoRA signal must NOT trigger an object_info refresh.
                         {"loras_requested_at": 1754000000}):
                with self.subTest(body=body):
                    self.assertIsNone(worker._maybe_refresh_object_info(
                        mock.AsyncMock(), mock.AsyncMock(), pairing, body, None))
        self.assertEqual(self.calls, [])
        self.assertEqual(pairing.object_info_synced_at, 0.0)

    async def test_no_comfy_client_means_no_refresh(self):
        pairing = _RefreshPairing()
        with self._patched():
            self.assertIsNone(worker._maybe_refresh_object_info(
                mock.AsyncMock(), None, pairing,
                {"object_info_requested_at": 1754000000}, None))
        self.assertEqual(self.calls, [])

    async def test_a_signal_starts_exactly_one_refresh(self):
        pairing = _RefreshPairing()
        body = {"object_info_requested_at": 1754000000}
        with self._patched():
            task = worker._maybe_refresh_object_info(
                mock.AsyncMock(), mock.AsyncMock(), pairing, body, None)
            await self._drain(task)
            # The relay replays the SAME timestamp on every beat until we report
            # back; the watermark is what keeps that at one re-capture.
            for _ in range(4):
                task = worker._maybe_refresh_object_info(
                    mock.AsyncMock(), mock.AsyncMock(), pairing, body, task)
                await self._drain(task)
        self.assertEqual(self.calls, [1754000000.0])

    async def test_a_newer_request_is_served_again(self):
        pairing = _RefreshPairing()
        with self._patched():
            t = worker._maybe_refresh_object_info(
                mock.AsyncMock(), mock.AsyncMock(), pairing,
                {"object_info_requested_at": 100}, None)
            await self._drain(t)
            t = worker._maybe_refresh_object_info(
                mock.AsyncMock(), mock.AsyncMock(), pairing,
                {"object_info_requested_at": 200}, t)
            await self._drain(t)
        self.assertEqual(self.calls, [100.0, 200.0])

    async def test_a_stale_request_is_ignored(self):
        pairing = _RefreshPairing(object_info_synced_at=500.0)
        with self._patched():
            self.assertIsNone(worker._maybe_refresh_object_info(
                mock.AsyncMock(), mock.AsyncMock(), pairing,
                {"object_info_requested_at": 499}, None))
        self.assertEqual(self.calls, [])

    async def test_an_in_flight_refresh_is_not_duplicated(self):
        pairing = _RefreshPairing()
        gate = asyncio.Event()

        async def slow(relay, comfy, pr, requested):
            self.calls.append(requested)
            await gate.wait()

        with mock.patch.object(worker, "_refresh_object_info", slow):
            task = worker._maybe_refresh_object_info(
                mock.AsyncMock(), mock.AsyncMock(), pairing,
                {"object_info_requested_at": 100}, None)
            await asyncio.sleep(0)
            for _ in range(3):
                same = worker._maybe_refresh_object_info(
                    mock.AsyncMock(), mock.AsyncMock(), pairing,
                    {"object_info_requested_at": 100}, task)
                self.assertIs(same, task)
            gate.set()
            await task
        self.assertEqual(self.calls, [100.0])

    def test_dispatch_is_synchronous(self):
        # ⚠️ It runs INSIDE the heartbeat loop: if it were awaitable it could
        # stall the beat and the relay's reaper would take this backend for dead.
        self.assertFalse(asyncio.iscoroutinefunction(worker._maybe_refresh_object_info))


class _Boom(Exception):
    pass


class TestRefreshObjectInfo(unittest.IsolatedAsyncioTestCase):
    """What one re-capture does — and what it refuses to do on failure."""

    async def test_uploads_and_advances_watermark_when_changed(self):
        oi = {"A": {}, "B": {}}
        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = oi
        pairing = _RefreshPairing()
        state = _FakeState()
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "STATUS") as status:
            await worker._refresh_object_info(relay, comfy, pairing, 900.0)
        relay.upload_object_info.assert_awaited_once_with("b1", oi)
        self.assertEqual(pairing.object_info_hash, object_info_hash(oi))
        self.assertEqual(pairing.object_info_synced_at, 900.0)
        # The panel's node count follows; its state/error do NOT (they belong to
        # the register and job paths — a background refresh must not clobber them).
        for call in status.set.call_args_list:
            self.assertNotIn("state", call.kwargs)
            self.assertNotIn("error", call.kwargs)
        self.assertIn(len(oi), [c.kwargs.get("node_count")
                                for c in status.set.call_args_list])

    async def test_unchanged_uploads_nothing_but_still_serves_the_request(self):
        oi = {"A": {}}
        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = oi
        pairing = _RefreshPairing(object_info_hash=object_info_hash(oi))
        with mock.patch.object(worker, "STATE", _FakeState()), \
                mock.patch.object(worker, "STATUS"):
            await worker._refresh_object_info(relay, comfy, pairing, 900.0)
        # Content hash says nothing moved => no multi-MB re-upload...
        relay.upload_object_info.assert_not_awaited()
        # ...but the request WAS served, so it must not re-run every 25 seconds.
        self.assertEqual(pairing.object_info_synced_at, 900.0)

    async def test_capture_failure_leaves_the_watermark_alone(self):
        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.side_effect = RuntimeError("ComfyUI not reachable")
        pairing = _RefreshPairing()
        with mock.patch.object(worker, "STATE", _FakeState()), \
                mock.patch.object(worker, "STATUS"):
            await worker._refresh_object_info(relay, comfy, pairing, 900.0)
        relay.upload_object_info.assert_not_awaited()
        self.assertEqual(pairing.object_info_synced_at, 0.0)   # next beat retries

    async def test_upload_failure_leaves_watermark_and_hash_alone(self):
        relay = mock.AsyncMock()
        relay.upload_object_info.side_effect = RuntimeError("R2 unconfigured")
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = {"A": {}}
        pairing = _RefreshPairing()
        with mock.patch.object(worker, "STATE", _FakeState()), \
                mock.patch.object(worker, "STATUS"):
            await worker._refresh_object_info(relay, comfy, pairing, 900.0)
        self.assertEqual(pairing.object_info_hash, "")
        self.assertEqual(pairing.object_info_synced_at, 0.0)

    async def test_never_raises(self):
        # It runs as a bare task off the heartbeat loop; an escaping exception
        # would surface as an unhandled task error, never as a fixed problem.
        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.side_effect = _Boom("weird")
        with mock.patch.object(worker, "STATE", _FakeState()), \
                mock.patch.object(worker, "STATUS"):
            await worker._refresh_object_info(relay, comfy, _RefreshPairing(), 1.0)


class TestRefreshNeverBlocksTheBeat(unittest.IsolatedAsyncioTestCase):
    async def test_heartbeat_keeps_beating_while_a_refresh_hangs(self):
        relay = mock.AsyncMock()
        relay.heartbeat.return_value = {"object_info_requested_at": 100}
        pairing = _RefreshPairing()
        started = asyncio.Event()

        async def stuck(*_a, **_kw):
            started.set()
            await asyncio.Event().wait()          # never finishes

        state = _FakeState()
        state.get_pairing = lambda bid: pairing
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0.001), \
                mock.patch.object(worker, "_refresh_object_info", stuck):
            hb = asyncio.create_task(worker._heartbeat_loop(
                relay, pairing, None, None, mock.AsyncMock()))
            await asyncio.wait_for(started.wait(), 2)
            await asyncio.sleep(0.05)
            beats = relay.heartbeat.await_count
            await asyncio.sleep(0.05)
            self.assertGreater(relay.heartbeat.await_count, beats)
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
        # Teardown takes the child with it — no task left holding a closing session.
        await asyncio.sleep(0)


class TestWatermarkPersistence(unittest.TestCase):
    """The watermark must survive a ComfyUI restart, and tolerate old files."""

    def test_round_trips_through_the_state_file(self):
        import tempfile
        from pathlib import Path

        from comfylink import config

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "comfylink_state.json"
            with mock.patch.object(config, "_state_path", lambda: path):
                st = config.State()
                pr = st.add_pairing("tok", "dev1")
                pr.object_info_synced_at = 1754000000.0
                st.save()
                again = config.State.load()
            self.assertEqual(again.pairings[0].object_info_synced_at, 1754000000.0)

    def test_a_state_file_without_the_field_loads_as_never_served(self):
        import tempfile
        from pathlib import Path

        from comfylink import config

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "comfylink_state.json"
            path.write_text(json.dumps({
                "backend_name": "dev",
                "pairings": [{"backend_id": "b1", "device_token": "t"}],
            }), "utf-8")
            with mock.patch.object(config, "_state_path", lambda: path):
                st = config.State.load()
        self.assertEqual(st.pairings[0].object_info_synced_at, 0.0)


class TestTheTwoWatermarksAreIndependent(unittest.IsolatedAsyncioTestCase):
    """⛔ ONE watermark for both signals would be a silent, unreproducible bug.

    The relay replays BOTH timestamps in every beat. If the plugin remembered a
    single "handled up to" value, serving the LoRA request would push that value
    past an object_info request the user made earlier — and their "refresh model
    list" press would simply never do anything, with no error anywhere. It only
    misfires once BOTH buttons have been pressed, which is exactly the kind of
    thing that never shows up in a manual smoke test.

    Hence two fields on the pairing: `loras_synced_at` and
    `object_info_synced_at`. These tests fail if they are ever merged.
    """

    def setUp(self):
        self.scans = []
        self.refreshes = []

    def _patched(self):
        async def _scan(relay, pairing, requested):
            self.scans.append(requested)
            pairing.loras_synced_at = max(pairing.loras_synced_at or 0, requested)

        async def _refresh(relay, comfy, pairing, requested):
            self.refreshes.append(requested)
            pairing.object_info_synced_at = max(
                pairing.object_info_synced_at or 0, requested)

        return (mock.patch.object(worker, "_scan_and_report", _scan),
                mock.patch.object(worker, "_refresh_object_info", _refresh))

    async def _beat(self, pairing, body, scan=None, refresh=None):
        scan = worker._maybe_scan_models(mock.AsyncMock(), pairing, body, scan)
        refresh = worker._maybe_refresh_object_info(
            mock.AsyncMock(), mock.AsyncMock(), pairing, body, refresh)
        for t in (scan, refresh):
            if t is not None:
                await t
        return scan, refresh

    async def test_serving_the_lora_request_does_not_bury_the_other(self):
        pairing = _RefreshPairing()
        p1, p2 = self._patched()
        with p1, p2:
            # Beat 1: only the LoRA button has been pressed.
            s, r = await self._beat(pairing, {"ok": True,
                                              "loras_requested_at": 100,
                                              "object_info_requested_at": 0})
            self.assertEqual(self.scans, [100.0])
            self.assertEqual(self.refreshes, [])
            # Beat 2: the user now presses "refresh model list" too. Its
            # timestamp is OLDER than the LoRA one that was already served — a
            # shared watermark would swallow it here.
            await self._beat(pairing, {"ok": True,
                                       "loras_requested_at": 100,
                                       "object_info_requested_at": 50}, s, r)
        self.assertEqual(self.scans, [100.0])          # not scanned twice
        self.assertEqual(self.refreshes, [50.0])       # ...and still refreshed

    async def test_serving_the_object_info_request_does_not_bury_the_other(self):
        pairing = _RefreshPairing()
        p1, p2 = self._patched()
        with p1, p2:
            s, r = await self._beat(pairing, {"ok": True,
                                              "loras_requested_at": 0,
                                              "object_info_requested_at": 100})
            self.assertEqual(self.refreshes, [100.0])
            self.assertEqual(self.scans, [])
            await self._beat(pairing, {"ok": True,
                                       "loras_requested_at": 50,
                                       "object_info_requested_at": 100}, s, r)
        self.assertEqual(self.refreshes, [100.0])
        self.assertEqual(self.scans, [50.0])

    async def test_the_idle_relay_body_does_nothing_at_all(self):
        # The relay always serializes both fields; 0 means "nothing pending",
        # and that is what ~140 beats an hour look like.
        pairing = _RefreshPairing()
        p1, p2 = self._patched()
        with p1, p2:
            for _ in range(5):
                s, r = await self._beat(pairing, {"ok": True,
                                                  "loras_requested_at": 0,
                                                  "object_info_requested_at": 0})
                self.assertIsNone(s)
                self.assertIsNone(r)
        self.assertEqual((self.scans, self.refreshes), ([], []))
        self.assertEqual(pairing.loras_synced_at, 0.0)
        self.assertEqual(pairing.object_info_synced_at, 0.0)


# ── the receipt: the beat carries the snapshot fingerprint ──────────────────

class TestHeartbeatCarriesTheFingerprint(unittest.IsolatedAsyncioTestCase):
    """The relay never sees the snapshot land (presigned PUT straight to R2),
    so the beat is how it learns which snapshot this backend is actually on."""

    def _client(self):
        relay = RelayClient.__new__(RelayClient)
        sent = []

        async def fake_json(method, path, body, timeout=None):
            sent.append((method, path, body))
            return {"ok": True}

        relay._json = fake_json
        return relay, sent

    async def test_current_fingerprint_is_sent(self):
        relay, sent = self._client()
        await relay.heartbeat("b1", "0123456789abcdef0123456789abcdef")
        _method, path, body = sent[0]
        self.assertEqual(path, "/v1/backends/heartbeat")
        self.assertEqual(body["backend_id"], "b1")
        self.assertEqual(body["object_info_hash"],
                         "0123456789abcdef0123456789abcdef")

    async def test_nothing_captured_yet_sends_no_fake_value(self):
        # ⛔ "" means "keep what you have" relay-side. A backend that has never
        # captured a snapshot must contribute NOTHING to that column — and in
        # particular must never pad it with some other identifier that happened
        # to be at hand (commit sha, version, backend_id).
        relay, sent = self._client()
        await relay.heartbeat("b1")                       # default: never captured
        _method, _path, body = sent[0]
        self.assertNotIn("object_info_hash", body)
        for stray in (body.get("commit"), body.get("version"), body["backend_id"]):
            self.assertNotIn(str(stray), str(body.get("object_info_hash", "")))
        # The rest of the body is untouched.
        self.assertIn("version", body)
        self.assertIn("commit", body)

    async def test_empty_string_is_omitted_not_sent_blank(self):
        relay, sent = self._client()
        await relay.heartbeat("b1", "")
        self.assertNotIn("object_info_hash", sent[0][2])

    # ── the OTHER half of the receipt: which refresh request we have served ──
    #
    # The fingerprint alone cannot say "your refresh is done": an unchanged
    # ComfyUI keeps the same fingerprint forever, so an app waiting on it waits
    # for a change that is never coming. The watermark is the completion signal.

    async def test_served_request_watermark_is_sent(self):
        relay, sent = self._client()
        await relay.heartbeat("b1", "abc", 1754000000.0)
        _method, _path, body = sent[0]
        self.assertEqual(body["object_info_synced_at"], 1754000000.0)
        # ⛔ ...and it goes out as an INT, not the float it is stored as. A
        # decimal point here makes the relay's int64 field reject the WHOLE body
        # (it answers "backend_id required", of all things) and this backend's
        # heartbeat dies permanently. Full story + the rest of the guards:
        # tests/test_heartbeat_wire.py.
        v = body["object_info_synced_at"]
        self.assertTrue(isinstance(v, int) and not isinstance(v, bool),
                        "watermark must be sent as int, got %r" % (v,))
        self.assertEqual(body["object_info_hash"], "abc")     # both, independently

    async def test_never_served_sends_no_fake_value(self):
        # ⛔ Same "empty = keep what you have" rule as the fingerprint: a backend
        # that has never served a refresh request must not write a 0 over
        # whatever the relay has.
        relay, sent = self._client()
        await relay.heartbeat("b1", "abc")                    # default: 0.0
        self.assertNotIn("object_info_synced_at", sent[0][2])
        for falsy in (0, 0.0, None):
            with self.subTest(value=falsy):
                relay, sent = self._client()
                await relay.heartbeat("b1", "abc", falsy)
                self.assertNotIn("object_info_synced_at", sent[0][2])

    async def test_the_watermark_is_sent_even_with_no_fingerprint(self):
        # The two fields are independent: a refresh can complete (watermark
        # moves) on a backend whose snapshot upload has never succeeded.
        relay, sent = self._client()
        await relay.heartbeat("b1", "", 900.0)
        _method, _path, body = sent[0]
        self.assertNotIn("object_info_hash", body)
        self.assertEqual(body["object_info_synced_at"], 900.0)

    async def test_the_loop_reads_the_pairing_fresh_each_beat(self):
        # A refresh that lands between beats must be reported by the NEXT beat,
        # so the value may not be captured once and cached.
        relay = mock.AsyncMock()
        relay.heartbeat.return_value = {"ok": True}
        pairing = _RefreshPairing(object_info_hash="aaa")
        state = _FakeState()
        state.get_pairing = lambda bid: pairing

        async def bump(*_a, **_kw):
            if relay.heartbeat.await_count == 2:
                pairing.object_info_hash = "bbb"
            return {"ok": True}

        relay.heartbeat.side_effect = bump
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0.001):
            hb = asyncio.create_task(worker._heartbeat_loop(relay, pairing, None))
            await asyncio.sleep(0.05)
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
        hashes = [c.args[1] for c in relay.heartbeat.await_args_list]
        self.assertEqual(hashes[0], "aaa")
        self.assertIn("bbb", hashes)                      # picked up, not cached

    async def test_the_loop_reports_the_served_request_watermark(self):
        # Same freshness rule for the watermark: a refresh that advances it
        # between beats must be reported by the NEXT beat, not one interval later.
        relay = mock.AsyncMock()
        pairing = _RefreshPairing(object_info_hash="aaa")
        state = _FakeState()
        state.get_pairing = lambda bid: pairing

        async def bump(*_a, **_kw):
            if relay.heartbeat.await_count == 2:
                pairing.object_info_synced_at = 900.0
            return {"ok": True}

        relay.heartbeat.side_effect = bump
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0.001):
            hb = asyncio.create_task(worker._heartbeat_loop(relay, pairing, None))
            await asyncio.sleep(0.05)
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
        marks = [c.args[2] for c in relay.heartbeat.await_args_list]
        self.assertEqual(marks[0], 0.0)                   # never served anything yet
        self.assertIn(900.0, marks)                       # picked up, not cached


class TestImmediateBeatAfterUpload(unittest.IsolatedAsyncioTestCase):
    """A finished refresh announces itself instead of waiting out the interval.

    ⛔ REGRESSION GUARD (user-reported, 2026-08-01): the announcement used to be
    conditional on `uploaded`. A ComfyUI whose node set had not changed uploads
    nothing — by design, that optimisation is correct — so it never beat, and the
    app, which waits for the refresh receipt, waited until it timed out on a
    refresh that had actually finished in milliseconds. "The refresh is done" and
    "the content changed" are two different facts carried by two different fields;
    the beat goes out on BOTH paths.
    """

    def _fixture(self, oi, stored_hash=""):
        relay = mock.AsyncMock()
        relay.heartbeat.return_value = {"ok": True}
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = oi
        return relay, comfy, _RefreshPairing(object_info_hash=stored_hash)

    async def test_upload_is_announced_at_once(self):
        oi = {"A": {}, "B": {}}
        relay, comfy, pairing = self._fixture(oi)
        with mock.patch.object(worker, "STATE", _FakeState()), \
                mock.patch.object(worker, "STATUS"):
            await worker._refresh_object_info(relay, comfy, pairing, 900.0)
        relay.upload_object_info.assert_awaited_once()
        relay.heartbeat.assert_awaited_once_with("b1", object_info_hash(oi), 900.0)
        # The beat reports the snapshot that is NOW in R2, not the old one.
        self.assertEqual(relay.heartbeat.await_args.args[1], pairing.object_info_hash)

    async def test_unchanged_snapshot_is_still_announced(self):
        # ⛔ THE BUG. Nothing to upload, but the user's refresh WAS served — and
        # the app is waiting for exactly that. Beat, carrying the watermark that
        # just advanced and the (unchanged) fingerprint.
        oi = {"A": {}}
        relay, comfy, pairing = self._fixture(oi, stored_hash=object_info_hash(oi))
        with mock.patch.object(worker, "STATE", _FakeState()), \
                mock.patch.object(worker, "STATUS"):
            await worker._refresh_object_info(relay, comfy, pairing, 900.0)
        # The R2 optimisation is untouched: an unchanged snapshot is NOT re-sent.
        relay.upload_object_info.assert_not_awaited()
        # ...but the completion receipt goes out anyway.
        relay.heartbeat.assert_awaited_once_with("b1", object_info_hash(oi), 900.0)

    async def test_the_announced_watermark_is_the_request_just_served(self):
        # The beat must carry the watermark AFTER it advanced — announcing the
        # old value would tell the app "still working" and it would keep waiting.
        oi = {"A": {}}
        relay, comfy, pairing = self._fixture(oi, stored_hash=object_info_hash(oi))
        pairing.object_info_synced_at = 100.0          # served an older request
        with mock.patch.object(worker, "STATE", _FakeState()), \
                mock.patch.object(worker, "STATUS"):
            await worker._refresh_object_info(relay, comfy, pairing, 900.0)
        self.assertEqual(relay.heartbeat.await_args.args[2], 900.0)
        self.assertEqual(relay.heartbeat.await_args.args[2],
                         pairing.object_info_synced_at)

    async def test_a_failed_upload_is_not_announced(self):
        relay, comfy, pairing = self._fixture({"A": {}})
        relay.upload_object_info.side_effect = RuntimeError("R2 unconfigured")
        with mock.patch.object(worker, "STATE", _FakeState()), \
                mock.patch.object(worker, "STATUS"):
            await worker._refresh_object_info(relay, comfy, pairing, 900.0)
        relay.heartbeat.assert_not_awaited()

    async def test_a_failed_announcement_is_swallowed_and_not_retried(self):
        # The next regular beat carries the same fingerprint, so this costs
        # latency and nothing else — retrying here would only pile on a relay
        # that is already struggling.
        oi = {"A": {}}
        relay, comfy, pairing = self._fixture(oi)
        relay.heartbeat.side_effect = RuntimeError("relay redeploying")
        with mock.patch.object(worker, "STATE", _FakeState()), \
                mock.patch.object(worker, "STATUS"):
            await worker._refresh_object_info(relay, comfy, pairing, 900.0)   # no raise
        self.assertEqual(relay.heartbeat.await_count, 1)     # exactly once
        # The refresh still counts as served — the watermark advanced.
        self.assertEqual(pairing.object_info_synced_at, 900.0)

    async def test_the_regular_rhythm_is_untouched(self):
        # The announcement is issued from the REFRESH task, so the heartbeat
        # loop's own sleep is neither cut short nor extended.
        relay = mock.AsyncMock()
        relay.heartbeat.return_value = {"ok": True}
        pairing = _RefreshPairing()
        state = _FakeState()
        state.get_pairing = lambda bid: pairing
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0.02), \
                mock.patch.object(worker, "_maybe_refresh_object_info",
                                  lambda *a, **kw: None):
            hb = asyncio.create_task(worker._heartbeat_loop(relay, pairing, None))
            await asyncio.sleep(0.11)
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
        # ~0.11s at a 0.02s cadence: a handful of beats, not a burst.
        self.assertLessEqual(relay.heartbeat.await_count, 8)
        self.assertGreaterEqual(relay.heartbeat.await_count, 3)


# ── the fingerprint must track REAL changes, and only real ones ─────────────
#
# Shapes below mirror what PM pulled off a real user's ComfyUI (2808 node types,
# two reads one second apart): three nodes rebuild an input `default` on every
# call, which made the fingerprint useless. See worker._VOLATILE_INPUT_OPTS.

def _checkpoint_loader(models):
    """A loader node — the candidate list is spec[0], NOT a `default`."""
    return {
        "input": {"required": {
            "ckpt_name": [list(models),
                          {"default": models[0] if models else None,
                           "tooltip": "The name of the checkpoint to load."}],
        }},
        "output": ["MODEL", "CLIP", "VAE"],
        "name": "CheckpointLoaderSimple",
        "category": "loaders",
    }


def _text_image(seed):
    """LayerUtility: TextImage — variation_seed.default IS the wall clock."""
    return {
        "input": {
            "required": {"text": ["STRING", {"default": "", "multiline": True}]},
            "optional": {"variation_seed": ["INT", {"default": seed, "min": 0,
                                                    "max": 99999999, "step": 1}]},
        },
        "name": "LayerUtility: TextImage",
    }


def _cache_node(suffix):
    """Cache Node — both suffixes are re-randomised on every call."""
    return {
        "input": {"required": {
            "conditioning_suffix": ["STRING", {"default": f"{suffix}_cache"}],
            "image_suffix": ["STRING", {"default": f"{suffix}_cache"}],
        }},
        "name": "Cache Node",
    }


def _snapshot(models=("checkpoints/a.safetensors", "checkpoints/b.safetensors"),
              seed=1785556904, suffix="3474403"):
    return {
        "CheckpointLoaderSimple": _checkpoint_loader(models),
        "LayerUtility: TextImage": _text_image(seed),
        "Cache Node": _cache_node(suffix),
    }


class TestVolatileDefaultsAreIgnored(unittest.TestCase):
    """① The bug: a snapshot that only ticked its clocks is NOT a new snapshot."""

    def test_two_reads_one_second_apart_hash_the_same(self):
        # Exactly the observed drift: variation_seed +1 (unix time), Cache Node
        # re-randomised. Same machine, same 3 nodes, nothing installed/removed.
        first = _snapshot(seed=1785556904, suffix="3474403")
        second = _snapshot(seed=1785556905, suffix="96440706")
        self.assertNotEqual(first, second)                 # the blobs DO differ
        self.assertEqual(object_info_hash(first), object_info_hash(second))

    def test_defaults_are_ignored_in_every_input_section(self):
        for section in ("required", "optional", "hidden"):
            with self.subTest(section=section):
                a = {"N": {"input": {section: {"x": ["INT", {"default": 1}]}}}}
                b = {"N": {"input": {section: {"x": ["INT", {"default": 2}]}}}}
                self.assertEqual(object_info_hash(a), object_info_hash(b))

    def test_hashing_does_not_mutate_the_snapshot_we_upload(self):
        # The upload must stay COMPLETE — the app renders form initial values
        # from `default`. Only the fingerprint's input is narrowed.
        oi = _snapshot()
        before = json.dumps(oi, sort_keys=True)
        object_info_hash(oi)
        self.assertEqual(json.dumps(oi, sort_keys=True), before)
        self.assertEqual(
            oi["LayerUtility: TextImage"]["input"]["optional"]["variation_seed"][1]
            ["default"], 1785556904)


class TestModelListChangesAreAlwaysDetected(unittest.TestCase):
    """⛔ THE GUARD RAIL. Widening the filter to spec[0] would blind us to the
    exact thing R-1.0.6-14 exists to detect — with no symptom at all."""

    def test_installing_a_model_changes_the_hash(self):
        before = _snapshot(models=("checkpoints/a.safetensors",))
        after = _snapshot(models=("checkpoints/a.safetensors",
                                  "checkpoints/new.safetensors"))
        self.assertNotEqual(object_info_hash(before), object_info_hash(after))

    def test_deleting_a_model_changes_the_hash(self):
        before = _snapshot(models=("checkpoints/a.safetensors",
                                   "checkpoints/b.safetensors"))
        after = _snapshot(models=("checkpoints/a.safetensors",))
        self.assertNotEqual(object_info_hash(before), object_info_hash(after))

    def test_renaming_a_model_changes_the_hash(self):
        before = _snapshot(models=("checkpoints/a.safetensors",))
        after = _snapshot(models=("checkpoints/renamed.safetensors",))
        self.assertNotEqual(object_info_hash(before), object_info_hash(after))

    def test_reordering_the_candidates_changes_the_hash(self):
        # The candidate list is ordered data (it drives the app's picker), not a
        # set — sort_keys must never be allowed to flatten it away.
        before = _snapshot(models=("a.safetensors", "b.safetensors"))
        after = _snapshot(models=("b.safetensors", "a.safetensors"))
        self.assertNotEqual(object_info_hash(before), object_info_hash(after))

    def test_a_model_change_wins_even_while_the_clocks_tick(self):
        # The realistic case: the user installed a model AND a second passed.
        before = _snapshot(models=("a.safetensors",), seed=1785556904)
        after = _snapshot(models=("a.safetensors", "new.safetensors"),
                          seed=1785556905, suffix="96440706")
        self.assertNotEqual(object_info_hash(before), object_info_hash(after))

    def test_lora_candidates_are_covered_too(self):
        before = {"LoraLoader": {"input": {"required": {
            "lora_name": [["style/x.safetensors"], {"default": None}]}}}}
        after = {"LoraLoader": {"input": {"required": {
            "lora_name": [["style/x.safetensors", "style/y.safetensors"],
                          {"default": None}]}}}}
        self.assertNotEqual(object_info_hash(before), object_info_hash(after))


class TestRealChangesStillMoveTheHash(unittest.TestCase):
    """③④ Everything else that genuinely describes this ComfyUI."""

    def test_installing_or_removing_a_node_changes_the_hash(self):
        base = _snapshot()
        added = dict(base, NewCustomNode={"input": {"required": {}}})
        removed = {k: v for k, v in base.items() if k != "Cache Node"}
        self.assertNotEqual(object_info_hash(base), object_info_hash(added))
        self.assertNotEqual(object_info_hash(base), object_info_hash(removed))

    def test_an_input_name_change_moves_the_hash(self):
        a = {"N": {"input": {"required": {"steps": ["INT", {"default": 20}]}}}}
        b = {"N": {"input": {"required": {"stepz": ["INT", {"default": 20}]}}}}
        self.assertNotEqual(object_info_hash(a), object_info_hash(b))

    def test_an_input_type_change_moves_the_hash(self):
        a = {"N": {"input": {"required": {"seed": ["INT", {"default": 0}]}}}}
        b = {"N": {"input": {"required": {"seed": ["FLOAT", {"default": 0}]}}}}
        self.assertNotEqual(object_info_hash(a), object_info_hash(b))

    def test_an_input_appearing_or_vanishing_moves_the_hash(self):
        a = {"N": {"input": {"required": {"seed": ["INT", {"default": 0}]}}}}
        b = {"N": {"input": {"required": {"seed": ["INT", {"default": 0}],
                                          "steps": ["INT", {"default": 20}]}}}}
        self.assertNotEqual(object_info_hash(a), object_info_hash(b))

    def test_other_options_are_still_hashed(self):
        # Only `default` is volatile. min/max/step/tooltip describe the node and
        # a node update that changes them IS a change worth re-uploading.
        base = {"N": {"input": {"required": {
            "steps": ["INT", {"default": 20, "min": 1, "max": 100,
                              "step": 1, "tooltip": "steps"}]}}}}
        for key, value in (("min", 2), ("max", 150), ("step", 2),
                           ("tooltip", "how many steps")):
            with self.subTest(key=key):
                other = json.loads(json.dumps(base))
                other["N"]["input"]["required"]["steps"][1][key] = value
                self.assertNotEqual(object_info_hash(base), object_info_hash(other))

    def test_every_input_section_keeps_contributing(self):
        # Guards against a filter that DROPS a section instead of filtering it.
        # `required` comes first, so such a bug would keep working there and go
        # silent on optional/hidden — where `variation_seed` actually lives.
        for section in ("optional", "hidden"):
            for a_spec, b_spec in (
                (["INT", {"min": 1}], ["INT", {"min": 2}]),          # constraint
                (["INT", {"default": 0}], ["FLOAT", {"default": 0}]),  # type
                ([["a.safetensors"], {}],
                 [["a.safetensors", "b.safetensors"], {}]),          # candidates
            ):
                with self.subTest(section=section, spec=a_spec):
                    a = {"N": {"input": {"required": {"seed": ["INT", {"default": 0}]},
                                         section: {"x": a_spec}}}}
                    b = {"N": {"input": {"required": {"seed": ["INT", {"default": 0}]},
                                         section: {"x": b_spec}}}}
                    self.assertNotEqual(object_info_hash(a), object_info_hash(b))

    def test_a_whole_input_section_appearing_moves_the_hash(self):
        a = {"N": {"input": {"required": {"seed": ["INT", {"default": 0}]}}}}
        b = {"N": {"input": {"required": {"seed": ["INT", {"default": 0}]},
                             "optional": {"mask": ["MASK", {}]}}}}
        self.assertNotEqual(object_info_hash(a), object_info_hash(b))

    def test_node_metadata_outside_input_is_still_hashed(self):
        a = _snapshot()
        b = json.loads(json.dumps(a))
        b["CheckpointLoaderSimple"]["category"] = "loaders/advanced"
        self.assertNotEqual(object_info_hash(a), object_info_hash(b))


class TestStableViewIsDefensive(unittest.TestCase):
    """Third-party node definitions are not a shape we control."""

    def test_odd_shapes_pass_through_instead_of_raising(self):
        for oi in (
            {},
            {"N": None},
            {"N": "not a dict"},
            {"N": {"input": None}},
            {"N": {"input": "boom"}},
            {"N": {"input": {"required": None}}},
            {"N": {"input": {"required": "boom"}}},
            {"N": {"input": {"required": {"x": None}}}},
            {"N": {"input": {"required": {"x": "INT"}}}},      # spec not a list
            {"N": {"input": {"required": {"x": []}}}},         # empty spec
            {"N": {"input": {"required": {"x": ["INT"]}}}},    # no options dict
            {"N": {"input": {"required": {"x": [{"default": 1}]}}}},
        ):
            with self.subTest(oi=oi):
                h = object_info_hash(oi)
                self.assertEqual(len(h), 32)
                int(h, 16)

    def test_an_unparseable_snapshot_is_hashed_whole_not_dropped(self):
        # Better to hash something we didn't understand (worst case: a needless
        # re-upload) than to ignore it (worst case: a change we never notice).
        a = {"N": {"input": {"weird_section": [1, 2, 3]}}}
        b = {"N": {"input": {"weird_section": [1, 2, 4]}}}
        self.assertNotEqual(object_info_hash(a), object_info_hash(b))

    def test_a_non_dict_snapshot_does_not_explode(self):
        for junk in ([], "nope", None, 7):
            with self.subTest(junk=junk):
                self.assertEqual(len(object_info_hash(junk)), 32)
