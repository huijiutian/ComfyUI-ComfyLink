"""Tests for the startup orphan-job sweep.

When ComfyUI is killed mid-job and restarted, the plugin re-registers and
heartbeats immediately, so the relay's reaper (which only reclaims a job that is
BOTH stale AND on an offline backend) never fires — the previous run's
claimed/running job stays a zombie forever and the app shows it "running"
indefinitely. The fix is plugin-side: ComfyUI's queue is in-memory, so a freshly
started process is running zero jobs. Right after the first successful register
the worker tells the relay to fail everything it still has claimed/running on
this backend. These tests pin:

  * RelayClient.abandon_jobs POSTs to /v1/backends/{backend_id}/jobs/abandon and
    parses the {"abandoned": N} count (defaulting to 0 when absent);
  * _abandon_orphans marks a backend swept on success (so a reconnect never
    sweeps it again) and swallows failures while leaving it unswept (so the next
    reconnect retries) — a failed sweep never tears down the serve loop;
  * _serve_pairing sweeps a backend exactly ONCE across a reconnect (a network
    blip → re-register must NOT clear a job that is genuinely running by then).

Run:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.relay import RelayClient  # noqa: E402


class _FakePairing:
    """Minimal stand-in for a config.Pairing (backend_id is all _abandon uses)."""

    def __init__(self, backend_id="b1"):
        self.backend_id = backend_id
        self.account = ""


class _SupervisorState:
    """Stand-in for config.STATE with the pairing lookups _serve_pairing needs."""

    def __init__(self, pairings):
        self.pairings = list(pairings)
        self.backend_name = "dev"

    def get_pairing(self, backend_id):
        return next((p for p in self.pairings if p.backend_id == backend_id), None)

    def remove_pairing(self, backend_id):
        pr = self.get_pairing(backend_id)
        self.pairings = [p for p in self.pairings if p.backend_id != backend_id]
        return pr

    def save(self):
        pass


class TestAbandonJobs(unittest.IsolatedAsyncioTestCase):
    async def test_posts_to_backend_abandon_path_and_parses_count(self):
        relay = RelayClient.__new__(RelayClient)

        async def fake_json(method, path, body):
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/v1/backends/b1/jobs/abandon")
            # No-body contract: an empty JSON object.
            self.assertEqual(body, {})
            return {"abandoned": 3}

        relay._json = fake_json
        n = await relay.abandon_jobs("b1")
        self.assertEqual(n, 3)

    async def test_missing_field_defaults_to_zero(self):
        relay = RelayClient.__new__(RelayClient)

        async def fake_json(method, path, body):
            return {}  # relay omitted the count

        relay._json = fake_json
        self.assertEqual(await relay.abandon_jobs("b9"), 0)


class TestAbandonOrphans(unittest.IsolatedAsyncioTestCase):
    async def test_sweeps_and_marks_backend_swept(self):
        relay = mock.AsyncMock()
        relay.abandon_jobs.return_value = 2
        swept = set()
        await worker._abandon_orphans(relay, _FakePairing("b1"), swept)
        relay.abandon_jobs.assert_awaited_once_with("b1")
        self.assertIn("b1", swept, "a successful sweep must mark the backend swept")

    async def test_failure_leaves_backend_unswept_and_does_not_raise(self):
        relay = mock.AsyncMock()
        relay.abandon_jobs.side_effect = RuntimeError("network down")
        swept = set()
        # Must swallow — a failed sweep can never tear down the serve loop.
        await worker._abandon_orphans(relay, _FakePairing("b1"), swept)
        self.assertNotIn("b1", swept, "a failed sweep must leave the backend unswept to retry")


class TestServePairingSweepsOnce(unittest.IsolatedAsyncioTestCase):
    async def test_sweeps_once_across_a_reconnect(self):
        # The shared per-process `swept` set spans reconnects: the first register
        # sweeps this backend, a later reconnect (after a transient error) must
        # NOT sweep again or it would kill a job that is live by then.
        from comfylink.config import Pairing

        relay = mock.AsyncMock()
        relay.abandon_jobs.return_value = 1
        pairing = Pairing(backend_id="b1", device_token="clr")
        state = _SupervisorState([pairing])
        swept = set()

        # _claim_loop: 1st call raises a transient error (forces a reconnect),
        # 2nd raises _Revoked so _serve_pairing returns.
        calls = {"n": 0}

        async def fake_claim_loop(relay_, worker_, pairing_, job_lock_, stop_):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("blip")
            raise worker._Revoked()

        with mock.patch.object(worker, "RelayClient", return_value=relay), \
                mock.patch.object(worker, "ComfyClient"), \
                mock.patch.object(worker, "Worker"), \
                mock.patch.object(worker, "STATUS"), \
                mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_register", mock.AsyncMock()), \
                mock.patch.object(worker, "_heartbeat_loop", mock.AsyncMock()), \
                mock.patch.object(worker, "_claim_loop", fake_claim_loop), \
                mock.patch.object(worker.asyncio, "sleep", mock.AsyncMock()):
            await worker._serve_pairing(
                pairing, asyncio.Lock(), None, "http://comfy", None, swept
            )

        # Swept exactly once despite registering twice (a reconnect happened).
        relay.abandon_jobs.assert_awaited_once_with("b1")
        self.assertIn("b1", swept)
        # The revoke removed the pairing from STATE.
        self.assertIsNone(state.get_pairing("b1"))


if __name__ == "__main__":
    unittest.main()
