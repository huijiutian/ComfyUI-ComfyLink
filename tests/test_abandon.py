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
  * _abandon_orphans latches on success (so a reconnect never sweeps again) and
    swallows failures while leaving the latch unset (so the next reconnect
    retries) — a failed sweep never tears down the serve loop;
  * the startup path sweeps exactly ONCE across two _serve_paired calls (a
    network blip → re-register must NOT clear a job that is genuinely running by
    then).

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


class _FakeState:
    """Minimal stand-in for config.STATE (only backend_id is needed here)."""

    def __init__(self, backend_id="b1"):
        self.backend_id = backend_id
        self.device_token = "clr_test"


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
    async def test_sweeps_and_latches_done(self):
        relay = mock.AsyncMock()
        relay.abandon_jobs.return_value = 2
        swept = {"done": False}
        with mock.patch.object(worker, "STATE", _FakeState(backend_id="b1")):
            await worker._abandon_orphans(relay, swept)
        relay.abandon_jobs.assert_awaited_once_with("b1")
        self.assertTrue(swept["done"], "a successful sweep must latch")

    async def test_failure_leaves_latch_unset_and_does_not_raise(self):
        relay = mock.AsyncMock()
        relay.abandon_jobs.side_effect = RuntimeError("network down")
        swept = {"done": False}
        with mock.patch.object(worker, "STATE", _FakeState(backend_id="b1")):
            # Must swallow — a failed sweep can never tear down the serve loop.
            await worker._abandon_orphans(relay, swept)
        self.assertFalse(swept["done"], "a failed sweep must leave the latch unset to retry")


class TestServePairedSweepsOnce(unittest.IsolatedAsyncioTestCase):
    async def test_abandon_called_once_across_two_serve_paired(self):
        # The same per-process latch is shared across reconnects: the first
        # _serve_paired sweeps, the second (a reconnect) must NOT.
        relay = mock.AsyncMock()
        relay.abandon_jobs.return_value = 1
        swept = {"done": False}

        with mock.patch.object(worker, "RelayClient", return_value=relay), \
                mock.patch.object(worker, "ComfyClient"), \
                mock.patch.object(worker, "Worker"), \
                mock.patch.object(worker, "STATUS"), \
                mock.patch.object(worker, "STATE", _FakeState(backend_id="b1")), \
                mock.patch.object(worker, "_register", mock.AsyncMock()), \
                mock.patch.object(worker, "_heartbeat_loop", mock.AsyncMock()), \
                mock.patch.object(worker, "_claim_loop", mock.AsyncMock()):
            await worker._serve_paired(None, "http://comfy", None, swept)
            await worker._serve_paired(None, "http://comfy", None, swept)
            await asyncio.sleep(0)  # let cancelled heartbeat tasks settle

        relay.abandon_jobs.assert_awaited_once_with("b1")
        self.assertTrue(swept["done"])

    async def test_failed_sweep_retried_on_next_serve_paired(self):
        # First sweep fails (latch stays unset); the reconnect retries and
        # succeeds. abandon is therefore attempted on BOTH calls.
        relay = mock.AsyncMock()
        relay.abandon_jobs.side_effect = [RuntimeError("blip"), 4]
        swept = {"done": False}

        with mock.patch.object(worker, "RelayClient", return_value=relay), \
                mock.patch.object(worker, "ComfyClient"), \
                mock.patch.object(worker, "Worker"), \
                mock.patch.object(worker, "STATUS"), \
                mock.patch.object(worker, "STATE", _FakeState(backend_id="b1")), \
                mock.patch.object(worker, "_register", mock.AsyncMock()), \
                mock.patch.object(worker, "_heartbeat_loop", mock.AsyncMock()), \
                mock.patch.object(worker, "_claim_loop", mock.AsyncMock()):
            await worker._serve_paired(None, "http://comfy", None, swept)
            self.assertFalse(swept["done"])
            await worker._serve_paired(None, "http://comfy", None, swept)
            await asyncio.sleep(0)

        self.assertEqual(relay.abandon_jobs.await_count, 2)
        self.assertTrue(swept["done"])


if __name__ == "__main__":
    unittest.main()
