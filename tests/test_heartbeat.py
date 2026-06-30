"""Tests for the execution-time job heartbeat.

While a job is claimed-but-not-finished the worker fires a background heartbeat
that re-POSTs a fixed (0, 0) "running" beat to the relay every
JOB_HEARTBEAT_INTERVAL, so the relay's updated_at keeps advancing and its reaper
never mistakes a long-but-healthy generation (or a slow collect/upload) for a
dead plugin. We no longer report real progress numbers (the app shows a spinner
+ elapsed time, not a bar). These tests pin:

  * the heartbeat fires while a job is running (even with no progress source);
  * it carries a fixed (0, 0);
  * it is stopped/awaited once the job ends — no leaked asyncio task.

Run:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.worker import Worker  # noqa: E402


def _make_worker():
    """A Worker with AsyncMock collaborators; progress() returns no-cancel."""
    relay = mock.AsyncMock()
    relay.progress.return_value = {"cancel": False}
    comfy = mock.AsyncMock()
    return Worker(relay, comfy), relay, comfy


class TestJobHeartbeat(unittest.IsolatedAsyncioTestCase):
    async def test_fires_during_a_quiet_job_and_stops_after(self):
        w, relay, _ = _make_worker()

        started = asyncio.Event()
        release = asyncio.Event()

        # Stand in for the real pipeline: signal we're "running", then block until
        # the test lets the job finish — simulating a long, silent generation.
        async def fake_run_prompt(job_id, prompt):
            started.set()
            await release.wait()
            return "pid-1"

        async def fake_collect(prompt_id, output_format):
            return [], 0

        async def fake_upload(job_id, items):
            return []

        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", 0.02), \
                mock.patch.object(w, "_run_prompt", fake_run_prompt), \
                mock.patch.object(w, "_collect_outputs", fake_collect), \
                mock.patch.object(w, "_upload_outputs", fake_upload):
            task = asyncio.create_task(w.handle_job({"id": "j1", "api_prompt": {}}))
            await asyncio.wait_for(started.wait(), 1)
            # Let several heartbeat ticks elapse while the job is blocked.
            await asyncio.sleep(0.1)
            beats_mid = _heartbeat_count(relay)
            self.assertGreaterEqual(beats_mid, 1, "heartbeat should fire during a quiet job")
            # Heartbeats carry a fixed (0, 0) — no real progress is reported.
            self.assertTrue(
                any(c.args[1:] == ("running", 0, 0) for c in relay.progress.call_args_list),
                f"heartbeat should re-send (0, 0); calls={relay.progress.call_args_list}",
            )

            # Let the job finish, then confirm the heartbeat stopped.
            release.set()
            self.assertEqual(await asyncio.wait_for(task, 1), "done")
            beats_after_done = _heartbeat_count(relay)
            await asyncio.sleep(0.1)  # well past several intervals
            self.assertEqual(
                _heartbeat_count(relay), beats_after_done,
                "heartbeat must stop once the job ends (no leaked task)",
            )

    async def test_heartbeat_stops_on_failure(self):
        w, relay, _ = _make_worker()

        async def boom(job_id, prompt):
            await asyncio.sleep(0.05)  # let at least one heartbeat fire first
            raise worker.JobFailed("kaboom")

        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", 0.02), \
                mock.patch.object(w, "_run_prompt", boom):
            self.assertEqual(
                await asyncio.wait_for(w.handle_job({"id": "j2", "api_prompt": {}}), 1),
                "failed",
            )
        # After the (failed) job returns, no further heartbeats land.
        before = _heartbeat_count(relay)
        await asyncio.sleep(0.1)
        self.assertEqual(_heartbeat_count(relay), before)
        # And no asyncio tasks were left running besides this test's own.
        leaked = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        self.assertEqual(leaked, [], f"leaked tasks: {leaked}")


def _heartbeat_count(relay) -> int:
    """How many progress() calls were 'running' heartbeats/reports so far."""
    return sum(
        1 for c in relay.progress.call_args_list
        if len(c.args) >= 2 and c.args[1] == "running"
    )


if __name__ == "__main__":
    unittest.main()
