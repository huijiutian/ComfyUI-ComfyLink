"""Tests for the execution-time job heartbeat.

While a job is claimed-but-not-finished the worker fires ONE background
heartbeat task that re-POSTs a fixed (0, 0) "running" beat to the relay, so the
relay's updated_at keeps advancing and its reaper never mistakes a
long-but-healthy generation (or a slow collect/upload) for a dead plugin. We no
longer report real progress numbers (the app shows a spinner + elapsed time, not
a bar).

Since the "带取消等待的长轮询" change that beat carries ``wait=True``: the relay
bumps updated_at on arrival and then HOLDS the request until a cancel shows up
(~25s max), so the same request that keeps the job alive is also how a cancel
reaches us — within seconds instead of at the next scheduled beat. These tests
pin:

  * the heartbeat fires while a job is running (even with no progress source);
  * it carries a fixed (0, 0) and wait=True;
  * a {"cancel": true} answer sets the (sticky) cancel event and STOPS the loop
    — otherwise the relay would answer instantly forever and we'd hot-loop;
  * a failed beat backs off instead of retrying instantly;
  * an old relay that ignores `wait` (returns immediately) is floored at
    JOB_HEARTBEAT_INTERVAL, not hammered;
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
        async def fake_run_prompt(job_id, prompt, canceled):
            started.set()
            await release.wait()
            return "pid-1"

        async def fake_collect(prompt_id, output_format, job_prompt=None):
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

        async def boom(job_id, prompt, canceled):
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


class TestHeartbeatCancelChannel(unittest.IsolatedAsyncioTestCase):
    """The beat is now ALSO the cancel channel (progress wait=True)."""

    async def test_every_beat_asks_the_relay_to_wait(self):
        # Without wait=True the relay answers instantly and we're back to the old
        # "cancel arrives at the next scheduled beat" (up to a full interval).
        w, relay, _ = _make_worker()
        canceled = asyncio.Event()
        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", 0.02):
            task = asyncio.create_task(w._job_heartbeat("j1", canceled))
            await asyncio.sleep(0.1)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        calls = relay.progress.call_args_list
        self.assertGreaterEqual(len(calls), 1, "no beat fired")
        for c in calls:
            self.assertEqual(c.args[1:], ("running", 0, 0))
            self.assertIs(c.kwargs.get("wait"), True,
                          f"beat must long-poll for a cancel; got {c}")

    async def test_cancel_sets_event_and_stops_beating(self):
        # A {"cancel": true} answer must (a) set the event _run_prompt reads and
        # (b) END the loop: the relay answers a canceled job's wait INSTANTLY, so
        # beating on would be a hot request loop against the relay.
        w, relay, _ = _make_worker()
        relay.progress.return_value = {"cancel": True}
        canceled = asyncio.Event()
        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", 0.01):
            await asyncio.wait_for(w._job_heartbeat("j1", canceled), 1)
        self.assertTrue(canceled.is_set(), "cancel must reach _run_prompt")
        beats = len(relay.progress.call_args_list)
        self.assertEqual(beats, 1, "should return on the first canceling beat")

    async def test_cancel_is_sticky_across_later_failures(self):
        # Once set, nothing may clear it — a later beat failure must not "un-cancel"
        # a job whose ComfyUI prompt we are already tearing down.
        w, relay, _ = _make_worker()
        relay.progress.side_effect = RuntimeError("relay down")
        canceled = asyncio.Event()
        canceled.set()
        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", 0.01), \
                mock.patch.object(worker, "JOB_HEARTBEAT_RETRY", 0.01):
            task = asyncio.create_task(w._job_heartbeat("j1", canceled))
            await asyncio.sleep(0.05)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        self.assertTrue(canceled.is_set())

    async def test_failed_beat_backs_off_instead_of_hot_looping(self):
        # A relay redeploy / network blip must not turn the beat into a spin.
        w, relay, _ = _make_worker()
        relay.progress.side_effect = RuntimeError("relay down")
        canceled = asyncio.Event()
        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", 0.01), \
                mock.patch.object(worker, "JOB_HEARTBEAT_RETRY", 0.05):
            task = asyncio.create_task(w._job_heartbeat("j1", canceled))
            await asyncio.sleep(0.25)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        beats = len(relay.progress.call_args_list)
        self.assertGreaterEqual(beats, 2, "must keep beating after a failure")
        self.assertLessEqual(beats, 8, f"failed beats hot-looped ({beats} in 0.25s)")
        self.assertFalse(canceled.is_set(), "a failed beat must never imply cancel")

    async def test_old_relay_returning_instantly_is_floored(self):
        # An old relay ignores `wait` and answers immediately. Without the
        # JOB_HEARTBEAT_INTERVAL floor that would be an unbounded request loop.
        w, relay, _ = _make_worker()  # returns {"cancel": False} instantly
        canceled = asyncio.Event()
        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", 0.05):
            task = asyncio.create_task(w._job_heartbeat("j1", canceled))
            await asyncio.sleep(0.25)
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        beats = len(relay.progress.call_args_list)
        self.assertLessEqual(beats, 8, f"instant answers hot-looped ({beats} in 0.25s)")
        self.assertGreaterEqual(beats, 2)


class TestOneBeaterPerJob(unittest.IsolatedAsyncioTestCase):
    """A running job pokes the relay from ONE place, not two.

    Before this change a generating job had two 20s beats — the background
    _job_heartbeat AND an inline one inside _run_prompt's poll loop — so the
    relay saw ~2 requests per interval. The inline one is gone; the job's whole
    relay traffic while generating is the single held beat.
    """

    async def test_generation_sends_one_beat_per_interval(self):
        w, relay, _ = _make_worker()
        started = asyncio.Event()
        release = asyncio.Event()

        async def fake_run_prompt(job_id, prompt, canceled):
            started.set()
            await release.wait()
            return "pid-1"

        async def fake_collect(prompt_id, output_format, job_prompt=None):
            return [], 0

        async def fake_upload(job_id, items):
            return []

        interval = 0.02
        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", interval), \
                mock.patch.object(w, "_run_prompt", fake_run_prompt), \
                mock.patch.object(w, "_collect_outputs", fake_collect), \
                mock.patch.object(w, "_upload_outputs", fake_upload):
            task = asyncio.create_task(w.handle_job({"id": "j1", "api_prompt": {}}))
            await asyncio.wait_for(started.wait(), 1)
            await asyncio.sleep(0.2)  # ~10 intervals of "generating"
            beats = _heartbeat_count(relay)
            release.set()
            self.assertEqual(await asyncio.wait_for(task, 1), "done")

        # ~10 intervals → ~11 requests (10 beats + handle_job's one-shot "running"
        # status flip). Two beaters would have made it ~21.
        self.assertLessEqual(beats, 15,
                             f"{beats} relay pokes in ~10 intervals — a second "
                             f"beater is back?")
        self.assertGreaterEqual(beats, 2)


def _heartbeat_count(relay) -> int:
    """How many progress() calls were 'running' heartbeats/reports so far."""
    return sum(
        1 for c in relay.progress.call_args_list
        if len(c.args) >= 2 and c.args[1] == "running"
    )


if __name__ == "__main__":
    unittest.main()
