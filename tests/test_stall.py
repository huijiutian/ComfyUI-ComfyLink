"""Tests for _run_prompt's version-independent REST polling model.

_run_prompt no longer parses the (version-drifting) ComfyUI websocket. It polls
the stable /history and /queue endpoints and decides the job's fate from REST
state. These tests pin the contract:

  * completion (history completed + outputs) -> returns the prompt id;
  * outputs that lag completion (#11540: completed-but-empty, then present) ->
    waits and returns once they appear;
  * an error status in /history -> JobFailed with the ComfyUI message;
  * the prompt vanishing (not in /history AND not in /queue) -> JobFailed(
    error_code="interrupted"), reported fast, with NO global interrupt;
  * a prompt wedged past the backstop -> JobFailed(error_code="execution_stalled"),
    again with NO global interrupt (never kill a user's local generation);
  * cancel of a PENDING prompt -> precise queue_delete (no interrupt);
  * cancel of a RUNNING prompt -> global interrupt (it IS ours on the GPU);
  * the heartbeat re-pokes the relay with a fixed (0, 0).

The crucial invariant throughout: the plugin issues a *global* interrupt ONLY
when it has confirmed OUR job is the one currently running on ComfyUI — never on
a mere stall, a pending job, or a vanished prompt — so it can never abort a
generation the user kicked off locally.

Run:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.worker import JobCanceled, JobFailed, Worker  # noqa: E402

PID = "pid-1"

# Fast timings so the polling loop spins quickly in tests (real defaults are
# ~1s poll / 5s grace / 30min backstop / 20s heartbeat).
_FAST = dict(POLL_INTERVAL=0.001, OUTPUTS_GRACE=0.05,
             EXECUTION_BACKSTOP_TIMEOUT=10.0, JOB_HEARTBEAT_INTERVAL=10.0)


def _fast(**overrides):
    """Patch the worker's timing constants (overrides win) for a test block."""
    return mock.patch.multiple(worker, **{**_FAST, **overrides})


def _completed(outputs=None):
    """A /history payload for PID: completed success with optional outputs."""
    return {PID: {"status": {"status_str": "success", "completed": True},
                  "outputs": outputs or {}}}


def _error(message):
    """A /history payload for PID: an execution error carrying `message`."""
    return {PID: {"status": {"status_str": "error", "completed": False,
                             "messages": [["execution_error",
                                           {"exception_message": message}]]},
                  "outputs": {}}}


def _running():
    return {"queue_running": [[0, PID, {}]], "queue_pending": []}


def _pending():
    return {"queue_running": [], "queue_pending": [[0, PID, {}]]}


def _empty_queue():
    return {"queue_running": [], "queue_pending": []}


class _Seq:
    """Returns successive values from a list; the LAST value repeats forever."""

    def __init__(self, values):
        self._values = list(values)

    def __call__(self):
        if len(self._values) > 1:
            return self._values.pop(0)
        return self._values[0]


class FakeComfy:
    """Stub ComfyClient: scripted history/queue, recorded submit/interrupt/delete."""

    def __init__(self, history=None, queue=None):
        self._history = _Seq(history if history is not None else [{}])
        self._queue = _Seq(queue if queue is not None else [_empty_queue()])
        self.submit = mock.AsyncMock(return_value=PID)
        self.interrupt = mock.AsyncMock()
        self.queue_delete = mock.AsyncMock()

    async def history(self, prompt_id):
        assert prompt_id == PID
        return self._history()

    async def queue(self):
        return self._queue()


def _make_worker(comfy, cancel=False):
    relay = mock.AsyncMock()
    relay.progress.return_value = {"cancel": cancel}
    return Worker(relay, comfy), relay


class TestRunPrompt(unittest.IsolatedAsyncioTestCase):
    async def test_completion_returns_pid(self):
        comfy = FakeComfy(history=[_completed({"9": {"images": [{"x": 1}]}})])
        w, _ = _make_worker(comfy)
        with _fast():
            pid = await asyncio.wait_for(w._run_prompt("j", {}), 3)
        self.assertEqual(pid, PID)
        comfy.interrupt.assert_not_awaited()
        comfy.queue_delete.assert_not_awaited()

    async def test_outputs_lag_then_appear(self):
        # completed-but-empty twice (#11540), then outputs land -> return.
        comfy = FakeComfy(history=[
            _completed({}),
            _completed({}),
            _completed({"9": {"images": [{"x": 1}]}}),
        ])
        w, _ = _make_worker(comfy)
        # Generous grace so we wait for the third poll rather than bailing early.
        with _fast(OUTPUTS_GRACE=1.0):
            pid = await asyncio.wait_for(w._run_prompt("j", {}), 3)
        self.assertEqual(pid, PID)
        comfy.interrupt.assert_not_awaited()

    async def test_outputs_never_appear_returns_after_grace(self):
        # completed but outputs stay empty forever -> return after grace (let
        # _collect_outputs report "no valid output"); no interrupt.
        comfy = FakeComfy(history=[_completed({})])
        w, _ = _make_worker(comfy)
        with _fast(OUTPUTS_GRACE=0.02):
            pid = await asyncio.wait_for(w._run_prompt("j", {}), 3)
        self.assertEqual(pid, PID)
        comfy.interrupt.assert_not_awaited()

    async def test_error_status_raises_jobfailed(self):
        comfy = FakeComfy(history=[_error("CUDA out of memory")])
        w, _ = _make_worker(comfy)
        with _fast(), self.assertRaises(JobFailed) as cm:
            await asyncio.wait_for(w._run_prompt("j", {}), 3)
        self.assertEqual(cm.exception.message, "CUDA out of memory")
        self.assertEqual(cm.exception.error_code, "")  # NOT a stall/interrupt
        comfy.interrupt.assert_not_awaited()

    async def test_interrupted_when_gone(self):
        # Not in history, not in queue -> interrupted, fast, NO global interrupt.
        comfy = FakeComfy(history=[{}], queue=[_empty_queue()])
        w, _ = _make_worker(comfy)
        with _fast(), self.assertRaises(JobFailed) as cm:
            await asyncio.wait_for(w._run_prompt("j", {}), 3)
        self.assertEqual(cm.exception.error_code, "interrupted")
        comfy.interrupt.assert_not_awaited()
        comfy.queue_delete.assert_not_awaited()

    async def test_backstop_fails_without_interrupt(self):
        # Always running, never completes -> backstop trips, but NO interrupt
        # (must never kill a user's local generation).
        comfy = FakeComfy(history=[{}], queue=[_running()])
        w, _ = _make_worker(comfy)
        with _fast(EXECUTION_BACKSTOP_TIMEOUT=0.03), self.assertRaises(JobFailed) as cm:
            await asyncio.wait_for(w._run_prompt("j", {}), 3)
        self.assertEqual(cm.exception.error_code, "execution_stalled")
        comfy.interrupt.assert_not_awaited()

    async def test_cancel_pending_uses_queue_delete(self):
        # Cancel while pending -> precise delete, NO global interrupt.
        comfy = FakeComfy(history=[{}], queue=[_pending()])
        w, _ = _make_worker(comfy, cancel=True)
        # heartbeat (and the cancel check) fires on the first iteration.
        with _fast(JOB_HEARTBEAT_INTERVAL=0.0), self.assertRaises(JobCanceled):
            await asyncio.wait_for(w._run_prompt("j", {}), 3)
        comfy.queue_delete.assert_awaited_once_with([PID])
        comfy.interrupt.assert_not_awaited()

    async def test_cancel_running_uses_interrupt(self):
        # Cancel while running -> global interrupt (the GPU job IS ours).
        comfy = FakeComfy(history=[{}], queue=[_running()])
        w, _ = _make_worker(comfy, cancel=True)
        with _fast(JOB_HEARTBEAT_INTERVAL=0.0), self.assertRaises(JobCanceled):
            await asyncio.wait_for(w._run_prompt("j", {}), 3)
        comfy.interrupt.assert_awaited_once()
        comfy.queue_delete.assert_not_awaited()

    async def test_heartbeat_reports_zero(self):
        # The loop's heartbeat re-pokes the relay with a fixed (0, 0).
        comfy = FakeComfy(history=[{}], queue=[_running()])
        w, relay = _make_worker(comfy, cancel=False)
        with _fast(JOB_HEARTBEAT_INTERVAL=0.0, EXECUTION_BACKSTOP_TIMEOUT=0.03), \
                self.assertRaises(JobFailed):  # eventually trips the backstop
            await asyncio.wait_for(w._run_prompt("j", {}), 3)
        self.assertTrue(
            any(c.args[1:] == ("running", 0, 0) for c in relay.progress.call_args_list),
            f"heartbeat should re-send (running, 0, 0); calls={relay.progress.call_args_list}",
        )

    async def test_jobfailed_default_error_code_is_empty(self):
        # A plain execution error must NOT masquerade as a stall/interrupt code.
        self.assertEqual(JobFailed("boom").error_code, "")
        self.assertEqual(JobFailed("boom", error_code="interrupted").error_code,
                         "interrupted")


if __name__ == "__main__":
    unittest.main()
