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
  * the loop makes NO relay request at all — the beat (and the cancel it picks
    up) moved to _job_heartbeat's held long-poll (see test_heartbeat.py); this
    loop only reads the `canceled` event that beat sets;
  * the local poll rate is decoupled from everything else — changing
    POLL_INTERVAL moves neither the backstop's real duration nor the (now zero)
    relay traffic (see TestPollRateDecoupled).

The crucial invariant throughout: the plugin issues a *global* interrupt ONLY
when it has confirmed OUR job is the one currently running on ComfyUI — never on
a mere stall, a pending job, or a vanished prompt — so it can never abort a
generation the user kicked off locally.

Run:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.worker import JobCanceled, JobFailed, Worker  # noqa: E402

PID = "pid-1"

# Fast timings so the polling loop spins quickly in tests (real defaults are
# 0.25s history poll / 1s queue poll / 5s grace / 30min backstop / 20s heartbeat).
_FAST = dict(POLL_INTERVAL=0.001, QUEUE_POLL_INTERVAL=0.001, OUTPUTS_GRACE=0.05,
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
    """Worker + its relay mock, plus the cancel event _run_prompt now reads.

    `cancel=True` pre-sets that event — i.e. "the background heartbeat already
    learned the relay wants this job canceled". _run_prompt itself no longer
    talks to the relay at all.
    """
    relay = mock.AsyncMock()
    relay.progress.return_value = {"cancel": False}
    canceled = asyncio.Event()
    if cancel:
        canceled.set()
    return Worker(relay, comfy), relay, canceled


class TestRunPrompt(unittest.IsolatedAsyncioTestCase):
    async def test_completion_returns_pid(self):
        comfy = FakeComfy(history=[_completed({"9": {"images": [{"x": 1}]}})])
        w, _, canceled = _make_worker(comfy)
        with _fast():
            pid = await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
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
        w, _, canceled = _make_worker(comfy)
        # Generous grace so we wait for the third poll rather than bailing early.
        with _fast(OUTPUTS_GRACE=1.0):
            pid = await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        self.assertEqual(pid, PID)
        comfy.interrupt.assert_not_awaited()

    async def test_outputs_never_appear_returns_after_grace(self):
        # completed but outputs stay empty forever -> return after grace (let
        # _collect_outputs report "no valid output"); no interrupt.
        comfy = FakeComfy(history=[_completed({})])
        w, _, canceled = _make_worker(comfy)
        with _fast(OUTPUTS_GRACE=0.02):
            pid = await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        self.assertEqual(pid, PID)
        comfy.interrupt.assert_not_awaited()

    async def test_error_status_raises_jobfailed(self):
        comfy = FakeComfy(history=[_error("CUDA out of memory")])
        w, _, canceled = _make_worker(comfy)
        with _fast(), self.assertRaises(JobFailed) as cm:
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        self.assertEqual(cm.exception.message, "CUDA out of memory")
        self.assertEqual(cm.exception.error_code, "")  # NOT a stall/interrupt
        comfy.interrupt.assert_not_awaited()

    async def test_interrupted_when_gone(self):
        # Not in history, not in queue -> interrupted, fast, NO global interrupt.
        comfy = FakeComfy(history=[{}], queue=[_empty_queue()])
        w, _, canceled = _make_worker(comfy)
        with _fast(), self.assertRaises(JobFailed) as cm:
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        self.assertEqual(cm.exception.error_code, "interrupted")
        comfy.interrupt.assert_not_awaited()
        comfy.queue_delete.assert_not_awaited()

    async def test_backstop_fails_without_interrupt(self):
        # Always running, never completes -> backstop trips, but NO interrupt
        # (must never kill a user's local generation).
        comfy = FakeComfy(history=[{}], queue=[_running()])
        w, _, canceled = _make_worker(comfy)
        with _fast(EXECUTION_BACKSTOP_TIMEOUT=0.03), self.assertRaises(JobFailed) as cm:
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        self.assertEqual(cm.exception.error_code, "execution_stalled")
        comfy.interrupt.assert_not_awaited()

    async def test_cancel_pending_uses_queue_delete(self):
        # Cancel while pending -> precise delete, NO global interrupt. The cancel
        # now arrives as a pre-set event (the heartbeat's held long-poll saw it);
        # the HANDLING below is unchanged, which is exactly what this pins.
        comfy = FakeComfy(history=[{}], queue=[_pending()])
        w, _, canceled = _make_worker(comfy, cancel=True)
        with _fast(), self.assertRaises(JobCanceled):
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        comfy.queue_delete.assert_awaited_once_with([PID])
        comfy.interrupt.assert_not_awaited()

    async def test_cancel_running_uses_interrupt(self):
        # Cancel while running -> global interrupt (the GPU job IS ours).
        comfy = FakeComfy(history=[{}], queue=[_running()])
        w, _, canceled = _make_worker(comfy, cancel=True)
        with _fast(), self.assertRaises(JobCanceled):
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        comfy.interrupt.assert_awaited_once()
        comfy.queue_delete.assert_not_awaited()

    async def test_cancel_set_mid_flight_is_honoured(self):
        # The event is read every iteration, so a cancel that lands WHILE the
        # loop is polling (the real case: the heartbeat task sets it) is picked
        # up within one poll — not only when it was already set on entry.
        comfy = FakeComfy(history=[{}], queue=[_running()])
        w, _, canceled = _make_worker(comfy)

        async def cancel_soon():
            await asyncio.sleep(0.05)
            canceled.set()

        with _fast(EXECUTION_BACKSTOP_TIMEOUT=5.0):
            asyncio.create_task(cancel_soon())
            with self.assertRaises(JobCanceled):
                await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        comfy.interrupt.assert_awaited_once()

    async def test_loop_never_touches_the_relay(self):
        # The inline beat MOVED OUT: this loop is local-only now. If a beat ever
        # creeps back in here we'd be back to two callers × one request each.
        comfy = FakeComfy(history=[{}], queue=[_running()])
        w, relay, canceled = _make_worker(comfy, cancel=False)
        with _fast(EXECUTION_BACKSTOP_TIMEOUT=0.03), \
                self.assertRaises(JobFailed):  # eventually trips the backstop
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        relay.progress.assert_not_awaited()
        self.assertEqual(relay.method_calls, [],
                         f"_run_prompt must make no relay call; got {relay.method_calls}")

    async def test_jobfailed_default_error_code_is_empty(self):
        # A plain execution error must NOT masquerade as a stall/interrupt code.
        self.assertEqual(JobFailed("boom").error_code, "")
        self.assertEqual(JobFailed("boom", error_code="interrupted").error_code,
                         "interrupted")


class TestPollRateDecoupled(unittest.IsolatedAsyncioTestCase):
    """POLL_INTERVAL must buy latency and nothing else.

    It is tempting to "just" lower the poll interval, but the loop used to
    derive both the backstop timeout and the relay heartbeat from an
    ``elapsed += POLL_INTERVAL`` accumulator — i.e. from the ITERATION COUNT.
    Under that model, polling 4× faster silently meant 4× the traffic to the
    relay (I-05's heartbeat is what keeps the reaper from failing a healthy
    long render) and a timeout that drifts with request latency.

    These two tests pin the decoupling by running the SAME scenario at two very
    different poll rates and asserting the wall-clock behaviour is unchanged.
    The relay half is now decoupled by construction — the beat moved to its own
    task — so the second test asserts the strongest form: ZERO relay requests
    from this loop at any poll rate.
    """

    async def _stall(self, poll_interval, backstop=0.2):
        """Run a never-completing prompt to the backstop; return (secs, beats)."""
        comfy = FakeComfy(history=[{}], queue=[_running()])
        w, relay, canceled = _make_worker(comfy)
        t0 = time.monotonic()
        with _fast(POLL_INTERVAL=poll_interval,
                   QUEUE_POLL_INTERVAL=poll_interval,
                   EXECUTION_BACKSTOP_TIMEOUT=backstop), \
                self.assertRaises(JobFailed) as cm:
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 10)
        self.assertEqual(cm.exception.error_code, "execution_stalled")
        return time.monotonic() - t0, len(relay.progress.call_args_list)

    async def test_backstop_duration_does_not_move_with_poll_rate(self):
        # 4× faster polling must NOT make the safety timeout trip 4× sooner.
        # Lower bound only in the strict sense (a loaded machine may overshoot);
        # the bug this guards against is an EARLY trip.
        for poll in (0.005, 0.02):
            with self.subTest(poll=poll):
                secs, _ = await self._stall(poll, backstop=0.2)
                self.assertGreaterEqual(secs, 0.2)
                self.assertLess(secs, 2.0)

    async def test_relay_beats_do_not_scale_with_poll_rate(self):
        # Was "beats must not multiply with the poll rate"; now the loop sends
        # none at all, at either rate (the beat lives in _job_heartbeat, paced by
        # the relay's own hold).
        _, fast_beats = await self._stall(0.005, backstop=0.2)
        _, slow_beats = await self._stall(0.02, backstop=0.2)
        self.assertEqual((fast_beats, slow_beats), (0, 0),
                         "_run_prompt must send no relay beat at any poll rate")


if __name__ == "__main__":
    unittest.main()
