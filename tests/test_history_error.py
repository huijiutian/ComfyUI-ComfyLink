"""Tests for _history_failure — how a /history error block becomes a result.

⭐ WHY THIS FILE EXISTS (2026-08-07). Production metrics over 14 days: 81% of
failures carried an EMPTY error_code, and once the free text was expanded the
second-biggest bucket was the bare fallback string "ComfyUI reported an
execution error" (5 failures / 5 users). The old ``_history_error`` recognised
exactly ONE payload shape — a ``["execution_error", {...}]`` pair with a
non-empty ``exception_message`` — and every other shape fell through to that
useless string. It had NO tests at all; the only shape ever exercised was the
one fixture in tests/test_stall.py.

So this file pins EVERY shape we know ComfyUI emits, plus the degenerate ones:

  * execution_error with exception_message                 -> message + node
  * execution_error with only exception_type               -> type + node
  * execution_error with exception_message: ""             -> falls to the TYPE,
                                                              not to the generic
  * execution_error with no node_type/node_id              -> plain detail
  * execution_interrupted (no exception_* at all)          -> "interrupted"
  * status.messages missing entirely                       -> generic + code
  * status.messages serialised as a dict                   -> handled
  * malformed entries (bare string / 1-tuple / non-dict payload) -> skipped
  * a giant exception_message                              -> bounded

and the CANCEL RACE in _run_prompt: an interrupt that is OURS must come out as
JobCanceled, never as a failed job.

Run:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.worker import (  # noqa: E402
    EXECUTION_ERROR,
    INTERRUPTED,
    JobCanceled,
    JobFailed,
    Worker,
    _history_debug,
    _history_failure,
    _history_messages,
)

GENERIC = "ComfyUI reported an execution error"
PID = "pid-1"

_FAST = dict(POLL_INTERVAL=0.001, QUEUE_POLL_INTERVAL=0.001, OUTPUTS_GRACE=0.05,
             EXECUTION_BACKSTOP_TIMEOUT=10.0, JOB_HEARTBEAT_INTERVAL=10.0)


def _fast(**overrides):
    return mock.patch.multiple(worker, **{**_FAST, **overrides})


def _status(messages):
    """A /history status block with status_str == "error"."""
    return {"status_str": "error", "completed": False, "messages": messages}


# ── the payload shapes ───────────────────────────────────────────────────────


class TestHistoryFailure(unittest.TestCase):
    def test_exception_message_with_node_type(self):
        msg, code = _history_failure(_status([
            ["execution_error", {"exception_message": "CUDA out of memory",
                                 "exception_type": "torch.OutOfMemoryError",
                                 "node_id": "12", "node_type": "KSampler",
                                 "traceback": ["Traceback...\n"],
                                 "current_inputs": {"seed": [1]}}],
        ]))
        # The node is what makes this actionable — that is the whole point.
        self.assertEqual(msg, "KSampler (#12): CUDA out of memory")
        self.assertEqual(code, EXECUTION_ERROR)

    def test_node_type_without_node_id(self):
        msg, _ = _history_failure(_status([
            ["execution_error", {"exception_message": "boom", "node_type": "VAEDecode"}],
        ]))
        self.assertEqual(msg, "VAEDecode: boom")

    def test_node_id_without_node_type(self):
        msg, _ = _history_failure(_status([
            ["execution_error", {"exception_message": "boom", "node_id": 7}],
        ]))
        self.assertEqual(msg, "#7: boom")

    def test_no_node_info_is_the_bare_detail(self):
        msg, code = _history_failure(_status([
            ["execution_error", {"exception_message": "boom"}],
        ]))
        self.assertEqual(msg, "boom")
        self.assertEqual(code, EXECUTION_ERROR)

    def test_empty_exception_message_falls_to_type_not_to_generic(self):
        # ⭐ The `or`-chain bug: exception_message is "" (ComfyUI does emit this),
        # and the old chain skipped straight past a perfectly good type.
        msg, code = _history_failure(_status([
            ["execution_error", {"exception_message": "   ",
                                 "exception_type": "ValueError",
                                 "node_type": "CLIPTextEncode"}],
        ]))
        self.assertEqual(msg, "CLIPTextEncode: ValueError")
        self.assertEqual(code, EXECUTION_ERROR)

    def test_no_detail_at_all_still_names_the_node(self):
        msg, code = _history_failure(_status([
            ["execution_error", {"node_type": "KSampler", "node_id": "3"}],
        ]))
        self.assertEqual(msg, f"{GENERIC} in KSampler (#3)")
        self.assertEqual(code, EXECUTION_ERROR)

    def test_execution_interrupted_is_not_an_execution_error(self):
        # The most likely of the five production cases: the browser's interrupt
        # button, or our own /interrupt. status_str is "error" but the event name
        # differs AND the payload has no exception_message at all.
        msg, code = _history_failure(_status([
            ["execution_start", {"prompt_id": PID}],
            ["execution_interrupted", {"prompt_id": PID, "node_id": "12",
                                       "node_type": "KSampler"}],
        ]))
        self.assertEqual(code, INTERRUPTED)
        self.assertEqual(msg, "job was interrupted on ComfyUI")

    def test_execution_error_wins_over_a_later_interrupt(self):
        # A node raised and the run was then torn down: the EXCEPTION is the
        # useful thing to tell the user, not the teardown.
        msg, code = _history_failure(_status([
            ["execution_error", {"exception_message": "boom", "node_type": "KSampler"}],
            ["execution_interrupted", {"prompt_id": PID}],
        ]))
        self.assertEqual(code, EXECUTION_ERROR)
        self.assertIn("boom", msg)

    def test_messages_missing_entirely(self):
        # Older / forked ComfyUI builds write only status_str.
        msg, code = _history_failure({"status_str": "error", "completed": False})
        self.assertEqual(msg, GENERIC)
        self.assertEqual(code, EXECUTION_ERROR)

    def test_messages_serialised_as_a_dict(self):
        msg, code = _history_failure(_status(
            {"execution_error": {"exception_message": "boom", "node_type": "KSampler"}}
        ))
        self.assertEqual(msg, "KSampler: boom")
        self.assertEqual(code, EXECUTION_ERROR)

    def test_dict_shaped_interrupt(self):
        msg, code = _history_failure(_status({"execution_interrupted": {"prompt_id": PID}}))
        self.assertEqual(code, INTERRUPTED)
        self.assertEqual(msg, "job was interrupted on ComfyUI")

    def test_malformed_entries_are_skipped_not_raised(self):
        for messages in (
            "execution_error",                      # a bare string
            ["execution_error"],                    # a list of strings
            [["execution_error"]],                  # 1-element pair
            [[None, {"exception_message": "boom"}]],  # non-str event name
            [[]],
            [42],
            None,
            123,
        ):
            msg, code = _history_failure(_status(messages))
            self.assertEqual(msg, GENERIC, messages)
            self.assertEqual(code, EXECUTION_ERROR, messages)

    def test_non_dict_payload_is_tolerated(self):
        msg, code = _history_failure(_status([["execution_error", "boom"]]))
        self.assertEqual(msg, GENERIC)
        self.assertEqual(code, EXECUTION_ERROR)
        # Same for the interrupt event.
        _msg, code = _history_failure(_status([["execution_interrupted", None]]))
        self.assertEqual(code, INTERRUPTED)

    def test_huge_message_is_bounded(self):
        msg, _ = _history_failure(_status([
            ["execution_error", {"exception_message": "x" * 5000}],
        ]))
        self.assertLessEqual(len(msg), worker._MAX_DETAIL_CHARS + 8)
        self.assertTrue(msg.endswith("…"))

    def test_never_raises_on_junk(self):
        for status in ({}, {"messages": object()}, {"messages": {1: 2}}):
            msg, code = _history_failure(status)
            self.assertEqual(code, EXECUTION_ERROR)
            self.assertEqual(msg, GENERIC)


class TestHistoryMessages(unittest.TestCase):
    def test_payload_is_always_a_dict(self):
        for _name, payload in _history_messages(_status([
            ["execution_error", None], ["execution_interrupted", "x"],
        ])):
            self.assertIsInstance(payload, dict)


class TestHistoryDebug(unittest.TestCase):
    """The traceback stays LOCAL — it is logged, never uploaded."""

    def test_traceback_extracted_for_the_local_log(self):
        detail = _history_debug(_status([
            ["execution_error", {"exception_message": "boom",
                                 "traceback": ["Traceback (most recent call last):\n",
                                               "  File \"/x/y.py\", line 1\n"]}],
        ]))
        self.assertIn("Traceback (most recent call last):", detail)

    def test_traceback_never_reaches_the_reported_message(self):
        status = _status([
            ["execution_error", {"exception_message": "boom",
                                 "traceback": ["/home/somebody/private/path.py\n"],
                                 "current_inputs": {"text": ["a private prompt"]}}],
        ])
        msg, _ = _history_failure(status)
        self.assertNotIn("private", msg)
        self.assertNotIn("path.py", msg)

    def test_no_traceback_is_empty(self):
        self.assertEqual(_history_debug(_status([["execution_error", {}]])), "")
        self.assertEqual(_history_debug({}), "")


# ── the cancel race in _run_prompt ───────────────────────────────────────────


class _Comfy:
    """Stub ComfyClient returning a fixed history/queue."""

    def __init__(self, history, queue=None):
        self._history = history
        self._queue = queue or {"queue_running": [], "queue_pending": []}
        self.submit = mock.AsyncMock(return_value=PID)
        self.interrupt = mock.AsyncMock()
        self.queue_delete = mock.AsyncMock()

    async def history(self, prompt_id):
        return self._history

    async def queue(self):
        return self._queue


def _worker(comfy, cancel=False):
    canceled = asyncio.Event()
    if cancel:
        canceled.set()
    return Worker(mock.AsyncMock(), comfy), canceled


class TestCancelBeatsFailure(unittest.IsolatedAsyncioTestCase):
    async def test_our_interrupt_in_history_reports_canceled(self):
        # ⭐ THE RACE. The cancel event is set by the heartbeat task, which can
        # flip it during ANY await in the poll loop. If ComfyUI has already
        # written execution_interrupted by the time we read /history, the old
        # code called that a FAILED job — the user pressed cancel and got an
        # error. Cancel must win.
        comfy = _Comfy({PID: {"status": _status([["execution_interrupted",
                                                  {"prompt_id": PID}]]),
                              "outputs": {}}})
        w, canceled = _worker(comfy)

        real_history = comfy.history

        async def racing_history(prompt_id):
            canceled.set()          # the heartbeat lands mid-await
            return await real_history(prompt_id)

        comfy.history = racing_history
        with _fast(), self.assertRaises(JobCanceled):
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)

    async def test_external_interrupt_is_still_a_failure(self):
        # Nobody asked us to cancel -> the user interrupted in their browser (or
        # something else did). That IS a failure of this job, and it now carries
        # the same "interrupted" code the vanished-from-queue path reports.
        comfy = _Comfy({PID: {"status": _status([["execution_interrupted",
                                                  {"prompt_id": PID}]]),
                              "outputs": {}}})
        w, canceled = _worker(comfy)
        with _fast(), self.assertRaises(JobFailed) as cm:
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        self.assertEqual(cm.exception.error_code, INTERRUPTED)

    async def test_vanished_prompt_during_a_cancel_reports_canceled(self):
        # Same race on the other branch: our own queue_delete/interrupt is
        # precisely what makes the prompt disappear from both history and queue.
        comfy = _Comfy({})
        w, canceled = _worker(comfy)

        async def racing_queue():
            canceled.set()
            return {"queue_running": [], "queue_pending": []}

        comfy.queue = racing_queue
        with _fast(), self.assertRaises(JobCanceled):
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)

    async def test_real_execution_error_during_a_cancel_stays_failed(self):
        # ⛔ The cancel override is scoped to interrupt-shaped terminations. A
        # node that genuinely raised must still be reported as a failure with
        # its message — swallowing it into "canceled" would hide a real bug.
        comfy = _Comfy({PID: {"status": _status([["execution_error",
                                                  {"exception_message": "boom",
                                                   "node_type": "KSampler"}]]),
                              "outputs": {}}})
        w, canceled = _worker(comfy)
        real_history = comfy.history

        async def racing_history(prompt_id):
            canceled.set()          # cancel lands mid-await, same race as above
            return await real_history(prompt_id)

        comfy.history = racing_history
        with _fast(), self.assertRaises(JobFailed) as cm:
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)
        self.assertEqual(cm.exception.error_code, EXECUTION_ERROR)


class TestCancelIsBestEffort(unittest.IsolatedAsyncioTestCase):
    """A cancel we couldn't deliver must never surface as a FAILED job."""

    async def test_interrupt_failure_does_not_escape(self):
        comfy = _Comfy({}, queue={"queue_running": [[0, PID, {}]], "queue_pending": []})
        comfy.interrupt = mock.AsyncMock(side_effect=OSError("connection reset"))
        w, _canceled = _worker(comfy)
        await w._cancel_comfy(PID)  # must not raise
        comfy.interrupt.assert_awaited_once()

    async def test_queue_delete_failure_does_not_escape(self):
        comfy = _Comfy({}, queue={"queue_running": [], "queue_pending": [[0, PID, {}]]})
        comfy.queue_delete = mock.AsyncMock(side_effect=OSError("connection reset"))
        w, _canceled = _worker(comfy)
        await w._cancel_comfy(PID)  # must not raise
        comfy.queue_delete.assert_awaited_once()

    async def test_cancel_path_end_to_end_still_reports_canceled(self):
        # The whole reason the above matters: an exception from interrupt() used
        # to escape _run_prompt, miss `except JobCanceled` in handle_job, and be
        # reported FAILED.
        comfy = _Comfy({}, queue={"queue_running": [[0, PID, {}]], "queue_pending": []})
        comfy.interrupt = mock.AsyncMock(side_effect=OSError("connection reset"))
        w, canceled = _worker(comfy, cancel=True)
        with _fast(), self.assertRaises(JobCanceled):
            await asyncio.wait_for(w._run_prompt("j", {}, canceled), 3)


if __name__ == "__main__":
    unittest.main()
