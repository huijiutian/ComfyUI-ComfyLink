"""Tests for the no-activity execution-stall watchdog in _run_prompt.

The relay's reaper deliberately won't reclaim a job whose worker is wedged
waiting on a silent ComfyUI while the backend process is otherwise alive. The
plugin self-heals: if NOTHING arrives on the job's websocket for
EXECUTION_STALL_TIMEOUT seconds it declares the execution stalled and reports
failed(error_code='execution_stalled'), freeing the worker. These tests pin:

  * complete silence past the threshold -> failed with execution_stalled, and
    ComfyUI is interrupted so the wedged prompt is abandoned;
  * a job that keeps emitting activity NEVER trips the watchdog (it finishes),
    even when each gap is shorter than the threshold.

Run:  python -m unittest discover -s tests
"""

import asyncio
import json
import os
import sys
import unittest
from unittest import mock

import aiohttp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.worker import JobFailed, Worker  # noqa: E402


class _Msg:
    """Minimal stand-in for an aiohttp WSMessage (type + data)."""

    def __init__(self, type_, data=""):
        self.type = type_
        self.data = data


def _text(payload: dict) -> _Msg:
    return _Msg(aiohttp.WSMsgType.TEXT, json.dumps(payload))


class _FakeWS:
    """A websocket whose receive() replays a script of (delay, message) steps.

    Frames are pushed onto an internal asyncio.Queue by a background feeder that
    honors each step's ``delay`` (simulating ComfyUI silence). receive() just
    awaits the queue. This faithfully mirrors real aiohttp behavior where
    receive() reads from a buffered reader: wrapping it in
    asyncio.wait_for(timeout=poll) and re-calling after a timeout RESUMES from
    the buffer rather than losing data — exactly the contract _run_prompt relies
    on. A ``None`` message means "feed nothing further" (stay silent forever),
    used to force a stall.
    """

    def __init__(self, script):
        self._script = list(script)
        self._queue: asyncio.Queue = asyncio.Queue()
        self._feeder: asyncio.Task | None = None

    async def __aenter__(self):
        self._feeder = asyncio.create_task(self._feed())
        return self

    async def __aexit__(self, *exc):
        if self._feeder:
            self._feeder.cancel()
        return False

    async def _feed(self):
        for delay, msg in self._script:
            if delay:
                await asyncio.sleep(delay)
            if msg is None:
                return  # go silent forever — nothing more is ever enqueued
            await self._queue.put(msg)

    async def receive(self, timeout=None):
        # Mirror aiohttp's receive(timeout=): a poll timeout raises
        # asyncio.TimeoutError without consuming a frame; the next call resumes
        # from the queue (no data lost).
        if timeout is None:
            return await self._queue.get()
        return await asyncio.wait_for(self._queue.get(), timeout)


def _make_worker(ws):
    relay = mock.AsyncMock()
    relay.progress.return_value = {"cancel": False}
    comfy = mock.AsyncMock()
    comfy.ws_connect = mock.MagicMock(return_value=ws)
    comfy.submit.return_value = "pid-1"
    return Worker(relay, comfy), relay, comfy


class TestExecutionStall(unittest.IsolatedAsyncioTestCase):
    async def test_silence_past_threshold_fails_with_stalled_code(self):
        # ComfyUI accepts the prompt then goes completely silent.
        ws = _FakeWS([(None, None)])
        w, relay, comfy = _make_worker(ws)

        # Fast timings: poll every 10ms, declare stalled after 50ms of
        # cumulative silence. Real defaults are 20s poll / 600s stall.
        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", 0.01), \
                mock.patch.object(worker, "EXECUTION_STALL_TIMEOUT", 0.05):
            status = await asyncio.wait_for(
                w.handle_job({"id": "j-stall", "api_prompt": {}}), 2
            )

        self.assertEqual(status, "failed")
        comfy.interrupt.assert_awaited()  # the wedged prompt is abandoned
        # The relay result carried the machine-readable stall code.
        relay.result.assert_awaited()
        call = relay.result.await_args
        self.assertEqual(call.args[0], "j-stall")
        self.assertEqual(call.args[1], "failed")
        self.assertEqual(call.kwargs.get("error_code"), "execution_stalled")

    async def test_steady_activity_never_trips_watchdog(self):
        # Each frame arrives after a gap SHORTER than the stall window; the
        # window resets every time, so the job finishes normally. The final
        # `executing node=None` frame signals done.
        script = [
            (0.02, _text({"type": "progress",
                          "data": {"value": 1, "max": 3, "prompt_id": "pid-1"}})),
            (0.02, _text({"type": "progress",
                          "data": {"value": 2, "max": 3, "prompt_id": "pid-1"}})),
            (0.02, _text({"type": "progress",
                          "data": {"value": 3, "max": 3, "prompt_id": "pid-1"}})),
            (0.02, _text({"type": "executing",
                          "data": {"node": None, "prompt_id": "pid-1"}})),
        ]
        ws = _FakeWS(script)
        w, relay, comfy = _make_worker(ws)

        async def fake_collect(prompt_id, output_format):
            return [], 0

        async def fake_upload(job_id, items):
            return []

        # Stall window (0.2s) >> each gap (0.02s), poll fine-grained at 0.01s,
        # so the silence clock resets on every frame and never reaches 0.2s.
        with mock.patch.object(worker, "JOB_HEARTBEAT_INTERVAL", 0.01), \
                mock.patch.object(worker, "EXECUTION_STALL_TIMEOUT", 0.2), \
                mock.patch.object(w, "_collect_outputs", fake_collect), \
                mock.patch.object(w, "_upload_outputs", fake_upload):
            status = await asyncio.wait_for(
                w.handle_job({"id": "j-ok", "api_prompt": {}}), 2
            )

        self.assertEqual(status, "done")
        comfy.interrupt.assert_not_awaited()  # never declared stalled
        # No failed result was sent for this job.
        for c in relay.result.await_args_list:
            self.assertNotEqual(c.args[1] if len(c.args) > 1 else None, "failed")

    async def test_jobfailed_default_error_code_is_empty(self):
        # A plain execution error must NOT masquerade as execution_stalled.
        e = JobFailed("boom")
        self.assertEqual(e.error_code, "")
        e2 = JobFailed("boom", error_code="execution_stalled")
        self.assertEqual(e2.error_code, "execution_stalled")


if __name__ == "__main__":
    unittest.main()
