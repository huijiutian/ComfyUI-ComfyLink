"""Tests for Worker._upload_outputs' bounded-concurrent R2 upload.

_upload_outputs fans the per-output (sign_upload + put_object) work out
concurrently, capped by UPLOAD_CONCURRENCY, instead of uploading one-at-a-time.
These tests pin the two contracts that matter:

  * ORDER: the returned payload list stays aligned with the input `items`
    order even when uploads complete out of order (the app renders outputs in
    sequence, so a reorder would misattribute images);
  * FAILURE: a single sign_upload/put_object exception propagates out of
    _upload_outputs (and up through handle_job as a failed result) — no image
    is silently dropped or reported as done.

Concurrency DEGREE itself is intentionally not asserted (flaky/timing-bound);
order + failure propagation are the load-bearing invariants.

Run:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.worker import Worker  # noqa: E402


def _item(i: int) -> dict:
    """A collected-output dict shaped exactly like _collect_outputs emits."""
    return {"data": f"bytes-{i}".encode(), "filename": f"out_{i}.webp",
            "content_type": "image/webp", "subfolder": "", "type": "output",
            "media_type": "image"}


class _OrderingRelay:
    """Fake relay whose put_object completes in REVERSE submission order.

    sign_upload returns a key derived from the filename; put_object sleeps so
    that the LAST-submitted upload finishes FIRST. If _upload_outputs leaked
    completion order into its result, the payloads would come back reversed —
    so a still-in-input-order result proves gather preserves `items` order.
    """

    def __init__(self, count: int):
        self._count = count
        self._n = 0

    async def sign_upload(self, job_id, kind, filename, ct):
        return (f"r2/{filename}", f"https://put/{filename}")

    async def put_object(self, url, data, ct):
        # Earlier submissions sleep longer → completions are reversed.
        idx = self._n
        self._n += 1
        await asyncio.sleep(0.02 * (self._count - idx))


class _FailingRelay:
    """Fake relay that raises on put_object for one specific filename."""

    def __init__(self, fail_filename: str):
        self._fail = fail_filename
        self.uploaded: list[str] = []

    async def sign_upload(self, job_id, kind, filename, ct):
        return (f"r2/{filename}", f"https://put/{filename}")

    async def put_object(self, url, data, ct):
        if url.endswith(self._fail):
            raise RuntimeError("R2 upload blew up")
        self.uploaded.append(url)


class TestCollectPassesJobPrompt(unittest.IsolatedAsyncioTestCase):
    """R-1.0.6-26:_collect_outputs 必须把 job 的 prompt 交给
    encode_output(直通字段/workflow JSON 的主源),不能靠源图元数据。"""

    async def test_job_prompt_reaches_encode_output(self):
        class _Comfy:
            async def history(self, pid):
                return {pid: {"outputs": {"17": {"images": [
                    {"filename": "a.webp", "subfolder": "", "type": "output"},
                ]}}}}

            async def view(self, filename, subfolder, typ):
                return b"raw-bytes"

        w = Worker.__new__(Worker)  # 只测 _collect_outputs,不走完整构造
        w.comfy = _Comfy()
        job_prompt = {"3": {"class_type": "KSampler", "inputs": {}}}
        seen = {}

        def fake_encode(data, filename, output_format, media_type="image",
                        job_prompt=None):
            seen["job_prompt"] = job_prompt
            return data, filename, "image/webp"

        with mock.patch("comfylink.worker.encode_output", fake_encode):
            items, total = await w._collect_outputs("pid", "webp", job_prompt)
        self.assertIs(seen["job_prompt"], job_prompt)
        self.assertEqual(total, len(b"raw-bytes"))
        self.assertEqual(items[0]["filename"], "a.webp")


class TestUploadOutputs(unittest.IsolatedAsyncioTestCase):
    async def test_result_preserves_input_order(self):
        items = [_item(i) for i in range(5)]
        w = Worker(_OrderingRelay(len(items)), mock.AsyncMock())
        out = await asyncio.wait_for(w._upload_outputs("job-1", items), 3)
        # Same length, same order, correct payload shape.
        self.assertEqual([p["filename"] for p in out],
                         [it["filename"] for it in items])
        self.assertEqual([p["r2_key"] for p in out],
                         [f"r2/{it['filename']}" for it in items])
        for p in out:
            self.assertEqual(
                set(p), {"r2_key", "filename", "subfolder", "type",
                         "media_type", "content_type"})

    async def test_single_item_still_works(self):
        # batchSize=1 must degrade to one task, no special-casing.
        items = [_item(0)]
        w = Worker(_OrderingRelay(1), mock.AsyncMock())
        out = await asyncio.wait_for(w._upload_outputs("job-1", items), 3)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["filename"], "out_0.webp")

    async def test_one_failure_propagates(self):
        items = [_item(i) for i in range(4)]
        relay = _FailingRelay(fail_filename="out_2.webp")
        w = Worker(relay, mock.AsyncMock())
        with self.assertRaises(RuntimeError):
            await asyncio.wait_for(w._upload_outputs("job-1", items), 3)

    async def test_failure_bubbles_through_handle_job(self):
        # A single upload failure ends the job as "failed" (never "done").
        relay = mock.AsyncMock()
        relay.sign_upload.return_value = ("r2/x", "https://put/x")
        relay.put_object.side_effect = RuntimeError("uplink died")
        w = Worker(relay, mock.AsyncMock())
        with mock.patch.object(
            Worker, "_stage_inputs", new=mock.AsyncMock()
        ), mock.patch.object(
            Worker, "_run_prompt", new=mock.AsyncMock(return_value="pid-x")
        ), mock.patch.object(
            Worker, "_collect_outputs",
            new=mock.AsyncMock(return_value=([_item(0), _item(1)], 100)),
        ):
            status = await asyncio.wait_for(
                w.handle_job({"id": "job-1", "max_output_bytes": 10_000_000}), 3)
        self.assertEqual(status, "failed")
        # The failed result was reported to the relay (not a "done").
        reported = [c.args[1] for c in relay.result.await_args_list]
        self.assertIn("failed", reported)
        self.assertNotIn("done", reported)


if __name__ == "__main__":
    unittest.main()
