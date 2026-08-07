"""随心跳上报「本机 ComfyUI 现状」—— 别人占了多少队列 + 已加载的 checkpoint。

中继此前对用户机器的了解只有「在线/离线」:它会往一台用户自己已经排了十个任务的
ComfyUI 上继续派活,也不知道哪台机器已经把某个 checkpoint 加载好了。这两项补的是
这个盲区。**本次只打通管道**(插件发、中继存),中继不拿它做任何决策。

⭐ 上报的是「**别人**占了多少」而不是队列总深度:中继自己就知道它往这台机器派了多少
活,总深度里我们那部分对它信息量为零。

这份用例钉住五件事,每一件都对应一种会在线上静默出问题的写法:

  ① **wire 形状**:两个键**无条件出现**,深度是真正的 int。
     中继每一拍无条件覆盖,所以「取到了才带上」会让中继侧的值闪断/残留;而一个
     float 形状的数字会让**整个请求体**解析失败,这台机器的心跳从此永久死掉
     (2026-08-01 的血教训,见 test_heartbeat_wire.py)。
  ② **0 与「不知道」分得开**:0 = 没有别人的活,-1 = 读不到。
  ③ ⛔ **减自己只在比对得上时减**:宁可多算一个,也不要减错(减多了 = 机器看起来
     比实际闲,那是将来调度最贵的一种错)。
  ④ **采集不在心跳的关键路径上**:ComfyUI 挂了/hang 住,心跳照发、照准时。
  ⑤ **失败是覆盖式的**:ComfyUI 挂了,下一拍必须报「不知道」,不能留着上一次的值。

Run:  python -m unittest discover -s tests
"""

import asyncio
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.jobs import checkpoint_of, last_checkpoint  # noqa: E402
from comfylink.relay import FOREIGN_QUEUE_UNKNOWN, RelayClient  # noqa: E402
from comfylink.worker import _ComfyProbe, _foreign_queue_depth  # noqa: E402

# ⚠️ 全部是编造的中性文件名/id —— 本仓库是**公开仓**,fixture 里不放任何真实用户的
# 模型名、路径或账号信息。
CKPT_A = "sdxl_base_1.0.safetensors"
CKPT_B = "anime_style_v2.safetensors"


def _prompt(*pairs) -> dict:
    """API 形态的 prompt:(node_id, ckpt_name) 若干。"""
    return {
        node: {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": name}}
        for node, name in pairs
    }


def _entry(prompt: dict, status_str: str = "success") -> dict:
    """一条 /history 记录(ComfyUI 的形状:prompt 是队列元组,graph 在下标 2)。"""
    return {
        "prompt": [0, "pid", prompt, {}, []],
        "outputs": {},
        "status": {"status_str": status_str, "completed": status_str == "success"},
    }


def _queue(running=(), pending=()):
    """ComfyUI /queue 的形状:每个条目的 prompt_id 在下标 1。"""
    def rows(ids):
        return [[i, pid, {}, {}, []] for i, pid in enumerate(ids)]
    return {"queue_running": rows(running), "queue_pending": rows(pending)}


# ── ① 纯函数:从 prompt / history 里认 checkpoint ────────────────────────────

class TestCheckpointOf(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(checkpoint_of(_prompt(("4", CKPT_A))), CKPT_A)

    def test_multiple_takes_the_lowest_node_id_deterministically(self):
        # 挑哪个都是猜,但必须挑得**确定** —— 否则同一个 workflow 会让这台机器的
        # 上报在两个模型之间来回跳,将来的调度会跟着抖。
        p = _prompt(("10", CKPT_B), ("4", CKPT_A))
        self.assertEqual(checkpoint_of(p), CKPT_A)
        self.assertEqual(checkpoint_of(dict(reversed(list(p.items())))), CKPT_A)

    def test_linked_input_is_not_a_filename(self):
        # 连线过来的输入是 ["4", 0],不是文件名。
        self.assertEqual(checkpoint_of({"5": {"inputs": {"ckpt_name": ["4", 0]}}}), "")

    def test_nothing_recognisable(self):
        self.assertEqual(checkpoint_of({}), "")
        self.assertEqual(checkpoint_of(None), "")
        # ⚠️ 已知缺口:走 UNETLoader(Flux 那类)的 workflow 报不出来 —— 那是另一个
        # 模型目录,不该混进同一个字段。
        self.assertEqual(checkpoint_of({"4": {"inputs": {"unet_name": "flux.sft"}}}), "")
        self.assertEqual(checkpoint_of({"4": "not a dict"}), "")
        self.assertEqual(checkpoint_of({"4": {"inputs": {"ckpt_name": "   "}}}), "")


class TestLastCheckpoint(unittest.TestCase):
    def test_newest_wins(self):
        # /history 是插入序,**最新的在最后**。
        hist = {"p1": _entry(_prompt(("4", CKPT_A))),
                "p2": _entry(_prompt(("4", CKPT_B)))}
        self.assertEqual(last_checkpoint(hist), CKPT_B)

    def test_skips_failed_runs(self):
        hist = {"p1": _entry(_prompt(("4", CKPT_A))),
                "p2": _entry(_prompt(("4", CKPT_B)), status_str="error")}
        self.assertEqual(last_checkpoint(hist), CKPT_A)

    def test_all_failed_reports_unknown(self):
        hist = {"p1": _entry(_prompt(("4", CKPT_A)), status_str="error")}
        self.assertEqual(last_checkpoint(hist), "")

    def test_missing_status_counts_as_success(self):
        # 老 ComfyUI 的 history 条目没有 status 键。一律判失败会让那些用户永远报空。
        self.assertEqual(
            last_checkpoint({"p1": {"prompt": [0, "pid", _prompt(("4", CKPT_A))]}}),
            CKPT_A)

    def test_garbage_never_raises(self):
        # 这个函数跑在心跳路径上,任何形状都只能归一成「不知道」。
        for bad in (None, [], "", {"p": None}, {"p": {"prompt": "x"}},
                    {"p": {"prompt": [0, "pid"]}}, {"p": {}}):
            self.assertEqual(last_checkpoint(bad), "", repr(bad))


# ── ② 纯函数:队列里有多少是别人的 ──────────────────────────────────────────

class TestForeignQueueDepth(unittest.TestCase):
    def test_nobody_else(self):
        self.assertEqual(_foreign_queue_depth(_queue(), set()), 0)

    def test_counts_running_and_pending(self):
        q = _queue(running=["a"], pending=["b", "c"])
        self.assertEqual(_foreign_queue_depth(q, set()), 3)

    def test_subtracts_exactly_our_own_prompt(self):
        q = _queue(running=["ours"], pending=["theirs-1", "theirs-2"])
        self.assertEqual(_foreign_queue_depth(q, {"ours"}), 2)

    def test_does_not_subtract_when_ours_is_not_in_the_queue(self):
        """⛔ 本类的核心用例:**比对不上就不减**。

        我们的 prompt 有两个窗口不在队列里:刚提交还没进队列,以及刚跑完已经离开。
        那时无脑减一就是**减多了**,中继会以为这台机器比实际更闲 —— 而「宁可多算
        一个」的代价只是少派一个任务。

        注入回归验证方式:把 _foreign_queue_depth 改成 `total - len(ours)`,
        本用例立刻红。
        """
        q = _queue(pending=["theirs-1", "theirs-2"])
        self.assertEqual(_foreign_queue_depth(q, {"ours"}), 2)

    def test_unrecognisable_entries_still_count(self):
        # 认不出 id 的条目仍然占着这台机器,跳过它 = 把机器说得比实际更闲。
        q = {"queue_running": [["short"]], "queue_pending": [None, [0, "theirs"]]}
        self.assertEqual(_foreign_queue_depth(q, set()), 3)

    def test_garbage_never_raises(self):
        for bad in (None, [], "", 3):
            self.assertEqual(_foreign_queue_depth(bad, set()), FOREIGN_QUEUE_UNKNOWN,
                             repr(bad))
        self.assertEqual(_foreign_queue_depth({}, set()), 0)


# ── ③ wire 形状:无条件发,int 出线,0 与「不知道」分得开 ────────────────────

def _client():
    """A RelayClient whose only wired collaborator is a recording _json."""
    relay = RelayClient.__new__(RelayClient)
    sent = []

    async def fake_json(method, path, body, timeout=None):
        sent.append((method, path, body))
        return {"ok": True}

    relay._json = fake_json
    return relay, sent


def _is_real_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


class TestHeartbeatWireShape(unittest.IsolatedAsyncioTestCase):
    async def test_both_keys_are_always_present_even_when_unknown(self):
        """⛔ 本文件的核心用例之一:**无条件发**。

        中继每一拍都用收到的值覆盖(读不到就覆盖成 NULL)。插件若写成「取到了才
        带上」,中继侧的值会在两拍之间闪断,或者留着一个陈旧的深度让将来的调度
        照着它派活。

        注入回归验证方式:把 RelayClient.heartbeat 里那两行改成条件带入
        (`if foreign_queue_depth >= 0:`),本用例立刻红。
        """
        relay, sent = _client()
        await relay.heartbeat("b1")           # 什么都没传 = 什么都不知道
        body = sent[0][2]
        self.assertIn("foreign_queue_depth", body,
                      "⛔ foreign_queue_depth 必须**每一拍都出现**,取不到就发 -1")
        self.assertIn("loaded_checkpoint", body,
                      "⛔ loaded_checkpoint 必须**每一拍都出现**,取不到就发 \"\"")
        self.assertEqual(body["foreign_queue_depth"], FOREIGN_QUEUE_UNKNOWN)
        self.assertEqual(body["loaded_checkpoint"], "")

    async def test_nobody_else_is_not_unknown(self):
        """⛔ 0(没有别人的活)和 -1(不知道)必须是两个不同的值。

        混成一件事会让一台我们其实什么都不知道的机器,在将来的调度里看起来像最闲
        的那台。
        """
        relay, sent = _client()
        await relay.heartbeat("b1", foreign_queue_depth=0)
        self.assertEqual(sent[0][2]["foreign_queue_depth"], 0)
        self.assertNotEqual(sent[0][2]["foreign_queue_depth"], FOREIGN_QUEUE_UNKNOWN)

    async def test_depth_goes_out_as_a_real_int(self):
        """同水位线那条血教训:float 形状会让中继的**整个请求体**解析失败。"""
        relay, sent = _client()
        for raw, want in ((3, 3), (3.0, 3), ("7", 7), (0, 0),
                          (None, -1), (-5, -1), (True, -1), ("x", -1), ([], -1)):
            sent.clear()
            await relay.heartbeat("b1", foreign_queue_depth=raw)
            v = sent[0][2]["foreign_queue_depth"]
            self.assertTrue(_is_real_int(v),
                            "深度必须以 int 出线,%r 变成了 %r" % (raw, v))
            self.assertEqual(v, want, repr(raw))

    async def test_checkpoint_is_always_a_string(self):
        relay, sent = _client()
        for raw, want in ((CKPT_A, CKPT_A), ("", ""), (None, "")):
            sent.clear()
            await relay.heartbeat("b1", loaded_checkpoint=raw)
            v = sent[0][2]["loaded_checkpoint"]
            self.assertIsInstance(v, str)
            self.assertEqual(v, want)

    async def test_existing_fields_are_untouched(self):
        """加这两个键不能动已有的 wire 契约(版本/commit/caps/水位线)。"""
        relay, sent = _client()
        await relay.heartbeat("b1", "abc", 1785561497.0, 2, CKPT_A)
        body = sent[0][2]
        self.assertEqual(body["backend_id"], "b1")
        self.assertEqual(body["object_info_hash"], "abc")
        self.assertEqual(body["object_info_synced_at"], 1785561497)
        self.assertTrue(_is_real_int(body["object_info_synced_at"]))
        self.assertIn("caps", body)
        self.assertIn("version", body)
        self.assertIn("commit", body)


# ── ④ 采集器本身 ────────────────────────────────────────────────────────────

def _comfy(queue=None, history=None, queue_exc=None, history_exc=None):
    c = mock.AsyncMock()
    c.queue.side_effect = queue_exc
    if queue_exc is None:
        c.queue.return_value = queue if queue is not None else _queue()
    c.recent_history.side_effect = history_exc
    if history_exc is None:
        c.recent_history.return_value = history if history is not None else {}
    return c


class TestComfyProbe(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        worker._OUR_PROMPTS.clear()
        self.addCleanup(worker._OUR_PROMPTS.clear)

    async def test_starts_out_unknown(self):
        self.assertEqual(_ComfyProbe(_comfy()).snapshot(), (FOREIGN_QUEUE_UNKNOWN, ""))

    async def test_refresh_populates_both(self):
        p = _ComfyProbe(_comfy(queue=_queue(pending=["x", "y", "z", "w"]),
                               history={"p": _entry(_prompt(("4", CKPT_A)))}))
        p.kick()
        await p.task
        self.assertEqual(p.snapshot(), (4, CKPT_A))

    async def test_our_own_running_job_is_not_counted_as_foreign(self):
        worker._OUR_PROMPTS.add("ours")
        p = _ComfyProbe(_comfy(queue=_queue(running=["ours"], pending=["theirs"])))
        p.kick()
        await p.task
        self.assertEqual(p.snapshot()[0], 1)

    async def test_one_side_failing_does_not_sink_the_other(self):
        p = _ComfyProbe(_comfy(queue=_queue(pending=["a"]),
                               history_exc=RuntimeError("boom")))
        p.kick()
        await p.task
        self.assertEqual(p.snapshot(), (1, ""))

        p = _ComfyProbe(_comfy(queue_exc=RuntimeError("boom"),
                               history={"p": _entry(_prompt(("4", CKPT_A)))}))
        p.kick()
        await p.task
        self.assertEqual(p.snapshot(), (FOREIGN_QUEUE_UNKNOWN, CKPT_A))

    async def test_failure_CLEARS_a_previously_good_value(self):
        """⛔ ComfyUI 挂了 ⇒ 下一拍必须报「不知道」,不能留着上一次的值。

        一个残留下来的旧队列深度比没有值更坏:将来的调度会照着它派活。
        """
        comfy = _comfy(queue=_queue(pending=["a", "b", "c"]),
                       history={"p": _entry(_prompt(("4", CKPT_A)))})
        p = _ComfyProbe(comfy)
        p.kick()
        await p.task
        self.assertEqual(p.snapshot(), (3, CKPT_A))

        comfy.queue.side_effect = OSError("connection refused")
        comfy.recent_history.side_effect = OSError("connection refused")
        p.kick()
        await p.task
        self.assertEqual(p.snapshot(), (FOREIGN_QUEUE_UNKNOWN, ""))

    async def test_absurd_queue_shape_is_unknown(self):
        for bad in ("nonsense", None, 7):
            comfy = _comfy()
            comfy.queue.return_value = bad
            p = _ComfyProbe(comfy)
            p.kick()
            await p.task
            self.assertEqual(p.snapshot()[0], FOREIGN_QUEUE_UNKNOWN, repr(bad))

    async def test_kick_is_deduped_while_one_is_in_flight(self):
        """一次采集跨了好几拍时不能堆出一串并发请求去踩一台本来就慢的 ComfyUI。"""
        gate = asyncio.Event()
        comfy = mock.AsyncMock()

        async def slow():
            await gate.wait()
            return _queue()

        comfy.queue.side_effect = slow
        comfy.recent_history.return_value = {}
        p = _ComfyProbe(comfy)
        for _ in range(5):
            p.kick()
            await asyncio.sleep(0)
        self.assertEqual(comfy.queue.await_count, 1)
        gate.set()
        await p.task

    async def test_no_comfy_client_is_a_no_op(self):
        p = _ComfyProbe(None)
        p.kick()
        self.assertIsNone(p.task)
        self.assertEqual(p.snapshot(), (FOREIGN_QUEUE_UNKNOWN, ""))

    async def test_never_raises(self):
        # _refresh 是 create_task 的目标,抛出去就是一个没人接的 exception。
        comfy = mock.AsyncMock()
        comfy.queue.side_effect = RuntimeError("boom")
        comfy.recent_history.side_effect = RuntimeError("boom")
        p = _ComfyProbe(comfy)
        p.kick()
        await p.task           # 不抛就是通过
        self.assertEqual(p.snapshot(), (FOREIGN_QUEUE_UNKNOWN, ""))


# ── ⑤ 心跳循环:采集绝不拖慢、绝不拖垮心跳 ──────────────────────────────────

class _Pairing:
    def __init__(self):
        self.backend_id = "b1"
        self.object_info_hash = ""
        self.object_info_synced_at = 0.0


class TestHeartbeatLoopIsNeverBlockedByTheProbe(unittest.IsolatedAsyncioTestCase):
    async def _run_beats(self, comfy, beats=3):
        """跑 `beats` 拍,返回每拍发给中继的 (foreign_queue_depth, checkpoint)。"""
        pairing = _Pairing()
        stop = asyncio.Event()
        seen = []
        relay = mock.AsyncMock()

        async def heartbeat(backend_id, oi_hash="", oi_synced=0.0,
                            foreign_queue_depth=FOREIGN_QUEUE_UNKNOWN,
                            loaded_checkpoint=""):
            seen.append((foreign_queue_depth, loaded_checkpoint))
            if len(seen) >= beats:
                stop.set()
            return {}

        relay.heartbeat.side_effect = heartbeat
        with mock.patch.object(worker.STATE, "get_pairing", return_value=pairing), \
                mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0), \
                mock.patch.object(worker, "_maybe_scan_models", lambda *a: None), \
                mock.patch.object(worker, "_maybe_refresh_object_info", lambda *a: None):
            await asyncio.wait_for(
                worker._heartbeat_loop(relay, pairing, stop, None, comfy), 5)
        return seen

    async def test_first_beat_is_unknown_then_the_value_lands(self):
        comfy = _comfy(queue=_queue(pending=["a", "b"]),
                       history={"p": _entry(_prompt(("4", CKPT_A)))})
        seen = await self._run_beats(comfy, beats=3)
        # 第一拍必然是「不知道」——采集排在发送之后,这完全可以接受(无条件覆盖语义)。
        self.assertEqual(seen[0], (FOREIGN_QUEUE_UNKNOWN, ""))
        self.assertIn((2, CKPT_A), seen[1:],
                      "第二拍起就该带上真值,实际:%r" % (seen,))

    async def test_a_hung_comfyui_never_stalls_or_breaks_the_beat(self):
        """⛔⛔ 本文件的另一个核心用例。

        ComfyUI **在自己的主事件循环上跑推理**,一步推理的间隙里它的 HTTP 就是不
        应答的 —— 也就是「队列很深」这个我们最想知道的时刻。心跳必须照发、照准时,
        报「不知道」。

        注入回归验证方式:把 _heartbeat_loop 里的 `probe.kick()` 换成
        `await probe._refresh()`,本用例会直接超时(5s)。
        """
        comfy = mock.AsyncMock()
        never = asyncio.Event()

        async def hang():
            await never.wait()

        comfy.queue.side_effect = hang
        comfy.recent_history.side_effect = hang
        seen = await self._run_beats(comfy, beats=3)
        self.assertEqual(len(seen), 3)
        for s in seen:
            self.assertEqual(s, (FOREIGN_QUEUE_UNKNOWN, ""))

    async def test_a_dead_comfyui_never_breaks_the_beat(self):
        comfy = _comfy(queue_exc=OSError("connection refused"),
                       history_exc=OSError("connection refused"))
        seen = await self._run_beats(comfy, beats=3)
        self.assertEqual(len(seen), 3)
        self.assertEqual(seen[-1], (FOREIGN_QUEUE_UNKNOWN, ""))

    async def test_probe_task_is_torn_down_with_the_loop(self):
        """采集 task 是这个循环的孩子 —— 配对拆掉时必须跟着走,不能泄漏。"""
        comfy = mock.AsyncMock()
        never = asyncio.Event()

        async def hang():
            await never.wait()

        comfy.queue.side_effect = hang
        comfy.recent_history.return_value = {}
        before = len(asyncio.all_tasks())
        await self._run_beats(comfy, beats=2)
        await asyncio.sleep(0)
        self.assertLessEqual(len(asyncio.all_tasks()), before + 1,
                             "采集 task 泄漏了")


# ── ⑥ 在途 prompt 的登记:任何出口都必须摘掉 ────────────────────────────────

class TestOurPromptsRegistry(unittest.IsolatedAsyncioTestCase):
    """⛔ 一个残留的 prompt_id 会让**后续所有心跳**把别人的一个任务误当成我们的,
    永久把这台机器说得比实际更闲。所以登记必须是 try/finally。
    """

    def setUp(self):
        worker._OUR_PROMPTS.clear()
        self.addCleanup(worker._OUR_PROMPTS.clear)

    async def _run(self, outcome):
        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.submit.return_value = "pid-1"
        w = worker.Worker(relay, comfy)
        seen = []

        async def fake_await(job_id, prompt_id, canceled):
            seen.append(set(worker._OUR_PROMPTS))
            if outcome is not None:
                raise outcome
            return prompt_id

        w._await_prompt = fake_await
        return w, seen

    async def test_registered_while_in_flight_and_removed_on_success(self):
        w, seen = await self._run(None)
        pid = await w._run_prompt("job-1", {}, None)
        self.assertEqual(pid, "pid-1")
        self.assertEqual(seen[0], {"pid-1"}, "在途期间必须登记")
        self.assertEqual(worker._OUR_PROMPTS, set(), "成功出口必须摘掉")

    async def test_removed_on_every_failure_exit(self):
        for exc in (worker.JobFailed("x"), worker.JobCanceled(), RuntimeError("boom")):
            w, seen = await self._run(exc)
            with self.assertRaises(type(exc)):
                await w._run_prompt("job-1", {}, None)
            self.assertEqual(seen[0], {"pid-1"})
            self.assertEqual(worker._OUR_PROMPTS, set(),
                             "⛔ %s 出口没有摘掉登记" % type(exc).__name__)


if __name__ == "__main__":
    unittest.main()
