"""心跳的**线上形状**与**失败可见性** —— 2026-08-01 那次静默失效的回归护栏。

用户报「App 点刷新一直干等」。在他的 ComfyUI 上直接向中继发心跳,实测到的分野是:

    {"backend_id": "...", "object_info_synced_at": 1785561497}    -> HTTP 200
    {"backend_id": "...", "object_info_synced_at": 1785561497.0}  -> HTTP 400
                                                {"error":"backend_id required"}

完整链条(每一环都在下面有对应的用例):

  1. ``Pairing.object_info_synced_at`` 是 Python ``float`` ⇒ 序列化成 ``1785561497.0``;
  2. 中继的字段是 Go ``int64`` ⇒ **整个请求体**解析失败 ⇒ 结构体全零值;
  3. ⇒ 中继回「backend_id required」,一句与真实原因毫无关系的话;
  4. 插件把心跳异常一律压成 ``log.debug`` ⇒ 日志级别是 INFO ⇒ **永远看不见**;
  5. ⇒ ``resp is None`` ⇒ 心跳循环整段分发被跳过 ⇒ 刷新信号永远读不到,
     ``last_seen`` 也不再推进(App 里这台机器会显示离线)。

⛔ 为什么之前一直没暴露:0 值**不发这个字段**,所以从没服务过刷新的 backend 一切
正常;**一旦某个 backend 成功服务过一次刷新,它的心跳就永久坏掉**。

Run:  python -m unittest discover -s tests
"""

import asyncio
import logging
import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.relay import RelayClient, RelayError  # noqa: E402


def _client():
    """A RelayClient whose only wired collaborator is a recording _json."""
    relay = RelayClient.__new__(RelayClient)
    sent = []

    async def fake_json(method, path, body, timeout=None, **kw):
        sent.append((method, path, body))
        return {"ok": True}

    relay._json = fake_json
    return relay, sent


def _is_real_int(v) -> bool:
    """True only for a genuine int. bool is an int subclass — exclude it."""
    return isinstance(v, int) and not isinstance(v, bool)


# ── ① 线材形状:水位线必须以整数出线 ─────────────────────────────────────────

class TestWatermarkGoesOutAsAnInteger(unittest.IsolatedAsyncioTestCase):
    """⭐ 本次修复的核心用例。

    注入回归验证方式:把 RelayClient.heartbeat 里的 ``int(object_info_synced_at)``
    改回 ``object_info_synced_at``,本类立刻红。
    """

    async def test_float_watermark_is_sent_as_int(self):
        # 插件持久化在 comfylink_state.json 里的正是 float(config.Pairing 的类型
        # 声明就是 float),所以这是**已装机版本真实会发出来的值**。
        relay, sent = _client()
        await relay.heartbeat("b1", "abc", 1785561497.0)
        v = sent[0][2]["object_info_synced_at"]
        self.assertTrue(
            _is_real_int(v),
            "⛔ 水位线是 unix 秒,必须以 int 出线,实际是 %r(%s)。"
            "float 形状会让中继的 int64 字段解析**整个请求体**失败,"
            "这台机器的心跳从此永久死掉 —— 而且错误信息会说「backend_id required」，"
            "把排查带到完全无关的方向。" % (v, type(v).__name__),
        )
        self.assertEqual(v, 1785561497)

    async def test_every_shape_of_watermark_lands_as_int(self):
        for raw, want in ((1785561497.0, 1785561497),
                          (1785561497, 1785561497),
                          (1785561497.9, 1785561497),   # 亚秒无意义,截断即可
                          (1.0, 1)):
            with self.subTest(raw=raw):
                relay, sent = _client()
                await relay.heartbeat("b1", "abc", raw)
                v = sent[0][2]["object_info_synced_at"]
                self.assertTrue(_is_real_int(v), "%r 出线成了 %r" % (raw, v))
                self.assertEqual(v, want)

    async def test_no_other_field_in_the_body_is_a_float(self):
        # ⚠️ 整条心跳请求体的**每一个数字**都要经得起同一条检查:中继那边
        # 只要有一个整数字段收到小数点,整个 body 就读不懂。这条是给以后新加字段的人
        # 留的闸门 —— 加一个 float 字段进来,这里当场红。
        relay, sent = _client()
        await relay.heartbeat("b1", "abc", 1785561497.0)
        for key, value in sent[0][2].items():
            with self.subTest(key=key):
                self.assertNotIsInstance(
                    value, float,
                    "心跳 body 的 %r 是 float(%r)。若中继侧对应字段是整数类型,"
                    "整个请求体会解析失败。" % (key, value),
                )

    async def test_zero_still_omits_the_key(self):
        # 「0 = 保持中继现有的值」这条规则不能被这次改动破坏:int(0) 仍然是假值,
        # 但要确认没有人把守卫改成 `is not None` 之类。
        relay, sent = _client()
        await relay.heartbeat("b1", "abc", 0.0)
        self.assertNotIn("object_info_synced_at", sent[0][2])


# ── ② 可见性:持续失败必须能被看见,抖动必须继续沉默 ─────────────────────────

class _Pairing:
    """Minimal Pairing stand-in for the heartbeat loop."""

    def __init__(self):
        self.backend_id = "b1"
        self.object_info_hash = ""
        self.object_info_synced_at = 0.0
        self.loras_hash = ""
        self.loras_synced_at = 0.0


class _State:
    """STATE stand-in: the loop only asks "is this pairing still there?"."""

    def __init__(self, pairing):
        self._pairing = pairing

    def get_pairing(self, backend_id):
        return self._pairing

    def save(self):
        pass


async def _run_loop(relay, pairing, beats):
    """Run _heartbeat_loop until ``relay.heartbeat`` has been awaited ``beats``
    times, then tear it down. HEARTBEAT_INTERVAL is squashed so the test is
    instant; STATE is faked so the loop's liveness check passes."""
    state = _State(pairing)
    with mock.patch.object(worker, "STATE", state), \
            mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0):
        task = asyncio.create_task(worker._heartbeat_loop(relay, pairing, None))
        try:
            for _ in range(2000):
                if relay.heartbeat.await_count >= beats:
                    break
                await asyncio.sleep(0)
            else:  # pragma: no cover - only on a broken loop
                raise AssertionError("loop never reached %d beats" % beats)
        finally:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass


class TestPersistentFailureIsVisible(unittest.IsolatedAsyncioTestCase):
    """⛔ 「中继一直 400」这种持续性故障不能零症状。

    注入回归验证方式:把两个 except 分支里的 ``_note_heartbeat_failure`` 换回
    ``log.debug("heartbeat error: %s", e)``,本类的前两个用例立刻红。
    """

    def setUp(self):
        # 生产日志级别是 INFO —— debug 那行在用户机器上根本不存在。用真实的
        # assertLogs 级别把这一点钉住:断言的是 WARNING/INFO,不是 DEBUG。
        self.pairing = _Pairing()
        self.relay = mock.AsyncMock()

    async def test_a_persistent_failure_warns_once(self):
        # 线上真实形状:中继对每一拍都回 400,而且错误信息还是误导性的那句。
        self.relay.heartbeat.side_effect = RelayError(
            'relay POST /v1/backends/heartbeat -> 400: {"error":"backend_id required"}',
            400)
        with self.assertLogs("comfylink", level=logging.WARNING) as cm:
            await _run_loop(self.relay, self.pairing,
                            worker._HEARTBEAT_WARN_AFTER + 1)
        warnings = [r for r in cm.records if r.levelno >= logging.WARNING]
        self.assertEqual(
            len(warnings), 1,
            "连续失败应当**恰好**警告一次(不是每拍一次): %r" %
            [r.getMessage() for r in warnings])
        msg = warnings[0].getMessage()
        self.assertIn("b1", msg)                       # 哪台机器
        self.assertIn("backend_id required", msg)      # 中继原话,便于排查
        self.assertIn("in a row", msg)                 # 说清是「持续」不是一次抖动

    async def test_a_non_relay_exception_is_also_visible(self):
        # 不只是 RelayError:超时、DNS、以及 P-11 那类「自己代码写错了」的
        # TypeError,持续发生时同样必须冒头。
        self.relay.heartbeat.side_effect = TimeoutError("connection timed out")
        with self.assertLogs("comfylink", level=logging.WARNING) as cm:
            await _run_loop(self.relay, self.pairing,
                            worker._HEARTBEAT_WARN_AFTER + 1)
        self.assertTrue(any("connection timed out" in r.getMessage()
                            for r in cm.records))

    async def test_a_blip_stays_silent(self):
        # ⚠️ 平衡的另一半:网络抖动是常态。少于阈值的连续失败,以及被成功打断的
        # 失败,都不能升级 —— 否则就回到「每 25 秒刷一行 WARNING」,那正是当初
        # 把它压成 debug 的原因。
        seq = [RelayError("blip", 502), {"ok": True},
               RelayError("blip", 502), RelayError("blip", 502), {"ok": True}]
        calls = []

        async def side_effect(*_a, **_kw):
            item = seq[min(len(calls), len(seq) - 1)]
            calls.append(item)
            if isinstance(item, Exception):
                raise item
            return item

        self.relay.heartbeat.side_effect = side_effect
        with mock.patch.object(worker.log, "warning") as warn:
            await _run_loop(self.relay, self.pairing, len(seq))
        warn.assert_not_called()

    async def test_recovery_is_announced_and_the_counter_resets(self):
        # 恢复要留一条 info:否则日志里只剩一句「一直在失败」,看的人不知道它
        # 什么时候好的;而且计数必须归零,不然下一次抖动会被误判成持续故障。
        n = worker._HEARTBEAT_WARN_AFTER
        calls = []

        async def side_effect(*_a, **_kw):
            calls.append(1)
            if len(calls) <= n:
                raise RelayError("down", 400)
            return {"ok": True}

        self.relay.heartbeat.side_effect = side_effect
        with self.assertLogs("comfylink", level=logging.INFO) as cm:
            await _run_loop(self.relay, self.pairing, n + 3)
        infos = [r.getMessage() for r in cm.records if r.levelno == logging.INFO]
        self.assertTrue(any("recovered" in m for m in infos),
                        "恢复后应当有一条 info: %r" % infos)
        # 恢复之后又跑了好几拍,不能再多出第二条 warning。
        self.assertEqual(
            len([r for r in cm.records if r.levelno >= logging.WARNING]), 1)


class TestFailureRateIsBounded(unittest.TestCase):
    """长时间故障也不能刷屏 —— 阈值之后是**按次数间隔**再说一次,不是每拍。"""

    def test_one_line_per_window_not_per_beat(self):
        after, every = worker._HEARTBEAT_WARN_AFTER, worker._HEARTBEAT_WARN_EVERY
        with mock.patch.object(worker.log, "warning") as warn, \
                mock.patch.object(worker.log, "debug"):
            fails = 0
            for _ in range(after + every * 2):
                fails = worker._note_heartbeat_failure("b1", fails, RuntimeError("x"))
        # 阈值那一次 + 之后每 every 次一次 = 3 次,而不是 after + 2*every 次。
        self.assertEqual(warn.call_count, 3, "刷屏了(或者反过来,再也不说话了)")
        # 用秒数说明这个节奏是可接受的:两条 warning 之间 ≈ 50 分钟。
        self.assertGreaterEqual(every * worker.HEARTBEAT_INTERVAL, 30 * 60)

    def test_the_first_warning_comes_fast_enough_to_be_useful(self):
        # 太晚说话等于没说。阈值折算成时间必须在几分钟以内。
        self.assertLessEqual(
            worker._HEARTBEAT_WARN_AFTER * worker.HEARTBEAT_INTERVAL, 5 * 60)


if __name__ == "__main__":
    unittest.main()
