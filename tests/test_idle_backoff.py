"""空闲退避(插件侧)。

中继在「这台机器没活儿 + 这个账号的 App 也很久没露面」时,会在响应头里捎带一个
建议:下一次 claim 之前先睡 N 秒 / 下一拍心跳隔 N 秒再打。

⭐ 这一组用例的重心是 **fail-safe 的方向**:所有说不准的情况都必须回到 0
(= 没有建议 = 照旧全速轮询)。少省一点流量只是钱,睡过头是用户点了生成却干等着。
"""

import asyncio
import unittest
from unittest import mock

from multidict import CIMultiDict

from comfylink.relay import (
    IDLE_SLEEP_HEADER,
    MAX_IDLE_SLEEP,
    _idle_sleep_hint,
    _sleep_or_stop,
)


class TestIdleSleepHint(unittest.TestCase):
    def test_reads_a_plain_value(self):
        self.assertEqual(_idle_sleep_hint({IDLE_SLEEP_HEADER: "17"}), 17)

    def test_missing_header_means_no_advice(self):
        """老中继根本不发这个头 ⇒ 行为必须与改造前逐字节一致。"""
        self.assertEqual(_idle_sleep_hint({}), 0)

    def test_go_canonicalised_casing_is_still_found(self):
        """⛔ 这条守着一个**会让整个功能静默失效**的坑。

        Go 的 ``w.Header().Set`` 会把头名规范化成 ``X-Comfylink-Idle-Sleep``
        —— 注意 ``L`` 变成了小写 ``l``,和插件侧的常量并不逐字相同。心跳那条路把
        响应头 update 进了一个**普通 dict**(大小写敏感),精确匹配会 miss。

        miss 的症状是「什么都没发生」:一切照常跑,流量一分不省,也不会有任何报错。
        """
        self.assertEqual(_idle_sleep_hint({"X-Comfylink-Idle-Sleep": "17"}), 17)

    def test_case_insensitive_container_also_works(self):
        """claim 那条路拿到的是 aiohttp 的 CIMultiDict(本来就不敏感)。"""
        self.assertEqual(
            _idle_sleep_hint(CIMultiDict({"x-comfylink-idle-sleep": "17"})), 17)

    def test_garbage_means_no_advice(self):
        for raw in ("", "   ", "soon", "17s", "1.5", "NaN"):
            with self.subTest(raw=raw):
                self.assertEqual(_idle_sleep_hint({IDLE_SLEEP_HEADER: raw}), 0)

    def test_zero_and_negative_mean_no_advice(self):
        for raw in ("0", "-1", "-3600"):
            with self.subTest(raw=raw):
                self.assertEqual(_idle_sleep_hint({IDLE_SLEEP_HEADER: raw}), 0)

    def test_absurd_value_is_clamped_locally(self):
        """⛔ 中继侧自己有上界,但那是**另一个进程里的假设**。

        万一它发疯下发一小时,用户的机器就一小时不领任务 —— 保险丝必须也留一份
        在自己这边。
        """
        self.assertEqual(_idle_sleep_hint({IDLE_SLEEP_HEADER: "999999"}),
                         MAX_IDLE_SLEEP)

    def test_broken_headers_object_does_not_raise(self):
        """headers 形状异常不值得让一次 claim 挂掉 —— 吞掉,回 0。"""
        class Exploding:
            def get(self, *_a, **_kw):
                raise RuntimeError("boom")

            def items(self):
                raise RuntimeError("boom")

        self.assertEqual(_idle_sleep_hint(Exploding()), 0)


class TestSleepOrStop(unittest.IsolatedAsyncioTestCase):
    async def test_returns_immediately_when_stop_is_already_set(self):
        stop = asyncio.Event()
        stop.set()
        # 真的睡满就会挂到测试超时;这里用一个大到不可能睡完的值。
        await asyncio.wait_for(_sleep_or_stop(3600, stop), timeout=2)

    async def test_wakes_up_when_stop_fires_midway(self):
        """关插件时不该干等一觉睡完。"""
        stop = asyncio.Event()

        async def fire():
            await asyncio.sleep(0)
            stop.set()

        await asyncio.wait_for(
            asyncio.gather(_sleep_or_stop(3600, stop), fire()), timeout=2)

    async def test_without_stop_it_is_a_plain_sleep(self):
        """⛔ 必须走 asyncio.sleep —— 测试正是靠 patch 它来让这些循环飞转的。

        换成 wait_for(…, timeout=…) 行为等价,但 patch 不到它的定时器,整套
        测试会真的按秒睡。**能被换掉的等待**是这类循环可测的前提。
        """
        with mock.patch.object(asyncio, "sleep", mock.AsyncMock()) as slept:
            await _sleep_or_stop(42, None)
        slept.assert_awaited_once_with(42)

    async def test_with_stop_it_still_goes_through_asyncio_sleep(self):
        """同上:带 stop 的那条路也必须可被 patch,否则心跳循环没法测。"""
        stop = asyncio.Event()
        with mock.patch.object(asyncio, "sleep", mock.AsyncMock()) as slept:
            await asyncio.wait_for(_sleep_or_stop(42, stop), timeout=2)
        slept.assert_awaited_once_with(42)


class _FakeState:
    """够 _heartbeat_loop 用的最小 STATE(它只问「这个配对还在不在」)。"""

    def __init__(self, pairing):
        self._p = pairing

    def get_pairing(self, backend_id):
        return self._p if self._p.backend_id == backend_id else None


class TestHeartbeatLoopHonoursTheAdvice(unittest.IsolatedAsyncioTestCase):
    """⭐ 上面那组只证明「读得对」,这一条证明「**真的用上了**」。

    两者缺一不可:解析对但没接进循环,和根本没实现是一样的 —— 而且同样没有症状。
    """

    def _pairing(self):
        from comfylink.config import Pairing
        return Pairing(backend_id="b1", device_token="clr")

    async def _run_one_beat(self, advice: str):
        """跑一拍心跳,返回它随后打算睡的秒数。"""
        from comfylink import worker

        pairing = self._pairing()
        stop = asyncio.Event()
        relay = mock.AsyncMock()

        async def beat(_bid, *_a, headers_out=None, **_kw):
            if headers_out is not None and advice:
                # 用 Go 规范化之后的大小写,和线上真实收到的一模一样。
                headers_out["X-Comfylink-Idle-Sleep"] = advice
            return {}

        relay.heartbeat.side_effect = beat

        slept: list = []

        async def fake_sleep(sec, *_a, **_kw):
            slept.append(sec)
            stop.set()  # 睡完这一拍就收工,循环顶部的守卫会让它退出

        with mock.patch.object(asyncio, "sleep", fake_sleep), \
                mock.patch.object(worker, "STATE", _FakeState(pairing)):
            await asyncio.wait_for(
                worker._heartbeat_loop(relay, pairing, stop), timeout=5)
        return slept

    async def test_advised_interval_replaces_the_default_rhythm(self):
        slept = await self._run_one_beat("60")
        self.assertEqual(slept, [60],
                         "中继建议了 60 秒,心跳却没照着睡 ⇒ 退避没有生效")

    async def test_without_advice_the_default_rhythm_is_kept(self):
        from comfylink.worker import HEARTBEAT_INTERVAL

        slept = await self._run_one_beat("")
        self.assertEqual(slept, [HEARTBEAT_INTERVAL],
                         "没有建议时必须保持原节奏(老中继就是这条路)")

    async def test_a_failed_beat_keeps_the_default_rhythm(self):
        """⛔ 心跳打不通的时候更该勤快点重试,而不是顺着一个旧建议睡过去。"""
        from comfylink import worker
        from comfylink.worker import HEARTBEAT_INTERVAL

        pairing = self._pairing()
        stop = asyncio.Event()
        relay = mock.AsyncMock()
        relay.heartbeat.side_effect = RuntimeError("relay down")

        slept: list = []

        async def fake_sleep(sec, *_a, **_kw):
            slept.append(sec)
            stop.set()

        with mock.patch.object(asyncio, "sleep", fake_sleep), \
                mock.patch.object(worker, "STATE", _FakeState(pairing)):
            await asyncio.wait_for(
                worker._heartbeat_loop(relay, pairing, stop), timeout=5)

        self.assertEqual(slept, [HEARTBEAT_INTERVAL])


if __name__ == "__main__":
    unittest.main()
