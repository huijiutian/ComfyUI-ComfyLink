"""The plugin reports its version AND its capabilities to the relay on register
+ every heartbeat.

The app uses the version to tell the user "your ComfyUI plugin has a new
version"; it uses ``caps`` to decide which entry points it may offer at all
(R-1.0.6-22 — 版本号在这个仓库里没随功能抬过,不能拿来判断功能有没有).
Heartbeat carries both (not just register) so a `git pull` + ComfyUI restart
refreshes what the relay sees without a re-pairing. Every field is optional
server-side, so an old relay just ignores them — but the plugin must always
send them, unconditionally: the relay overwrites caps on every beat (empty
included), so anything conditional here shows up in the app as a capability
that flickers.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.relay import RelayClient  # noqa: E402
from comfylink.version import __caps__, __commit__, __version__  # noqa: E402


class TestVersionReporting(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        # Bare RelayClient with _json stubbed — we only care about the body.
        self.relay = RelayClient.__new__(RelayClient)
        self.calls: list[tuple[str, str, dict]] = []

        async def fake_json(method, path, body, **kw):
            self.calls.append((method, path, body))
            return {}

        self.relay._json = fake_json

    async def test_register_sends_version_and_commit(self):
        await self.relay.register("b1", "my-box")
        self.assertEqual(self.calls, [(
            "POST", "/v1/backends/register",
            {"backend_id": "b1", "name": "my-box",
             "version": __version__, "commit": __commit__,
             "caps": ["models"]},
        )])

    async def test_heartbeat_sends_version_and_commit(self):
        await self.relay.heartbeat("b1")
        self.assertEqual(self.calls, [(
            "POST", "/v1/backends/heartbeat",
            {"backend_id": "b1", "version": __version__, "commit": __commit__,
             "caps": ["models"],
             # ⭐ 2026-08-07 起心跳无条件携带本机 ComfyUI 现状(R-1.0.8-21):
             # -1/"" 是「不知道」哨兵 —— 无 ComfyUI 客户端的测试装配正落在哨兵上。
             "foreign_queue_depth": -1,
             "loaded_checkpoint": ""},
        )])

    async def test_register_declares_the_models_capability(self):
        # ⭐ 钉住线上形状:字面量 ["models"],**不带版本号**。
        # 若有人把它改成 "models.v3" 之类(比如从 loras.MANIFEST_SCHEMA 派生),
        # 老 App 会一夜之间认不出这个能力 —— 这条用例就是拦那个改动的。
        await self.relay.register("b1", "my-box")
        self.assertEqual(self.calls[0][2]["caps"], ["models"])

    async def test_heartbeat_declares_the_models_capability(self):
        await self.relay.heartbeat("b1")
        self.assertEqual(self.calls[0][2]["caps"], ["models"])

    def test_the_capability_name_carries_no_version(self):
        # caps 回答「支不支持这件事」,不回答「支持到第几版」。manifest schema 的演进
        # 不该让老 App 失能,所以能力名里不能出现数字/点号后缀。
        for cap in __caps__:
            with self.subTest(cap=cap):
                self.assertNotIn(".", cap, "能力名不该带版本后缀: %r" % cap)
                self.assertFalse(any(ch.isdigit() for ch in cap),
                                 "能力名不该带版本号: %r" % cap)


if __name__ == "__main__":
    unittest.main()
