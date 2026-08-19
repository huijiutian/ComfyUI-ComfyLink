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
import re
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
             "caps": ["models", "img2img"]},
        )])

    async def test_heartbeat_sends_version_and_commit(self):
        await self.relay.heartbeat("b1")
        self.assertEqual(self.calls, [(
            "POST", "/v1/backends/heartbeat",
            {"backend_id": "b1", "version": __version__, "commit": __commit__,
             "caps": ["models", "img2img"],
             # ⭐ 2026-08-07 起心跳无条件携带本机 ComfyUI 现状(R-1.0.8-21):
             # -1/"" 是「不知道」哨兵 —— 无 ComfyUI 客户端的测试装配正落在哨兵上。
             "foreign_queue_depth": -1,
             "loaded_checkpoint": ""},
        )])

    async def test_register_declares_the_capabilities(self):
        # ⭐ 钉住线上形状:字面量 ["models", "img2img"],**都不带版本号**。
        # 若有人把它改成 "models.v3" 之类(比如从 loras.MANIFEST_SCHEMA 派生),
        # 老 App 会一夜之间认不出这个能力 —— 这条用例就是拦那个改动的。
        await self.relay.register("b1", "my-box")
        self.assertEqual(self.calls[0][2]["caps"], ["models", "img2img"])

    async def test_heartbeat_declares_the_capabilities(self):
        await self.relay.heartbeat("b1")
        self.assertEqual(self.calls[0][2]["caps"], ["models", "img2img"])

    async def test_img2img_is_declared_unconditionally(self):
        # ⭐ R-1.0.11-16:App 靠这个名字决定放不放行「带参考图生成」。它必须
        # **每一拍都在**,而且不受这台机器的任何状态影响 —— 中继每拍无条件覆盖,
        # 写成「满足条件才带」会让入口在 App 里闪断。
        await self.relay.register("b1", "my-box")
        await self.relay.heartbeat("b1")
        for method, _path, body in self.calls:
            with self.subTest(method=method):
                self.assertIn("img2img", body["caps"])

    def test_the_capability_names_carry_no_version(self):
        # caps 回答「支不支持这件事」,不回答「支持到第几版」。manifest schema 的演进
        # 不该让老 App 失能,所以能力名里不能带版本后缀。
        #
        # ⚠️ 判据是「**结尾**是不是数字」,不是「含不含数字」——「含数字就算版本号」
        # 那条旧判据会把 `img2img` 这种**名字里就有数字**的正当能力名一起打死。
        #
        # ⛔ 别改成 `([._-]|\bv)\d+$` 那种「认版本记号」的写法(2026-08-19 实测栽过):
        # Python 的 `\w` **包含下划线**,于是 `models_v3` 里 `_` 与 `v` 之间**不是词边界**,
        # `\bv` 匹配不上 ⇒ 最典型的 `models_v3` 反而漏过去,而 `models-v3` 因为 `-`
        # 不是单词字符才被挡住。同一条规则对两种写法给出相反结论,是最糟的那种规则。
        # 「结尾不是数字」没有这个陷阱,且照样放行 `img2img`(结尾是 g)。
        version_suffix = re.compile(r"\d+$")
        for cap in __caps__:
            with self.subTest(cap=cap):
                self.assertNotIn(".", cap, "能力名不该带版本后缀: %r" % cap)
                self.assertIsNone(version_suffix.search(cap),
                                  "能力名不该以数字结尾(那多半是版本号): %r" % cap)


if __name__ == "__main__":
    unittest.main()
