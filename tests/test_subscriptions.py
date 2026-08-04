"""Unit tests for subscription size-cap + WebP conversion helpers.

Pure helpers only — no aiohttp / ComfyUI / relay needed.

Run:  python -m unittest discover -s tests

The WebP tests need Pillow (the same PIL the plugin uses in ComfyUI). If PIL
is genuinely absent they skip; install with `pip install pillow` to run them.
"""

import os
import sys
import unittest
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.jobs import encode_output, within_cap  # noqa: E402

try:
    from PIL import Image  # noqa: F401
    HAVE_PIL = True
except ImportError:  # pragma: no cover - env-dependent
    HAVE_PIL = False


def _tiny_png(prompt: str | None = None) -> bytes:
    """A real PNG with pseudo-random pixel noise. Noise is near-incompressible
    for PNG's lossless filter but compresses well under WebP's lossy encoder,
    so the WebP re-encode comes out clearly smaller (which is the whole point
    of the feature).

    When ``prompt`` is given it is written as a ComfyUI-style ``prompt`` tEXt
    chunk so we can assert it survives the WebP re-encode."""
    import random

    from PIL import PngImagePlugin

    rng = random.Random(1234)  # deterministic so the test is stable
    im = Image.new("RGB", (128, 128))
    px = im.load()
    for y in range(128):
        for x in range(128):
            px[x, y] = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
    buf = BytesIO()
    pnginfo = None
    if prompt is not None:
        pnginfo = PngImagePlugin.PngInfo()
        pnginfo.add_text("prompt", prompt)
        # A bulky workflow tEXt we explicitly do NOT want carried into the WebP.
        pnginfo.add_text("workflow", '{"big": "' + "x" * 5000 + '"}')
    im.save(buf, format="PNG", pnginfo=pnginfo)
    return buf.getvalue()


def _xmp_text(webp: bytes) -> str:
    x = Image.open(BytesIO(webp)).info.get("xmp", b"")
    return x.decode("utf-8", "replace") if isinstance(x, bytes) else (x or "")


@unittest.skipUnless(HAVE_PIL, "Pillow not installed (pip install pillow)")
class TestEncodeOutput(unittest.TestCase):
    def test_png_to_webp_smaller_and_retagged(self):
        png = _tiny_png()
        data, filename, ct = encode_output(png, "ComfyUI_00001_.png", "webp")
        self.assertLess(len(data), len(png))  # webp helps fit under the cap
        self.assertEqual(filename, "ComfyUI_00001_.webp")
        self.assertEqual(ct, "image/webp")
        # And it really is a WebP we can re-open.
        self.assertEqual(Image.open(BytesIO(data)).format, "WEBP")

    def test_uppercase_format_still_converts(self):
        png = _tiny_png()
        data, filename, ct = encode_output(png, "a.png", "WEBP")
        self.assertEqual(ct, "image/webp")
        self.assertEqual(filename, "a.webp")

    def test_png_passthrough_when_format_png(self):
        png = _tiny_png()
        data, filename, ct = encode_output(png, "a.png", "png")
        self.assertEqual(data, png)
        self.assertEqual(filename, "a.png")
        self.assertEqual(ct, "image/png")

    def test_default_format_passthrough(self):
        png = _tiny_png()
        # empty/missing format string => no conversion
        data, filename, ct = encode_output(png, "a.png", "")
        self.assertEqual(data, png)
        self.assertEqual(filename, "a.png")
        self.assertEqual(ct, "image/png")

    def test_webp_conversion_failure_falls_back_to_original(self):
        # WebP is best-effort: undecodable bytes → fall back to the original
        # (job still succeeds) rather than crash. Content-type by extension.
        garbage = b"not an image at all"
        data, filename, ct = encode_output(garbage, "weird.dat", "webp")
        self.assertEqual(data, garbage)
        self.assertEqual(filename, "weird.dat")
        self.assertEqual(ct, "application/octet-stream")

    def test_webp_preserves_prompt_but_not_workflow(self):
        # The prompt has XML-special chars so we also exercise escaping.
        prompt = '{"3": {"class_type": "KSampler", "text": "a <b> & \\"c\\""}}'
        png = _tiny_png(prompt=prompt)
        data, _, ct = encode_output(png, "a.png", "webp")
        self.assertEqual(ct, "image/webp")
        xmp = _xmp_text(data)
        self.assertIn("comfylink:prompt", xmp)
        # The escaped prompt round-trips out of the XMP attribute.
        from xml.sax.saxutils import unescape
        import re
        m = re.search(r'comfylink:prompt="(.*?)"', xmp, re.S)
        self.assertIsNotNone(m)
        recovered = unescape(m.group(1), {"&quot;": '"'})
        self.assertEqual(recovered, prompt)
        # The bulky workflow blob must NOT have been carried along.
        self.assertNotIn("xxxxx", xmp)

    def test_webp_without_prompt_still_encodes(self):
        # No prompt tEXt in the source -> still a valid WebP, just no comfylink xmp.
        png = _tiny_png(prompt=None)
        data, filename, ct = encode_output(png, "a.png", "webp")
        self.assertEqual(ct, "image/webp")
        self.assertEqual(Image.open(BytesIO(data)).format, "WEBP")

    def test_video_skips_conversion_even_with_webp_format(self):
        # A video item must never be run through Pillow, even under webp format.
        fake_mp4 = b"\x00\x00\x00\x18ftypmp42fake-video-bytes"
        data, filename, ct = encode_output(fake_mp4, "out.mp4", "webp", media_type="video")
        self.assertEqual(data, fake_mp4)
        self.assertEqual(filename, "out.mp4")
        self.assertEqual(ct, "video/mp4")


def _sampler_prompt_json(positive: str = "masterpiece, 1girl",
                         negative: str = "lowres") -> str:
    """API prompt with sampler links so prompt_texts can assign roles."""
    import json

    return json.dumps({
        "3": {"class_type": "KSampler",
              "inputs": {"positive": ["6", 0], "negative": ["7", 0], "seed": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": positive}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": negative}},
    })


@unittest.skipUnless(HAVE_PIL, "Pillow not installed (pip install pillow)")
class TestXmpDirectFields(unittest.TestCase):
    """R-1.0.6-30:自产图的直通提示词字段(comfylink:positive/negative)。"""

    def test_direct_fields_written_with_generator_and_prompt(self):
        png = _tiny_png(prompt=_sampler_prompt_json())
        data, _, _ = encode_output(png, "a.png", "webp")
        xmp = _xmp_text(data)
        # 直通字段:最终正/负向全文,元素形式。
        self.assertIn("<comfylink:positive>masterpiece, 1girl</comfylink:positive>", xmp)
        self.assertIn("<comfylink:negative>lowres</comfylink:negative>", xmp)
        # 来源标记 + 原有 workflow JSON 属性照写不动(向后兼容)。
        self.assertIn('comfylink:generator="ComfyLink ', xmp)
        self.assertIn('comfylink:prompt="', xmp)

    def test_direct_fields_omitted_when_role_untraceable(self):
        # 没有采样器连线 ⇒ prompt_texts 定不了角色 ⇒ 宁缺毋假:不写直通字段,
        # workflow JSON 照写。
        import json

        loose = json.dumps({
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "floating words"}},
        })
        png = _tiny_png(prompt=loose)
        data, _, _ = encode_output(png, "a.png", "webp")
        xmp = _xmp_text(data)
        self.assertNotIn("<comfylink:positive>", xmp)
        self.assertNotIn("<comfylink:negative>", xmp)
        self.assertIn('comfylink:prompt="', xmp)

    def test_special_chars_escaped_in_element(self):
        # 引号/尖括号/换行是提示词常态:元素内容只转义 & < >,引号与换行原样。
        tricky = 'he said "hi" & <tag>\nline2'
        png = _tiny_png(prompt=_sampler_prompt_json(positive=tricky))
        data, _, _ = encode_output(png, "a.png", "webp")
        xmp = _xmp_text(data)
        self.assertIn(
            '<comfylink:positive>he said "hi" &amp; &lt;tag&gt;\nline2'
            "</comfylink:positive>",
            xmp,
        )

    def test_direct_fields_written_through_concat_chain(self):
        # R-1.0.6-26:正向可以藏在 CR Text Concatenate → CR Prompt Text
        # 拼接链后。_role_texts 经 prompt_texts 顺文本连线定角色 ⇒
        # 直通字段必须把两段都带上,负向照常。
        import json

        chain = json.dumps({
            "3": {"class_type": "KSampler",
                  "inputs": {"positive": ["6", 0], "negative": ["7", 0],
                             "seed": 1}},
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": ["37", 0], "clip": ["4", 1]}},
            "7": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "lowres, watermark, ", "clip": ["4", 1]}},
            "37": {"class_type": "CR Text Concatenate",
                   "inputs": {"separator": "", "text1": ["38", 0],
                              "text2": ["39", 0]}},
            "38": {"class_type": "CR Prompt Text",
                   "inputs": {"prompt": "masterpiece, 8K, "}},
            "39": {"class_type": "CR Prompt Text",
                   "inputs": {"prompt": "landscape, winter, "}},
        })
        png = _tiny_png(prompt=chain)
        data, _, _ = encode_output(png, "a.png", "webp")
        xmp = _xmp_text(data)
        self.assertIn("<comfylink:positive>", xmp)
        self.assertIn("masterpiece, 8K, ", xmp)
        self.assertIn("landscape, winter, ", xmp)
        self.assertIn("<comfylink:negative>lowres, watermark, </comfylink:negative>",
                      xmp)

    def test_job_prompt_writes_fields_when_source_has_no_metadata(self):
        # R-1.0.6-26(Image Saver 类保存节点):源图可以完全没有可读元数据
        # (tEXt/XMP 都没有)—— 直通字段与 workflow JSON 必须来自 **job 的
        # prompt**(claim/R-20 日志同源),不能依赖源图。
        import json

        png = _tiny_png()  # 刻意零元数据
        job = json.loads(_sampler_prompt_json())
        data, _, _ = encode_output(png, "a.png", "webp", job_prompt=job)
        xmp = _xmp_text(data)
        self.assertIn("<comfylink:positive>masterpiece, 1girl</comfylink:positive>",
                      xmp)
        self.assertIn("<comfylink:negative>lowres</comfylink:negative>", xmp)
        self.assertIn('comfylink:prompt="', xmp)
        self.assertIn("KSampler", xmp)  # workflow JSON 真的是 job 那份

    def test_webp_source_still_gets_comfylink_xmp(self):
        # 源直接是 WebP(Image Saver lossless 直出)⇒ 重编后同样带 comfylink XMP。
        import json

        src = BytesIO()
        Image.open(BytesIO(_tiny_png())).save(src, format="WEBP", lossless=True)
        job = json.loads(_sampler_prompt_json())
        data, filename, ct = encode_output(
            src.getvalue(), "2026-01-01-000000__42.webp", "webp",
            job_prompt=job)
        self.assertEqual(ct, "image/webp")
        self.assertEqual(filename, "2026-01-01-000000__42.webp")
        xmp = _xmp_text(data)
        self.assertIn("<comfylink:positive>", xmp)
        self.assertIn("<comfylink:negative>", xmp)

    def test_job_prompt_wins_over_source_text(self):
        # 源图 tEXt 与 job prompt 都在 ⇒ 以 job 为准(它才是产出这张图的那次
        # 提交);源图元数据只作 job_prompt 缺席时的兜底。
        import json

        src_wf = json.dumps({
            "6": {"class_type": "CLIPTextEncode",
                  "inputs": {"text": "src words"}},
        })
        png = _tiny_png(prompt=src_wf)
        job = json.loads(_sampler_prompt_json(positive="job words"))
        data, _, _ = encode_output(png, "a.png", "webp", job_prompt=job)
        xmp = _xmp_text(data)
        self.assertIn("job words", xmp)
        self.assertNotIn("src words", xmp)

    def test_source_metadata_is_the_fallback_without_job_prompt(self):
        # 防御路径:不带 job_prompt(单测/旧调用)时保持旧行为 —— 源图 tEXt 兜底。
        png = _tiny_png(prompt=_sampler_prompt_json())
        data, _, _ = encode_output(png, "a.png", "webp")
        self.assertIn("<comfylink:positive>masterpiece, 1girl</comfylink:positive>",
                      _xmp_text(data))

    def test_pillow_xmp_degrade_sets_status_and_ships_promptless_webp(self):
        # 老 Pillow 不认 xmp= ⇒ 降级为无 prompt 的 WebP,但**不再零症状**:
        # STATUS.webp_xmp_degraded 置真,面板据此亮软警告(R-1.0.6-26 复诊)。
        from unittest import mock

        from comfylink.status import STATUS

        STATUS.set(webp_xmp_degraded=False)
        self.addCleanup(lambda: STATUS.set(webp_xmp_degraded=False))
        orig_save = Image.Image.save

        def old_pillow_save(self_im, fp, *a, **kw):
            if "xmp" in kw:
                raise TypeError("save() got an unexpected keyword argument 'xmp'")
            return orig_save(self_im, fp, *a, **kw)

        png = _tiny_png(prompt=_sampler_prompt_json())
        with mock.patch.object(Image.Image, "save", old_pillow_save):
            data, filename, ct = encode_output(png, "a.png", "webp")
        self.assertEqual(ct, "image/webp")
        self.assertEqual(Image.open(BytesIO(data)).format, "WEBP")
        self.assertEqual(_xmp_text(data), "")  # prompt 真的没进图
        self.assertTrue(STATUS.snapshot()["webp_xmp_degraded"])

    def test_status_snapshot_defaults_flag_false(self):
        from comfylink.status import Status

        self.assertFalse(Status().snapshot()["webp_xmp_degraded"])


@unittest.skipUnless(HAVE_PIL, "Pillow not installed (pip install pillow)")
class TestWebpCapability(unittest.TestCase):
    """环境能力探测:面板一眼定案「出的图带不带 prompt」(R-1.0.6-26 复诊)。"""

    def test_probe_reports_this_env_and_caches(self):
        from comfylink.jobs import webp_capability

        caps = webp_capability()
        self.assertTrue(caps["pillow_version"])  # 本环境有 Pillow
        self.assertTrue(caps["webp_ok"])
        self.assertTrue(caps["webp_xmp_ok"])  # 矩阵已证 >=9.1 都支持
        self.assertIs(caps, webp_capability())  # 缓存:同一对象


class TestWithinCap(unittest.TestCase):
    def test_zero_cap_always_allows(self):
        # 0/absent = unlimited (legacy safety).
        self.assertTrue(within_cap(10_000_000, 0))
        self.assertTrue(within_cap(0, 0))

    def test_negative_cap_allows(self):
        self.assertTrue(within_cap(999, -1))

    def test_under_and_at_cap_allowed(self):
        self.assertTrue(within_cap(500, 1000))
        self.assertTrue(within_cap(1000, 1000))  # exactly at cap is allowed

    def test_over_cap_rejected(self):
        self.assertFalse(within_cap(1001, 1000))


if __name__ == "__main__":
    unittest.main()
