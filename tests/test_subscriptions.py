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

    def test_garbage_bytes_fall_back_to_original(self):
        garbage = b"not an image at all"
        data, filename, ct = encode_output(garbage, "weird.dat", "webp")
        # Conversion failed -> original bytes/filename, content-type by extension.
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
