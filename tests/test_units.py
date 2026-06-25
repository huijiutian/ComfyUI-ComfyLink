"""Unit tests for pure helpers — no aiohttp / ComfyUI / relay needed.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.jobs import (  # noqa: E402
    apply_inputs,
    content_type_for,
    extract_outputs,
    progress_event,
)


class TestContentType(unittest.TestCase):
    def test_known(self):
        self.assertEqual(content_type_for("a.PNG"), "image/png")
        self.assertEqual(content_type_for("a.webp"), "image/webp")
        self.assertEqual(content_type_for("a.jpeg"), "image/jpeg")
        self.assertEqual(content_type_for("a.bin"), "application/octet-stream")

    def test_video_mime(self):
        self.assertEqual(content_type_for("clip.mp4"), "video/mp4")
        self.assertEqual(content_type_for("clip.WEBM"), "video/webm")
        self.assertEqual(content_type_for("anim.gif"), "image/gif")
        self.assertEqual(content_type_for("c.mov"), "video/quicktime")


class TestExtractOutputs(unittest.TestCase):
    def test_only_output_type(self):
        history = {
            "p1": {"outputs": {
                "9": {"images": [
                    {"filename": "final.png", "subfolder": "", "type": "output"},
                    {"filename": "prev.png", "subfolder": "", "type": "temp"},
                ]},
                "10": {"images": [{"filename": "b.png", "subfolder": "s", "type": "output"}]},
            }}
        }
        out = extract_outputs(history, "p1")
        names = sorted(i["filename"] for i in out)
        self.assertEqual(names, ["b.png", "final.png"])
        # Stills are tagged image.
        self.assertTrue(all(i["media_type"] == "image" for i in out))

    def test_collects_gifs_and_videos_as_video(self):
        history = {
            "p1": {"outputs": {
                "9": {"gifs": [
                    {"filename": "vhs_00001.mp4", "subfolder": "", "type": "output"},
                    {"filename": "preview.mp4", "subfolder": "", "type": "temp"},
                ]},
                "12": {"videos": [
                    {"filename": "native.webm", "subfolder": "v", "type": "output"},
                ]},
                "20": {"images": [
                    {"filename": "still.png", "subfolder": "", "type": "output"},
                ]},
            }}
        }
        out = extract_outputs(history, "p1")
        by_name = {i["filename"]: i for i in out}
        # temp preview dropped; the three outputs kept.
        self.assertEqual(sorted(by_name), ["native.webm", "still.png", "vhs_00001.mp4"])
        self.assertEqual(by_name["vhs_00001.mp4"]["media_type"], "video")
        self.assertEqual(by_name["native.webm"]["media_type"], "video")
        self.assertEqual(by_name["still.png"]["media_type"], "image")

    def test_animated_webp_in_gifs_is_video(self):
        # VHS_VideoCombine can drop an animated .webp under "gifs" — the source
        # key makes it a video so we never WebP-flatten it.
        history = {
            "p1": {"outputs": {
                "9": {"gifs": [{"filename": "anim.webp", "subfolder": "", "type": "output"}]},
            }}
        }
        out = extract_outputs(history, "p1")
        self.assertEqual(out[0]["media_type"], "video")

    def test_missing_prompt(self):
        self.assertEqual(extract_outputs({}, "nope"), [])


class TestApplyInputs(unittest.TestCase):
    def test_rewrites_node_field(self):
        prompt = {"10": {"class_type": "LoadImage", "inputs": {"image": "old.png"}}}
        inputs = [{"r2_key": "k1", "node": "10", "field": "image", "name": "ref.png"}]
        apply_inputs(prompt, inputs, {"k1": "ref_uploaded.png"})
        self.assertEqual(prompt["10"]["inputs"]["image"], "ref_uploaded.png")

    def test_ignores_unknown_node(self):
        prompt = {"10": {"inputs": {}}}
        apply_inputs(prompt, [{"r2_key": "k", "node": "99", "field": "image"}], {"k": "x.png"})
        self.assertEqual(prompt["10"]["inputs"], {})


class TestProgressEvent(unittest.TestCase):
    def test_progress(self):
        ev = progress_event({"type": "progress", "data": {"value": 3, "max": 20}})
        self.assertEqual(ev["kind"], "progress")
        self.assertEqual((ev["value"], ev["max"]), (3, 20))

    def test_done_on_node_null(self):
        ev = progress_event({"type": "executing", "data": {"node": None, "prompt_id": "p1"}})
        self.assertEqual(ev["kind"], "done")
        self.assertEqual(ev["prompt_id"], "p1")

    def test_executing_a_node_is_ignored(self):
        self.assertIsNone(progress_event({"type": "executing", "data": {"node": "5"}}))

    def test_error(self):
        ev = progress_event({"type": "execution_error", "data": {"exception_message": "boom"}})
        self.assertEqual(ev["kind"], "error")
        self.assertEqual(ev["message"], "boom")

    def test_unknown_ignored(self):
        self.assertIsNone(progress_event({"type": "status", "data": {}}))


class TestState(unittest.TestCase):
    def test_save_load_roundtrip(self):
        import tempfile

        d = tempfile.mkdtemp()
        os.environ["COMFYLINK_STATE"] = os.path.join(d, "state.json")
        try:
            # fresh import with the env in place
            import importlib

            import comfylink.config as cfg
            importlib.reload(cfg)
            st = cfg.State.load()
            self.assertTrue(st.backend_id)  # auto-generated
            self.assertFalse(st.paired)
            st.device_token = "clr_abc"
            st.device_id = "d1"
            st.save()
            again = cfg.State.load()
            self.assertEqual(again.device_token, "clr_abc")
            self.assertTrue(again.paired)
            again.clear_pairing()
            self.assertFalse(again.paired)
        finally:
            del os.environ["COMFYLINK_STATE"]


if __name__ == "__main__":
    unittest.main()
