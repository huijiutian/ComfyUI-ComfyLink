"""Tests for R-1.0.6-20: the final prompt is visible in the local log.

The user story: the prompt on the phone is assembled from a preset tree, then
flattened to tokens, then LoRA trigger words are layered on, then it becomes one
string, then it crosses the relay, then it reaches this plugin, then ComfyUI.
When the picture is wrong, every one of those hops looks identical from the
outside. This log line is the only place that can say what the FAR END actually
received — so "is the prompt manager's output correct?" stops being a guess.

Pinned here:

  * the positive/negative texts are logged IN FULL, at INFO, before submit;
  * each line says WHERE the text is wired (node id, class, input name), because
    a workflow can have several text boxes and an unlabeled dump can't be tied
    back to the one on screen;
  * the role comes from the GRAPH (traced back from the sampler's
    positive/negative inputs, through conditioning helpers), never from wording,
    and never crosses branches;
  * when we cannot identify a text input we SAY SO — "no text input identified"
    plus what the workflow was made of. ⛔ Never a guess, never a crash, never a
    silent empty line: a plausible-looking wrong prompt in the log would send
    the reader chasing a bug that isn't there;
  * ⛔ the text goes to the user's own console ONLY — nothing about it is ever
    reported to the relay.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.comfy import PromptRejected  # noqa: E402
from comfylink.jobs import format_prompt_log, prompt_texts  # noqa: E402
from comfylink.worker import Worker  # noqa: E402

POS = "masterpiece, best quality, 1girl, long_hair, (blue eyes:1.2)"
NEG = "worst quality, bad hands, watermark"


def _txt2img(pos=POS, neg=NEG, pos_title=None, neg_title=None):
    """The canonical ComfyUI default workflow, in API format."""
    p = {
        "3": {"class_type": "KSampler", "inputs": {
            "seed": 42, "steps": 20, "cfg": 8.0, "sampler_name": "euler",
            "scheduler": "normal", "denoise": 1.0,
            "model": ["4", 0], "positive": ["6", 0], "negative": ["7", 0],
            "latent_image": ["5", 0]}},
        "4": {"class_type": "CheckpointLoaderSimple",
              "inputs": {"ckpt_name": "waiNSFW_v11.safetensors"}},
        "5": {"class_type": "EmptyLatentImage",
              "inputs": {"width": 512, "height": 512, "batch_size": 1}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"text": pos, "clip": ["4", 1]}},
        "7": {"class_type": "CLIPTextEncode", "inputs": {"text": neg, "clip": ["4", 1]}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage",
              "inputs": {"filename_prefix": "ComfyUI", "images": ["8", 0]}},
    }
    if pos_title:
        p["6"]["_meta"] = {"title": pos_title}
    if neg_title:
        p["7"]["_meta"] = {"title": neg_title}
    return p


def _roles(prompt):
    """{role: text} for the texts we could identify."""
    return {t["role"]: t["text"] for t in prompt_texts(prompt) if t["role"]}


class TestTheOrdinaryCase(unittest.TestCase):
    """A stock txt2img workflow: both prompts, in full, correctly labeled."""

    def test_positive_and_negative_are_identified(self):
        self.assertEqual(_roles(_txt2img()), {"positive": POS, "negative": NEG})

    def test_the_log_line_carries_both_texts_verbatim(self):
        out = format_prompt_log("job-abc", _txt2img())
        self.assertIn("job-abc", out)
        self.assertIn(f"[positive] node 6 CLIPTextEncode.text: {POS}", out)
        self.assertIn(f"[negative] node 7 CLIPTextEncode.text: {NEG}", out)

    def test_it_says_where_the_text_is_wired(self):
        # A workflow can have several text boxes; without node + input name a
        # reader cannot tell which one on screen a line belongs to.
        out = format_prompt_log("j", _txt2img(pos_title="Positive Prompt"))
        self.assertIn("node 6 CLIPTextEncode.text (Positive Prompt)", out)

    def test_nodes_are_listed_in_workflow_order(self):
        out = format_prompt_log("j", _txt2img())
        self.assertLess(out.index("node 6"), out.index("node 7"))

    def test_an_empty_negative_is_still_reported(self):
        # "the negative prompt is empty" is a real answer. Dropping the line
        # would read as "no negative found", which is a different diagnosis.
        out = format_prompt_log("j", _txt2img(neg=""))
        self.assertIn("[negative] node 7 CLIPTextEncode.text: ", out)

    def test_output_is_deterministic(self):
        shuffled = dict(reversed(list(_txt2img().items())))
        self.assertEqual(format_prompt_log("j", _txt2img()),
                         format_prompt_log("j", shuffled))


class TestNotTruncated(unittest.TestCase):
    """Truncating is what would destroy the value of this log."""

    def test_a_long_real_prompt_is_logged_whole(self):
        long_prompt = ", ".join(f"tag_{i}" for i in range(200))  # ~1.5k chars
        self.assertLess(len(long_prompt), 4000)
        out = format_prompt_log("j", _txt2img(pos=long_prompt))
        self.assertIn(long_prompt, out)
        self.assertNotIn("TRUNCATED", out)

    def test_a_pathological_prompt_is_capped_and_says_so(self):
        huge = "x" * 9000
        out = format_prompt_log("j", _txt2img(pos=huge))
        self.assertIn("TRUNCATED", out)
        self.assertIn("9000 chars total", out)   # the real length is still stated
        self.assertNotIn("x" * 4100, out)


class TestRoleTracing(unittest.TestCase):
    """The role comes from the graph, through however many helpers sit between."""

    def test_through_a_guidance_node(self):
        p = _txt2img()
        p["3"]["inputs"]["positive"] = ["20", 0]
        p["20"] = {"class_type": "FluxGuidance",
                   "inputs": {"conditioning": ["6", 0], "guidance": 3.5}}
        self.assertEqual(_roles(p)["positive"], POS)

    def test_through_a_conditioning_combine(self):
        p = _txt2img()
        p["3"]["inputs"]["positive"] = ["20", 0]
        p["20"] = {"class_type": "ConditioningCombine",
                   "inputs": {"conditioning_1": ["6", 0], "conditioning_2": ["21", 0]}}
        p["21"] = {"class_type": "CLIPTextEncode",
                   "inputs": {"text": "extra style", "clip": ["4", 1]}}
        texts = {t["node"]: t["role"] for t in prompt_texts(p)}
        self.assertEqual(texts["6"], "positive")
        self.assertEqual(texts["21"], "positive")

    def test_a_controlnet_never_crosses_the_branches(self):
        # ⛔ ControlNetApplyAdvanced carries BOTH conditionings. Walking back
        # from `negative` must follow only its `negative` input — otherwise the
        # positive prompt gets logged as the negative one, which is worse than
        # no log at all.
        p = _txt2img()
        p["3"]["inputs"]["positive"] = ["20", 0]
        p["3"]["inputs"]["negative"] = ["20", 1]
        p["20"] = {"class_type": "ControlNetApplyAdvanced", "inputs": {
            "positive": ["6", 0], "negative": ["7", 0],
            "control_net": ["21", 0], "image": ["22", 0], "strength": 1.0}}
        p["21"] = {"class_type": "ControlNetLoader",
                   "inputs": {"control_net_name": "canny.pth"}}
        p["22"] = {"class_type": "LoadImage", "inputs": {"image": "pose.png"}}
        self.assertEqual(_roles(p), {"positive": POS, "negative": NEG})

    def test_a_custom_sampler_is_recognised_by_its_inputs(self):
        # Matching on class names would miss every fork; we match on the
        # positive/negative link inputs they all share.
        p = _txt2img()
        p["3"]["class_type"] = "SomeVendorKSamplerXL"
        self.assertEqual(_roles(p), {"positive": POS, "negative": NEG})

    def test_sdxl_dual_encoders_are_both_shown(self):
        p = _txt2img()
        p["6"] = {"class_type": "CLIPTextEncodeSDXL",
                  "inputs": {"text_g": POS, "text_l": "short tags", "clip": ["4", 1]}}
        out = format_prompt_log("j", p)
        self.assertIn(f"[positive] node 6 CLIPTextEncodeSDXL.text_g: {POS}", out)
        self.assertIn("[positive] node 6 CLIPTextEncodeSDXL.text_l: short tags", out)


class TestWhenWeCannotTell(unittest.TestCase):
    """⛔ Say "I don't know". A confident wrong answer costs the reader hours."""

    def test_an_untraceable_text_is_unlabeled_not_guessed(self):
        p = _txt2img()
        p["30"] = {"class_type": "PrimitiveStringMultiline",
                   "inputs": {"value": "a note nobody wired up"}}
        out = format_prompt_log("j", p)
        self.assertIn("[unlabeled] node 30 PrimitiveStringMultiline.value: "
                      "a note nobody wired up", out)
        # ...and it did not steal either role.
        self.assertEqual(_roles(p), {"positive": POS, "negative": NEG})

    def test_a_workflow_with_no_text_input_says_so(self):
        p = {"1": {"class_type": "LoadImage", "inputs": {"image": "a.png"}},
             "2": {"class_type": "ImageUpscaleWithModel",
                   "inputs": {"image": ["1", 0], "upscale_model": ["3", 0]}},
             "3": {"class_type": "UpscaleModelLoader",
                   "inputs": {"model_name": "4x.pth"}}}
        out = format_prompt_log("j7", p)
        self.assertIn("no text input identified", out)
        self.assertNotIn("[positive]", out)
        # The node types are named so a bug report still shows what ran.
        self.assertIn("3 node(s)", out)
        self.assertIn("LoadImage", out)

    def test_an_empty_workflow_says_so(self):
        out = format_prompt_log("j", {})
        self.assertIn("no text input identified", out)
        self.assertIn("empty workflow", out)

    def test_the_class_summary_is_bounded(self):
        p = {str(i): {"class_type": f"Node{i:02d}", "inputs": {"n": i}}
             for i in range(20)}
        out = format_prompt_log("j", p)
        self.assertIn("20 node(s)", out)
        self.assertIn("more", out)
        self.assertLess(len(out), 400)

    def test_model_and_file_names_are_never_mistaken_for_a_prompt(self):
        # An allowlist of INPUT names is what guarantees this: a ckpt/lora/
        # sampler/filename value must never surface as "the prompt".
        p = {"1": {"class_type": "CheckpointLoaderSimple",
                   "inputs": {"ckpt_name": "waiNSFW.safetensors"}},
             "2": {"class_type": "LoraLoader",
                   "inputs": {"lora_name": "style.safetensors", "model": ["1", 0]}},
             "3": {"class_type": "SaveImage",
                   "inputs": {"filename_prefix": "my_renders/batch", "images": ["2", 0]}}}
        self.assertEqual(prompt_texts(p), [])
        out = format_prompt_log("j", p)
        self.assertIn("no text input identified", out)
        for value in ("waiNSFW.safetensors", "style.safetensors", "my_renders/batch"):
            self.assertNotIn(value, out)

    def test_stray_empty_strings_are_not_noise(self):
        # An unlabeled EMPTY text says nothing; an unlabeled non-empty one might
        # be the answer. Only the former is dropped.
        p = _txt2img()
        p["30"] = {"class_type": "Whatever", "inputs": {"text": "   "}}
        self.assertNotIn("node 30", format_prompt_log("j", p))


class TestNeverRaises(unittest.TestCase):
    """A workflow is user data, not a contract — and this is only a log line."""

    def test_malformed_workflows_degrade(self):
        for p in (
            None, [], "boom", 42,
            {"3": None},
            {"3": "not a dict"},
            {"3": {"inputs": None}},
            {"3": {"inputs": "boom"}},
            {"3": {"inputs": {"positive": "not-a-link"}}},
            {"3": {"inputs": {"positive": []}}},
            {"3": {"inputs": {"positive": [None, 0]}}},
            {"3": {"inputs": {"positive": ["nope", 0]}}},     # dangling link
            {"3": {"inputs": {"text": 12345}}},               # text not a string
            {"3": {"class_type": 7, "inputs": {"text": "hi", "positive": ["3", 0]}}},
            {"3": {"inputs": {"text": "hi"}, "_meta": "boom"}},
            {"3": {"inputs": {"text": "hi"}, "_meta": {"title": 9}}},
            {3: {"class_type": "CLIPTextEncode", "inputs": {"text": "int key"}}},
        ):
            with self.subTest(p=p):
                out = format_prompt_log("j", p)
                self.assertTrue(out.startswith("job j final prompt"))

    def test_a_cyclic_graph_terminates(self):
        p = {"3": {"class_type": "KSampler",
                   "inputs": {"positive": ["4", 0], "negative": ["4", 0]}},
             "4": {"class_type": "Loop", "inputs": {"conditioning": ["5", 0]}},
             "5": {"class_type": "Loop", "inputs": {"conditioning": ["4", 0]}},
             "6": {"class_type": "CLIPTextEncode", "inputs": {"text": POS}}}
        out = format_prompt_log("j", p)          # must return, not hang
        self.assertIn("[unlabeled] node 6", out)


class TestWorkerWiring(unittest.IsolatedAsyncioTestCase):
    """It fires on the real job path — before submit — and stays local."""

    def _worker(self, submit=None):
        relay = mock.AsyncMock()
        relay.progress.return_value = {"cancel": False}
        relay.result.return_value = {}
        comfy = mock.AsyncMock()
        if submit is not None:
            comfy.submit.side_effect = submit
        return Worker(relay, comfy), relay, comfy

    async def test_logged_at_info_before_the_prompt_is_submitted(self):
        # submit blows up => if the line is still in the log, it was emitted
        # BEFORE the prompt went to ComfyUI, which is the requirement.
        w, _relay, comfy = self._worker(submit=PromptRejected("/prompt 500: boom"))
        with self.assertLogs("comfylink", level="INFO") as cm:
            self.assertEqual(
                await w.handle_job({"id": "j9", "api_prompt": _txt2img()}), "failed")
        blob = "\n".join(cm.output)
        self.assertIn("job j9 final prompt", blob)
        self.assertIn(POS, blob)
        self.assertIn(NEG, blob)
        comfy.submit.assert_awaited_once()

    async def test_an_unreadable_prompt_does_not_break_the_job(self):
        w, _relay, comfy = self._worker(submit=PromptRejected("/prompt 500: boom"))
        with self.assertLogs("comfylink", level="INFO") as cm:
            await w.handle_job({"id": "j9", "api_prompt": {"3": "not a dict"}})
        self.assertIn("no text input identified", "\n".join(cm.output))
        comfy.submit.assert_awaited_once()

    async def test_the_prompt_text_is_never_sent_to_the_relay(self):
        # ⛔ Local log only. Logging what a user typed on their own machine is
        # not the same thing as us holding their content.
        w, relay, _comfy = self._worker(submit=PromptRejected("/prompt 500: boom"))
        with self.assertLogs("comfylink", level="INFO"):
            await w.handle_job({"id": "j9", "api_prompt": _txt2img()})
        for call in relay.mock_calls:
            self.assertNotIn(POS, repr(call))
            self.assertNotIn(NEG, repr(call))


if __name__ == "__main__":
    unittest.main()
