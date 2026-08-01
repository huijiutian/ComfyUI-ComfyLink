"""Tests for diagnosing a ComfyUI /prompt rejection.

The user story: they add or delete models on the PC, the app is still showing
the inventory captured when the plugin registered, so they pick a model that is
gone. ComfyUI answers /prompt with a 400 whose body says EXACTLY what is wrong —
which the plugin used to paste into the failure message verbatim, so the app
showed a wall of raw JSON (with the whole candidate list embedded) and the user
had no idea what happened.

Pinned here:

  * a ``value_not_in_list`` node error becomes error_code ``model_not_found``
    and a message naming the input and the value;
  * ⛔ NO other validation failure is ever mistaken for it — a missing required
    input, an out-of-range number, an unknown node type, a validation exception
    and a prompt-level error all stay ordinary, code-less failures;
  * the parser degrades instead of raising, at every layer, for every shape
    ComfyUI could plausibly drift into (it is not a contract we control);
  * the diagnosis reaches the relay result through the EXISTING failure path
    (JobFailed -> relay.result(error_code=...)), with no new channel;
  * the failure path issues NO interrupt — a rejected prompt never ran, and the
    user's own local generation must not be touched.

Run:  python -m unittest discover -s tests
"""

import os
import sys
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink.comfy import ComfyClient, PromptRejected  # noqa: E402
from comfylink.jobs import MODEL_NOT_FOUND, prompt_rejection  # noqa: E402
from comfylink.worker import JobFailed, Worker  # noqa: E402


def _node_error(errors, class_type="CheckpointLoaderSimple", node="4"):
    """A ComfyUI 400 body carrying `errors` on one node."""
    return {
        "error": {"type": "prompt_outputs_failed_validation",
                  "message": "Prompt outputs failed validation"},
        "node_errors": {node: {"class_type": class_type, "errors": errors,
                               "dependent_outputs": []}},
    }


def _value_not_in_list(field="ckpt_name", value="waiNSFW_v11.safetensors",
                       with_extra_info=True, details=None):
    """One ComfyUI ``value_not_in_list`` error, as validate_inputs emits it."""
    err = {
        "type": "value_not_in_list",
        "message": "Value not in list",
        "details": (details if details is not None
                    else f"{field}: '{value}' not in (list of length 42)"),
    }
    if with_extra_info:
        err["extra_info"] = {
            "input_name": field,
            # ComfyUI puts the candidate list here when it is short. We must
            # never read it — that is the wall of JSON we are getting rid of.
            "input_config": [["a.safetensors", "b.safetensors"]],
            "received_value": value,
        }
    return err


def _json(payload):
    import json
    return json.dumps(payload)


class TestModelNotFound(unittest.TestCase):
    """The happy path: a real ComfyUI rejection becomes a usable diagnosis."""

    def test_extra_info_is_the_source_of_truth(self):
        msg, code = prompt_rejection(400, _json(_node_error([_value_not_in_list()])))
        self.assertEqual(code, MODEL_NOT_FOUND)
        # Everything the app needs to talk to the user: which value, which input.
        self.assertIn('"waiNSFW_v11.safetensors"', msg)
        self.assertIn("ckpt_name", msg)
        self.assertIn("node 4", msg)
        self.assertIn("CheckpointLoaderSimple", msg)

    def test_details_string_is_the_fallback(self):
        # A ComfyUI (or fork) that omits extra_info: parse the human string.
        body = _node_error([_value_not_in_list(
            field="lora_name", value="style/xl_art.safetensors",
            with_extra_info=False)], class_type="LoraLoader", node="10")
        msg, code = prompt_rejection(400, _json(body))
        self.assertEqual(code, MODEL_NOT_FOUND)
        self.assertIn('"style/xl_art.safetensors"', msg)
        self.assertIn("lora_name", msg)

    def test_details_value_may_contain_an_apostrophe(self):
        body = _node_error([_value_not_in_list(
            field="ckpt_name", with_extra_info=False,
            details="ckpt_name: 'it's_a.safetensors' not in ['a.safetensors']")])
        msg, code = prompt_rejection(400, _json(body))
        self.assertEqual(code, MODEL_NOT_FOUND)
        self.assertIn('"it\'s_a.safetensors"', msg)

    def test_never_leaks_the_candidate_list(self):
        # The whole point: the app must not receive ComfyUI's enum dump.
        body = _node_error([_value_not_in_list(
            details="ckpt_name: 'gone.safetensors' not in "
                    "['a.safetensors', 'b.safetensors', 'c.safetensors']")])
        msg, _ = prompt_rejection(400, _json(body))
        for candidate in ("a.safetensors", "b.safetensors", "c.safetensors"):
            self.assertNotIn(candidate, msg)

    def test_several_missing_models_are_all_named(self):
        body = {"node_errors": {
            "4": {"class_type": "CheckpointLoaderSimple",
                  "errors": [_value_not_in_list(value="gone_ckpt.safetensors")]},
            "10": {"class_type": "LoraLoader",
                   "errors": [_value_not_in_list(field="lora_name",
                                                 value="gone_lora.safetensors")]},
        }}
        msg, code = prompt_rejection(400, _json(body))
        self.assertEqual(code, MODEL_NOT_FOUND)
        self.assertIn("gone_ckpt.safetensors", msg)
        self.assertIn("gone_lora.safetensors", msg)

    def test_message_is_bounded_and_deterministic(self):
        many = {str(i): {"class_type": "LoraLoader",
                         "errors": [_value_not_in_list(field="lora_name",
                                                       value=f"m{i}.safetensors")]}
                for i in range(9)}
        msg, code = prompt_rejection(400, _json({"node_errors": many}))
        self.assertEqual(code, MODEL_NOT_FOUND)
        self.assertIn("(+4 more)", msg)          # 9 found, 5 spelled out
        # Deterministic regardless of the dict order JSON happened to carry.
        shuffled = dict(reversed(list(many.items())))
        again, _ = prompt_rejection(400, _json({"node_errors": shuffled}))
        self.assertEqual(msg, again)

    def test_value_is_sanitized_for_the_app_contract(self):
        # The app reads every quoted run in the message as one model name, so a
        # value may never carry a quote of its own; newlines would break the
        # one-line message; length is capped.
        body = _node_error([_value_not_in_list(
            value='we"ird\nname' + "x" * 300)])
        msg, code = prompt_rejection(400, _json(body))
        self.assertEqual(code, MODEL_NOT_FOUND)
        self.assertEqual(msg.count('"'), 2)      # exactly the delimiters
        self.assertNotIn("\n", msg)
        self.assertLess(len(msg), 300)


class TestNotAModelProblem(unittest.TestCase):
    """⛔ The guard rail: nothing else may be reported as a missing model.

    Sending a user after a model that was never the problem is worse than the
    generic failure — they will delete and re-download a model that was fine.
    """

    def test_other_validation_types_get_no_code(self):
        for err in (
            {"type": "required_input_missing", "message": "Required input is missing",
             "details": "ckpt_name", "extra_info": {"input_name": "ckpt_name"}},
            {"type": "value_smaller_than_min", "message": "Value smaller than min",
             "details": "steps: 0 < 1",
             "extra_info": {"input_name": "steps", "received_value": 0}},
            {"type": "value_bigger_than_max", "message": "Value bigger than max",
             "details": "steps: 9999 > 1000",
             "extra_info": {"input_name": "steps", "received_value": 9999}},
            {"type": "bad_linked_input", "message": "Bad linked input",
             "details": "image", "extra_info": {"input_name": "image"}},
            {"type": "return_type_mismatch", "message": "Return type mismatch",
             "details": "image, IMAGE != LATENT"},
            {"type": "exception_during_validation", "message": "Exception when validating",
             "details": "ZeroDivisionError", "extra_info": {"exception_type": "..."}},
            {"type": "invalid_input_type", "message": "Failed to convert",
             "details": "seed, -1 -> INT",
             "extra_info": {"input_name": "seed", "received_value": "-1"}},
        ):
            with self.subTest(kind=err["type"]):
                msg, code = prompt_rejection(400, _json(_node_error([err])))
                self.assertEqual(code, "", f"{err['type']} misjudged as a model problem")
                self.assertNotIn("model not found", msg)

    def test_unknown_node_type_gets_no_code(self):
        body = {"error": {"type": "invalid_prompt",
                          "message": "Cannot execute because node WeirdNode does not exist."},
                "node_errors": {}}
        _msg, code = prompt_rejection(400, _json(body))
        self.assertEqual(code, "")

    def test_prompt_level_errors_get_no_code(self):
        for body in (
            {"error": {"type": "prompt_no_outputs", "message": "Prompt has no outputs"}},
            {"error": {"type": "no_prompt", "message": "No prompt provided"}},
            {"error": {"type": "prompt_outputs_failed_validation"}, "node_errors": {}},
        ):
            with self.subTest(body=body):
                _msg, code = prompt_rejection(400, _json(body))
                self.assertEqual(code, "")

    def test_a_mixed_response_still_reports_only_the_model(self):
        # A workflow can fail validation for several reasons at once. Only the
        # value_not_in_list entries may drive the code and the message.
        body = {"node_errors": {
            "3": {"class_type": "KSampler",
                  "errors": [{"type": "required_input_missing",
                              "details": "model",
                              "extra_info": {"input_name": "model"}}]},
            "4": {"class_type": "CheckpointLoaderSimple",
                  "errors": [_value_not_in_list(value="gone.safetensors")]},
        }}
        msg, code = prompt_rejection(400, _json(body))
        self.assertEqual(code, MODEL_NOT_FOUND)
        self.assertIn("gone.safetensors", msg)
        self.assertNotIn("required_input_missing", msg)


class TestDefensiveParsing(unittest.TestCase):
    """ComfyUI's body shape is not ours; every layer must degrade, never raise."""

    def test_unstructured_bodies_degrade_to_a_plain_failure(self):
        for body in (
            "",                                        # empty
            "Internal Server Error",                   # plain text
            "<html><body>502</body></html>",           # a proxy page
            "not json {{{",                            # broken JSON
            "[1, 2, 3]",                               # JSON, but not an object
            '"just a string"',
            "null",
        ):
            with self.subTest(body=body):
                msg, code = prompt_rejection(400, body)
                self.assertEqual(code, "")
                self.assertTrue(msg.startswith("/prompt 400:"))

    def test_malformed_node_errors_degrade(self):
        for payload in (
            {"node_errors": None},
            {"node_errors": []},                                   # list, not dict
            {"node_errors": "boom"},
            {"node_errors": {"4": None}},                          # node not a dict
            {"node_errors": {"4": ["errors"]}},
            {"node_errors": {"4": {"errors": None}}},              # errors not a list
            {"node_errors": {"4": {"errors": "value_not_in_list"}}},
            {"node_errors": {"4": {"errors": [None, 7, "x"]}}},    # entries not dicts
            {"node_errors": {"4": {"errors": [{}]}}},              # no type at all
            {"node_errors": {"4": {"errors": [{"type": None}]}}},
        ):
            with self.subTest(payload=payload):
                msg, code = prompt_rejection(400, _json(payload))
                self.assertEqual(code, "")
                self.assertTrue(msg.startswith("/prompt 400:"))

    def test_right_type_but_nothing_nameable_degrades(self):
        # Recognised as value_not_in_list, yet neither the input nor the value
        # can be read => the app could tell the user nothing, so we don't claim
        # a model problem.
        for err in (
            {"type": "value_not_in_list"},
            {"type": "value_not_in_list", "details": None, "extra_info": None},
            {"type": "value_not_in_list", "details": "totally unparseable"},
            {"type": "value_not_in_list", "extra_info": {}},
            {"type": "value_not_in_list", "extra_info": "nope", "details": 12},
            {"type": "value_not_in_list",
             "extra_info": {"input_name": None, "received_value": None}},
        ):
            with self.subTest(err=err):
                _msg, code = prompt_rejection(400, _json(_node_error([err])))
                self.assertEqual(code, "")

    def test_structured_received_value_is_not_spilled(self):
        # received_value is a dict/list => not a model name. The input name still
        # identifies the problem, so this stays a model problem, but nothing
        # structured leaks into the message.
        err = {"type": "value_not_in_list",
               "extra_info": {"input_name": "ckpt_name",
                              "received_value": {"weird": ["shape"]}}}
        msg, code = prompt_rejection(400, _json(_node_error([err])))
        self.assertEqual(code, MODEL_NOT_FOUND)
        self.assertIn("ckpt_name", msg)
        self.assertNotIn("weird", msg)

    def test_non_string_received_value_still_names_the_value(self):
        err = {"type": "value_not_in_list",
               "extra_info": {"input_name": "steps", "received_value": 17}}
        msg, code = prompt_rejection(400, _json(_node_error([err])))
        self.assertEqual(code, MODEL_NOT_FOUND)
        self.assertIn('"17"', msg)

    def test_missing_class_type_is_fine(self):
        body = {"node_errors": {"4": {"errors": [_value_not_in_list()]}}}
        msg, code = prompt_rejection(400, _json(body))
        self.assertEqual(code, MODEL_NOT_FOUND)
        self.assertIn("node 4", msg)

    def test_plain_message_is_truncated(self):
        # The old code inlined the whole body — for a validation failure that is
        # ComfyUI's entire model list. Bound it.
        msg, code = prompt_rejection(500, "x" * 5000)
        self.assertEqual(code, "")
        self.assertLess(len(msg), 500)
        self.assertTrue(msg.endswith("…"))

    def test_any_status_is_accepted(self):
        for status in (400, 403, 500, 502):
            with self.subTest(status=status):
                msg, code = prompt_rejection(status, "boom")
                self.assertEqual(code, "")
                self.assertIn(str(status), msg)


# ── the wiring: parse -> PromptRejected -> JobFailed -> relay result ─────────

class _Resp:
    def __init__(self, status, text):
        self.status = status
        self._text = text

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def text(self):
        return self._text

    async def json(self):
        import json
        return json.loads(self._text)


class _Session:
    def __init__(self, resp):
        self._resp = resp

    def post(self, url, json=None):        # noqa: A002 - mirrors aiohttp's API
        return self._resp


class TestSubmitRaisesStructured(unittest.IsolatedAsyncioTestCase):
    async def test_rejection_carries_the_code(self):
        body = _json(_node_error([_value_not_in_list()]))
        comfy = ComfyClient(_Session(_Resp(400, body)), "http://127.0.0.1:8188")
        with self.assertRaises(PromptRejected) as ctx:
            await comfy.submit({"4": {}}, "cid")
        self.assertEqual(ctx.exception.error_code, MODEL_NOT_FOUND)
        self.assertIn("waiNSFW_v11.safetensors", ctx.exception.message)

    async def test_undiagnosable_rejection_carries_no_code(self):
        comfy = ComfyClient(_Session(_Resp(500, "boom")), "http://127.0.0.1:8188")
        with self.assertRaises(PromptRejected) as ctx:
            await comfy.submit({"4": {}}, "cid")
        self.assertEqual(ctx.exception.error_code, "")

    async def test_success_is_untouched(self):
        comfy = ComfyClient(_Session(_Resp(200, '{"prompt_id": "p1"}')),
                            "http://127.0.0.1:8188")
        self.assertEqual(await comfy.submit({"4": {}}, "cid"), "p1")


class TestReportedToRelay(unittest.IsolatedAsyncioTestCase):
    """The code must reach the app through the EXISTING failure path."""

    def _worker(self, exc):
        relay = mock.AsyncMock()
        relay.progress.return_value = {"cancel": False}
        comfy = mock.AsyncMock()
        comfy.submit.side_effect = exc
        return Worker(relay, comfy), relay, comfy

    async def test_run_prompt_turns_it_into_a_coded_failure(self):
        w, _relay, _comfy = self._worker(
            PromptRejected('model not found: "gone.safetensors"', MODEL_NOT_FOUND))
        with self.assertRaises(JobFailed) as ctx:
            await w._run_prompt("j1", {})
        self.assertEqual(ctx.exception.error_code, MODEL_NOT_FOUND)
        self.assertIn("gone.safetensors", ctx.exception.message)

    async def test_result_carries_status_message_and_code(self):
        w, relay, comfy = self._worker(
            PromptRejected('model not found: "gone.safetensors" (input ckpt_name)',
                           MODEL_NOT_FOUND))
        self.assertEqual(await w.handle_job({"id": "j1", "api_prompt": {"4": {}}}),
                         "failed")
        relay.result.assert_awaited_once()
        args, kwargs = relay.result.await_args
        self.assertEqual(args[0], "j1")
        self.assertEqual(args[1], "failed")
        self.assertEqual(args[2], [])                       # no images
        self.assertIn("gone.safetensors", args[3])
        self.assertEqual(kwargs["error_code"], MODEL_NOT_FOUND)
        # ⛔ R-05 guard: a rejected prompt never ran, so nothing may be
        # interrupted — a user's own local generation must survive untouched.
        comfy.interrupt.assert_not_awaited()
        comfy.queue_delete.assert_not_awaited()

    async def test_undiagnosable_rejection_reports_as_before(self):
        w, relay, _comfy = self._worker(PromptRejected("/prompt 500: boom"))
        self.assertEqual(await w.handle_job({"id": "j1", "api_prompt": {}}), "failed")
        _args, kwargs = relay.result.await_args
        self.assertEqual(kwargs["error_code"], "")


if __name__ == "__main__":
    unittest.main()
