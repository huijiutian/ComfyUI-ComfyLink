"""Pure helpers for turning ComfyUI history/inputs into relay payloads.

Kept free of I/O so they are unit-testable without aiohttp or a running
ComfyUI.
"""

from __future__ import annotations

import json
import os
import re
from io import BytesIO
from typing import Any
from xml.sax.saxutils import escape

from .log import log

# Quality for PNG->WebP re-encode. 90 is a good size/fidelity tradeoff: it
# shrinks typical diffusion outputs well below their PNG size while staying
# visually lossless enough for the subscription "convert to WebP" feature.
WEBP_QUALITY = 90

# Extensions we treat as video. ComfyUI's animated/video savers land in
# outputs[node]["gifs"] (VHS_VideoCombine, animated GIF/WebP) or ["videos"]
# (native SaveVideo). We classify by extension because the history rows don't
# carry a media type, only filenames.
_VIDEO_EXTS = (".mp4", ".webm", ".mkv", ".mov", ".avi", ".gif")
# Animated WebP from VHS lands under "gifs" with a .webp name; it is NOT a
# still image we should re-encode (Pillow would flatten it to a single frame),
# so the source key — not just the extension — decides image-vs-video too.


def content_type_for(filename: str) -> str:
    """Best-effort MIME type from a filename extension."""
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".gif"):
        return "image/gif"
    if lower.endswith(".mp4"):
        return "video/mp4"
    if lower.endswith(".webm"):
        return "video/webm"
    if lower.endswith(".mkv"):
        return "video/x-matroska"
    if lower.endswith(".mov"):
        return "video/quicktime"
    if lower.endswith(".avi"):
        return "video/x-msvideo"
    return "application/octet-stream"


def _xmp_with_prompt(prompt: str) -> bytes:
    """Build a minimal XMP packet carrying only ComfyUI's ``prompt`` string.

    We deliberately embed *just* the prompt (not the much larger ``workflow``
    blob): users want the generation prompt to round-trip, and keeping the
    packet tiny keeps the WebP small. The prompt is JSON, so it can contain
    ``<``/``&``/``"`` — XML-escape it before placing it in an attribute.
    """
    esc = escape(prompt, {'"': "&quot;"})
    packet = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        '<rdf:Description xmlns:comfylink="http://comfylink.app/ns/1.0/" '
        'comfylink:prompt="' + esc + '"/>'
        "</rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
    return packet.encode("utf-8")


def encode_output(
    data: bytes, filename: str, output_format: str, media_type: str = "image"
) -> tuple[bytes, str, str]:
    """Optionally re-encode an output to WebP, preserving ComfyUI's prompt.

    Returns ``(data, filename, content_type)``. When ``output_format`` is
    ``"webp"`` *and* the item is a still image, the bytes are decoded with
    Pillow and re-saved as WebP — and ComfyUI's ``prompt`` text (PNG ``tEXt`` /
    WebP XMP) is carried into the new WebP's XMP so the prompt survives the
    re-encode. The filename extension is swapped to ``.webp`` and the
    content-type becomes ``image/webp``.

    Videos (``media_type == "video"``) and any non-webp ``output_format`` pass
    through unchanged, with the content-type derived from the filename. We never
    run Pillow over a video — it would flatten/corrupt an animation.

    WebP is BEST-EFFORT. The whole conversion is wrapped in try/except: if Pillow
    can't open/encode the bytes, is too old to accept ``xmp=``, or is missing
    entirely, we fall back to the original bytes/filename so a job is never
    crashed by the WebP step (a working PNG beats a failed job) — logged at
    WARNING so the cause is diagnosable, and the app copy sets the expectation
    that WebP only applies when the ComfyUI env supports it. ``prompt`` only
    rides along when present in the source; its absence is not an error.

    Note: we deliberately do NOT re-encode PNGs that stay PNG — ComfyUI already
    writes the prompt into the PNG ``tEXt`` and re-saving would risk dropping it.
    """
    if media_type == "video":
        return data, filename, content_type_for(filename)
    if (output_format or "").lower() != "webp":
        return data, filename, content_type_for(filename)
    try:
        from PIL import Image  # imported lazily — only needed when converting

        im = Image.open(BytesIO(data))
        # Pull ComfyUI's prompt (PNG tEXt or source-WebP XMP land it in .info).
        prompt = im.info.get("prompt")
        save_kwargs: dict[str, Any] = {"format": "WEBP", "quality": WEBP_QUALITY}
        if isinstance(prompt, (str, bytes)):
            if isinstance(prompt, bytes):
                prompt = prompt.decode("utf-8", "replace")
            save_kwargs["xmp"] = _xmp_with_prompt(prompt)
        buf = BytesIO()
        try:
            im.save(buf, **save_kwargs)
        except TypeError:
            # Old Pillow without xmp= support: re-encode without metadata
            # rather than crash. Better a prompt-less WebP than a failed job.
            buf = BytesIO()
            im.save(buf, format="WEBP", quality=WEBP_QUALITY)
            log.info("Pillow too old for xmp=; shipping WebP without prompt for %s", filename)
        webp = buf.getvalue()
        new_name = _swap_ext(filename, ".webp")
        return webp, new_name, "image/webp"
    except Exception as e:  # noqa: BLE001 - never let conversion crash a job
        # WebP is BEST-EFFORT: it only helps when the ComfyUI env can do it
        # (Pillow with libwebp). If conversion fails (Pillow missing, no libwebp
        # encoder, undecodable bytes), fall back to shipping the original bytes so
        # the job still SUCCEEDS — a working PNG beats a failed job. Logged at
        # WARNING (not silent) so the cause is diagnosable; the app copy tells
        # users WebP is applied only when the environment supports it.
        log.warning(
            "webp conversion unavailable for %s (%s); shipping original — "
            "install Pillow with WebP support in ComfyUI to enable it",
            filename, e,
        )
        return data, filename, content_type_for(filename)


def _swap_ext(filename: str, new_ext: str) -> str:
    """Replace a filename's extension (e.g. ``a.png`` -> ``a.webp``)."""
    root, _ = os.path.splitext(filename)
    return (root or filename) + new_ext


def within_cap(total_bytes: int, max_output_bytes: int) -> bool:
    """Decide whether outputs totaling ``total_bytes`` may be uploaded.

    A non-positive ``max_output_bytes`` means "unlimited" (0/absent = legacy
    safety), so it always allows. Otherwise the total must not exceed the cap.
    Pure + separately testable (no I/O).
    """
    if max_output_bytes <= 0:
        return True
    return total_bytes <= max_output_bytes


def _media_type_for(filename: str, source_key: str) -> str:
    """Classify an output item as ``"image"`` or ``"video"``.

    ComfyUI files videos/animations under the ``gifs``/``videos`` output keys
    and stills under ``images`` — but VHS_VideoCombine can also drop an animated
    WebP/GIF into ``gifs``. So an item from a video-ish key is a video, and so
    is anything with a known video extension. Everything else is an image.
    """
    if source_key in ("gifs", "videos"):
        return "video"
    if filename.lower().endswith(_VIDEO_EXTS):
        return "video"
    return "image"


def extract_outputs(history: dict, prompt_id: str) -> list[dict]:
    """Pull the final output items (images + videos) for a prompt from /history.

    Collects ComfyUI's ``images`` (stills) plus ``gifs``/``videos`` (animations
    and video clips, e.g. VHS_VideoCombine -> ``gifs``, native SaveVideo ->
    ``videos``). Only items of type ``output`` are returned — ``temp``/preview
    artifacts are skipped (we never ship previews). Each item carries a
    ``media_type`` ("image"|"video") so the rest of the pipeline knows whether
    to WebP-convert it and what to tell the relay/app.
    """
    entry = history.get(prompt_id) or {}
    outputs = entry.get("outputs") or {}
    out: list[dict] = []
    for node_output in outputs.values():
        for source_key in ("images", "gifs", "videos"):
            for it in node_output.get(source_key, []) or []:
                if it.get("type") != "output":
                    continue
                filename = it.get("filename", "")
                out.append(
                    {
                        "filename": filename,
                        "subfolder": it.get("subfolder", ""),
                        "type": it.get("type", "output"),
                        "media_type": _media_type_for(filename, source_key),
                    }
                )
    return out


def apply_inputs(prompt: dict, inputs: list[dict], key_to_name: dict[str, str]) -> dict:
    """Rewrite a workflow's node inputs to reference uploaded local filenames.

    ``inputs`` is the relay's claim payload list ([{r2_key, node, field, ...}]);
    ``key_to_name`` maps each r2_key to the filename returned by the local
    ComfyUI /upload/image. Mutates and returns ``prompt``.
    """
    for inp in inputs:
        node = str(inp.get("node", ""))
        field = inp.get("field", "")
        key = inp.get("r2_key", "")
        name = key_to_name.get(key)
        if node and field and name and node in prompt:
            prompt[node].setdefault("inputs", {})[field] = name
    return prompt


def progress_event(msg: dict) -> dict[str, Any] | None:
    """Classify a ComfyUI websocket text message.

    DEPRECATED: the worker no longer drives jobs over the websocket (its message
    format drifts across ComfyUI versions); _run_prompt now polls the stable
    /history + /queue REST endpoints instead. This pure classifier is retained
    only for reference/tests and is not used in the live job path.

    Returns one of:
      {"kind": "progress", "value": int, "max": int}
      {"kind": "done"}                       (executing with node == null)
      {"kind": "error", "message": str}
    or None for messages we ignore. ``prompt_id`` correlation is left to the
    caller (it knows the active prompt).
    """
    t = msg.get("type")
    d = msg.get("data") or {}
    if t == "progress":
        return {"kind": "progress", "value": int(d.get("value", 0)), "max": int(d.get("max", 0)),
                "prompt_id": d.get("prompt_id")}
    if t == "executing" and d.get("node") is None:
        return {"kind": "done", "prompt_id": d.get("prompt_id")}
    if t == "execution_error":
        return {"kind": "error", "message": d.get("exception_message") or "execution error",
                "prompt_id": d.get("prompt_id")}
    return None


# ── /prompt rejections: structured diagnosis ────────────────────────────────
#
# When ComfyUI refuses a prompt it answers 400 with a STRUCTURED body:
#
#   {"error": {"type": "prompt_outputs_failed_validation", "message": "..."},
#    "node_errors": {"4": {"class_type": "CheckpointLoaderSimple", "errors": [
#       {"type": "value_not_in_list", "message": "Value not in list",
#        "details": "ckpt_name: 'x.safetensors' not in [...]",
#        "extra_info": {"input_name": "ckpt_name", "received_value": "x.safetensors",
#                       "input_config": ...}}]}}}
#
# The one failure users actually hit is "the model this workflow asks for is not
# on this machine any more" — they added/removed models on the PC while the app
# still shows the inventory captured earlier. That is `value_not_in_list`, and
# turning it into a machine-readable code is what lets the app say "model X is
# gone, refresh" instead of showing the raw JSON above.

# Failure code reported to the relay (and read by the app) when a workflow
# references a value that is not in this ComfyUI's list for that input.
MODEL_NOT_FOUND = "model_not_found"

# The only ComfyUI validation type we map to MODEL_NOT_FOUND. Every other type
# (required_input_missing, value_not_in_range, invalid_prompt, exception_during
# _validation, bad_linked_input, return_type_mismatch, ...) stays an ordinary
# failure with NO code — misreporting a missing widget as a missing model would
# send the user chasing a model that was never the problem.
_VALUE_NOT_IN_LIST = "value_not_in_list"

# Caps. `details`/`received_value` are attacker-free but not size-bounded (a
# pathological workflow could carry a huge string), and the message ends up in a
# DB column and a phone UI, so everything that leaves here is bounded.
_MAX_ITEM_CHARS = 120   # per extracted field name / value
_MAX_LISTED = 5         # missing models spelled out before "(+N more)"
_MAX_BODY_CHARS = 400   # raw body kept in the fallback (unstructured) message

# "ckpt_name: 'x.safetensors' not in [...]" — the human `details` string, used
# only when `extra_info` is missing (older/forked ComfyUI). Non-greedy so a
# value containing an apostrophe still resolves against the literal " not in ".
_DETAILS_RE = re.compile(r"\s*([^:\s]+)\s*:\s*'(.*?)'\s+not in\b", re.DOTALL)


def prompt_rejection(status: int, body: str) -> tuple[str, str]:
    """Turn a non-200 ComfyUI /prompt response into ``(message, error_code)``.

    Returns ``(message, MODEL_NOT_FOUND)`` when the body proves at least one
    input value is not in ComfyUI's list for that input, else
    ``(message, "")`` — an ordinary failure, exactly as before this existed.

    DEFENSIVE BY CONSTRUCTION. ComfyUI's response shape is not a contract we
    control and it has changed before, so every layer degrades instead of
    raising: a non-JSON body, a JSON non-object, a missing/oddly-typed
    ``node_errors``, a node entry that isn't a dict, an ``errors`` value that
    isn't a list, an entry with no usable input name AND no usable value — each
    of those simply means "no structured diagnosis", and the caller reports the
    plain failure it would have reported anyway. NEVER raises. Pure => testable
    without a ComfyUI.
    """
    missing = _missing_values(_as_dict(body))
    if not missing:
        return _plain(status, body), ""
    listed = "; ".join(_describe(m) for m in missing[:_MAX_LISTED])
    extra = len(missing) - _MAX_LISTED
    if extra > 0:
        listed += f" (+{extra} more)"
    return f"model not found: {listed}", MODEL_NOT_FOUND


def _as_dict(body: str) -> dict:
    """Parse a response body into a dict; ``{}`` for anything else. Never raises."""
    try:
        data = json.loads(body)
    except Exception:  # noqa: BLE001 - any parse failure => no structured info
        return {}
    return data if isinstance(data, dict) else {}


def _plain(status: int, body: str) -> str:
    """The unstructured fallback message — the historical shape, but BOUNDED.

    The old code inlined the whole body, and for a validation failure that body
    embeds ComfyUI's candidate list, so the user was shown kilobytes of JSON.
    Truncating costs nothing diagnostically (the caller logs the full body).
    """
    text = (body or "").strip()
    if len(text) > _MAX_BODY_CHARS:
        text = text[:_MAX_BODY_CHARS] + "…"
    return f"/prompt {status}: {text}"


def _missing_values(payload: dict) -> list[dict]:
    """Every ``value_not_in_list`` entry in ``node_errors``, in node-id order.

    Each item is ``{"node", "class_type", "field", "value"}`` with missing parts
    as "". Sorted so the message is deterministic for a given response (dict
    order from JSON is insertion order, which we don't want to depend on).
    """
    node_errors = payload.get("node_errors")
    if not isinstance(node_errors, dict):
        return []
    found: list[dict] = []
    for node_id in sorted(node_errors, key=str):
        node = node_errors[node_id]
        if not isinstance(node, dict):
            continue
        errors = node.get("errors")
        if not isinstance(errors, list):
            continue
        class_type = node.get("class_type")
        for err in errors:
            if not isinstance(err, dict) or err.get("type") != _VALUE_NOT_IN_LIST:
                continue
            field, value = _field_and_value(err)
            if not field and not value:
                # Recognised as the right TYPE but we can name neither the input
                # nor the value => the app could not tell the user anything
                # useful. Drop it rather than report an empty "model not found".
                continue
            found.append({
                "node": str(node_id),
                "class_type": _trim(class_type) if isinstance(class_type, str) else "",
                "field": field,
                "value": value,
            })
    return found


def _field_and_value(err: dict) -> tuple[str, str]:
    """``(input_name, received_value)`` for one validation error; "" when unknown.

    ``extra_info`` is the authoritative source (ComfyUI puts the raw values
    there); the human ``details`` string is only a fallback for a ComfyUI that
    doesn't send it. ``extra_info["input_config"]`` is deliberately never read —
    it can hold the whole candidate list.
    """
    extra = err.get("extra_info")
    if isinstance(extra, dict):
        name = extra.get("input_name")
        got = extra.get("received_value")
        field = _trim(name) if isinstance(name, str) else ""
        # A model name is a string. Anything structured is not one, so we don't
        # spill it into a user-facing message.
        value = "" if got is None or isinstance(got, (dict, list)) else _trim(str(got))
        if field or value:
            return field, value
    details = err.get("details")
    if isinstance(details, str):
        m = _DETAILS_RE.match(details)
        if m:
            return _trim(m.group(1)), _trim(m.group(2))
    return "", ""


def _trim(s: str) -> str:
    """Normalize one extracted token for display.

    Collapses whitespace (a stray newline would break the one-line message),
    drops double quotes so the app can read every quoted run in the message as
    exactly one model name, and caps the length.
    """
    out = " ".join(s.split()).replace('"', "")
    return out[:_MAX_ITEM_CHARS]


def _describe(item: dict) -> str:
    '''One missing value as `"<value>" (input <field>, node <id> <class>)`.

    The quoted value comes FIRST and contains no quote character of its own, so
    the app can localize the copy and still pull the model name(s) out with a
    trivial scan for quoted runs — while a user (or an older app that just shows
    the message) reads a sentence.
    '''
    bits = []
    if item["field"]:
        bits.append(f"input {item['field']}")
    if item["node"]:
        node = f"node {item['node']}"
        if item["class_type"]:
            node += f" {item['class_type']}"
        bits.append(node)
    suffix = f" ({', '.join(bits)})" if bits else ""
    return f'"{item["value"] or "?"}"{suffix}'
