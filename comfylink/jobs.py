"""Pure helpers for turning ComfyUI history/inputs into relay payloads.

Kept free of I/O so they are unit-testable without aiohttp or a running
ComfyUI.
"""

from __future__ import annotations

import os
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

    The whole conversion is wrapped in try/except: if Pillow can't open/encode
    the bytes, is too old to accept ``xmp=``, or is missing entirely, we fall
    back to the original bytes/filename so a job is never crashed by the WebP
    step. ``prompt`` only rides along when present in the source; its absence is
    not an error.

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
        log.warning("webp conversion failed for %s (%s); shipping original", filename, e)
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
