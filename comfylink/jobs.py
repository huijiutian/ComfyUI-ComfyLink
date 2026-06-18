"""Pure helpers for turning ComfyUI history/inputs into relay payloads.

Kept free of I/O so they are unit-testable without aiohttp or a running
ComfyUI.
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import Any

from .log import log

# Quality for PNG->WebP re-encode. 90 is a good size/fidelity tradeoff: it
# shrinks typical diffusion outputs well below their PNG size while staying
# visually lossless enough for the subscription "convert to WebP" feature.
WEBP_QUALITY = 90


def content_type_for(filename: str) -> str:
    """Best-effort MIME type from a filename extension."""
    lower = filename.lower()
    if lower.endswith(".png"):
        return "image/png"
    if lower.endswith(".webp"):
        return "image/webp"
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    return "application/octet-stream"


def convert_if_webp(data: bytes, filename: str, output_format: str) -> tuple[bytes, str, str]:
    """Optionally re-encode output image bytes to WebP.

    Returns ``(data, filename, content_type)``. When ``output_format`` is
    ``"webp"`` the bytes are decoded with Pillow and re-saved as WebP, the
    filename extension is swapped to ``.webp`` and the content-type becomes
    ``image/webp``. For any other format (default ``"png"``) the inputs pass
    through unchanged, with the content-type derived from the filename.

    Conversion is wrapped in try/except: if Pillow can't open/encode the bytes
    (odd format, truncated data, PIL missing) we fall back to the original
    bytes/filename so a job is never crashed by the WebP step.
    """
    if (output_format or "").lower() != "webp":
        return data, filename, content_type_for(filename)
    try:
        from PIL import Image  # imported lazily — only needed when converting

        im = Image.open(BytesIO(data))
        buf = BytesIO()
        im.save(buf, format="WEBP", quality=WEBP_QUALITY)
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


def extract_output_images(history: dict, prompt_id: str) -> list[dict]:
    """Pull the final output images for a prompt from a /history response.

    Only images of type 'output' are returned — 'temp'/preview artifacts are
    skipped (we never ship previews).
    """
    entry = history.get(prompt_id) or {}
    outputs = entry.get("outputs") or {}
    images: list[dict] = []
    for node_output in outputs.values():
        for im in node_output.get("images", []) or []:
            if im.get("type") == "output":
                images.append(
                    {
                        "filename": im.get("filename", ""),
                        "subfolder": im.get("subfolder", ""),
                        "type": im.get("type", "output"),
                    }
                )
    return images


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
