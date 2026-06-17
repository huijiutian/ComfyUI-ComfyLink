"""Pure helpers for turning ComfyUI history/inputs into relay payloads.

Kept free of I/O so they are unit-testable without aiohttp or a running
ComfyUI.
"""

from __future__ import annotations

from typing import Any


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
