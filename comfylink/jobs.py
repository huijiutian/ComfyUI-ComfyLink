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
from .status import STATUS  # pure in-memory flags(不引 I/O,单测无碍)
from .version import __version__

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


def _xmp_with_prompt(prompt: str, positive: str = "", negative: str = "") -> bytes:
    """Build a minimal XMP packet carrying ComfyUI's ``prompt`` string, plus —
    when extractable — the final positive/negative prompt texts verbatim.

    We deliberately embed the prompt (not the much larger ``workflow`` blob):
    users want the generation prompt to round-trip, and keeping the packet
    tiny keeps the WebP small. The prompt is JSON, so it can contain
    ``<``/``&``/``"`` — XML-escape it before placing it in an attribute.

    R-1.0.6-30(直通字段): [positive]/[negative] 是提交时算好的**最终提示词
    全文**(与 R-20 提交日志同源,见 :func:`prompt_texts`)。写成**元素形式**
    ``<comfylink:positive>``/``<comfylink:negative>`` —— 提示词里引号/换行是
    常态,元素内容只需转义 ``&``/``<``/``>``,不用跟属性引号纠缠。空的不写
    (提不出 = 宁缺毋假);``comfylink:prompt``(workflow JSON)照写不动,
    读不到直通字段的旧 App 与高级场景仍可整树解析。来源标记
    ``comfylink:generator`` 恒写,让「这图是 ComfyLink 产的」可辨认。
    """
    esc = escape(prompt, {'"': "&quot;"})
    fields = ""
    if positive.strip():
        fields += "<comfylink:positive>" + escape(positive) + "</comfylink:positive>"
    if negative.strip():
        fields += "<comfylink:negative>" + escape(negative) + "</comfylink:negative>"
    desc_open = (
        '<rdf:Description xmlns:comfylink="http://comfylink.app/ns/1.0/" '
        'comfylink:generator="ComfyLink ' + escape(__version__, {'"': "&quot;"}) + '" '
        'comfylink:prompt="' + esc + '"'
    )
    desc = desc_open + (">" + fields + "</rdf:Description>" if fields else "/>")
    packet = (
        '<?xpacket begin="﻿" id="W5M0MpCehiHzreSzNTczkc9d"?>'
        '<x:xmpmeta xmlns:x="adobe:ns:meta/">'
        '<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">'
        + desc +
        "</rdf:RDF></x:xmpmeta>"
        '<?xpacket end="w"?>'
    )
    return packet.encode("utf-8")


def _role_texts(prompt_json: str) -> tuple[str, str]:
    """R-1.0.6-30:从 API prompt JSON 提最终正/负向全文(R-20 同源)。

    复用 :func:`prompt_texts` 的连线判角色(不猜节点名);同角色多段按节点序
    换行拼接;**只收 role 明确的** —— 提不出角色的文本一概不进直通字段
    (宁缺毋假,workflow JSON 里仍有全量)。任何异常都当「提不出」。
    """
    try:
        d = json.loads(prompt_json)
        if not isinstance(d, dict):
            return "", ""
        texts = prompt_texts(d)
        pos = "\n".join(
            t["text"] for t in texts
            if t["role"] == "positive" and t["text"].strip())
        neg = "\n".join(
            t["text"] for t in texts
            if t["role"] == "negative" and t["text"].strip())
        return pos, neg
    except Exception:  # noqa: BLE001 - 直通字段是锦上添花,绝不连累转换
        return "", ""


def encode_output(
    data: bytes,
    filename: str,
    output_format: str,
    media_type: str = "image",
    job_prompt: dict | None = None,
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
        # ⭐ workflow 主源 = **job 的 prompt**(claim/staged、R-20 日志同源,
        # R-1.0.6-26):Image Saver 这类保存节点出的图,源字节里
        # 可能根本没有可读的 prompt(它写 EXIF 272,不写 tEXt/XMP)—— 靠源图
        # 元数据,直通字段就会静默缺席。源图 tEXt(PNG)只作 job_prompt 缺席时
        # 的兜底(防御/旧调用路径;worker 正常调用总带 job_prompt)。
        # 源是 WebP(如 Image Saver lossless 直出)同样走解码→重编:XMP 一并
        # 附上;代价是 lossless 源经 quality=90 有一次有损重编(与历来行为一致,
        # 用户要的 output_format=webp 本就是「有损换体积」)。
        prompt_str: str | None = None
        if isinstance(job_prompt, dict) and job_prompt:
            try:
                prompt_str = json.dumps(job_prompt, ensure_ascii=False)
            except (TypeError, ValueError):
                prompt_str = None
        if prompt_str is None:
            src = im.info.get("prompt")
            if isinstance(src, bytes):
                src = src.decode("utf-8", "replace")
            if isinstance(src, str):
                prompt_str = src
        save_kwargs: dict[str, Any] = {"format": "WEBP", "quality": WEBP_QUALITY}
        if prompt_str is not None:
            # R-1.0.6-30:workflow JSON 照写,另附直通的正/负向全文(提得出才写)。
            positive, negative = _role_texts(prompt_str)
            save_kwargs["xmp"] = _xmp_with_prompt(
                prompt_str, positive=positive, negative=negative)
        buf = BytesIO()
        try:
            im.save(buf, **save_kwargs)
        except TypeError:
            # Old Pillow without xmp= support: re-encode without metadata
            # rather than crash. Better a prompt-less WebP than a failed job.
            # ⚠️ 不再零症状(R-1.0.6-26 复诊):置 status 标记 → 面板出软警告,
            # 否则用户直到「从图片建预设/看提示词」失败都不知道 prompt 没进图。
            buf = BytesIO()
            im.save(buf, format="WEBP", quality=WEBP_QUALITY)
            log.info("Pillow too old for xmp=; shipping WebP without prompt for %s", filename)
            STATUS.set(webp_xmp_degraded=True)
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


_webp_caps: dict | None = None


def webp_capability() -> dict:
    """This env's WebP capability, for the panel status(R-1.0.6-26 复诊)。

    排查「出的图没带 prompt」一眼定案:Pillow 版本 + 能不能编 WebP + ``xmp=``
    是否真的写进去(有的 build 会**静默忽略**关键字,也算不行)。首次调用探测
    一次(内存 1x1 编码,毫秒级)后缓存 —— 环境不会在运行中途自愈。绝不抛。
    """
    global _webp_caps
    if _webp_caps is not None:
        return _webp_caps
    caps = {"pillow_version": "", "webp_ok": False, "webp_xmp_ok": False}
    try:
        import PIL
        from PIL import Image

        caps["pillow_version"] = getattr(PIL, "__version__", "")
        im = Image.new("RGB", (1, 1))
        buf = BytesIO()
        im.save(buf, format="WEBP")
        caps["webp_ok"] = True
        probe = BytesIO()
        try:
            im.save(probe, format="WEBP", xmp=b"<x/>")
            caps["webp_xmp_ok"] = b"<x/>" in probe.getvalue()
        except TypeError:
            pass
    except Exception:  # noqa: BLE001 - 能力探测绝不连累任何调用方
        pass
    _webp_caps = caps
    return caps


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


# ── Final prompt visibility (local log only) ────────────────────────────────
#
# A prompt travels a long way before it reaches ComfyUI: the app's preset tree →
# flattened tokens → LoRA trigger words → one string → relay → this plugin →
# ComfyUI. When the picture comes back wrong, EVERY one of those hops looks the
# same from the outside. This is the only place that can say what the far end
# actually received, so we log it verbatim right before submitting.
#
# ⛔ LOCAL ONLY. This goes to the user's own ComfyUI console and is NEVER
# reported to the relay: a user reading their own log is not the same thing as
# us holding their content.

# Per-text safety valve. NOT a display cap — a real prompt is a few hundred
# chars, so this only ever fires on something pathological, and when it does the
# line SAYS it was truncated instead of silently misrepresenting what we sent.
# Truncating by default would destroy the entire point of the log.
_MAX_PROMPT_CHARS = 4000

# Input names that carry prompt text in ComfyUI's API-format JSON. Deliberately
# an ALLOWLIST of INPUT names, not node types: the node types that hold text are
# open-ended (CLIPTextEncode is only the most common one), while the input names
# are conventional — and an allowlist can never mistake `ckpt_name`,
# `sampler_name` or `filename_prefix` for a prompt. Every hit is additionally
# required to be a `str`, which is what keeps a wired-up link input named
# `positive` (a list) or a numeric `value` out.
_TEXT_INPUTS = frozenset({
    "text", "text_g", "text_l", "text_positive", "text_negative",
    "prompt", "positive_prompt", "negative_prompt",
    "positive", "negative", "string", "value",
    "wildcard_text", "populated_text",
    "clip_l", "t5xxl",  # CLIPTextEncodeFlux / CLIPTextEncodeSD3
})

# 文本拼接节点的编号输入(text1..textN / text_1..text_N,如 CR Text
# Concatenate)。按键形认、不认类名 —— 各家 concat 变体都长这样。
_TEXT_N = re.compile(r"^text_?\d+$")

# Link inputs we follow when walking BACK from a sampler towards the node that
# holds the text. Conditioning rarely goes straight from the encoder to the
# sampler — it passes through FluxGuidance / ConditioningCombine /
# ControlNetApply / … — and these are the names those helpers give the
# conditioning they wrap. Following ONLY conditioning-ish links is what keeps
# the walk from wandering into an unrelated branch and mislabeling some other
# node's text as the prompt.
_COND_INPUTS = frozenset({"positive", "negative", "cond"})

# Bound on one backwards walk. A real chain is 1-3 hops; this only stops a
# cyclic or absurd graph from spinning.
_MAX_TRACE_NODES = 32

# Node class names listed in the "couldn't find anything" line, so a bug report
# still shows what the workflow was made of.
_MAX_SUMMARY_CLASSES = 8


def format_prompt_log(job_id: str, prompt: dict) -> str:
    """The one-line-per-text block logged just before a prompt is submitted.

    Shape (one ``log.info`` so the console keeps it together)::

        job <id> final prompt (as submitted to ComfyUI):
          [positive] node 6 CLIPTextEncode.text (Positive): masterpiece, 1girl
          [negative] node 7 CLIPTextEncode.text: bad hands

    Roles come from the workflow itself (see :func:`prompt_texts`); a text we
    cannot tie to a sampler is printed as ``[unlabeled]`` rather than guessed at,
    and a workflow with no recognisable text input says so — with a summary of
    the node types it did contain, which is what a bug report needs. Guessing
    would be worse than admitting we don't know: the whole value of this log is
    that it is literally true.

    NEVER raises. It runs in the job path purely for observability, and a
    logging bug must not be able to fail a generation.
    """
    header = f"job {job_id} final prompt (as submitted to ComfyUI):"
    try:
        texts = prompt_texts(prompt)
        if not texts:
            return f"{header} no text input identified — {_node_summary(prompt)}"
        lines = [header]
        for t in texts:
            lines.append(f"  [{t['role'] or 'unlabeled'}] {_where(t)}: {_cap(t['text'])}")
        return "\n".join(lines)
    except Exception as e:  # noqa: BLE001 - observability must never fail a job
        return f"{header} could not be rendered ({e})"


def prompt_texts(prompt: dict) -> list[dict]:
    """Every prompt text in an API-format workflow, with where it is wired.

    Returns ``[{"node", "class_type", "title", "field", "text", "role"}]`` in
    node order, where ``role`` is ``"positive"``/``"negative"`` when the text
    can be traced to a sampler input and ``""`` when it cannot.

    HOW THE ROLE IS DECIDED — from the graph, never from the wording. Any node
    with LINK inputs named ``positive``/``negative`` is treated as a sampler
    (KSampler, KSamplerAdvanced, SamplerCustom and every fork of them share that
    convention, and matching on class names would miss all the forks). From each
    such link we walk backwards along conditioning links until we reach a node
    that holds text. When a node on the way has an input with the same name as
    the role being traced (ControlNetApplyAdvanced has both), we follow ONLY
    that one — otherwise a negative trace could wander into the positive branch
    and mislabel it.

    Empty texts are kept only when they got a role: "the negative prompt is
    empty" is a real answer and dropping it would read as "no negative found",
    but an unrelated empty string somewhere in the graph is just noise.

    Defensive throughout — a workflow is user data, not a contract, so a node
    that isn't a dict, inputs that aren't a dict, a link that isn't a pair, etc.
    all just mean "nothing to report here".
    """
    nodes = _nodes(prompt)
    found: list[dict] = []
    by_node: dict[str, list[int]] = {}
    for node_id in sorted(nodes, key=_node_order):
        node = nodes[node_id]
        class_type = node.get("class_type")
        meta = node.get("_meta")
        title = meta.get("title") if isinstance(meta, dict) else None
        for field, value in _inputs(node).items():
            if not isinstance(field, str) or field.lower() not in _TEXT_INPUTS:
                continue
            if not isinstance(value, str):
                continue
            by_node.setdefault(node_id, []).append(len(found))
            found.append({
                "node": node_id,
                "class_type": class_type if isinstance(class_type, str) else "",
                "title": title if isinstance(title, str) else "",
                "field": field,
                "text": value,
                "role": "",
            })
    if not found:
        return []
    for node_id in sorted(nodes, key=_node_order):
        for role in ("positive", "negative"):
            link = _inputs(nodes[node_id]).get(role)
            if _link_target(link):
                _trace(nodes, _link_target(link), role, found, by_node)
    return [t for t in found if t["role"] or t["text"].strip()]


def _trace(nodes: dict, start: str, role: str, found: list[dict],
           by_node: dict[str, list[int]]) -> None:
    """Walk back from a sampler input and label the first text nodes reached."""
    seen: set[str] = set()
    queue = [start]
    while queue and len(seen) < _MAX_TRACE_NODES:
        node_id = queue.pop(0)
        if node_id in seen or node_id not in nodes:
            continue
        seen.add(node_id)
        hits = by_node.get(node_id)
        if hits:
            # This node holds the text for that branch; stop here rather than
            # keep walking (whatever feeds IT is not the prompt itself).
            for i in hits:
                if not found[i]["role"]:
                    found[i]["role"] = role
            continue
        inputs = _inputs(nodes[node_id])
        same_name = _link_target(inputs.get(role))
        if same_name:
            # e.g. ControlNetApplyAdvanced: follow only the branch of our own
            # role, never its sibling.
            queue.append(same_name)
            continue
        for field, value in inputs.items():
            if not isinstance(field, str):
                continue
            lower = field.lower()
            # 顺两类线走:conditioning 类(编码之后的链),以及**文本类**——
            # 编码节点的 text 本身可以是连线(CR Text Concatenate 的
            # text1/text2 → CR Prompt Text 等拼接链,R-1.0.6-26:提示词
            # 常藏在这类拼接链后,不顺 text 线角色就永远定不到)。
            is_cond = "conditioning" in lower or lower in _COND_INPUTS
            is_text = lower in _TEXT_INPUTS or _TEXT_N.match(lower)
            if not (is_cond or is_text):
                continue
            target = _link_target(value)
            if target:
                queue.append(target)


def _nodes(prompt: dict) -> dict[str, dict]:
    """The workflow's nodes keyed by string id; ``{}`` for anything unusable."""
    if not isinstance(prompt, dict):
        return {}
    return {str(k): v for k, v in prompt.items() if isinstance(v, dict)}


def _node_order(node_id: str) -> tuple[int, int, str]:
    """Sort node ids numerically when they are numbers (ComfyUI's are), else by
    name — so the log reads in workflow order (3, 4, 10) rather than (10, 3, 4),
    and is deterministic whatever order the JSON happened to arrive in."""
    return (0, int(node_id), "") if node_id.isdigit() else (1, 0, node_id)


def _inputs(node: dict) -> dict:
    inputs = node.get("inputs")
    return inputs if isinstance(inputs, dict) else {}


def _link_target(value: Any) -> str:
    """The source node id of a ComfyUI link ``["6", 0]``; "" if not a link."""
    if isinstance(value, list) and len(value) == 2 and isinstance(value[0], (str, int)):
        return str(value[0])
    return ""


def _where(item: dict) -> str:
    """``node 6 CLIPTextEncode.text (Positive)`` — which box this text is in.

    A workflow can have several text inputs; without the node id and input name
    there is no way to tell which one on screen a line corresponds to.
    """
    out = f"node {item['node']}"
    if item["class_type"]:
        out += f" {item['class_type']}"
    out += f".{item['field']}"
    if item["title"]:
        out += f" ({item['title']})"
    return out


def _cap(text: str) -> str:
    """Full text, unless it is absurdly long — and then say so explicitly."""
    if len(text) <= _MAX_PROMPT_CHARS:
        return text
    return f"{text[:_MAX_PROMPT_CHARS]}… [TRUNCATED, {len(text)} chars total]"


def _node_summary(prompt: dict) -> str:
    """What the workflow was made of, for the "identified nothing" line."""
    nodes = _nodes(prompt)
    if not nodes:
        return "empty workflow"
    classes = sorted({
        n["class_type"] for n in nodes.values()
        if isinstance(n.get("class_type"), str) and n.get("class_type")
    })
    listed = ", ".join(classes[:_MAX_SUMMARY_CLASSES])
    if len(classes) > _MAX_SUMMARY_CLASSES:
        listed += f", +{len(classes) - _MAX_SUMMARY_CLASSES} more"
    return f"{len(nodes)} node(s){': ' + listed if listed else ''}"


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
