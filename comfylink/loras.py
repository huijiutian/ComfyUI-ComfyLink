"""Local model inventory: enumerate → stat → read the safetensors header.

LoRAs and checkpoints are collected by ONE code path (``_build_entries``) and
produce IDENTICAL entries — ``{name, size, mtime, meta}``. The only thing that
differs is which metadata keys are allowlisted out of the header, because the
two kinds answer different questions ("what fires this LoRA" vs "which model
family is this checkpoint").

⛔ NO MODEL FILE'S CONTENT IS EVER READ. Not a digest, not a partial digest,
nothing. Per file this module does exactly one ``os.stat`` plus — for
``.safetensors`` only — two bounded reads: the 8-byte header length and that
many bytes of JSON. A few hundred models scan in milliseconds, which is why the
whole thing can be a foreground, user-initiated action with no progress display.

⚠️ WHY THERE IS NO ``sha256`` (schema 3, 2026-08-01) — DO NOT ADD IT BACK.
LoRA entries used to carry a whole-file SHA256. It had exactly ONE consumer:
CivitAI indexes models by the digest of the whole file, and the relay used that
to resolve trigger words. The product dropped the CivitAI lookup entirely —
trigger words are now something the user picks or types, with the model's own
``modelspec.trigger_phrase`` as the sole suggestion — so the digest lost its
only reason to exist. Its price, by contrast, was never small: 100–300 MB per
LoRA and hundreds of them for a heavy user means tens of GB of reads and minutes
of wall clock on a cold scan, plus a persisted digest cache (keyed on
path+size+mtime) whose entire job was to avoid paying that twice, plus a
progress callback whose entire job was to prove the plugin had not hung. All
three are gone with it. Checkpoints were NEVER hashed — 2–7 GB each makes it
absurd — so deleting the LoRA digest is precisely what collapses the two kinds
onto one code path. That collapse IS this change. Anything that wants a digest
back has to justify the tens of GB again, from scratch.

⚠️ NAME KEY. ``name`` MUST be exactly what ComfyUI's
``folder_paths.get_filename_list("loras")`` returns — the same strings that end
up as ``LoraLoader.lora_name``'s enum in ``/object_info`` and hence inside the
workflow (relative to the loras dir, possibly with a subdirectory prefix such as
``style/foo.safetensors``). Scanning the directory ourselves and reporting
absolute paths or bare basenames would produce a manifest the app can never
match — and the symptom is SILENT: no error anywhere, just "no trigger words for
any LoRA, ever". ``tests/test_loras.py`` guards this. The SAME discipline
applies to checkpoints via ``get_filename_list("checkpoints")``
(``CheckpointLoaderSimple.ckpt_name``), and is guarded the same way. Now that
the digest is gone, ``name`` is the ONLY join key there is.

Everything here degrades to "no inventory" rather than raising: ``folder_paths``
is a ComfyUI module that simply is not importable outside ComfyUI, the loras
directory may not exist, and a file may be unreadable or truncated mid-write.
None of that is worth a traceback in a user's ComfyUI log, and none of it may
ever reach the worker's control flow.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import struct
import time
from typing import Callable, Optional

from .log import log

# Bumped whenever the manifest's shape changes in a way consumers must notice.
# The relay and the app both read this first.
#   1 — loras only
#   2 — adds the top-level `checkpoints` array (header-only, never hashed)
#   3 — drops `sha256` from the lora entries. LoRAs and checkpoints are now
#       collected identically: name + size + mtime + header meta. See the
#       module docstring for why the digest went away and must not come back.
#
# Note on the R2 key: the manifest lives at
# ``users/<uid>/backends/<bid>/loras/index.json`` even though it now also carries
# checkpoints. That name is HISTORICAL — schema 1 predated checkpoints — and
# renaming it would mean a relay change (new endpoint + key) for zero user-facing
# gain, so it stays. Compatibility falls out for free in both directions: an old
# app reads only `loras` and ignores what it does not know, and a new app reading
# a schema-1 manifest finds no `checkpoints`, cannot tell the family, and stays
# silent — which is exactly the required behaviour ("can't tell ⇒ say nothing").
# Schema 3 is a REMOVAL, so it needs the same care in the other direction: a
# consumer that still wants `sha256` finds the key ABSENT (never an empty
# string), which is the only shape that lets it tell "old plugin" from "this
# plugin does not do digests any more".
MANIFEST_SCHEMA = 3

# Hard ceiling on entries in one manifest. A pathological loras directory must
# not turn into an unbounded JSON upload; past this we report the first N (sorted
# by name, so the cut is at least deterministic) and set ``truncated``.
MAX_ENTRIES = 5000

# safetensors declares its header length in the first 8 bytes. A sane header is
# a few hundred KB at most (ss_tag_frequency is the fat one); anything bigger is
# a corrupt/hostile file and we refuse to allocate for it. This is also the ONLY
# ceiling on how much of a model file this module will ever read.
MAX_HEADER_BYTES = 16 * 1024 * 1024

# Per-value and whole-``meta`` budgets. Training metadata routinely runs to
# hundreds of KB; the manifest must stay small enough to fetch on mobile.
META_VALUE_MAX = 256
META_TOTAL_MAX = 4096
MAX_TOP_TAGS = 20
MAX_TAG_LEN = 64

# Only these extensions have a header worth parsing. A .pt/.ckpt LoRA still gets
# reported (name/size/mtime are all real), it just carries ``meta = {}``.
_SAFETENSORS_EXT = (".safetensors", ".sft")

# Keys lifted out of the safetensors ``__metadata__`` blob, in priority order —
# the trim below drops from the END, so the first entry is the best protected.
#
# ``modelspec.trigger_phrase`` leads because it is now the ONLY automatic
# trigger-word suggestion in the entire product: the CivitAI lookup is gone, so
# if this field is not in the manifest the user has nothing to accept and must
# type the word themselves. Everything after it is identification (what model is
# this) and then training detail, which is trivia by comparison. Kohya/sd-scripts
# writes the ``ss_*`` family; ``modelspec.*`` is the SAI standard that
# ComfyUI-side tooling writes.
_META_KEYS = (
    "modelspec.trigger_phrase",
    "modelspec.title",
    "modelspec.architecture",
    "modelspec.author",
    "modelspec.date",
    "modelspec.resolution",
    "ss_output_name",
    "ss_base_model_version",
    "ss_sd_model_name",
    "ss_network_module",
    "ss_network_dim",
    "ss_network_alpha",
    "ss_resolution",
    "ss_clip_skip",
    "ss_num_train_images",
    "ss_epoch",
    "ss_training_comment",
    "modelspec.description",
)

# Checkpoint metadata is a DIFFERENT question, so it gets its own allowlist
# rather than reusing the LoRA one. What the app needs from a checkpoint is
# "which model family / prompt dialect is this", so the list is ordered by how
# directly each field answers that:
#
#   * modelspec.architecture   — THE field. The SAI model-spec identifier
#                                ("stable-diffusion-xl-v1-base", "flux-1-dev",
#                                …). Primary evidence, and free.
#   * modelspec.title          — human name; feeds the same filename-ish
#                                heuristics the app already has, but written by
#                                the author rather than by whoever renamed the
#                                file.
#   * modelspec.prediction_type— epsilon vs v_prediction. Two checkpoints can
#                                share an architecture and still be different
#                                beasts (v-pred NoobAI vs eps SDXL).
#   * modelspec.implementation — which codebase/encoder stack wrote it; corrob-
#                                orates architecture when the latter is vague.
#   * modelspec.resolution     — 512 vs 1024 vs 1536 is a strong family tell and
#                                survives merges better than free text.
#   * modelspec.merged_from    — merges are exactly the case where the family is
#                                ambiguous; the lineage is the only clue left.
#   * modelspec.encoder_layer  — clip-skip convention, an anime/SDXL-lineage hint
#   * ss_base_model_version    — kohya writes this on FINE-TUNED checkpoints too,
#                                so it is a useful fallback when modelspec.* is
#                                absent entirely (very common for older merges).
#   * sai_model_spec/author/date/description — provenance; cheap, occasionally
#                                the only thing present.
#
# DELIBERATELY ABSENT: everything training-shaped. ss_network_*/ss_epoch/
# ss_tag_frequency (=> no `top_tags`) and modelspec.trigger_phrase describe how a
# LoRA was trained and what fires it; a checkpoint has neither notion, so
# carrying them would burn the meta budget on fields the app can never use.
# Also absent: modelspec.hash_sha256. Some writers embed a self-reported digest,
# but it is unverified and (per spec) not necessarily the digest of the file.
# Nothing in this product consumes a digest any more (schema 3), and publishing
# an unverified one would be an open invitation to build something on it.
_CKPT_META_KEYS = (
    "modelspec.architecture",
    "modelspec.title",
    "modelspec.prediction_type",
    "modelspec.implementation",
    "modelspec.resolution",
    "modelspec.merged_from",
    "modelspec.encoder_layer",
    "ss_base_model_version",
    "modelspec.sai_model_spec",
    "modelspec.author",
    "modelspec.date",
    "modelspec.description",
)


# ── ComfyUI folder_paths (the ONLY source of truth for model names) ───────────

def _folder_paths():
    """ComfyUI's ``folder_paths`` module, or None outside ComfyUI.

    Defensive import in the spirit of ``config.detect_comfy_url``: the plugin is
    imported by ComfyUI at startup, and an ImportError escaping here would take
    the whole ComfyUI process's node loading with it.
    """
    try:
        import folder_paths  # type: ignore

        return folder_paths
    except Exception:  # noqa: BLE001 - not running inside ComfyUI
        return None


def folder_paths_available() -> bool:
    """True when ComfyUI's folder_paths is importable.

    Callers use this to skip the whole inventory feature permanently: a module
    that cannot be imported now will not become importable later in the same
    process, so a caller should give up rather than retry forever.
    """
    return _folder_paths() is not None


def list_model_names(folder: str) -> list[str]:
    """Model names exactly as ComfyUI enumerates them — the manifest's join key.

    This is ``folder_paths.get_filename_list(folder)``: the SAME call whose
    result becomes the node widget's enum in ``/object_info`` (``lora_name`` for
    "loras", ``ckpt_name`` for "checkpoints") and hence the string stored in a
    workflow. Deduped and sorted for a deterministic manifest; the strings
    themselves are passed through untouched.
    """
    fp = _folder_paths()
    if fp is None:
        return []
    try:
        names = fp.get_filename_list(folder)
    except Exception as e:  # noqa: BLE001 - folder not configured, no dir, etc.
        log.debug("model inventory: cannot list %s: %s", folder, e)
        return []
    return sorted({str(n) for n in (names or []) if n})


def model_path(folder: str, name: str) -> Optional[str]:
    """Absolute path for an enumerated name, or None if it cannot be resolved.

    Resolution goes back through ``folder_paths`` too, so a multi-root install
    (several ``loras``/``checkpoints`` directories) resolves the same way ComfyUI
    itself would.
    """
    fp = _folder_paths()
    if fp is None:
        return None
    try:
        p = fp.get_full_path(folder, name)
    except Exception:  # noqa: BLE001
        return None
    return str(p) if p else None


def list_lora_names() -> list[str]:
    """LoRA names, in ``/object_info``'s ``LoraLoader.lora_name`` spelling."""
    return list_model_names("loras")


def lora_path(name: str) -> Optional[str]:
    return model_path("loras", name)


def list_checkpoint_names() -> list[str]:
    """Checkpoint names, in ``CheckpointLoaderSimple.ckpt_name`` spelling."""
    return list_model_names("checkpoints")


def checkpoint_path(name: str) -> Optional[str]:
    return model_path("checkpoints", name)


# ── safetensors header ────────────────────────────────────────────────────────

def read_header_metadata(f) -> dict:
    """``__metadata__`` from an open safetensors file — HEADER BYTES ONLY.

    Layout: 8 bytes little-endian uint64 header length, then that many bytes of
    JSON. We read those two spans and stop; the tensor payload (the other 99.9%
    of a 300 MB file) is never touched. Takes a file object rather than a path
    so a test can assert exactly how much was read.

    Any malformed input — short file, absurd declared length, non-UTF8, non-JSON,
    JSON that isn't an object, missing/!dict ``__metadata__`` — yields ``{}``.
    """
    raw = f.read(8)
    if len(raw) < 8:
        return {}
    (n,) = struct.unpack("<Q", raw)
    if n <= 0 or n > MAX_HEADER_BYTES:
        return {}
    blob = f.read(n)
    if len(blob) < n:
        return {}
    try:
        header = json.loads(blob.decode("utf-8"))
    except Exception:  # noqa: BLE001 - corrupt/partial file
        return {}
    if not isinstance(header, dict):
        return {}
    md = header.get("__metadata__")
    return md if isinstance(md, dict) else {}


def read_safetensors_metadata(path: str) -> dict:
    """``read_header_metadata`` for a path; ``{}`` for anything unreadable.

    ⛔ This is the ONLY place this module opens a model file, and it hands the
    handle straight to the bounded reader above. There is no other read path,
    and there must never be one.
    """
    try:
        with open(path, "rb") as f:
            return read_header_metadata(f)
    except Exception as e:  # noqa: BLE001 - permissions, mid-download file, ...
        log.debug("model inventory: cannot read header of %s: %s", path, e)
        return {}


def _top_tags(raw, limit: int = MAX_TOP_TAGS) -> list[str]:
    """Most frequent training tags, summed across dataset folders.

    ``ss_tag_frequency`` is a JSON *string* holding ``{dataset: {tag: count}}``
    and is by far the biggest thing in a Kohya header (hundreds of KB). It is
    also, for character/style LoRAs, often where the trigger word actually lives
    — it is the tag that appears in every caption. So we keep a bounded top-N
    instead of either dumping it or throwing it away. It is a WEAK signal (for a
    style LoRA the top tag is usually something useless like "1girl"), which is
    why it is never presented as an answer — but it comes free with a header we
    are already reading, so it costs nothing to hand the user the clue.
    """
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception:  # noqa: BLE001
            return []
    if not isinstance(raw, dict):
        return []
    counts: dict[str, int] = {}
    for per_dir in raw.values():
        if not isinstance(per_dir, dict):
            continue
        for tag, c in per_dir.items():
            try:
                n = int(c)
            except Exception:  # noqa: BLE001
                continue
            t = str(tag).strip()[:MAX_TAG_LEN]
            if t:
                counts[t] = counts.get(t, 0) + n
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return [t for t, _ in ranked[:limit]]


def _trim_meta(meta: dict, keys: tuple) -> dict:
    """Drop the lowest-priority keys until ``meta`` fits META_TOTAL_MAX.

    ``keys`` is ordered most- to least-identifying, so dropping from the end
    sheds trivia (training detail, free-text description) before it sheds the
    model's identity — or, at the very front of the LoRA list, its trigger word.
    """
    droppable = [k for k in reversed(keys) if k in meta]
    while droppable and len(json.dumps(meta, separators=(",", ":"))) > META_TOTAL_MAX:
        meta.pop(droppable.pop(0), None)
    return meta


def _extract(md: dict, keys: tuple) -> dict:
    """Allowlisted, per-value-truncated, size-capped subset of ``__metadata__``.

    Deliberately an allowlist, not a filter: an unknown trainer that writes a
    500 KB field must not be able to inflate the manifest just because we did not
    think to exclude its key.
    """
    if not isinstance(md, dict):
        return {}
    out: dict = {}
    for k in keys:
        v = md.get(k)
        if v is None:
            continue
        s = (v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)).strip()
        if not s or s in ("None", "null", "{}", "[]"):
            continue
        out[k] = s[:META_VALUE_MAX]
    return _trim_meta(out, keys)


def extract_meta(md: dict) -> dict:
    """LoRA ``meta``: identification fields plus a bounded top-tag digest."""
    out = _extract(md, _META_KEYS)
    tags = _top_tags(md.get("ss_tag_frequency")) if isinstance(md, dict) else []
    if tags:
        out["top_tags"] = tags
    return out


def extract_checkpoint_meta(md: dict) -> dict:
    """Checkpoint ``meta``: model-family evidence only (see _CKPT_META_KEYS).

    No ``top_tags`` — training tag frequencies say nothing about a checkpoint —
    and no digest of any kind.
    """
    return _extract(md, _CKPT_META_KEYS)


# ── manifest ─────────────────────────────────────────────────────────────────

def _build_entries(folder: str,
                   extract: Callable[[dict], dict]) -> tuple[list[dict], bool]:
    """One model folder → ``(entries, truncated)``. THE shared collection path.

    LoRAs and checkpoints run through this same function and come out the same
    shape; only ``extract`` differs, because the two kinds want different keys
    out of the header. Keeping it as one function is not tidiness — it is the
    mechanical guarantee that the two can never drift apart again, which is what
    schema 3 is for.

    Per file: ``os.stat`` (size + mtime), and for a safetensors file the two
    bounded header reads. Nothing else. No file's content is read.
    """
    entries: list[dict] = []
    names = list_model_names(folder)
    truncated = len(names) > MAX_ENTRIES
    for name in names[:MAX_ENTRIES]:
        path = model_path(folder, name)
        if not path:
            continue
        try:
            st = os.stat(path)
        except OSError as e:
            # Enumerated a moment ago, gone now (deleted, or a broken symlink).
            log.debug("model inventory: cannot stat %s: %s", path, e)
            continue
        meta: dict = {}
        if path.lower().endswith(_SAFETENSORS_EXT):
            meta = extract(read_safetensors_metadata(path))
        entries.append({
            "name": name,
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "meta": meta,
        })
    return entries, truncated


def build_loras() -> tuple[list[dict], bool]:
    """LoRA half of the inventory."""
    return _build_entries("loras", extract_meta)


def build_checkpoints() -> tuple[list[dict], bool]:
    """Checkpoint half of the inventory. Identical work, different allowlist."""
    return _build_entries("checkpoints", extract_checkpoint_meta)


def build_manifest() -> dict:
    """Build the whole inventory. Blocking — call via ``asyncio.to_thread``.

    Blocking only in the sense that it touches the filesystem: a few hundred
    stats plus a few hundred KB of header reads. On any normal disk that is
    milliseconds; on a network-mounted models directory it is not, which is the
    reason it still goes to a thread rather than running inline.
    """
    entries, truncated = build_loras()
    # The checkpoint half is guarded so that a surprise there can never cost us
    # the LoRA inventory. The guard is deliberately ONE-SIDED: if the LoRA half
    # blows up we want the exception, because the caller's response is "don't
    # advance the watermark, retry next beat", whereas swallowing it would
    # upload an empty `loras` array over a perfectly good one.
    try:
        checkpoints, ckpt_truncated = build_checkpoints()
    except Exception as e:  # noqa: BLE001 - LoRAs still ship
        log.warning("model inventory: checkpoint scan failed: %s", e)
        checkpoints, ckpt_truncated = [], False
    return {
        "schema": MANIFEST_SCHEMA,
        "generated_at": int(time.time()),
        # `count`/`truncated` describe the LORAS array and always have. Schema 2
        # kept them LoRA-scoped rather than making them totals, so an old app
        # reading a new manifest sees exactly the numbers it saw before.
        "count": len(entries),
        "truncated": truncated,
        "loras": entries,
        "checkpoints": checkpoints,
        "checkpoints_count": len(checkpoints),
        "checkpoints_truncated": ckpt_truncated,
    }


def manifest_hash(manifest: dict) -> str:
    """Content hash used to skip an unchanged re-upload.

    Covers the schema + entries ONLY — ``generated_at`` moves every run and
    including it would defeat the whole point (same mirror of the disk, new
    hash, pointless upload). Same shape/role as ``worker.object_info_hash``.
    ``schema`` is in there on purpose: the schema-2 → 3 bump changes the hash by
    itself, so every paired backend re-uploads once after the upgrade instead of
    leaving a stale manifest with ``sha256`` fields sitting in R2.

    Checkpoints are part of the hash: adding or replacing a checkpoint without
    touching a single LoRA must still trigger an upload, or the app would keep
    warning against a model list that no longer exists.
    """
    payload = {
        "schema": manifest.get("schema"),
        "loras": manifest.get("loras") or [],
        "checkpoints": manifest.get("checkpoints") or [],
    }
    return hashlib.md5(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


# ── on-demand collection ─────────────────────────────────────────────────────
#
# ⛔ NOTHING IN THIS MODULE RUNS ON ITS OWN. Reading a user's model folders is
# not something a plugin should do behind their back on a timer, and this plugin
# has said so since before the feature existed ("No background auto-sync",
# web/comfylink.js). An earlier revision ran a 10-minute background refresh; it
# was removed deliberately. The scan being cheap now does NOT reinstate the
# case for a timer — the objection was consent, not cost.
#
# The single trigger is the user pressing refresh IN THE APP: the relay records
# the request, hands the plugin the request's timestamp on the next heartbeat,
# and worker._maybe_scan_models turns that into exactly one scan. Consent is
# explicit and remote — the user does not have to walk over to the PC to give it.


# Only one build at a time in this process. A machine paired to several accounts
# gets one scan request per account; serialising means they queue behind each
# other instead of contending for the same disk. There is no result cache, and
# therefore no staleness question to answer.
_build_lock = asyncio.Lock()


async def collect() -> dict:
    """Build the inventory off the event loop. Only ever called on demand.

    ``asyncio.to_thread`` is not about CPU — there is none — it is about the
    filesystem. The calling coroutine shares its event loop with the relay
    heartbeat, the cancel long-poll and the claim loop, and a models directory on
    a slow or network-mounted disk would stall all three if it were stat'ed
    inline.
    """
    async with _build_lock:
        return await asyncio.to_thread(build_manifest)


async def upload_for(relay, pairing, manifest: dict) -> str:
    """Upload the manifest for ONE pairing. Returns what happened.

    ``"uploaded"`` — shipped and the content hash remembered.
    ``"unchanged"`` — identical to what this pairing already has; nothing sent.
    ``"unsupported"`` — the relay has no such endpoint (404). An older relay is
    an expected state during a staged rollout, not an error: the user gets told
    plainly and everything else keeps working.

    Anything else propagates so the caller can report a real per-account failure
    (the same shape /comfylink/sync uses). The content hash is stored ONLY after
    a successful upload, so a failure re-uploads on the next scan.
    """
    from .config import STATE
    from .relay import RelayError

    new_hash = manifest_hash(manifest)
    if pairing.loras_hash and pairing.loras_hash == new_hash:
        return "unchanged"
    try:
        await relay.upload_loras(pairing.backend_id, manifest)
    except RelayError as e:
        if e.status == 404:
            log.warning(
                "relay has no model inventory endpoint (404) — nothing uploaded; "
                "this works once the relay is updated"
            )
            return "unsupported"
        raise
    pairing.loras_hash = new_hash
    STATE.save()
    log.info("uploaded model inventory (%d LoRA(s), %d checkpoint(s), hash %s)",
             len(manifest.get("loras") or []),
             len(manifest.get("checkpoints") or []), new_hash[:12])
    return "uploaded"
