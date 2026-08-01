"""Local model inventory: enumerate → read safetensors header → manifest.

Two model kinds, deliberately treated DIFFERENTLY:

  * **LoRAs** — enumerate, read the header, and compute the whole-file SHA256 so
    the relay can resolve trigger words against CivitAI.
  * **checkpoints** — enumerate and read the header ONLY. ⛔ NEVER hashed. A LoRA
    is 100–300 MB and a cold full-hash pass is a one-off cost worth paying;
    a checkpoint is 2–7 GB, so a user with a few dozen of them would be hundreds
    of GB of disk reads. That premise simply does not survive the two orders of
    magnitude, so the checkpoint side is header-only (a few hundred KB per file)
    and answers a much smaller question: which model family is this, from
    ``modelspec.architecture`` and friends. The app uses that to warn when a
    preset's prompt dialect does not match the selected checkpoint, and stays
    SILENT whenever it cannot tell.

The app shows a LoRA's trigger words. It gets them by taking the LoRA file names
out of the *workflow* and looking them up in the manifest this module produces;
the relay then resolves ``sha256`` against CivitAI. Two things therefore matter
more than anything else here:

⚠️ NAME KEY. ``name`` MUST be exactly what ComfyUI's
``folder_paths.get_filename_list("loras")`` returns — the same strings that end
up as ``LoraLoader.lora_name``'s enum in ``/object_info`` and hence inside the
workflow (relative to the loras dir, possibly with a subdirectory prefix such as
``style/foo.safetensors``). Scanning the directory ourselves and reporting
absolute paths or bare basenames would produce a manifest the app can never
match — and the symptom is SILENT: no error anywhere, just "no trigger words for
any LoRA, ever". ``tests/test_loras.py`` guards this. The SAME discipline
applies to checkpoints via ``get_filename_list("checkpoints")``
(``CheckpointLoaderSimple.ckpt_name``), and is guarded the same way.

⚠️ COST. CivitAI indexes by the SHA256 of the WHOLE file, so we cannot hash a
prefix. LoRAs are 100–300 MB each and a heavy user has hundreds, so a cold run
reads tens of GB. Every digest is therefore cached by ``path|size|mtime_ns`` in
the plugin state file and only recomputed when the file actually changes; and
the whole build runs off the event loop (``asyncio.to_thread``) so pairing,
registration, heartbeats and job claims are never delayed by it.

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
#
# Note on the R2 key: the manifest lives at
# ``users/<uid>/backends/<bid>/loras/index.json`` even though it now also carries
# checkpoints. That name is HISTORICAL — schema 1 predated checkpoints — and
# renaming it would mean a relay change (new endpoint + key) for zero user-facing
# gain, so it stays. Compatibility falls out for free in both directions: an old
# app reads only `loras` and ignores what it does not know, and a new app reading
# a schema-1 manifest finds no `checkpoints`, cannot tell the family, and stays
# silent — which is exactly the required behaviour ("can't tell ⇒ say nothing").
MANIFEST_SCHEMA = 2

# Hard ceiling on entries in one manifest. A pathological loras directory must
# not turn into an unbounded JSON upload; past this we report the first N (sorted
# by name, so the cut is at least deterministic) and set ``truncated``.
MAX_ENTRIES = 5000

# Read size for the SHA256 pass. Large enough that a 300 MB file is ~75 reads,
# small enough to stay off the large-object heap.
_HASH_CHUNK = 4 * 1024 * 1024

# safetensors declares its header length in the first 8 bytes. A sane header is
# a few hundred KB at most (ss_tag_frequency is the fat one); anything bigger is
# a corrupt/hostile file and we refuse to allocate for it.
MAX_HEADER_BYTES = 16 * 1024 * 1024

# Per-value and whole-``meta`` budgets. Training metadata routinely runs to
# hundreds of KB; the manifest must stay small enough to fetch on mobile.
META_VALUE_MAX = 256
META_TOTAL_MAX = 4096
MAX_TOP_TAGS = 20
MAX_TAG_LEN = 64

# Cached digests are keyed on identity+mtime, so re-hashing only happens when the
# file really changed. Kept alongside this module's other tunables so the
# "what makes a cache entry stale" answer lives in one place.
_SAFETENSORS_EXT = (".safetensors", ".sft")

# Keys lifted out of the safetensors ``__metadata__`` blob, in priority order —
# identification (what model is this) first, training detail last, because the
# total-size trim below drops from the end. Kohya/sd-scripts writes the ``ss_*``
# family; ``modelspec.*`` is the SAI standard that ComfyUI-side tooling writes,
# and ``modelspec.trigger_phrase`` is the one field that may hold the answer
# without any CivitAI round-trip at all.
_META_KEYS = (
    "modelspec.title",
    "modelspec.trigger_phrase",
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
# but it is unverified and (per spec) not necessarily the digest of the file, so
# surfacing it next to the LoRA entries' real `sha256` would invite a consumer to
# treat the two as interchangeable. Checkpoints have no `sha256` field, full stop.
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


# ── ComfyUI folder_paths (the ONLY source of truth for LoRA names) ────────────

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
    process, so a background refresh loop should stop rather than spin.
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
    """``read_header_metadata`` for a path; ``{}`` for anything unreadable."""
    try:
        with open(path, "rb") as f:
            return read_header_metadata(f)
    except Exception as e:  # noqa: BLE001 - permissions, mid-download file, ...
        log.debug("lora inventory: cannot read header of %s: %s", path, e)
        return {}


def _top_tags(raw, limit: int = MAX_TOP_TAGS) -> list[str]:
    """Most frequent training tags, summed across dataset folders.

    ``ss_tag_frequency`` is a JSON *string* holding ``{dataset: {tag: count}}``
    and is by far the biggest thing in a Kohya header (hundreds of KB). It is
    also, for character/style LoRAs, often where the trigger word actually lives
    — it is the tag that appears in every caption. So we keep a bounded top-N
    instead of either dumping it or throwing it away; it is the only trigger-word
    signal available when CivitAI has never heard of the file (self-trained and
    private LoRAs, which are common).
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
    model's identity.
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


# ── hashing + manifest ────────────────────────────────────────────────────────

def file_sha256(path: str) -> str:
    """SHA256 of the WHOLE file, streamed.

    Whole-file because that is the digest CivitAI indexes by — a prefix hash
    would be cheap and useless. Streamed in _HASH_CHUNK blocks because these
    files are hundreds of MB and must never be materialised in memory.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_HASH_CHUNK)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def cache_key(path: str, size: int, mtime_ns: int) -> str:
    """Identity of a *specific version* of a file: path + size + mtime.

    A hit means "the bytes are the same as when we hashed them", which is the
    whole reason a restart does not re-read tens of GB. Nanosecond mtime is used
    (rather than whole seconds) so an edit within the same second still misses.
    """
    return f"{path}|{size}|{mtime_ns}"


def _notify(on_progress, phase: str, done: int, total: int, name: str) -> None:
    """Fire a progress callback without ever letting it break the scan.

    The callback is invoked from the worker THREAD (build_manifest is blocking)
    while the panel reads the same object from the event loop. Both sides only
    touch plain attributes, so the GIL is sufficient — but a raising callback
    must not cost the user a multi-minute scan, hence the guard.
    """
    if on_progress is None:
        return
    try:
        on_progress(phase, done, total, name)
    except Exception:  # noqa: BLE001 - progress display is never worth a failure
        pass


def build_checkpoints(on_progress=None) -> tuple[list[dict], bool]:
    """Header-only checkpoint inventory. Returns ``(entries, truncated)``.

    ⛔ NO HASHING HAPPENS HERE, and no code path may add it. Checkpoints are
    2–7 GB each; hashing a few dozen of them means hundreds of GB of reads, which
    is categorically different from the LoRA pass and is not a cost the user
    agreed to by installing a plugin. So each entry is ``name``/``size``/
    ``mtime`` plus whatever the safetensors header already told us — a read of a
    few hundred KB per file, essentially free.

    The absence of a digest is why the app must fall back to "can't tell ⇒ say
    nothing": we deliberately trade accuracy for cost here, and the silence
    absorbs the difference.
    """
    entries: list[dict] = []
    names = list_checkpoint_names()
    truncated = len(names) > MAX_ENTRIES
    todo = names[:MAX_ENTRIES]
    for i, name in enumerate(todo):
        _notify(on_progress, "checkpoints", i, len(todo), name)
        path = model_path("checkpoints", name)
        if not path:
            continue
        try:
            st = os.stat(path)
        except OSError as e:
            log.debug("model inventory: cannot stat %s: %s", path, e)
            continue
        meta: dict = {}
        if path.lower().endswith(_SAFETENSORS_EXT):
            meta = extract_checkpoint_meta(read_safetensors_metadata(path))
        entries.append({
            "name": name,
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "meta": meta,
        })
    _notify(on_progress, "checkpoints", len(todo), len(todo), "")
    return entries, truncated


def build_manifest(
    cache: Optional[dict] = None,
    hasher: Callable[[str], str] = file_sha256,
    on_progress=None,
) -> tuple[dict, dict]:
    """Build the inventory. BLOCKING — call via ``asyncio.to_thread``.

    Returns ``(manifest, new_cache)``. ``new_cache`` is rebuilt from the files
    that exist right now rather than mutated in place, so digests for deleted or
    replaced LoRAs are pruned instead of accumulating in the state file forever.

    ``hasher`` is injectable purely so tests can count how often a digest is
    actually computed (the "unchanged file is not re-hashed" guarantee).

    ``on_progress(phase, done, total, name)`` is called per file. The scan is
    user-initiated and a cold run can take minutes, so the panel needs something
    truthful to show — a button that looks dead for three minutes is a button
    users press twice.
    """
    cache = cache or {}
    new_cache: dict = {}
    entries: list[dict] = []
    names = list_lora_names()
    truncated = len(names) > MAX_ENTRIES
    todo = names[:MAX_ENTRIES]
    for i, name in enumerate(todo):
        _notify(on_progress, "loras", i, len(todo), name)
        path = lora_path(name)
        if not path:
            continue
        try:
            st = os.stat(path)
        except OSError as e:
            # Enumerated a moment ago, gone now (deleted, or a broken symlink).
            log.debug("lora inventory: cannot stat %s: %s", path, e)
            continue
        key = cache_key(path, st.st_size, st.st_mtime_ns)
        digest = cache.get(key)
        if not digest:
            try:
                digest = hasher(path)
            except OSError as e:
                # Still being downloaded, or unreadable. Skip this one file;
                # the next refresh picks it up.
                log.debug("lora inventory: cannot hash %s: %s", path, e)
                continue
        new_cache[key] = digest
        meta: dict = {}
        if path.lower().endswith(_SAFETENSORS_EXT):
            meta = extract_meta(read_safetensors_metadata(path))
        entries.append({
            "name": name,
            "size": st.st_size,
            "mtime": int(st.st_mtime),
            "sha256": digest,
            "meta": meta,
        })
    _notify(on_progress, "loras", len(todo), len(todo), "")
    # Checkpoints are collected in a guard of their own so that a surprise on the
    # newer, less-exercised path can never cost us the LoRA inventory — which is
    # the part users already depend on.
    try:
        checkpoints, ckpt_truncated = build_checkpoints(on_progress)
    except Exception as e:  # noqa: BLE001 - LoRAs still ship
        log.warning("model inventory: checkpoint scan failed: %s", e)
        checkpoints, ckpt_truncated = [], False
    manifest = {
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
    return manifest, new_cache


def manifest_hash(manifest: dict) -> str:
    """Content hash used to skip an unchanged re-upload.

    Covers the schema + entries ONLY — ``generated_at`` moves every run and
    including it would defeat the whole point (same mirror of the disk, new
    hash, pointless upload). Same shape/role as ``worker.object_info_hash``.

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
# ⛔ NOTHING IN THIS MODULE RUNS ON ITS OWN. Reading a user's model folders and
# fingerprinting their files is not something a plugin should do behind their
# back on a timer, and this plugin has said so since before the feature existed
# ("No background auto-sync", web/comfylink.js). An earlier revision ran a
# 10-minute background refresh; it was removed deliberately.
#
# The single trigger is the user pressing refresh IN THE APP: the relay records
# the request, hands the plugin the request's timestamp on the next heartbeat,
# and worker._maybe_scan_models turns that into exactly one scan. Consent is
# explicit and remote — the user does not have to walk over to the PC to give it.


# Only one build at a time in this process. A machine paired to several accounts
# gets one scan request per account, and hashing tens of GB once per account
# would be absurd. Serialising is enough on its own: whoever gets the lock second
# finds every digest already in the cache, so their "rebuild" is a stat per file
# (milliseconds). No result cache, therefore no staleness question to answer.
_build_lock = asyncio.Lock()


async def collect(on_progress=None) -> dict:
    """Build the inventory off the event loop. Only ever called on demand.

    ``asyncio.to_thread`` is not optional: a cold run reads tens of GB, and the
    calling coroutine shares its event loop with the relay heartbeat, the cancel
    long-poll and the claim loop. Blocking it for minutes would look exactly like
    a dead backend. The digest cache is read from and written back to the 0600
    plugin state file, and only persisted when it actually changed.
    """
    from .config import STATE

    async with _build_lock:
        cache = dict(STATE.lora_hashes)
        manifest, new_cache = await asyncio.to_thread(
            build_manifest, cache, file_sha256, on_progress)
        if new_cache != cache:
            STATE.lora_hashes = new_cache
            STATE.save()
        return manifest


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
