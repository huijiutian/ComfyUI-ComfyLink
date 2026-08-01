"""Tests for the model inventory (comfylink/loras.py + its worker wiring).

What these pin, in order of how badly a regression would hurt:

  * ⚠️ THE NAME KEY. Every reported ``name`` is verbatim what ComfyUI's
    ``folder_paths.get_filename_list("loras")`` returned — subdirectory prefix
    and all. That is the string the app finds inside a workflow (it comes from
    the same call, via ``/object_info``'s ``LoraLoader.lora_name`` enum), so it
    is the ONLY thing the two sides join on. Report a basename or an absolute
    path instead and nothing errors anywhere — the app just never matches a
    single LoRA and no trigger word ever appears. Nothing else in this feature
    fails this quietly, hence the tests come first.
  * HEADER-ONLY READS. A LoRA is hundreds of MB; the metadata parser must read
    the 8-byte length and the JSON header and then stop.
  * THE DIGEST CACHE. CivitAI indexes by whole-file SHA256, so a cold run reads
    tens of GB. An unchanged file must never be hashed twice; a changed one must
    be. Without this every ComfyUI restart re-reads the lot.
  * NON-BLOCKING / NON-FATAL. Collection failures, upload failures, and an old
    relay's 404 all leave the worker registering, heart-beating and claiming
    jobs exactly as before.
  * ⛔ CHECKPOINTS ARE NEVER HASHED. Schema 2 added a `checkpoints` array that is
    header-only: 2–7 GB per file means a full-hash pass would be hundreds of GB
    of reads. There is a test asserting no digest is computed for them, and one
    asserting the `loras` section is byte-identical at schema 2 so the older app
    that only reads `loras` cannot be affected by any of it.

Run:  python -m unittest discover -s tests
"""

import asyncio
import io
import json
import os
import struct
import sys
import tempfile
import time
import unittest
from contextlib import ExitStack, contextmanager
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import loras, worker  # noqa: E402


# ── fixtures ──────────────────────────────────────────────────────────────────

class _FakeFolderPaths:
    """Stand-in for ComfyUI's folder_paths module.

    Deliberately mimics the two behaviours that make the name key subtle:
    ``get_filename_list`` returns RELATIVE names that may carry a subdirectory
    prefix, and ``get_full_path`` is the only way back to an absolute path.
    Folder-aware, because "loras" and "checkpoints" are separate namespaces that
    must never be crossed.
    """

    def __init__(self, folders: dict):
        self.folders = folders          # folder -> {relative name: absolute path}
        self.list_calls: list[str] = []

    def get_filename_list(self, folder: str):
        self.list_calls.append(folder)
        return list(self.folders.get(folder) or {})

    def get_full_path(self, folder: str, name: str):
        return (self.folders.get(folder) or {}).get(name)


@contextmanager
def _folder_paths(loras: dict, checkpoints: dict = None):
    """Install a fake ``folder_paths`` module for the duration of the block."""
    fake = _FakeFolderPaths({"loras": loras, "checkpoints": checkpoints or {}})
    sys.modules["folder_paths"] = fake
    try:
        yield fake
    finally:
        sys.modules.pop("folder_paths", None)


def _write_safetensors(path: str, metadata: dict, tail: bytes = b"") -> int:
    """Write a minimal but real safetensors file; returns the header length."""
    header = {
        "__metadata__": {k: str(v) for k, v in metadata.items()},
        "fake.weight": {"dtype": "F16", "shape": [1], "data_offsets": [0, 2]},
    }
    blob = json.dumps(header).encode()
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        f.write(tail)
    return len(blob)


class _Boom(Exception):
    """Throwaway failure for "this must never raise" tests."""


class _RecordingReader:
    """File-like that remembers every read size (to prove we stop after the header)."""

    def __init__(self, data: bytes):
        self._buf = io.BytesIO(data)
        self.reads: list[int] = []

    def read(self, n: int = -1) -> bytes:
        self.reads.append(n)
        return self._buf.read(n)


# ── ⚠️ the name key ───────────────────────────────────────────────────────────

class TestNameKeyMatchesFolderPaths(unittest.TestCase):
    """The join key with /object_info. A silent-failure guard — see module docstring."""

    def test_names_are_folder_paths_strings_including_subdirectory(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "style")
            os.makedirs(sub)
            flat = os.path.join(d, "flat.safetensors")
            nested = os.path.join(sub, "foo.safetensors")
            _write_safetensors(flat, {})
            _write_safetensors(nested, {})
            # Exactly the shape folder_paths hands back: relative, subdir-prefixed.
            mapping = {"flat.safetensors": flat, "style/foo.safetensors": nested}
            with _folder_paths(mapping) as fake:
                manifest, _ = loras.build_manifest(hasher=lambda p: "d" * 64)

        names = [e["name"] for e in manifest["loras"]]
        self.assertEqual(sorted(names), ["flat.safetensors", "style/foo.safetensors"])
        # The subdirectory prefix must survive — a bare "foo.safetensors" would
        # never match the workflow's value.
        self.assertIn("style/foo.safetensors", names)
        # And no absolute paths leaked into the key.
        for n in names:
            self.assertFalse(os.path.isabs(n), f"{n} must be relative to the loras dir")
            self.assertNotIn(d, n)
        # It really came from get_filename_list("loras"), not our own walk.
        self.assertEqual(fake.list_calls, ["loras", "checkpoints"])

    def test_list_lora_names_passes_through_verbatim(self):
        weird = {
            "sub dir/Ünïcode Name.safetensors": "/nope",
            "a.safetensors": "/nope",
        }
        with _folder_paths(weird):
            self.assertEqual(loras.list_lora_names(),
                             sorted(weird))


class TestDegradesWithoutComfyUI(unittest.TestCase):
    def test_missing_folder_paths_is_silent(self):
        sys.modules.pop("folder_paths", None)
        self.assertFalse(loras.folder_paths_available())
        self.assertEqual(loras.list_lora_names(), [])
        self.assertIsNone(loras.lora_path("x.safetensors"))
        manifest, cache = loras.build_manifest()
        self.assertEqual(manifest["loras"], [])
        self.assertEqual(cache, {})

    def test_folder_paths_that_raises_is_silent(self):
        class _Broken:
            def get_filename_list(self, folder):
                raise RuntimeError("no loras directory configured")

        sys.modules["folder_paths"] = _Broken()
        try:
            self.assertEqual(loras.list_lora_names(), [])
        finally:
            sys.modules.pop("folder_paths", None)

    def test_vanished_file_is_skipped_not_fatal(self):
        # Enumerated, then deleted before we stat it (or a broken symlink).
        with _folder_paths({"gone.safetensors": "/definitely/not/here.safetensors"}):
            manifest, cache = loras.build_manifest(hasher=lambda p: "x" * 64)
        self.assertEqual(manifest["loras"], [])
        self.assertEqual(cache, {})


# ── safetensors header ────────────────────────────────────────────────────────

class TestSafetensorsHeader(unittest.TestCase):
    def test_reads_only_the_header_bytes(self):
        meta = {"ss_output_name": "mylora"}
        header = {"__metadata__": meta}
        blob = json.dumps(header).encode()
        tail = b"\x00" * (2 * 1024 * 1024)  # stands in for the tensor payload
        r = _RecordingReader(struct.pack("<Q", len(blob)) + blob + tail)

        got = loras.read_header_metadata(r)

        self.assertEqual(got, meta)
        # Exactly two bounded reads: the 8-byte length, then the header. The
        # multi-MB tail is never touched, and read() is never called unbounded.
        self.assertEqual(r.reads, [8, len(blob)])
        self.assertNotIn(-1, r.reads)
        self.assertLess(sum(r.reads), len(tail))

    def test_malformed_inputs_yield_empty(self):
        cases = {
            "short file": b"\x00\x01",
            "zero length": struct.pack("<Q", 0),
            "absurd length": struct.pack("<Q", loras.MAX_HEADER_BYTES + 1) + b"{}",
            "truncated header": struct.pack("<Q", 100) + b"{}",
            "not json": struct.pack("<Q", 3) + b"abc",
            "json not an object": struct.pack("<Q", 2) + b"[]",
            "no __metadata__": struct.pack("<Q", 2) + b"{}",
        }
        for label, data in cases.items():
            with self.subTest(label):
                self.assertEqual(loras.read_header_metadata(_RecordingReader(data)), {})

    def test_unreadable_path_yields_empty(self):
        self.assertEqual(loras.read_safetensors_metadata("/no/such/file.safetensors"), {})

    def test_non_safetensors_gets_empty_meta_but_is_still_reported(self):
        # .pt/.ckpt LoRAs have no readable header, but CivitAI indexes by hash
        # regardless, so they must still appear in the manifest.
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "old.pt")
            with open(p, "wb") as f:
                f.write(b"not safetensors at all")
            with _folder_paths({"old.pt": p}):
                manifest, _ = loras.build_manifest(hasher=lambda _p: "c" * 64)
        self.assertEqual(len(manifest["loras"]), 1)
        self.assertEqual(manifest["loras"][0]["meta"], {})
        self.assertEqual(manifest["loras"][0]["sha256"], "c" * 64)


# ── meta subset ───────────────────────────────────────────────────────────────

class TestMetaSubset(unittest.TestCase):
    def test_tag_frequency_is_not_dumped_verbatim(self):
        # A realistic Kohya header: ss_tag_frequency is the fat field.
        freq = {"10_subject": {f"tag_{i}": 1000 - i for i in range(4000)}}
        md = {
            "ss_output_name": "mylora",
            "ss_tag_frequency": json.dumps(freq),
            "ss_unknown_gigantic_field": "z" * 500000,
        }
        meta = loras.extract_meta(md)

        self.assertNotIn("ss_tag_frequency", meta)
        self.assertNotIn("ss_unknown_gigantic_field", meta,
                         "extraction is an allowlist; unknown fields never ride along")
        # A bounded, ranked digest of the tags survives instead.
        self.assertEqual(meta["top_tags"][0], "tag_0")
        self.assertLessEqual(len(meta["top_tags"]), loras.MAX_TOP_TAGS)
        # And the whole thing stays small enough to ship to a phone.
        self.assertLess(len(json.dumps(meta)), 8192)

    def test_keeps_the_identifying_fields(self):
        md = {
            "modelspec.title": "My Style",
            "modelspec.trigger_phrase": "mystyle",
            "ss_base_model_version": "sdxl_base_v1-0",
        }
        meta = loras.extract_meta(md)
        self.assertEqual(meta["modelspec.title"], "My Style")
        self.assertEqual(meta["modelspec.trigger_phrase"], "mystyle")
        self.assertEqual(meta["ss_base_model_version"], "sdxl_base_v1-0")

    def test_long_values_are_truncated(self):
        meta = loras.extract_meta({"ss_training_comment": "x" * 5000})
        self.assertLessEqual(len(meta["ss_training_comment"]), loras.META_VALUE_MAX)

    def test_empty_and_placeholder_values_dropped(self):
        meta = loras.extract_meta(
            {"ss_output_name": "  ", "ss_clip_skip": "None", "ss_epoch": "{}"})
        self.assertEqual(meta, {})

    def test_non_dict_metadata_is_tolerated(self):
        self.assertEqual(loras.extract_meta(None), {})
        self.assertEqual(loras.extract_meta("nope"), {})

    def test_top_tags_ignores_junk(self):
        self.assertEqual(loras._top_tags("not json"), [])
        self.assertEqual(loras._top_tags(None), [])
        self.assertEqual(loras._top_tags({"d": {"a": "NaN"}}), [])


# ── digest cache ──────────────────────────────────────────────────────────────

class TestDigestCache(unittest.TestCase):
    def test_unchanged_file_is_not_rehashed(self):
        calls = []

        def counting_hasher(path):
            calls.append(path)
            return "a" * 64

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.safetensors")
            _write_safetensors(p, {"ss_output_name": "a"})
            with _folder_paths({"a.safetensors": p}):
                m1, cache = loras.build_manifest({}, hasher=counting_hasher)
                self.assertEqual(len(calls), 1, "cold run hashes once")
                # Second pass with the cache from the first: same path, same
                # size, same mtime => the digest must be reused, not recomputed.
                m2, cache2 = loras.build_manifest(cache, hasher=counting_hasher)

        self.assertEqual(len(calls), 1, "an unchanged file must never be re-hashed")
        self.assertEqual(m1["loras"][0]["sha256"], m2["loras"][0]["sha256"])
        self.assertEqual(cache, cache2)

    def test_changed_mtime_forces_a_rehash(self):
        calls = []

        def counting_hasher(path):
            calls.append(path)
            return f"{len(calls):064d}"

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.safetensors")
            _write_safetensors(p, {})
            with _folder_paths({"a.safetensors": p}):
                _m1, cache = loras.build_manifest({}, hasher=counting_hasher)
                # Same path, new content/mtime — the cache key must miss.
                _write_safetensors(p, {"ss_output_name": "changed"}, tail=b"\x01" * 32)
                os.utime(p, (0, 0))
                m2, _ = loras.build_manifest(cache, hasher=counting_hasher)

        self.assertEqual(len(calls), 2, "a modified file must be re-hashed")
        self.assertEqual(m2["loras"][0]["sha256"], f"{2:064d}")

    def test_cache_is_pruned_not_accumulated(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.safetensors")
            _write_safetensors(p, {})
            stale = {"/deleted/lora.safetensors|1|1": "b" * 64}
            with _folder_paths({"a.safetensors": p}):
                _m, new_cache = loras.build_manifest(stale, hasher=lambda _p: "a" * 64)
        # Rebuilt from what exists now: the deleted LoRA's entry is gone, so the
        # cache cannot grow without bound across a LoRA collection's lifetime.
        self.assertNotIn("/deleted/lora.safetensors|1|1", new_cache)
        self.assertEqual(len(new_cache), 1)

    def test_cache_key_includes_size_and_mtime(self):
        a = loras.cache_key("/p", 1, 2)
        self.assertNotEqual(a, loras.cache_key("/p", 1, 3))
        self.assertNotEqual(a, loras.cache_key("/p", 9, 2))
        self.assertNotEqual(a, loras.cache_key("/q", 1, 2))

    def test_file_sha256_streams_and_is_correct(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "blob.bin")
            data = os.urandom(1024 * 64)
            with open(p, "wb") as f:
                f.write(data)
            self.assertEqual(loras.file_sha256(p), hashlib.sha256(data).hexdigest())


# ── manifest shape ────────────────────────────────────────────────────────────

class TestManifestShape(unittest.TestCase):
    def test_entry_fields_and_schema(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.safetensors")
            _write_safetensors(p, {"ss_output_name": "a"})
            with _folder_paths({"a.safetensors": p}):
                manifest, _ = loras.build_manifest(hasher=lambda _p: "a" * 64)

        self.assertEqual(manifest["schema"], loras.MANIFEST_SCHEMA)
        self.assertEqual(manifest["count"], 1)
        self.assertFalse(manifest["truncated"])
        self.assertIsInstance(manifest["generated_at"], int)
        e = manifest["loras"][0]
        self.assertEqual(set(e), {"name", "size", "mtime", "sha256", "meta"})
        self.assertIsInstance(e["size"], int)
        self.assertIsInstance(e["mtime"], int)
        self.assertEqual(e["meta"]["ss_output_name"], "a")
        # JSON-serialisable end to end (it is uploaded verbatim).
        json.loads(json.dumps(manifest))

    def test_manifest_hash_ignores_generated_at(self):
        base = {"schema": 1, "generated_at": 1, "loras": [{"name": "a", "sha256": "x"}]}
        later = {"schema": 1, "generated_at": 999999, "loras": [{"name": "a", "sha256": "x"}]}
        self.assertEqual(loras.manifest_hash(base), loras.manifest_hash(later))

    def test_manifest_hash_changes_with_entries(self):
        a = {"schema": 1, "loras": [{"name": "a", "sha256": "x"}]}
        b = {"schema": 1, "loras": [{"name": "a", "sha256": "y"}]}
        self.assertNotEqual(loras.manifest_hash(a), loras.manifest_hash(b))


class TestCollectOffThread(unittest.IsolatedAsyncioTestCase):
    async def test_build_runs_in_a_thread_and_persists_the_cache(self):
        state = mock.Mock()
        state.lora_hashes = {}

        seen_threads = []
        real_build = loras.build_manifest

        def spy_build(cache, hasher=loras.file_sha256, on_progress=None):
            import threading
            seen_threads.append(threading.current_thread().name)
            return real_build(cache, hasher=hasher, on_progress=on_progress)

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.safetensors")
            _write_safetensors(p, {})
            with _folder_paths({"a.safetensors": p}), \
                    mock.patch.object(loras, "build_manifest", spy_build), \
                    mock.patch("comfylink.config.STATE", state):
                m = await loras.collect()

        self.assertEqual(m["count"], 1)
        # The hashing pass must NOT run on the event loop thread — a cold run is
        # tens of GB of reads and would stall the heartbeat, the cancel long-poll
        # and the claim loop, which all share that loop.
        import threading
        self.assertNotIn(threading.current_thread().name, seen_threads)
        # New digests were persisted for the next process start.
        self.assertTrue(state.lora_hashes)
        state.save.assert_called()

    async def test_progress_callback_reaches_the_caller(self):
        # The scan is the only multi-minute thing this plugin does; without
        # per-file progress there is no way to tell hashing from a hang.
        state = mock.Mock()
        state.lora_hashes = {}
        seen = []
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.safetensors")
            _write_safetensors(p, {})
            with _folder_paths({"a.safetensors": p}), \
                    mock.patch("comfylink.config.STATE", state):
                await loras.collect(lambda *a: seen.append(a))
        phases = {a[0] for a in seen}
        self.assertIn("loras", phases)
        self.assertIn("checkpoints", phases)
        self.assertIn(("loras", 0, 1, "a.safetensors"), seen)

    async def test_a_raising_progress_callback_cannot_break_the_scan(self):
        state = mock.Mock()
        state.lora_hashes = {}

        def boom(*_a):
            raise RuntimeError("display blew up")

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "a.safetensors")
            _write_safetensors(p, {})
            with _folder_paths({"a.safetensors": p}), \
                    mock.patch("comfylink.config.STATE", state):
                m = await loras.collect(boom)
        self.assertEqual(m["count"], 1)

    async def test_concurrent_collects_are_serialised(self):
        # A machine paired to several accounts gets one request per account.
        # Hashing tens of GB once per account would be absurd, so builds must
        # never overlap; the second one finds a warm cache and is cheap.
        state = mock.Mock()
        state.lora_hashes = {}
        overlap = {"now": 0, "max": 0}

        def spy_build(cache, hasher=loras.file_sha256, on_progress=None):
            overlap["now"] += 1
            overlap["max"] = max(overlap["max"], overlap["now"])
            time.sleep(0.02)  # real thread work, so an overlap would be visible
            overlap["now"] -= 1
            return {"schema": 2, "generated_at": 0, "count": 0, "truncated": False,
                    "loras": [], "checkpoints": [], "checkpoints_count": 0,
                    "checkpoints_truncated": False}, {}

        with mock.patch.object(loras, "build_manifest", spy_build), \
                mock.patch("comfylink.config.STATE", state):
            await asyncio.gather(loras.collect(), loras.collect())
        self.assertEqual(overlap["max"], 1, "builds must not run concurrently")


# ── checkpoints (schema 2): header-only, NEVER hashed ─────────────────────────

class TestCheckpointNameKey(unittest.TestCase):
    """Same silent-failure guard as the LoRA one — the app joins on this string."""

    def test_names_are_folder_paths_strings_including_subdirectory(self):
        with tempfile.TemporaryDirectory() as d:
            sub = os.path.join(d, "sdxl")
            os.makedirs(sub)
            flat = os.path.join(d, "base.safetensors")
            nested = os.path.join(sub, "anime.safetensors")
            _write_safetensors(flat, {})
            _write_safetensors(nested, {})
            mapping = {"base.safetensors": flat, "sdxl/anime.safetensors": nested}
            with _folder_paths({}, mapping) as fake:
                manifest, _ = loras.build_manifest(hasher=lambda p: "d" * 64)

        names = [e["name"] for e in manifest["checkpoints"]]
        self.assertEqual(sorted(names), ["base.safetensors", "sdxl/anime.safetensors"])
        for n in names:
            self.assertFalse(os.path.isabs(n))
            self.assertNotIn(d, n)
        # Enumerated through folder_paths under the "checkpoints" folder key —
        # never our own directory walk, and never crossed with "loras".
        self.assertIn("checkpoints", fake.list_calls)

    def test_list_checkpoint_names_uses_the_checkpoints_folder(self):
        with _folder_paths({"a.safetensors": "/l"}, {"b.safetensors": "/c"}):
            self.assertEqual(loras.list_checkpoint_names(), ["b.safetensors"])
            self.assertEqual(loras.list_lora_names(), ["a.safetensors"])
            self.assertEqual(loras.checkpoint_path("b.safetensors"), "/c")
            # Namespaces must not leak into each other.
            self.assertIsNone(loras.checkpoint_path("a.safetensors"))
            self.assertIsNone(loras.lora_path("b.safetensors"))


class TestCheckpointsAreNeverHashed(unittest.TestCase):
    """⛔ The whole point of the checkpoint path: 2–7 GB files stay unread."""

    def test_no_sha256_field_and_no_hasher_call(self):
        hashed = []

        with tempfile.TemporaryDirectory() as d:
            lora = os.path.join(d, "l.safetensors")
            ckpt = os.path.join(d, "c.safetensors")
            _write_safetensors(lora, {})
            _write_safetensors(ckpt, {"modelspec.architecture": "flux-1-dev"})
            with _folder_paths({"l.safetensors": lora}, {"c.safetensors": ckpt}):
                manifest, cache = loras.build_manifest(
                    hasher=lambda p: hashed.append(p) or ("a" * 64))

        # The LoRA was hashed; the checkpoint was not — not once, not ever.
        self.assertEqual(hashed, [lora])
        self.assertNotIn(ckpt, hashed)
        entry = manifest["checkpoints"][0]
        self.assertNotIn("sha256", entry)
        self.assertEqual(set(entry), {"name", "size", "mtime", "meta"})
        # And nothing about a checkpoint entered the digest cache either.
        self.assertEqual(len(cache), 1)
        self.assertNotIn(ckpt, next(iter(cache)))

    def test_reported_digest_in_the_header_is_not_promoted(self):
        # Some writers embed modelspec.hash_sha256. It is self-reported and not
        # necessarily the file's digest, so it must never surface where a
        # consumer could mistake it for the LoRA entries' verified `sha256`.
        md = {"modelspec.architecture": "x", "modelspec.hash_sha256": "0" * 64}
        meta = loras.extract_checkpoint_meta(md)
        self.assertNotIn("modelspec.hash_sha256", meta)
        self.assertNotIn("sha256", meta)


class TestCheckpointMeta(unittest.TestCase):
    def test_keeps_family_evidence(self):
        md = {
            "modelspec.architecture": "stable-diffusion-xl-v1-base",
            "modelspec.title": "Some Anime Mix",
            "modelspec.prediction_type": "v_prediction",
            "modelspec.resolution": "1024x1024",
        }
        meta = loras.extract_checkpoint_meta(md)
        self.assertEqual(meta["modelspec.architecture"], "stable-diffusion-xl-v1-base")
        self.assertEqual(meta["modelspec.prediction_type"], "v_prediction")
        self.assertEqual(meta["modelspec.title"], "Some Anime Mix")
        self.assertEqual(meta["modelspec.resolution"], "1024x1024")

    def test_drops_lora_training_fields(self):
        # A checkpoint has no trigger phrase and no training tags; carrying them
        # would spend the meta budget on things the app cannot use.
        freq = {"10_x": {f"t{i}": 100 - i for i in range(500)}}
        md = {
            "modelspec.architecture": "flux-1-dev",
            "modelspec.trigger_phrase": "irrelevant",
            "ss_tag_frequency": json.dumps(freq),
            "ss_network_dim": "32",
            "ss_training_comment": "x" * 4000,
        }
        meta = loras.extract_checkpoint_meta(md)
        self.assertEqual(meta, {"modelspec.architecture": "flux-1-dev"})
        self.assertNotIn("top_tags", meta)

    def test_stays_small(self):
        md = {k: "y" * 5000 for k in loras._CKPT_META_KEYS}
        meta = loras.extract_checkpoint_meta(md)
        self.assertLessEqual(len(json.dumps(meta)), loras.META_TOTAL_MAX)
        # The single most important field survives the trim.
        self.assertIn("modelspec.architecture", meta)

    def test_non_safetensors_checkpoint_still_listed(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "old.ckpt")
            with open(p, "wb") as f:
                f.write(b"pickle junk")
            with _folder_paths({}, {"old.ckpt": p}):
                manifest, _ = loras.build_manifest()
        self.assertEqual(manifest["checkpoints"][0]["name"], "old.ckpt")
        self.assertEqual(manifest["checkpoints"][0]["meta"], {})


class TestSchema2DoesNotDisturbSchema1Consumers(unittest.TestCase):
    """An app that only reads `loras` must not be able to tell schema 2 happened."""

    def test_loras_section_is_byte_identical_with_and_without_checkpoints(self):
        with tempfile.TemporaryDirectory() as d:
            lora = os.path.join(d, "style", "l.safetensors")
            os.makedirs(os.path.dirname(lora))
            _write_safetensors(lora, {"ss_output_name": "l", "modelspec.title": "L"})
            ckpt = os.path.join(d, "c.safetensors")
            _write_safetensors(ckpt, {"modelspec.architecture": "flux-1-dev"})
            lora_map = {"style/l.safetensors": lora}

            with _folder_paths(lora_map, {}):
                without, _ = loras.build_manifest(hasher=lambda p: "a" * 64)
            with _folder_paths(lora_map, {"c.safetensors": ckpt}):
                with_ckpt, _ = loras.build_manifest(hasher=lambda p: "a" * 64)

        # Byte-for-byte: same entries, same key order, same values.
        self.assertEqual(json.dumps(without["loras"]), json.dumps(with_ckpt["loras"]))
        # ...and the top-level fields an old app reads keep their LoRA-scoped
        # meaning rather than silently becoming totals.
        self.assertEqual(without["count"], with_ckpt["count"])
        self.assertEqual(without["truncated"], with_ckpt["truncated"])
        self.assertEqual(with_ckpt["count"], 1)
        self.assertEqual(with_ckpt["checkpoints_count"], 1)

    def test_schema_is_2_and_checkpoints_key_always_present(self):
        with _folder_paths({}, {}):
            manifest, _ = loras.build_manifest()
        self.assertEqual(manifest["schema"], 2)
        self.assertEqual(loras.MANIFEST_SCHEMA, 2)
        # Always an array, never absent — "no checkpoints" and "old plugin" must
        # stay distinguishable for the app (absent key ⇒ schema 1 ⇒ stay silent).
        self.assertEqual(manifest["checkpoints"], [])
        self.assertFalse(manifest["checkpoints_truncated"])

    def test_manifest_hash_covers_checkpoints(self):
        a = {"schema": 2, "loras": [], "checkpoints": [{"name": "a", "meta": {}}]}
        b = {"schema": 2, "loras": [], "checkpoints": [{"name": "b", "meta": {}}]}
        # A new checkpoint with no LoRA change must still trigger a re-upload.
        self.assertNotEqual(loras.manifest_hash(a), loras.manifest_hash(b))
        # And a schema-1-shaped dict (no checkpoints key) must not explode.
        loras.manifest_hash({"schema": 1, "loras": []})


class TestCheckpointFailuresDoNotCostTheLoras(unittest.TestCase):
    def test_missing_checkpoints_folder_is_not_fatal(self):
        class _LorasOnly:
            def get_filename_list(self, folder):
                if folder != "loras":
                    raise KeyError("checkpoints folder not configured")
                return ["l.safetensors"]

            def get_full_path(self, folder, name):
                return None

        sys.modules["folder_paths"] = _LorasOnly()
        try:
            manifest, _ = loras.build_manifest()
        finally:
            sys.modules.pop("folder_paths", None)
        self.assertEqual(manifest["checkpoints"], [])
        self.assertEqual(loras.list_checkpoint_names(), [])

    def test_checkpoint_scan_blowing_up_still_ships_the_loras(self):
        with tempfile.TemporaryDirectory() as d:
            lora = os.path.join(d, "l.safetensors")
            _write_safetensors(lora, {})
            with _folder_paths({"l.safetensors": lora}), \
                    mock.patch.object(loras, "build_checkpoints",
                                      side_effect=RuntimeError("boom")):
                manifest, _ = loras.build_manifest(hasher=lambda p: "a" * 64)
        # The older, load-bearing half survives a failure in the newer half.
        self.assertEqual(len(manifest["loras"]), 1)
        self.assertEqual(manifest["checkpoints"], [])


# ── worker wiring: never fatal, never blocking ────────────────────────────────

class _FakePairing:
    def __init__(self, backend_id="b1", loras_hash="", loras_synced_at=0.0):
        self.backend_id = backend_id
        self.loras_hash = loras_hash
        # Watermark of the last app refresh request this pairing has served.
        self.loras_synced_at = loras_synced_at
        self.account = ""


class _FakeState:
    def __init__(self, pairings=()):
        self.pairings = list(pairings)
        self.backend_name = "dev"
        self.save_calls = 0

    def get_pairing(self, backend_id):
        return next((p for p in self.pairings if p.backend_id == backend_id), None)

    def remove_pairing(self, backend_id):
        pr = self.get_pairing(backend_id)
        self.pairings = [p for p in self.pairings if p.backend_id != backend_id]
        return pr

    def save(self):
        self.save_calls += 1


_MANIFEST = {"schema": 2, "generated_at": 1, "count": 1, "truncated": False,
             "loras": [{"name": "a.safetensors", "size": 1, "mtime": 1,
                        "sha256": "a" * 64, "meta": {}}],
             "checkpoints": [], "checkpoints_count": 0,
             "checkpoints_truncated": False}

# A machine with checkpoints but zero LoRAs still has something worth reporting
# (the app's model-family check), so this must NOT be treated as "empty".
_MANIFEST_CKPT_ONLY = {"schema": 2, "generated_at": 1, "count": 0,
                       "truncated": False, "loras": [],
                       "checkpoints": [{"name": "c.safetensors", "size": 1,
                                        "mtime": 1, "meta": {}}],
                       "checkpoints_count": 1, "checkpoints_truncated": False}


class TestUploadFor(unittest.IsolatedAsyncioTestCase):
    """loras.upload_for — one pairing's half of a scan."""

    async def test_uploads_and_remembers_the_hash(self):
        relay = mock.AsyncMock()
        pairing = _FakePairing()
        state = _FakeState([pairing])
        with mock.patch("comfylink.config.STATE", state):
            self.assertEqual(await loras.upload_for(relay, pairing, _MANIFEST),
                             "uploaded")
        relay.upload_loras.assert_awaited_once_with("b1", _MANIFEST)
        self.assertEqual(pairing.loras_hash, loras.manifest_hash(_MANIFEST))
        self.assertEqual(state.save_calls, 1)

    async def test_unchanged_manifest_is_not_re_uploaded(self):
        relay = mock.AsyncMock()
        pairing = _FakePairing(loras_hash=loras.manifest_hash(_MANIFEST))
        state = _FakeState([pairing])
        with mock.patch("comfylink.config.STATE", state):
            self.assertEqual(await loras.upload_for(relay, pairing, _MANIFEST),
                             "unchanged")
        relay.upload_loras.assert_not_awaited()
        self.assertEqual(state.save_calls, 0)

    async def test_checkpoints_only_machine_still_uploads(self):
        relay = mock.AsyncMock()
        pairing = _FakePairing()
        with mock.patch("comfylink.config.STATE", _FakeState([pairing])):
            self.assertEqual(
                await loras.upload_for(relay, pairing, _MANIFEST_CKPT_ONLY),
                "uploaded")
        relay.upload_loras.assert_awaited_once_with("b1", _MANIFEST_CKPT_ONLY)

    async def test_old_relay_404_is_reported_not_raised(self):
        # BACK-COMPAT: new plugin, relay without the endpoint. No exception, no
        # stored hash (so a later relay upgrade still uploads).
        relay = mock.AsyncMock()
        relay.upload_loras.side_effect = worker.RelayError("not found", 404)
        pairing = _FakePairing()
        state = _FakeState([pairing])
        with mock.patch("comfylink.config.STATE", state):
            self.assertEqual(await loras.upload_for(relay, pairing, _MANIFEST),
                             "unsupported")
        self.assertEqual(pairing.loras_hash, "")
        self.assertEqual(state.save_calls, 0)

    async def test_other_relay_errors_propagate(self):
        # A real failure must reach the caller, which decides not to advance the
        # watermark so the next beat retries.
        relay = mock.AsyncMock()
        relay.upload_loras.side_effect = worker.RelayError("R2 unconfigured", 503)
        pairing = _FakePairing()
        state = _FakeState([pairing])
        with mock.patch("comfylink.config.STATE", state):
            with self.assertRaises(worker.RelayError):
                await loras.upload_for(relay, pairing, _MANIFEST)
        self.assertEqual(pairing.loras_hash, "")
        self.assertEqual(state.save_calls, 0)


# ── the app-initiated signal (heartbeat body → exactly one scan) ─────────────

class TestRequestedAt(unittest.TestCase):
    """_requested_at — back-compat and junk tolerance, in one pure function."""

    def test_old_relay_body_reads_as_no_request(self):
        # The whole back-compat story for "new plugin, old relay": no field,
        # nothing happens, no error.
        self.assertEqual(worker._requested_at({}), 0.0)
        self.assertEqual(worker._requested_at({"other": 1}), 0.0)

    def test_a_timestamp_is_read(self):
        self.assertEqual(worker._requested_at({"loras_requested_at": 1753900000}),
                         1753900000.0)
        self.assertEqual(worker._requested_at({"loras_requested_at": "1753900000"}),
                         1753900000.0)

    def test_junk_degrades_to_zero(self):
        for junk in (None, "", "soon", {}, [], object()):
            with self.subTest(junk=junk):
                self.assertEqual(worker._requested_at({"loras_requested_at": junk}),
                                 0.0)

    def test_non_dict_body_degrades_to_zero(self):
        self.assertEqual(worker._requested_at(None), 0.0)
        self.assertEqual(worker._requested_at("nope"), 0.0)


class TestMaybeScanModels(unittest.IsolatedAsyncioTestCase):
    """⛔ The consent gate: no signal ⇒ no scan. And one signal ⇒ one scan."""

    def setUp(self):
        self.calls = []

    def _fake_scan(self, state):
        async def _scan(relay, pairing, requested):
            self.calls.append(requested)
            worker._remember_scan(pairing, requested)
        return _scan

    async def _drain(self, task):
        if task is not None:
            await task

    async def test_no_signal_means_no_scan(self):
        pairing = _FakePairing()
        state = _FakeState([pairing])
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_scan_and_report", self._fake_scan(state)):
            for body in ({}, {"loras_requested_at": 0}, {"other": 5}):
                self.assertIsNone(worker._maybe_scan_models(
                    mock.AsyncMock(), pairing, body, None))
        self.assertEqual(self.calls, [])
        self.assertEqual(pairing.loras_synced_at, 0.0)

    async def test_a_signal_starts_exactly_one_scan(self):
        pairing = _FakePairing()
        state = _FakeState([pairing])
        body = {"loras_requested_at": 1753900000}
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_scan_and_report", self._fake_scan(state)):
            task = worker._maybe_scan_models(mock.AsyncMock(), pairing, body, None)
            await self._drain(task)
        self.assertEqual(self.calls, [1753900000.0])
        self.assertEqual(pairing.loras_synced_at, 1753900000.0)

    async def test_the_same_timestamp_redelivered_scans_only_once(self):
        # ⚠️ IDEMPOTENCE. The relay replays the same value on EVERY ~25s beat
        # until we report back — roughly 140 redeliveries an hour. The watermark
        # is the only thing between that and 140 full rescans.
        pairing = _FakePairing()
        state = _FakeState([pairing])
        body = {"loras_requested_at": 1753900000}
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_scan_and_report", self._fake_scan(state)):
            task = None
            for _ in range(10):
                task = worker._maybe_scan_models(mock.AsyncMock(), pairing, body, task)
                await self._drain(task)
        self.assertEqual(self.calls, [1753900000.0], "one request ⇒ one scan")

    async def test_a_newer_timestamp_scans_again(self):
        pairing = _FakePairing()
        state = _FakeState([pairing])
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_scan_and_report", self._fake_scan(state)):
            t = worker._maybe_scan_models(
                mock.AsyncMock(), pairing, {"loras_requested_at": 100}, None)
            await self._drain(t)
            t = worker._maybe_scan_models(
                mock.AsyncMock(), pairing, {"loras_requested_at": 200}, t)
            await self._drain(t)
        self.assertEqual(self.calls, [100.0, 200.0])

    async def test_an_older_timestamp_is_ignored(self):
        # Clock skew / a relay rollback must not re-trigger an already-served
        # request, and must not drag the watermark backwards.
        pairing = _FakePairing(loras_synced_at=500.0)
        state = _FakeState([pairing])
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_scan_and_report", self._fake_scan(state)):
            self.assertIsNone(worker._maybe_scan_models(
                mock.AsyncMock(), pairing, {"loras_requested_at": 499}, None))
        self.assertEqual(self.calls, [])
        self.assertEqual(pairing.loras_synced_at, 500.0)

    async def test_a_running_scan_is_never_duplicated(self):
        # A cold first scan takes minutes, spanning many beats. Those beats must
        # not each pile on another scan of the same folders.
        pairing = _FakePairing()
        state = _FakeState([pairing])
        release = asyncio.Event()
        started = []

        async def slow_scan(relay, pairing_, requested):
            started.append(requested)
            await release.wait()

        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_scan_and_report", slow_scan):
            task = worker._maybe_scan_models(
                mock.AsyncMock(), pairing, {"loras_requested_at": 100}, None)
            await asyncio.sleep(0)
            for _ in range(5):
                same = worker._maybe_scan_models(
                    mock.AsyncMock(), pairing, {"loras_requested_at": 100}, task)
                self.assertIs(same, task, "must return the in-flight task, not a new one")
            release.set()
            await task
        self.assertEqual(started, [100.0])

    def test_dispatch_is_synchronous(self):
        # It runs INSIDE the heartbeat loop, so it must be a plain function that
        # cannot await anything — the type system is the guard here.
        self.assertFalse(asyncio.iscoroutinefunction(worker._maybe_scan_models))


class TestScanAndReport(unittest.IsolatedAsyncioTestCase):
    """Watermark discipline: advance only when the request was actually served."""

    async def _run(self, state, pairing, collect, upload=None):
        """Drive one scan with the collaborators stubbed. requested = 900.0."""
        with ExitStack() as stack:
            stack.enter_context(mock.patch.object(worker, "STATE", state))
            stack.enter_context(mock.patch.object(worker.loras, "collect", collect))
            stack.enter_context(mock.patch.object(
                worker.loras, "folder_paths_available", lambda: True))
            if upload is not None:
                stack.enter_context(
                    mock.patch.object(worker.loras, "upload_for", upload))
            await worker._scan_and_report(mock.AsyncMock(), pairing, 900.0)

    async def test_success_advances_the_watermark(self):
        pairing = _FakePairing()
        state = _FakeState([pairing])
        await self._run(state, pairing,
                        mock.AsyncMock(return_value=_MANIFEST),
                        mock.AsyncMock(return_value="uploaded"))
        self.assertEqual(pairing.loras_synced_at, 900.0)

    async def test_collection_failure_does_not_advance(self):
        # The user pressed refresh and got nothing; the next beat must retry.
        pairing = _FakePairing()
        state = _FakeState([pairing])
        await self._run(state, pairing,
                        mock.AsyncMock(side_effect=OSError("disk on fire")))
        self.assertEqual(pairing.loras_synced_at, 0.0)

    async def test_upload_failure_does_not_advance(self):
        pairing = _FakePairing()
        state = _FakeState([pairing])
        await self._run(state, pairing,
                        mock.AsyncMock(return_value=_MANIFEST),
                        mock.AsyncMock(side_effect=RuntimeError("relay down")))
        self.assertEqual(pairing.loras_synced_at, 0.0)
        self.assertEqual(pairing.loras_hash, "")

    async def test_old_relay_404_advances(self):
        # Nothing more we can do for this request; retrying it every 25s forever
        # would be pure noise.
        pairing = _FakePairing()
        state = _FakeState([pairing])
        await self._run(state, pairing,
                        mock.AsyncMock(return_value=_MANIFEST),
                        mock.AsyncMock(return_value="unsupported"))
        self.assertEqual(pairing.loras_synced_at, 900.0)

    async def test_empty_inventory_advances_without_uploading(self):
        empty = {"schema": 2, "generated_at": 1, "count": 0, "truncated": False,
                 "loras": [], "checkpoints": [], "checkpoints_count": 0,
                 "checkpoints_truncated": False}
        pairing = _FakePairing()
        state = _FakeState([pairing])
        upload = mock.AsyncMock()
        await self._run(state, pairing, mock.AsyncMock(return_value=empty), upload)
        upload.assert_not_awaited()
        self.assertEqual(pairing.loras_synced_at, 900.0)

    async def test_outside_comfyui_serves_the_request_and_gives_up(self):
        pairing = _FakePairing()
        state = _FakeState([pairing])
        collect = mock.AsyncMock()
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker.loras, "collect", collect), \
                mock.patch.object(worker.loras, "folder_paths_available",
                                  lambda: False):
            await worker._scan_and_report(mock.AsyncMock(), pairing, 900.0)
        collect.assert_not_awaited()
        # folder_paths cannot appear later in this process, so retrying forever
        # would be pointless — the request is served with a shrug.
        self.assertEqual(pairing.loras_synced_at, 900.0)

    async def test_never_raises(self):
        pairing = _FakePairing()
        state = _FakeState([pairing])
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker.loras, "folder_paths_available",
                                  lambda: True), \
                mock.patch.object(worker.loras, "collect",
                                  mock.AsyncMock(side_effect=_Boom)):
            await worker._scan_and_report(mock.AsyncMock(), pairing, 900.0)


# ── ⚠️ the scan must never hold up the heartbeat ─────────────────────────────

class TestHeartbeatIsNeverBlocked(unittest.IsolatedAsyncioTestCase):
    async def test_beats_keep_flowing_while_a_scan_is_stuck(self):
        # A cold scan runs for minutes. If the heartbeat waited for it, the
        # relay's reaper would mark this backend dead and the app would show the
        # PC as offline — which is exactly the failure this design must avoid.
        pairing = _FakePairing()
        state = _FakeState([pairing])
        relay = mock.AsyncMock()
        relay.heartbeat.return_value = {"loras_requested_at": 100}
        release = asyncio.Event()
        scans = []

        async def stuck_scan(relay_, pairing_, requested):
            scans.append(requested)
            await release.wait()          # never finishes during the test

        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0.01), \
                mock.patch.object(worker, "_scan_and_report", stuck_scan):
            hb = asyncio.create_task(worker._heartbeat_loop(relay, pairing, None))
            await asyncio.sleep(0.15)
            beats = relay.heartbeat.await_count
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
        release.set()
        self.assertGreaterEqual(beats, 3,
                                "the heartbeat must keep its rhythm during a scan")
        self.assertEqual(scans, [100.0], "and must not pile on more scans")

    async def test_scan_task_is_cancelled_with_the_pairing(self):
        # Tearing down a pairing (unpair / reconnect) must take its scan with it,
        # rather than leaving it holding the build lock and a dying session.
        pairing = _FakePairing()
        state = _FakeState([pairing])
        relay = mock.AsyncMock()
        relay.heartbeat.return_value = {"loras_requested_at": 100}
        entered = asyncio.Event()
        cancelled = []

        async def slow_scan(relay_, pairing_, requested):
            entered.set()
            try:
                await asyncio.sleep(60)
            except asyncio.CancelledError:
                cancelled.append(True)
                raise

        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0.01), \
                mock.patch.object(worker, "_scan_and_report", slow_scan):
            hb = asyncio.create_task(worker._heartbeat_loop(relay, pairing, None))
            await asyncio.wait_for(entered.wait(), 1)
            hb.cancel()
            try:
                await hb
            except asyncio.CancelledError:
                pass
        self.assertEqual(cancelled, [True], "the scan must not outlive its pairing")


class TestNothingScansWithoutTheSignal(unittest.IsolatedAsyncioTestCase):
    """⛔ THE CONSENT GUARD. Regression target for "we scan on a timer again"."""

    async def test_a_full_pairing_lifecycle_scans_nothing(self):
        # Register → heartbeat → claim, with a relay that never asks for an
        # inventory (an old relay, or a user who simply never pressed refresh).
        # Not one file may be read and not one byte uploaded.
        from comfylink.config import Pairing

        relay = mock.AsyncMock()
        relay.heartbeat.return_value = {}          # old relay: no such field
        pairing = Pairing(backend_id="b1", device_token="clr")
        state = _FakeState([pairing])
        beats = asyncio.Event()

        async def fake_claim_loop(relay_, worker_, pairing_, job_lock_, stop_, blocked_):
            await asyncio.wait_for(beats.wait(), 1)
            raise worker._Revoked()

        async def counting_heartbeat(bid):
            beats.set()
            return {}

        relay.heartbeat.side_effect = counting_heartbeat

        with _folder_paths({"a.safetensors": "/nope"}), \
                mock.patch.object(worker.loras, "collect", mock.AsyncMock()) as collect, \
                mock.patch.object(worker, "REVOKED_CONFIRM_STRIKES", 1), \
                mock.patch.object(worker, "HEARTBEAT_INTERVAL", 0.01), \
                mock.patch.object(worker, "RelayClient", return_value=relay), \
                mock.patch.object(worker, "ComfyClient"), \
                mock.patch.object(worker, "Worker"), \
                mock.patch.object(worker, "STATUS"), \
                mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_register", mock.AsyncMock()), \
                mock.patch.object(worker, "_abandon_orphans", mock.AsyncMock()), \
                mock.patch.object(worker, "_claim_loop", fake_claim_loop):
            await asyncio.wait_for(
                worker._serve_pairing(pairing, asyncio.Lock(), None,
                                      "http://comfy", None, set()),
                5,
            )

        collect.assert_not_awaited()
        relay.upload_loras.assert_not_awaited()
        self.assertEqual(pairing.loras_synced_at, 0.0)
        self.assertEqual(pairing.loras_hash, "")

    async def test_register_does_not_collect(self):
        # Registration latency must stay independent of the inventory.
        from comfylink.config import Pairing

        relay = mock.AsyncMock()
        comfy = mock.AsyncMock()
        comfy.object_info.return_value = {"A": {}}
        with mock.patch.object(worker.loras, "collect", mock.AsyncMock()) as collect, \
                mock.patch.object(worker, "STATUS"), \
                mock.patch.object(worker, "STATE", _FakeState()):
            await worker._register(relay, comfy,
                                   Pairing(backend_id="b1", device_token="t"))
        collect.assert_not_awaited()
        relay.upload_loras.assert_not_awaited()


# ── relay client ──────────────────────────────────────────────────────────────

class TestRelayLoraEndpoint(unittest.IsolatedAsyncioTestCase):
    async def test_sign_uses_the_contracted_path(self):
        from comfylink.relay import RelayClient

        relay = RelayClient.__new__(RelayClient)

        async def fake_json(method, path, body):
            self.assertEqual(method, "POST")
            self.assertEqual(path, "/v1/backends/loras/sign-put")
            self.assertEqual(body, {"backend_id": "b7"})
            return {"key": "k", "url": "https://r2.example.com/u"}

        relay._json = fake_json
        self.assertEqual(await relay.sign_put_loras("b7"),
                         ("k", "https://r2.example.com/u"))

    async def test_upload_signs_then_puts_json_without_delivery_retry(self):
        from comfylink.relay import RelayClient

        relay = RelayClient.__new__(RelayClient)
        calls = []

        async def fake_sign(backend_id):
            calls.append(("sign", backend_id))
            return "k", "https://r2.example.com/u"

        async def fake_put(url, data, content_type, retry=True):
            calls.append(("put", url, content_type, retry, json.loads(data.decode())))

        relay.sign_put_loras = fake_sign
        relay.put_object = fake_put
        await relay.upload_loras("b7", _MANIFEST)

        self.assertEqual(calls[0], ("sign", "b7"))
        _, url, ctype, retry, body = calls[1]
        self.assertEqual(url, "https://r2.example.com/u")
        self.assertEqual(ctype, "application/json")
        # Not a delivery-critical call: no 90s of backoff on a background task.
        self.assertFalse(retry)
        self.assertEqual(body, _MANIFEST)

    async def test_put_object_still_retries_by_default(self):
        # Guard: adding the `retry` switch must not have changed the delivery path.
        from comfylink.relay import RelayClient

        relay = RelayClient.__new__(RelayClient)
        delivered = []

        async def fake_deliver(what, thunk):
            delivered.append(what)
            return await thunk()

        relay._deliver = fake_deliver
        relay._session = mock.MagicMock()
        put_ctx = mock.MagicMock()
        put_ctx.__aenter__ = mock.AsyncMock(return_value=mock.MagicMock(status=200))
        put_ctx.__aexit__ = mock.AsyncMock(return_value=False)
        relay._session.put = mock.MagicMock(return_value=put_ctx)

        with mock.patch("comfylink.relay._validate_url"):
            await relay.put_object("https://r2.example.com/u", b"x", "application/json")
        self.assertEqual(delivered, ["put_object"])


if __name__ == "__main__":
    unittest.main()
