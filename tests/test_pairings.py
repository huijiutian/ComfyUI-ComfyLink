"""Tests for one-ComfyUI → many-account pairing.

A single machine can be paired to several accounts at once. Each pairing carries
its own backend_id + device token; the supervisor runs one service task per
pairing, and one global lock serializes generations across accounts on the lone
local GPU. These tests pin:

  * config.State pairing list — add/get/remove, persistence round-trip, account
    NOT persisted, and back-compat load of the OLD single-pairing state file;
  * the Pairing model defaults;
  * worker._reconcile — start a task per pairing, cancel one whose pairing was
    removed, and never restart a pairing that revoked itself;
  * worker._run_locked — the global job lock lets only one generation run at a
    time even when several accounts have a job ready.

Run:  python -m unittest discover -s tests
"""

import asyncio
import json
import os
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from comfylink import worker  # noqa: E402
from comfylink.config import Pairing, State  # noqa: E402


class _TempState(unittest.TestCase):
    """Point COMFYLINK_STATE at a throwaway file so save() never touches the
    real on-disk state."""

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._path = os.path.join(self._dir, "state.json")
        self._prev = os.environ.get("COMFYLINK_STATE")
        os.environ["COMFYLINK_STATE"] = self._path

    def tearDown(self):
        if self._prev is None:
            os.environ.pop("COMFYLINK_STATE", None)
        else:
            os.environ["COMFYLINK_STATE"] = self._prev

    def _write(self, data: dict):
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(data, f)

    def _read(self) -> dict:
        with open(self._path, encoding="utf-8") as f:
            return json.load(f)


class TestPairingModel(unittest.TestCase):
    def test_defaults(self):
        pr = Pairing(backend_id="b1", device_token="t1")
        self.assertEqual(pr.device_id, "")
        self.assertEqual(pr.account, "")
        self.assertEqual(pr.object_info_hash, "")


class TestStatePairings(_TempState):
    def test_add_get_remove(self):
        st = State()
        self.assertFalse(st.paired)
        p1 = st.add_pairing("t1", "d1")
        p2 = st.add_pairing("t2", "d2")
        self.assertTrue(st.paired)
        self.assertEqual(len(st.pairings), 2)
        # Each pairing gets its own backend_id (one per account).
        self.assertNotEqual(p1.backend_id, p2.backend_id)
        self.assertIs(st.get_pairing(p1.backend_id), p1)
        self.assertIsNone(st.get_pairing("nope"))
        removed = st.remove_pairing(p1.backend_id)
        self.assertIs(removed, p1)
        self.assertEqual([p.backend_id for p in st.pairings], [p2.backend_id])
        # Removing an unknown id is a no-op returning None.
        self.assertIsNone(st.remove_pairing("nope"))

    def test_persistence_roundtrip_two_pairings(self):
        st = State()
        st.backend_name = "rig"
        p1 = st.add_pairing("t1", "d1")
        p2 = st.add_pairing("t2", "d2")
        p1.object_info_hash = "h1"
        st.save()
        again = State.load()
        self.assertEqual(again.backend_name, "rig")
        self.assertEqual({p.backend_id for p in again.pairings},
                         {p1.backend_id, p2.backend_id})
        loaded1 = again.get_pairing(p1.backend_id)
        self.assertEqual(loaded1.device_token, "t1")
        self.assertEqual(loaded1.device_id, "d1")
        self.assertEqual(loaded1.object_info_hash, "h1")

    def test_account_is_never_persisted(self):
        st = State()
        pr = st.add_pairing("t1", "d1")
        pr.account = "alice@example.com"
        st.save()
        self.assertNotIn("account", self._read()["pairings"][0])
        again = State.load()
        self.assertEqual(again.pairings[0].account, "")  # in-memory only

    def test_backcompat_old_single_format(self):
        # Old state file: top-level backend_id/device_token (one pairing).
        self._write({
            "backend_id": "old-b",
            "device_token": "old-t",
            "device_id": "old-d",
            "backend_name": "oldrig",
            "object_info_hash": "oldh",
        })
        st = State.load()
        self.assertEqual(st.backend_name, "oldrig")
        self.assertEqual(len(st.pairings), 1)
        pr = st.pairings[0]
        self.assertEqual(pr.backend_id, "old-b")
        self.assertEqual(pr.device_token, "old-t")
        self.assertEqual(pr.device_id, "old-d")
        self.assertEqual(pr.object_info_hash, "oldh")

    def test_backcompat_old_unpaired_loads_zero_pairings(self):
        # Old file from an unpaired machine: had a backend_id but no token.
        self._write({"backend_id": "old-b", "device_token": "", "backend_name": "m"})
        st = State.load()
        self.assertFalse(st.paired)
        self.assertEqual(st.pairings, [])
        self.assertEqual(st.backend_name, "m")  # name still honored

    def test_missing_file_loads_empty(self):
        st = State.load()  # nothing written at the temp path yet
        self.assertFalse(st.paired)
        self.assertEqual(st.pairings, [])


class _SupervisorState:
    """Stand-in for config.STATE with the lookups _reconcile/_serve_pairing use."""

    def __init__(self, pairings):
        self.pairings = list(pairings)
        self.backend_name = "dev"

    def get_pairing(self, backend_id):
        return next((p for p in self.pairings if p.backend_id == backend_id), None)

    def remove_pairing(self, backend_id):
        pr = self.get_pairing(backend_id)
        self.pairings = [p for p in self.pairings if p.backend_id != backend_id]
        return pr

    def save(self):
        pass


class TestReconcile(unittest.IsolatedAsyncioTestCase):
    async def test_starts_per_pairing_and_cancels_removed(self):
        p1 = Pairing(backend_id="b1", device_token="t1")
        p2 = Pairing(backend_id="b2", device_token="t2")
        state = _SupervisorState([p1, p2])
        started = []

        async def fake_serve_pairing(pairing, job_lock, session, comfy_url, stop, swept):
            started.append(pairing.backend_id)
            await asyncio.sleep(3600)  # block until cancelled

        tasks: dict = {}
        lock = asyncio.Lock()
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_serve_pairing", fake_serve_pairing):
            worker._reconcile(tasks, lock, None, "http://c", None, set())
            await asyncio.sleep(0.01)  # let the tasks start
            self.assertEqual(set(tasks), {"b1", "b2"})
            self.assertEqual(sorted(started), ["b1", "b2"])
            # Drop b1; reconcile must cancel + remove its task, keep b2.
            state.pairings = [p2]
            worker._reconcile(tasks, lock, None, "http://c", None, set())
            await asyncio.sleep(0.01)
            self.assertEqual(set(tasks), {"b2"})
            # cleanup
            for t in tasks.values():
                t.cancel()
            await asyncio.gather(*tasks.values(), return_exceptions=True)

    async def test_does_not_restart_a_revoked_pairing(self):
        # A pairing that removed itself (revoked) is absent from STATE, so a later
        # reconcile must NOT recreate its task.
        p1 = Pairing(backend_id="b1", device_token="t1")
        state = _SupervisorState([p1])
        starts = {"n": 0}

        async def fake_serve_pairing(pairing, job_lock, session, comfy_url, stop, swept):
            starts["n"] += 1
            state.remove_pairing(pairing.backend_id)  # simulate revoke + return

        tasks: dict = {}
        with mock.patch.object(worker, "STATE", state), \
                mock.patch.object(worker, "_serve_pairing", fake_serve_pairing):
            worker._reconcile(tasks, asyncio.Lock(), None, "http://c", None, set())
            await asyncio.sleep(0.01)  # task runs, removes self
            worker._reconcile(tasks, asyncio.Lock(), None, "http://c", None, set())
            await asyncio.sleep(0.01)
            self.assertEqual(tasks, {})
            self.assertEqual(starts["n"], 1)


class TestRunLockedSerializes(unittest.IsolatedAsyncioTestCase):
    async def test_one_job_at_a_time_across_accounts(self):
        lock = asyncio.Lock()
        live = {"cur": 0, "max": 0}

        class _W:
            async def handle_job(self, job):
                live["cur"] += 1
                live["max"] = max(live["max"], live["cur"])
                await asyncio.sleep(0.02)
                live["cur"] -= 1
                return "done"

        with mock.patch.object(worker, "STATUS"):
            await asyncio.gather(
                worker._run_locked(lock, _W(), {"id": "1"}),
                worker._run_locked(lock, _W(), {"id": "2"}),
                worker._run_locked(lock, _W(), {"id": "3"}),
            )
        # The global lock guarantees the three never overlap on the single GPU.
        self.assertEqual(live["max"], 1)


if __name__ == "__main__":
    unittest.main()
