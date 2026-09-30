"""Offline guards complement tools/check_s6.py's isolated real ES/HTTP checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

from api import APIError, APIServer
from runtime import CatalogStore, atomic_json, legacy_setup
from tools import evaluate_search as evaluate, sync_playbook as sync
from tools.check_s6 import synthetic_snapshot, BOTH


class S6Checks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = CatalogStore(True, self.temp.name)

    def test_frozen_metrics_exclude_inaccessible_and_do_not_judge_unknowns(self):
        self.assertIsNone(evaluate.score([25], {}))
        self.assertEqual(evaluate.score([9, 25], {"25": 2, "99": 1}),
                         {"hit_at_5": 1, "mrr_at_5": .5, "judged_recall_at_5": .5})
        self.assertEqual(evaluate.score([9] * 5 + [25], {"25": 2})["hit_at_5"], 0)
        approved = evaluate.reviewed_golden()
        self.assertEqual(approved["hit_at_5_target_per_profile"], .9)
        approved["questions"][0]["by_profile"]["pb_msft_01"] = {}
        path = Path(self.temp.name) / "tampered.json"
        atomic_json(path, approved)
        with patch.object(evaluate, "GOLDEN", path), self.assertRaises(ValueError):
            evaluate.reviewed_golden()
        self.assertEqual(evaluate.latency(list(range(1, 101)))["p95_ms"], 95)

    def test_snapshot_validation_before_any_es_and_real_export_is_supported(self):
        original = sync.export_snapshot(CatalogStore(False, self.temp.name), sync.SQL, True)
        records, snapshot = sync.validate_snapshot(original, CatalogStore(False, self.temp.name))
        self.assertEqual(len(records), 130)
        self.assertNotEqual(snapshot, original["origin"]["sql_sha256"])
        batch = synthetic_snapshot(self.store)
        variants = []
        for change in (lambda b: b.update(complete=False), lambda b: b["tables"].pop("Role"),
                       lambda b: b["tables"]["Role_MenuItem"].append(b["tables"]["Role_MenuItem"][0]),
                       lambda b: b["profiles"][BOTH]["authentication_info"]["Roles"].append({"Id": 1, "Name": "MSFT"}),
                       lambda b: b["tables"]["MenuItem"][0].update(UnderControl=True)):
            value = copy.deepcopy(batch)
            change(value)
            variants.append(value)
        with patch.object(sync.es, "request") as request:
            for variant in variants:
                with self.assertRaises((RuntimeError, ValueError, KeyError)):
                    sync.prepare(self.store, variant)
            request.assert_not_called()

    def test_maintenance_drains_existing_reader_and_rejects_new_ones(self):
        entered, finished = threading.Event(), threading.Event()
        def writer():
            with self.store.maintenance("a" * 64):
                entered.set()
            finished.set()
        with self.store.read():
            thread = threading.Thread(target=writer)
            thread.start()
            for _ in range(100):
                if self.store.gate.exists():
                    break
                finished.wait(.01)
            self.assertFalse(entered.is_set())
            with self.assertRaises(RuntimeError), self.store.read():
                pass
        thread.join(3)
        self.assertTrue(entered.is_set() and finished.is_set())
        self.assertFalse(self.store.gate.exists())

    def test_process_death_keeps_gate_until_same_plan_retry(self):
        script = "from runtime import CatalogStore; import os, sys\ns=CatalogStore(True,sys.argv[1])\nwith s.maintenance('a'*64): os._exit(7)\n"
        result = subprocess.run([sys.executable, "-B", "-c", script, self.temp.name], cwd=sync.ROOT)
        self.assertEqual(result.returncode, 7)
        with self.assertRaises(RuntimeError), self.store.read():
            pass
        with self.assertRaises(RuntimeError), self.store.maintenance("b" * 64):
            pass
        with self.store.maintenance("a" * 64) as recovering:
            self.assertTrue(recovering)
        self.assertFalse(self.store.gate.exists())

    def test_generation_invalidates_existing_cookie_and_legacy_writes(self):
        server = APIServer(0, self.store)
        self.addCleanup(server.server_close)
        server.sessions["test"] = {"generation": "old", "expires": float("inf")}
        with self.assertRaises(APIError) as error:
            server.session("test", "new")
        self.assertEqual(error.exception.status, 401)
        self.assertNotIn("test", server.sessions)
        atomic_json(self.store.directory / "active.json", {"generation": "new"})
        with patch("runtime.CatalogStore", return_value=self.store), self.assertRaises(RuntimeError), legacy_setup():
            pass

    def test_content_rollback_never_restores_policy_or_missing_documents(self):
        old = synthetic_snapshot(self.store)
        latest = copy.deepcopy(old)
        latest["tables"]["Document"] = latest["tables"]["Document"][:1]
        latest["tables"]["Document"][0]["Content"] = "New content"
        latest["tables"]["Role_MenuItem"] = []
        latest["withdrawn_document_ids"] = [99001]
        latest["profiles"][BOTH]["authentication_info"]["BusinessCategories"] = [0]
        for key, batch in (("a" * 64, old), ("b" * 64, latest)):
            atomic_json(self.store.directory / "batches" / (key + ".json"), batch)
        atomic_json(self.store.directory / "active.json", {"batch": "b" * 64})
        rolled = sync.content_rollback(self.store, "a" * 64)
        self.assertEqual(len(rolled["tables"]["Document"]), 1)
        self.assertEqual(rolled["tables"]["Document"][0]["Content"], old["tables"]["Document"][0]["Content"])
        for key in ("profiles", "withdrawn_document_ids"):
            self.assertEqual(rolled[key], latest[key])
        self.assertEqual(rolled["tables"]["Role_MenuItem"], [])

    def test_http_200_partial_bulk_and_expired_license_do_not_publish(self):
        with patch.object(sync, "maintenance_request", return_value={"errors": True, "items": [{"create": {"status": 409}}]}):
            with self.assertRaisesRegex(RuntimeError, "Bulk incomplete"):
                sync.write_batch(self.store, {}, {"new": {"x": 1}})
        batch = synthetic_snapshot(self.store)
        with patch.object(sync, "sources", return_value={}):
            plan, _ = sync.prepare(self.store, batch)
            with patch.object(sync, "maintenance_request", return_value={"license": {"status": "expired", "type": "trial"}}), patch.object(sync.es, "request") as request:
                with self.assertRaisesRegex(RuntimeError, "DLS license unavailable"):
                    sync.apply(self.store, plan, "offline-check", True)
                request.assert_not_called()
        self.assertFalse(self.store.gate.exists())
        self.assertIsNone(self.store.current())

    def test_rollback_prepared_after_another_policy_change_is_rejected(self):
        batch = synthetic_snapshot(self.store)
        batch["origin"] = {"kind": "content_only_rollback", "policy_batch": "a" * 64}
        active = {"batch": "b" * 64}
        with patch.object(sync, "sources", return_value={}), patch.object(sync, "current_catalog", return_value=active):
            with self.assertRaisesRegex(RuntimeError, "Rollback policy changed"):
                sync.prepare(self.store, batch)


if __name__ == "__main__":
    unittest.main()
