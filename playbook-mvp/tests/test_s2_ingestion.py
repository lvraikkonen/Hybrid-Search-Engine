"""S2 offline regression; live ES checks are explicit CLI commands."""
import copy
import json
import unittest
from unittest.mock import patch

from documents import CLOSED, SQL, document_record, index_definition, language_record, load_records
from tools.extract_playbook_sql import decode_sql, parse_dump
from tools import ingest_playbook as ingest


class S2IngestionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, cls.audit = load_records()
        cls.by_id = {r["document_id"]: r for r in cls.records}
        cls.tables = parse_dump(decode_sql(SQL.read_bytes())[0])

    def test_all_source_fields_and_acl_provenance(self):
        self.assertEqual(len(self.records), 130)
        self.assertEqual(self.audit["quarantine"], [])
        fixture = json.loads((SQL.parents[1] / "fixtures/document_acl_snapshot.json").read_text())
        expected = {d["document_id"]: d for d in fixture["documents"]}
        for original in self.tables["Document"]:
            r = self.by_id[original["Id"]]
            self.assertEqual(r["lab_release"], CLOSED)
            self.assertIsNone(r["timestamp_timezone"])
            self.assertEqual(r["publication_state"], "unverified")
            self.assertEqual(r["source"]["sql_lines"], original["_sql_lines"])
            self.assertEqual(r["upstream_metadata"]["CreatedTime"], original["CreatedTime"])
            for lang, field in (("zh", "Content"), ("en", "ContentEn")):
                self.assertEqual(r["languages"][lang]["raw"], original[field])
            for field in ("state", "menu_id", "is_public", "allowed_role_ids", "required_business_category", "role_status_raw"):
                self.assertEqual(r["acl"][field], expected[r["document_id"]]["acl"][field])
            self.assertEqual(r["acl_provenance"]["MenuItem_Document"][0]["DocumentId"], r["document_id"])

    def test_real_merged_table_and_language_identity(self):
        r = self.by_id[148]
        self.assertEqual((r["business_category"], r["acl"]["menu_id"]), (1, 174))
        self.assertTrue(r["same_content"])
        self.assertEqual(r["languages"]["zh"]["raw"], r["languages"]["en"]["raw"])
        for lang in ("zh", "en"):
            v = r["languages"][lang]
            row = next(line for line in v["search_text"].splitlines() if "Title[3]：5.5 |" in line)
            self.assertIn("如何签署世纪互联在线服务高级协议", row)
            self.assertIn("合同的生成 Contract Creation", row)
            self.assertIn("Customer[5]：〔空白〕 | Partner[6]：✔️ | Sales[7]：✔️", row)
            self.assertIn('rowspan="8"', v["html"])
            self.assertEqual(len(v["table_warnings"]), 2)
            self.assertTrue(all("rowspan_8_exceeds_7" in w for w in v["table_warnings"]))
        nce = self.by_id[63]["languages"]["zh"]["search_text"]
        self.assertIn("Catalogue[2]：21VCA-E简介 Introduction of 21VCA-E", nce)
        self.assertIn("Catalogue[2]：21VCA-E简介 Introduction of 21VCA-E", nce.splitlines()[5])

    def test_sanitization_steps_and_null_are_separate_from_raw(self):
        raw = '# 标题\r\n\r\n3. first **step**\r\n4. second\r\n\r\n<script>UNSAFE_SCRIPT</script><style>UNSAFE_CSS</style><svg><path d="SVG_PATH"/></svg><img src="https://example.invalid/x" onerror="alert(1)">'
        v = language_record(raw, "Content", "test")
        self.assertEqual(v["raw"], raw)
        self.assertEqual(v["headings"], ["标题"])
        self.assertIn("3. first step", v["search_text"])
        self.assertIn("4. second", v["search_text"])
        for marker in ("UNSAFE_SCRIPT", "UNSAFE_CSS", "SVG_PATH", "onerror=", "<img"):
            self.assertNotIn(marker, v["html"])
            self.assertNotIn(marker, v["search_text"])
        self.assertIsNone(language_record(None, "ContentEn", "null")["raw_utf8_sha256"])
        self.assertIsNone(language_record(None, "ContentEn", "null")["raw"])
        self.assertEqual(language_record("", "ContentEn", "empty")["raw"], "")

    def test_ambiguous_acl_never_gets_active_roles(self):
        original = next(d for d in self.tables["Document"] if d["Id"] == 25)
        variants = []
        t = copy.deepcopy(self.tables)
        t["MenuItem_Document"].append({"Id": 9999, "DocumentId": 25, "MenuItemId": 114, "_sql_lines": [1, 1]})
        variants.append(t)
        for field, value in (("Status", None), ("Status", True), ("Id", 4)):
            t = copy.deepcopy(self.tables)
            role = next(r for r in t["Role"] if r["Id"] == 2)
            role[field] = value
            variants.append(t)
        t = copy.deepcopy(self.tables)
        next(m for m in t["MenuItem"] if m["Id"] == 31)["UnderControl"] = True
        variants.append(t)
        for t in variants:
            r = document_record(original, t, "test-snapshot")
            self.assertEqual(r["acl"]["state"], "unresolved")
            self.assertEqual(r["acl"]["active_role_ids"], [])
            self.assertEqual(r["lab_release"], CLOSED)

    def test_identical_import_preserves_release_changed_source_closes(self):
        r = copy.deepcopy(self.by_id[25])
        approved = copy.deepcopy(r)
        approved["lab_release"] = {"approved": True, "source_snapshot_sha256": r["source"]["snapshot_sha256"], "approved_by": "test", "approved_at": "2026-09-29T00:00:00Z"}
        current = {r["document_key"]: {"_source": approved, "_seq_no": 7, "_primary_term": 1}}
        self.assertEqual(ingest.import_actions([r], current), ([], 1))
        r["source"]["snapshot_sha256"] = "changed-source"
        actions, unchanged = ingest.import_actions([r], current)
        self.assertEqual(unchanged, 0)
        self.assertEqual(actions[0]["index"]["if_seq_no"], 7)
        self.assertEqual(actions[1]["lab_release"], CLOSED)
        self.assertTrue(approved["lab_release"]["approved"])
        self.assertEqual(r["lab_release"], CLOSED)

    def test_bulk_item_errors_are_not_hidden_by_http_200(self):
        response = {"errors": True, "items": [
            {"create": {"_id": "ok", "status": 201}},
            {"index": {"_id": "bad", "status": 409, "error": {"type": "version_conflict_engine_exception"}}},
        ]}
        with patch.object(ingest.es, "request", return_value=response) as request:
            result = ingest.bulk([{"create": {"_id": "ok"}}, {}, {"index": {"_id": "bad"}}, {}])
        self.assertEqual(result["written"], 1)
        self.assertEqual(result["errors"], [{"operation": "index", "document_key": "bad", "status": 409, "error_type": "version_conflict_engine_exception"}])
        payload = request.call_args.args[3]
        self.assertTrue(payload.endswith(b"\n"))
        self.assertEqual(len(payload.splitlines()), 4)

    def test_unreviewed_synonyms_and_wrong_release_snapshot_fail_before_es(self):
        with self.assertRaisesRegex(ValueError, "synonyms_require"):
            index_definition({"review_status": "pending", "approved_rules": ["cancel, delete"]})
        with patch.object(ingest.es, "request") as request:
            with self.assertRaisesRegex(RuntimeError, "Approval snapshot differs"):
                ingest.release(self.records, self.audit, "wrong-snapshot", "test", {})
            request.assert_not_called()


if __name__ == "__main__":
    unittest.main()
