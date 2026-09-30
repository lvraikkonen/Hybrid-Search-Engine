"""Actual loopback HTTP with a fault-injected ES transport; live DLS is checked separately."""
import copy
from concurrent.futures import ThreadPoolExecutor
import io
import json
import os
from pathlib import Path
import ssl
import sys
import threading
import time
import unittest
from unittest.mock import patch
from urllib.error import URLError
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import api
from documents import load_records
from tools.check_api import call, expect, login


class S4APIChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = {doc["document_key"]: doc for doc in load_records()[0]}

    def setUp(self):
        self.environment = patch.dict(os.environ, APP_ENV="test", MOCK_AUTH_ENABLED="1")
        self.environment.start()
        self.addCleanup(self.environment.stop)
        self.server = api.APIServer(0)
        self.server.audit_stream = io.StringIO()
        self.port = self.server.server_port
        self.worker = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.worker.start()
        self.addCleanup(self.stop_server)
        self.secret = patch.object(api.permissions, "credential", return_value="unit-test-only-not-a-credential")
        self.secret.start()
        self.addCleanup(self.secret.stop)
        self.transport = patch.object(api.es, "request", side_effect=self.fake_es)
        self.transport.start()
        self.addCleanup(self.transport.stop)
        self.calls = []
        self.fault = None
        self.auth_override = {}
        self.privilege_override = None
        self.raw_override = {}
        self.started, self.release = None, None

    def stop_server(self):
        self.server.shutdown()
        self.worker.join()
        self.server.server_close()

    def fake_es(self, user, method, path, body=None, expected=200, *, password_value=None):
        self.assertTrue(user.startswith("pb_"))
        self.assertEqual(password_value, "unit-test-only-not-a-credential")
        self.calls.append((user, method, path, body))
        principal = api.permissions.principals()[user]
        if path == "/_security/_authenticate":
            return {"username": user, "roles": [api.permissions.READER], "enabled": True,
                    "authentication_realm": {"type": "native"}, "metadata": principal.es_metadata_candidate(), **self.auth_override}
        if path == "/_security/user/_privileges":
            if self.privilege_override is not None:
                return self.privilege_override
            return {"cluster": [], "global": [], "run_as": [], "applications": [], "indices": [
                {**item, "query": [json.dumps(item["query"])]} for item in api.permissions.reader_role(self.server.snapshot)["indices"]]}
        if self.started:
            self.started.set()
            if not self.release.wait(10):
                raise RuntimeError("Test release timed out")
        if isinstance(self.fault, Exception):
            raise self.fault
        if self.fault is not None:
            return self.fault
        if "/_doc/" in path:
            key = unquote(path.rsplit("/", 1)[1])
            if key not in self.sources or user == "pb_partner_nce_01" and key.endswith(":99"):
                return {"found": False}
            doc = copy.deepcopy(self.sources[key])
            for language, value in self.raw_override.items():
                doc["languages"][language]["raw"] = value
            return {"found": True, "_source": doc}
        return {"timed_out": False, "_shards": {"failed": 0}, "took": 1, "hits": {"total": {"value": 0, "relation": "eq"}, "hits": []}}

    def test_session_expiry_disabled_mock_and_strict_inputs(self):
        cookie, revision = login(self.port, "pb_partner_nce_01")
        expect(call(self.port, "POST", "/api/dev/session", {"profile_id": "elastic"}, cookie), 400)
        expect(call(self.port, "POST", "/api/dev/session", {"profile_id": "pb_msft_01"}, cookie, {"Origin": "null"}), 403)
        expect(call(self.port, "POST", "/api/search", raw='{"query":"x","query":"y"}', cookie=cookie), 400)
        expect(call(self.port, "POST", "/api/search", {"query": "x", "business_scope": True}, cookie), 400)
        expect(call(self.port, "POST", "/api/search", {"query": "x", "role_ids": [1]}, cookie), 400)
        with self.server.lock:
            self.server.sessions[cookie.split("=", 1)[1]]["expires"] = time.monotonic() - 1
        expect(call(self.port, "GET", "/api/me", cookie=cookie), 401)
        self.server.enabled = False
        calls = len(self.calls)
        expect(call(self.port, "GET", "/api/dev/profiles"), 404)
        expect(call(self.port, "POST", "/api/dev/session", {"profile_id": "pb_msft_01"}), 404)
        expect(call(self.port, "GET", "/api/me", cookie=cookie), 401)
        self.assertEqual(len(self.calls), calls)
        with patch.dict(os.environ, APP_ENV="production", MOCK_AUTH_ENABLED="1"):
            with api.APIServer(0) as production:
                self.assertFalse(production.enabled)

    def test_faults_are_errors_on_all_read_routes_without_fallback(self):
        faults = [(api.es.ESHTTPError(403, "license fault secret payload"), 503),
                  (api.es.ESHTTPError(500, "server fault secret payload"), 503),
                  (TimeoutError("secret payload"), 504), (URLError(TimeoutError()), 504),
                  (URLError(ConnectionRefusedError()), 503), (ssl.SSLCertVerificationError(), 503)]
        cookie, revision = login(self.port, "pb_partner_ospa_01")
        for fault, status in faults:
            self.fault = fault
            for method, path, body in (("POST", "/api/search", {"query": "must-not-be-logged"}), ("GET", "/api/menu", None),
                         ("GET", "/api/documents/playbook-test:Document:99", None),
                         ("GET", "/api/documents/playbook-test:Document:99/raw", None),
                         ("GET", "/api/documents/playbook-test:Document:99/download", None)):
                with self.subTest(status=status, path=path, fault=type(fault).__name__):
                    _, body = expect(call(self.port, method, path, body, cookie), status)
                    self.assertNotIn("secret", json.dumps(body))
        for fault, status in (({"timed_out": True}, 504), ({"timed_out": False, "_shards": {"failed": 1}}, 503)):
            self.fault = fault
            expect(call(self.port, "POST", "/api/search", {"query": "合同"}, cookie), status)
        self.assertNotIn("must-not-be-logged", self.server.audit_stream.getvalue())
        self.assertNotIn("secret", self.server.audit_stream.getvalue())
        self.assertNotIn(cookie, self.server.audit_stream.getvalue())
        self.assertEqual({c[0] for c in self.calls}, {"pb_partner_ospa_01"})
        self.assertTrue(all(not path.startswith("/playbook-s3") for _, _, path, _ in self.calls))

    def test_native_identity_credential_and_dls_drift_fail_closed(self):
        for change in ({"username": "elastic"}, {"roles": [api.permissions.READER, "superuser"]}, {"metadata": {}}, {"enabled": False}):
            cookie, _ = login(self.port, "pb_partner_ospa_01")
            self.auth_override = change
            before = len(self.calls)
            expect(call(self.port, "GET", "/api/documents/playbook-test:Document:99", cookie=cookie), 401)
            self.assertEqual(len(self.calls), before + 1)
            self.auth_override = {}
        cookie, _ = login(self.port, "pb_partner_ospa_01")
        self.privilege_override = {"indices": [{"names": [api.INDEX], "privileges": ["read"]}], "cluster": []}
        expect(call(self.port, "GET", "/api/menu", cookie=cookie), 503)
        self.privilege_override = None
        with patch.object(api.permissions, "credential", side_effect=FileNotFoundError()):
            before = len(self.calls)
            expect(call(self.port, "GET", "/api/me", cookie=cookie), 401)
            self.assertEqual(len(self.calls), before)
        cookie, _ = login(self.port, "pb_partner_ospa_01")
        self.fault = api.es.ESHTTPError(401, "credential expired")
        expect(call(self.port, "GET", "/api/documents/playbook-test:Document:99/raw", cookie=cookie), 401)

    def test_switch_and_logout_discard_in_flight_body(self):
        for action in ("switch", "logout"):
            cookie, old_revision = login(self.port, "pb_partner_ospa_01")
            self.started, self.release = threading.Event(), threading.Event()
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(call, self.port, "GET", "/api/documents/playbook-test:Document:99/raw", None, cookie)
                try:
                    self.assertTrue(self.started.wait(5))
                    if action == "switch":
                        new_cookie, revision = login(self.port, "pb_partner_nce_01", cookie)
                        self.assertNotEqual(revision, old_revision)
                    else:
                        expect(call(self.port, "DELETE", "/api/session", cookie=cookie), 204)
                finally:
                    self.release.set()
                _, value = expect(future.result(timeout=10), 401)
                self.assertEqual(set(value), {"request_id", "error"})
            self.started = None
            expect(call(self.port, "GET", "/api/me", cookie=cookie), 401)
            if action == "switch":
                expect(call(self.port, "GET", "/api/documents/playbook-test:Document:99/raw", cookie=new_cookie), 404)

    def test_null_empty_raw_are_distinct_and_attachment_safe(self):
        cookie, _ = login(self.port, "pb_partner_ospa_01")
        self.raw_override = {"zh": None, "en": ""}
        _, detail = expect(call(self.port, "GET", "/api/documents/playbook-test:Document:99", cookie=cookie))
        self.assertEqual(set(detail), {"request_id", "session_revision", "document"})
        self.assertTrue(detail["document"]["languages"]["zh"]["raw_is_null"])
        self.assertTrue(detail["document"]["languages"]["en"]["raw_is_empty"])
        for route in ("raw", "download"):
            headers, body = expect(call(self.port, "GET", f"/api/documents/playbook-test:Document:99/{route}?language=zh", cookie=cookie), 204)
            self.assertEqual(headers["x-source-field-null"], "true")
            self.assertEqual(body, b"")
            headers, body = expect(call(self.port, "GET", f"/api/documents/playbook-test:Document:99/{route}?language=en", cookie=cookie))
            self.assertNotIn("x-source-field-null", headers)
            self.assertEqual(body, b"")
        self.raw_override = {"zh": "<script>alert('test')</script>"}
        headers, body = expect(call(self.port, "GET", "/api/documents/playbook-test:Document:99/download", cookie=cookie))
        self.assertEqual(headers["content-type"], "text/plain; charset=utf-8")
        self.assertTrue(headers["content-disposition"].startswith("attachment;"))
        self.assertIn("sandbox", headers["content-security-policy"])

    def test_hidden_parent_menu_and_bm25_language_structure(self):
        cookie, _ = login(self.port, "pb_partner_nce_01")
        self.server.tables = {"Role": [{"Id": 2, "Status": 1}], "Role_MenuItem": [], "MenuItem": [
            {"Id": 10, "ParentId": 0, "Name": "hidden-parent", "NameEn": "hidden-parent-en", "Sort": 1,
             "Type": 1, "Active": 1, "UnderControl": 1, "BusinessCategory": 1},
            {"Id": 11, "ParentId": 10, "Name": "public-child", "NameEn": "public-child-en", "Sort": 2,
             "Type": 0, "Active": 1, "UnderControl": 0, "BusinessCategory": 1},
            {"Id": 12, "ParentId": 10, "Name": "hidden-sibling", "NameEn": "hidden-sibling-en", "Sort": 3,
             "Type": 0, "Active": 1, "UnderControl": 1, "BusinessCategory": 1}]}
        self.fault = {"timed_out": False, "_shards": {"failed": 0}, "hits": {"total": {"value": 1, "relation": "eq"}, "hits": [
            {"_source": {"document_key": "synthetic-public-child", "acl": {"menu_id": 11}, "source": {"snapshot_sha256": self.server.snapshot}}}]}}
        _, result = expect(call(self.port, "GET", "/api/menu", cookie=cookie))
        self.assertEqual([n["menu_id"] for n in result["nodes"]], [10, 11])
        self.assertTrue(result["nodes"][0]["placeholder"])
        self.assertIsNone(result["nodes"][0]["business_category"])
        self.assertIsNone(result["nodes"][0]["is_public"])
        self.assertTrue(result["nodes"][1]["is_public"])
        self.assertEqual(result["nodes"][1]["business_category"], 1)
        self.assertEqual(result["nodes"][0]["document_keys"], [])
        self.assertEqual(result["nodes"][0]["visible_document_count"], 1)
        self.assertNotIn("hidden-", json.dumps(result))
        self.fault = None
        expect(call(self.port, "POST", "/api/search", {"query": "RI", "business_scope": 1}, cookie))
        body = self.calls[-1][3]
        query = body["query"]["bool"]
        self.assertEqual(query["must"][0]["dis_max"]["tie_breaker"], 0)
        self.assertEqual([q["multi_match"]["_name"] for q in query["must"][0]["dis_max"]["queries"]], ["zh", "en"])
        self.assertEqual(query["filter"], [{"term": {"business_category": 1}}])
        self.assertFalse(body["track_total_hits"])
        self.assertEqual(body["highlight"]["pre_tags"], ["<mark>"])
        self.assertEqual(body["highlight"]["encoder"], "html")
        self.assertEqual(body["sort"], [{"_score": "desc"}, {"document_id": "asc"}])
        self.assertNotIn("roles", json.dumps(body))

    def test_search_scores_and_highlights_preserve_text_without_executable_markup(self):
        cookie, _ = login(self.port, "pb_partner_nce_01")
        hit = {"_source": self.sources["playbook-test:Document:25"], "_score": 7.125,
               "matched_queries": ["zh"], "highlight": {
                   "title": ["<mark>合同</mark>的生成"],
                   "languages.zh.search_text": ["&lt;img src=x onerror=alert(1)&gt; &amp; <mark>合同</mark> &lt;mark&gt;原样&lt;/mark&gt; 😀"]}}
        self.fault = {"timed_out": False, "_shards": {"failed": 0}, "took": 1, "hits": {"hits": [hit]}}
        _, response = expect(call(self.port, "POST", "/api/search", {"query": "合同", "language": "zh"}, cookie))
        item = response["items"][0]
        self.assertEqual(item["score"], 7.125)
        self.assertEqual(item["title_highlights"]["title"], [{"text": "合同", "highlighted": True}, {"text": "的生成", "highlighted": False}])
        snippet = item["snippets"][0]
        self.assertEqual(snippet["text"], '<img src=x onerror=alert(1)> & 合同 <mark>原样</mark> 😀')
        self.assertEqual("".join(p["text"] for p in snippet["parts"]), snippet["text"])
        self.assertEqual([p["text"] for p in snippet["parts"] if p["highlighted"]], ["合同"])
        hit["highlight"]["languages.zh.search_text"] = ["😀" * 299 + "<mark>合同</mark>"]
        _, response = expect(call(self.port, "POST", "/api/search", {"query": "合同"}, cookie))
        self.assertEqual(response["items"][0]["snippets"][0]["parts"][-1], {"text": "合", "highlighted": True})
        for score in (None, True, -1, float("nan"), float("inf")):
            hit["_score"] = score
            expect(call(self.port, "POST", "/api/search", {"query": "合同"}, cookie), 503)


if __name__ == "__main__":
    unittest.main()
