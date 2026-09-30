#!/usr/bin/env python3
"""S4 real HTTP + native ES checks; no browser or relevance-quality claim."""
from concurrent.futures import ThreadPoolExecutor
from http.client import HTTPConnection
import json
import math
from pathlib import Path
import sys
import threading

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from api import APIServer
from documents import load_records
from tools import es_permissions as permissions


def call(port, method, path, body=None, cookie=None, headers=None, raw=None):
    connection = HTTPConnection("127.0.0.1", port, timeout=40)
    request_headers = {"Origin": f"http://127.0.0.1:{port}"}
    if cookie:
        request_headers["Cookie"] = cookie
    if body is not None or raw is not None:
        request_headers["Content-Type"] = "application/json"
    request_headers.update(headers or {})
    try:
        connection.request(method, path, raw if raw is not None else json.dumps(body) if body is not None else None, request_headers)
        response = connection.getresponse()
        data = response.read()
        response_headers = {k.lower(): v for k, v in response.getheaders()}
        value = json.loads(data) if data and "application/json" in response_headers.get("content-type", "") else data
        return response.status, response_headers, value
    finally:
        connection.close()


def check(condition, message):
    if not condition:
        raise AssertionError(message)


def expect(response, status=200):
    code, headers, body = response
    check(code == status, f"HTTP {code}, expected {status}")
    check(headers.get("cache-control") == "no-store" and headers.get("x-content-type-options") == "nosniff", "Unsafe response headers")
    check(bool(headers.get("x-request-id")), "Missing request ID")
    if isinstance(body, dict):
        check(body["request_id"] == headers["x-request-id"], "Request ID mismatch")
    if status >= 400:
        check(set(body) == {"request_id", "error"} and set(body["error"]) == {"code", "message"}, "Error leaks internal data")
    return headers, body


def login(port, profile_id, cookie=None):
    headers, body = expect(call(port, "POST", "/api/dev/session", {"profile_id": profile_id}, cookie))
    check("HttpOnly" in headers["set-cookie"] and "SameSite=Strict" in headers["set-cookie"], "Unsafe cookie")
    check(body["principal"]["profile_id"] == profile_id, "Wrong session identity")
    return headers["set-cookie"].split(";", 1)[0], body["session_revision"]


def check_items(items, allowed, sources):
    check(len(items) <= 20 and len({i["document_id"] for i in items}) == len(items), "Invalid document-level K")
    scores = [item["score"] for item in items]
    check(all(type(score) in (int, float) and math.isfinite(score) and score >= 0 for score in scores), "Invalid BM25 score")
    check(scores == sorted(scores, reverse=True), "Results are not sorted by descending score")
    check([(i["score"], -i["document_id"]) for i in items] ==
          sorted(((i["score"], -i["document_id"]) for i in items), reverse=True), "Score ties are not ordered by document ID")

    def check_parts(text, parts):
        check(all(set(p) == {"text", "highlighted"} and isinstance(p["text"], str) and type(p["highlighted"]) is bool
                  for p in parts) and "".join(p["text"] for p in parts) == text, "Highlight changed source text")

    for item in items:
        did = item["document_id"]
        check(did in allowed, f"Forbidden search result {did}")
        check(set(item) == {"document_key", "document_id", "title", "title_en", "business_category", "is_public", "source",
                            "content_revision", "acl_revision", "matched_language", "snippets", "score", "title_highlights"}, "Unsafe search projection")
        check(set(item["source"]) == {"kb_id", "instance_key", "table", "primary_key", "snapshot_sha256"}, "Unsafe source projection")
        check(item["matched_language"] in ("zh", "en", "both"), "Wrong language")
        check(item["title"] == sources[did]["title"], "Title changed")
        check(set(item["title_highlights"]) <= {"title", "title_en"}, "Unexpected title highlight field")
        for field, parts in item["title_highlights"].items():
            check_parts(item[field] or "", parts)
        for snippet in item["snippets"]:
            lang = "zh" if snippet["source_field"] == "Content" else "en"
            check(snippet["source_field"] in ("Content", "ContentEn") and snippet["text"] in sources[did]["languages"][lang]["search_text"], "Excerpt not source evidence")
            check_parts(snippet["text"], snippet["parts"])


def run(server, report):
    port = server.server_port
    records, audit = load_records()
    sources = {r["document_id"]: r for r in records}
    oracle = json.loads((ROOT / "fixtures/expected_access.json").read_text())
    questions = json.loads((ROOT / "evaluation/candidate_questions.json").read_text())
    check(audit["source_snapshot_sha256"] == server.snapshot == oracle["source"]["sha256"], "Source snapshot mismatch")
    profiles = expect(call(port, "GET", "/api/dev/profiles"))[1]["profiles"]
    check(len(profiles) == 20 and all(set(p) == {"profile_id", "label"} for p in profiles), "Profile list leaks permissions")
    for path in ("/api/me", "/api/menu", "/api/documents/playbook-test:Document:3", "/api/documents/playbook-test:Document:3/raw",
                 "/api/documents/playbook-test:Document:3/download"):
        expect(call(port, "GET", path), 401)
    expect(call(port, "POST", "/api/search", {"query": "合同"}), 401)
    report["anonymous_requests"] = 6
    sessions, allowed_by_pid = {}, {}
    for entry in oracle["expectations"]:
        pid = entry["profile_id"]
        allowed = set(entry["allowed_document_ids"])
        allowed_by_pid[pid] = allowed
        cookie, revision = login(port, pid)
        sessions[pid] = (cookie, revision)
        me = expect(call(port, "GET", "/api/me", cookie=cookie))[1]
        check(me["session_revision"] == revision and me["principal"]["user_name"] == pid, "Wrong me identity")
        menu = expect(call(port, "GET", "/api/menu", cookie=cookie))[1]
        actual = {int(key.rsplit(":", 1)[1]) for node in menu["nodes"] for key in node["document_keys"]}
        check(actual == allowed, f"Menu ID set differs: {pid}")
        for node in menu["nodes"]:
            check(set(node) == {"menu_id", "parent_id", "label", "label_en", "type", "placeholder", "document_keys", "visible_document_count", "business_category", "is_public"}, "Unsafe menu fields")
            if node["placeholder"]:
                check(node["label"] == node["label_en"] == "上级目录" and node["type"] is None, "Hidden parent name exposed")
        routes = ("", "/raw?language=zh", "/raw?language=en", "/download?kind=markdown&language=zh",
                  "/download?kind=markdown&language=en", "/download?kind=metadata")
        for did in (25, 99, 148, 3, 81, 999999):
            denied = []
            for suffix in routes:
                response = call(port, "GET", f"/api/documents/playbook-test:Document:{did}{suffix}", cookie=cookie)
                headers, body = expect(response, 200 if did in allowed else 404)
                if did not in allowed:
                    denied.append(body["error"])
                    continue
                check(headers["x-session-revision"] == revision, "Read lost session revision")
                if "download" in suffix:
                    check(headers["content-disposition"].startswith("attachment;"), "Download not attachment")
                if suffix and "metadata" not in suffix:
                    lang = "en" if "language=en" in suffix else "zh"
                    check(body == sources[did]["languages"][lang]["raw"].encode(), "Raw bytes changed")
                    check(headers["content-type"] == "text/plain; charset=utf-8", "Executable raw HTML")
                else:
                    if not suffix:
                        check(set(body) == {"request_id", "session_revision", "document"}, "Wrong detail envelope")
                        body = body["document"]
                    check(body["document_id"] == did and "acl" not in body and "acl_provenance" not in body and "sql_lines" not in body["source"], "Unsafe detail metadata")
            check(not denied or all(error == denied[0] for error in denied), "Inconsistent unreadable response")
        result = expect(call(port, "POST", "/api/search", {"query": "合同是如何生成的？"}, cookie))[1]
        check_items(result["items"], allowed, sources)
        check(result["session_revision"] == revision and result["retrieval_mode"] == "bm25" and len(result["items"]) <= 5, "Wrong search envelope")
        report.setdefault("profiles", {})[pid] = {"menu_ids_match_oracle": True, "visible_documents": len(allowed),
                    "contract_query_result_ids": [i["document_id"] for i in result["items"]], "read_routes_checked": len(routes) * 6}
        print(f"{pid}: menu/search + 36 document HTTP checks passed", flush=True)
    report["document_http_checks"] = 20 * 6 * 6
    # Approved V1 input only. Store actual result IDs, never label them relevance gold.
    rows = questions["questions"]
    report["v1_queries"] = []
    for pid in ("pb_msft_01", "pb_21v_01", "pb_partner_nce_01", "pb_partner_ospa_01", "pb_partner_both_01"):
        cookie, revision = sessions[pid]
        for question in rows:
            request = {k: question[k] for k in ("query", "language", "business_scope")}
            result = expect(call(port, "POST", "/api/search", request, cookie))[1]
            check_items(result["items"], allowed_by_pid[pid], sources)
            report["v1_queries"].append({"profile_id": pid, "question_id": question["id"],
                                       "document_ids": [i["document_id"] for i in result["items"]]})
    report["v1_status"] = "approved_inputs_executed_without_frozen_relevance_labels"
    cookie, _ = sessions["pb_partner_nce_01"]
    invalid = [{}, {"query": " "}, {"query": "x" * 1001}, {"query": 1}, {"query": "x", "top_k": True},
               {"query": "x", "top_k": 21}, {"query": "x", "top_k": "5"}, {"query": "x", "top_k": 0},
               {"query": "x", "language": ["zh"]}, {"query": "x", "business_scope": True}, {"query": "x", "business_scope": "0"}]
    invalid += [{"query": "x", key: value} for key, value in (("role_ids", [1]), ("business_categories", [0, 1]),
                ("index", "*"), ("provider", "llm"), ("password", "bad"), ("dsl", {"match_all": {}}))]
    for body in invalid:
        expect(call(port, "POST", "/api/search", body, cookie), 400)
    for raw in ('{"query":"x","query":"y"}', '{', '[]', '{"query":NaN}', '{"query":"\\ud800"}'):
        expect(call(port, "POST", "/api/search", cookie=cookie, raw=raw), 400)
    for path in ("/api/documents/elastic:Document:25", "/api/documents/../data/playbook-sampledata.sql",
                 "/api/documents/playbook-test:Document:25/raw?language=both", "/api/documents/playbook-test:Document:25?index=other",
                 "/api/search?role_ids=1", "/api/menu?business_categories=1"):
        expect(call(port, "GET", path, cookie=cookie), 400 if not path.startswith("/api/search") else 404)
    report["invalid_requests"] = len(invalid) + 5 + 6
    for path in ("/src/template.html", "/playbook.html", "/data/portal_data.json", "/fixtures/expected_access.json", "/.local/s3/pb_partner_nce_01_password",
                 "/data/playbook-sampledata.sql", "/reports/S3_CHECKS.json", "/tools/extract_playbook_sql.py", "/raw/document-25.md", "/%2e%2e/.local/"):
        expect(call(port, "GET", path, cookie=cookie), 404)
    report["blocked_static_paths"] = 10
    for method, path, body in (("POST", "/api/dev/session", {"profile_id": "pb_msft_01"}), ("DELETE", "/api/session", None),
                               ("POST", "/api/search", {"query": "合同"})):
        expect(call(port, method, path, body, cookie, {"Origin": "https://other.invalid"}), 403)
    expect(call(port, "GET", "/api/dev/profiles", headers={"Host": "other.invalid"}), 403)
    report["origin_host_rejections"] = 4
    expect(call(port, "POST", "/api/dev/session", {"profile_id": "elastic"}), 400)
    expect(call(port, "POST", "/api/dev/session", {"profile_id": "pb_msft_01", "role_ids": [1]}), 400)
    # One alphabetic token: Smart Chinese splits mixed letters/digits, which
    # can legitimately match short numeric tokens in the source documents.
    check(expect(call(port, "POST", "/api/search", {"query": "zzqnomatchxylophonezzq"}, cookie))[1]["items"] == [], "Empty results incorrectly expanded")
    scoped = {}
    for scope in ("all", 0, 1):
        result = expect(call(port, "POST", "/api/search", {"query": "合同", "top_k": 20, "business_scope": scope}, cookie))[1]
        check_items(result["items"], allowed_by_pid["pb_partner_nce_01"], sources)
        check(scope == "all" or all(i["business_category"] == scope for i in result["items"]), "Scope broadened results")
        scoped[str(scope)] = [i["document_id"] for i in result["items"]]
    report["scope_query_ids"] = scoped
    report["language_alias_queries"] = []
    for query in ("RI", "预留实例", "Savings Plan", "节省计划", "CICR", "Contact Information Change Request", "取消", "删除", "暂停"):
        for lang in ("zh", "en", "both"):
            result = expect(call(port, "POST", "/api/search", {"query": query, "language": lang}, cookie))[1]
            check_items(result["items"], allowed_by_pid["pb_partner_nce_01"], sources)
            check(lang == "both" or all(i["matched_language"] == lang for i in result["items"]), "Language selector ignored")
            report["language_alias_queries"].append({"query": query, "language": lang, "document_ids": [i["document_id"] for i in result["items"]]})

    def parallel(n):
        pid = "pb_partner_nce_01" if n % 2 else "pb_partner_ospa_01"
        cookie, revision = sessions[pid]
        result = expect(call(port, "POST", "/api/search", {"query": "合同是如何生成的？", "top_k": 20}, cookie))[1]
        check_items(result["items"], allowed_by_pid[pid], sources)
        check(result["session_revision"] == revision, "Concurrent session crossed")
        expect(call(port, "GET", f"/api/documents/playbook-test:Document:{99 if n % 2 else 25}/download", cookie=cookie), 404)
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(parallel, range(40)))
    report["concurrent_http_requests"] = 80
    old, old_revision = sessions["pb_partner_ospa_01"]
    new, revision = login(port, "pb_partner_nce_01", old)
    check(new != old and revision != old_revision, "Session not rotated")
    expect(call(port, "GET", "/api/me", cookie=old), 401)
    expect(call(port, "GET", "/api/documents/playbook-test:Document:99/download", cookie=new), 404)
    expect(call(port, "DELETE", "/api/session", cookie=new), 204)
    expect(call(port, "GET", "/api/me", cookie=new), 401)
    report["switch_logout"] = "old_cookie_invalid_and_new_identity_enforced"


def main():
    report = {"stage": "S4", "started_at": permissions.es.now(), "status": "running"}
    path = ROOT / "reports/S4_CHECKS.json"
    logdir = ROOT / ".local/s4"
    logdir.mkdir(mode=0o700, exist_ok=True)
    permissions.es.write_json(path, report)
    try:
        with (logdir / "api-check.jsonl").open("w") as log, APIServer(0) as server:
            server.audit_stream = log
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                run(server, report)
            finally:
                server.shutdown()
                thread.join()
        entries = [json.loads(line) for line in (logdir / "api-check.jsonl").read_text().splitlines()]
        expected = {"request_id", "profile_id", "route", "status", "index", "policy_version", "transform_version", "analyzer", "synonyms_version", "synonyms_sha256", "result_ids", "elapsed_ms"}
        check(all(set(e) == expected for e in entries), "Unexpected audit payload")
        report.update(status="passed", audit_records=len(entries), audit_fields=sorted(expected))
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, message=str(error))
        failures = ROOT / "reports/S4_FAILURES.json"
        history = json.loads(failures.read_text()) if failures.exists() else []
        history.append({"at": permissions.es.now(), **report})
        permissions.es.write_json(failures, history)
        raise
    finally:
        report["completed_at"] = permissions.es.now()
        permissions.es.write_json(path, report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("profiles", "v1_queries", "language_alias_queries")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
