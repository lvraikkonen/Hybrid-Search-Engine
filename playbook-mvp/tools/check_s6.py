#!/usr/bin/env python3
"""Real ES + HTTP change/revocation acceptance in a fixed, synthetic-only namespace."""
import argparse
import copy
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
from unittest.mock import patch
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import APIServer
from runtime import CatalogStore
from tools import sync_playbook as sync
from tools.check_api import call, expect, login
from tools.extract_playbook_sql import canonical_hash
from tools.ingest_playbook import current_documents, history

es = sync.es
MSFT, NCE, BOTH = "pb_s6_msft_01", "pb_s6_partner_nce_01", "pb_s6_partner_both_01"
IDS = (99001, 99002, 99003)


def synthetic_snapshot(store):
    """These formal rows are fabricated TEST inputs, never real SQL/ACL or oracle edits."""
    tables = {k: [] for k in sync.TABLES}
    tables["Role"] = [{"Id": i, "Name": name, "Status": 1, "_sql_lines": []}
                      for i, name in ((1, "MSFT"), (2, "Partner"), (3, "21V"))]
    for did, category, controlled in ((99001, 0, 1), (99002, 1, 1), (99003, 1, 0)):
        title = f"S6 隔离验收 {did}"
        tables["Document"].append({"Id": did, "Title": title, "TitleEn": f"S6 isolated check {did}",
            "Slug": f"s6-{did}", "BusinessCategory": category, "Content": f"# {title}\n\nS6 共通知识 Common validation source v1.\n",
            "ContentEn": f"# S6 isolated check {did}\n\nS6 Common validation source v1.\n", "Status": 1, "TextFormatType": 1,
            "CreatedTime": "2000-01-01 00:00:00", "UpdatedTime": "2000-01-01 00:00:00", "_sql_lines": []})
        tables["MenuItem"].append({"Id": did, "Name": title, "NameEn": f"S6 isolated check {did}", "ParentId": 0,
            "Type": 0, "Active": 1, "UnderControl": controlled, "BusinessCategory": category, "Sort": did, "_sql_lines": []})
        tables["MenuItem_Document"].append({"Id": did, "MenuItemId": did, "DocumentId": did, "_sql_lines": []})
        if controlled:
            tables["Role_MenuItem"].extend({"Id": did * 10 + role, "RoleId": role, "MenuItemId": did, "_sql_lines": []}
                                           for role in (1, 2, 3))
    return {"schema": "s6-full-snapshot-v1", "complete": True, "origin": {"kind": "synthetic_s6_only", "sql_executed": False},
            "tables": tables, "profiles": store.initial_profiles(), "withdrawn_document_ids": []}


def run(restart=False):
    es.require(os.environ.get("APP_ENV") in ("development", "test") and os.environ.get("MOCK_AUTH_ENABLED") == "1", "Local mock environment required")
    store = CatalogStore(validation=True)
    report = {"stage": "S6", "started_at": es.now(), "status": "running", "namespace": store.index,
              "synthetic_only": True, "cases": [], "http_checks": 0, "native_es_checks": 0,
              "container_restart": "not_run"}
    previous = ROOT / "reports/S6_CHANGES.json"
    if previous.exists() and not (ROOT / "reports/S6_CHANGE_RUNS.json").exists():
        history("S6_CHANGE_RUNS.json", sync.read_json(previous))
    baseline_real = canonical_hash({k: h["_source"] for k, h in current_documents().items()})
    sync.init_validation(store)
    server = APIServer(0, store)
    server.audit_stream = io.StringIO()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_port
    cookies = {}

    def http(method, path, body=None, cookie=None, status=200):
        response = call(port, method, path, body, cookie)
        report["http_checks"] += 1
        expect(response, status)
        return response[2]

    def native(pid, method, path, body=None, status=200):
        result = es.request(pid, method, path, body, status, password_value=sync.permissions.credential(pid, store.credentials))
        report["native_es_checks"] += 1
        return result

    def paths(expected):
        evidence = {}
        for pid, allowed in expected.items():
            cookie, _ = login(port, pid)
            report["http_checks"] += 1
            cookies[pid] = cookie
            menu = http("GET", "/api/menu", cookie=cookie)
            keys = {key for node in menu["nodes"] for key in node["document_keys"]}
            assert keys == {f"playbook-test:Document:{did}" for did in allowed}
            result = http("POST", "/api/search", {"query": "S6 Common 共通知识", "language": "both", "top_k": 20}, cookie)
            assert {d["document_id"] for d in result["items"]} == set(allowed)
            found = native(pid, "POST", f"/{store.index}/_search", {"size": 20, "query": {"match_all": {}}})
            assert {h["_source"]["document_id"] for h in found["hits"]["hits"]} == set(allowed)
            keys = [f"playbook-test:Document:{did}" for did in IDS]
            batch = native(pid, "POST", f"/{store.index}/_mget", {"ids": keys})
            for did, key, item in zip(IDS, keys, batch["docs"]):
                status = 200 if did in allowed else 404
                assert item.get("found") is (did in allowed) and ("_source" in item) is (did in allowed)
                get = native(pid, "GET", f"/{store.index}/_doc/{quote(key, safe='')}", status=status)
                assert get.get("found") is (did in allowed)
                for suffix in ("", "/raw?language=zh", "/raw?language=en", "/download?kind=markdown&language=zh",
                               "/download?kind=markdown&language=en", "/download?kind=metadata"):
                    http("GET", "/api/documents/" + quote(key, safe="") + suffix, cookie=cookie, status=status)
            native(pid, "PUT", f"/{store.index}/_doc/forbidden", {"x": 1}, status=403)
            evidence[pid] = sorted(allowed)
        return evidence

    def commit(name, batch, approve=True, delete=False):
        plan, diff = sync.prepare(store, batch)
        previous = dict(cookies)
        result = sync.apply(store, plan, "s6:isolated-acceptance", approve, delete)
        if result["status"] != "unchanged":
            for cookie in previous.values():
                http("GET", "/api/me", cookie=cookie, status=401)
        report["cases"].append({"name": name, "plan_id": plan, "diff": diff, **result})
        print(json.dumps({"case": name, "status": result["status"]}), flush=True)
        return plan

    try:
        batch = synthetic_snapshot(store)
        initial = commit("complete_snapshot_explicit_release", batch)
        report["cases"][-1]["paths"] = paths({MSFT: IDS, NCE: (99001, 99003), BOTH: IDS})
        initial_sources = sync.source_values(sync.sources(store))
        # Old physical copy remains inaccessible even when it contains previously approved records.
        archive = copy.copy(store)
        archive.index = "playbook-s6-archive"
        sync.ensure_index(archive, sync.index_definition(sync.read_json(ROOT / "config/synonyms.json")))
        sync.write_batch(archive, sync.sources(archive), initial_sources)
        generation = store.current()["generation"]
        commit("repeat_import_noop", batch)
        assert store.current()["generation"] == generation
        for cookie in cookies.values():
            http("GET", "/api/me", cookie=cookie)

        stale_plan, _ = sync.prepare(store, batch)
        batch["tables"]["Document"][0]["Content"] += "\nChanged source v2 without UpdatedTime modification.\n"
        commit("content_update_defaults_closed", batch, approve=False)
        assert batch["tables"]["Document"][0]["UpdatedTime"] == "2000-01-01 00:00:00"
        report["cases"][-1]["paths"] = paths({MSFT: (), NCE: (), BOTH: ()})
        try:
            sync.apply(store, stale_plan, "s6:isolated-acceptance", True)
            raise AssertionError("Stale plan applied")
        except RuntimeError as error:
            assert "Stale plan" in str(error)
        assert not store.gate.exists()
        report["cases"].append({"name": "stale_plan_rejected_without_blocking_current", "status": "passed"})
        commit("content_snapshot_explicit_reopen", batch)
        report["cases"][-1]["paths"] = paths({MSFT: IDS, NCE: (99001, 99003), BOTH: IDS})
        assert "Changed source v2" in sync.source_values(sync.sources(store))["playbook-test:Document:99001"]["languages"]["zh"]["raw"]

        before = sync.source_values(sync.sources(store))
        batch["tables"]["Role_MenuItem"] = [r for r in batch["tables"]["Role_MenuItem"] if (r["RoleId"], r["MenuItemId"]) != (2, 99001)]
        commit("acl_only_revoke_partner_unchanged_document_timestamp", batch)
        after = sync.source_values(sync.sources(store))
        assert before["playbook-test:Document:99001"]["content_revision"] == after["playbook-test:Document:99001"]["content_revision"]
        report["cases"][-1]["paths"] = paths({MSFT: IDS, NCE: (99003,), BOTH: (99002, 99003)})

        batch["profiles"][BOTH]["authentication_info"]["BusinessCategories"] = [0]
        commit("user_business_revocation_invalidates_old_sessions", batch)
        report["cases"][-1]["paths"] = paths({MSFT: IDS, NCE: (99003,), BOTH: (99003,)})
        batch["withdrawn_document_ids"] = [99003]
        commit("withdraw_public_document", batch)
        report["cases"][-1]["paths"] = paths({MSFT: (99001, 99002), NCE: (), BOTH: ()})

        batch["tables"]["Document"] = [r for r in batch["tables"]["Document"] if r["Id"] != 99002]
        batch["tables"]["MenuItem"] = [r for r in batch["tables"]["MenuItem"] if r["Id"] != 99002]
        batch["tables"]["MenuItem_Document"] = [r for r in batch["tables"]["MenuItem_Document"] if r["DocumentId"] != 99002]
        batch["tables"]["Role_MenuItem"] = [r for r in batch["tables"]["Role_MenuItem"] if r["MenuItemId"] != 99002]
        plan, _ = sync.prepare(store, batch)
        try:
            sync.apply(store, plan, "s6:isolated-acceptance", True)
            raise AssertionError("Deletion lacked explicit approval")
        except RuntimeError as error:
            assert "allow-delete" in str(error)
        commit("complete_snapshot_explicit_delete", batch, delete=True)
        assert "playbook-test:Document:99002" not in sync.sources(store)
        report["cases"][-1]["paths"] = paths({MSFT: (99001,), NCE: (), BOTH: ()})

        batch["tables"]["Document"][0]["Content"] += "\nRetryable content v3.\n"
        plan, _ = sync.prepare(store, batch)
        real_write = sync.write_batch
        def interrupted_write(site, current, proposed):
            partial = sync.source_values(current)
            key = sorted(proposed)[0]
            partial[key] = proposed[key]
            real_write(site, current, partial)
            raise RuntimeError("Injected interruption after one real record write")
        try:
            with patch.object(sync, "write_batch", side_effect=interrupted_write):
                sync.apply(store, plan, "s6:isolated-acceptance", True)
            raise AssertionError("Injected failure escaped")
        except RuntimeError as error:
            assert "Injected interruption" in str(error)
        assert store.gate.exists()
        for cookie in cookies.values():
            http("GET", "/api/menu", cookie=cookie, status=503)
            http("GET", "/api/documents/playbook-test%3ADocument%3A99001/raw", cookie=cookie, status=503)
        for pid in (MSFT, NCE, BOTH):
            native(pid, "POST", f"/{store.index}/_search", {"query": {"match_all": {}}}, status=403)
        reopened = CatalogStore(validation=True)
        result = sync.apply(reopened, plan, "s6:isolated-acceptance", True)
        assert result["status"] == "passed" and not store.gate.exists()
        report["cases"].append({"name": "injected_partial_write_failure_durable_gate_same_plan_retry", **result,
                                "paths": paths({MSFT: (99001,), NCE: (), BOTH: ()})})

        latest = copy.deepcopy(batch)
        rollback = sync.content_rollback(store, initial)
        for key in ("profiles", "withdrawn_document_ids"):
            assert rollback[key] == latest[key]
        assert rollback["tables"]["Role_MenuItem"] == latest["tables"]["Role_MenuItem"]
        assert {r["Id"] for r in rollback["tables"]["Document"]} == {99001, 99003}
        commit("content_rollback_preserves_latest_denials", rollback)
        assert sync.source_values(sync.sources(store))["playbook-test:Document:99001"]["languages"]["zh"]["raw"] == initial_sources["playbook-test:Document:99001"]["languages"]["zh"]["raw"]
        report["cases"][-1]["paths"] = paths({MSFT: (99001,), NCE: (), BOTH: ()})
        for pid in (MSFT, NCE, BOTH):
            for method, suffix, body in (("GET", "/_search", None), ("GET", "/_doc/playbook-test%3ADocument%3A99001", None),
                                         ("POST", "/_mget", {"ids": ["playbook-test:Document:99001"]})):
                result = native(pid, method, "/playbook-s6-archive" + suffix, body, status=200 if suffix == "/_mget" else 403)
                if suffix == "/_mget":
                    assert len(result["docs"]) == 1 and result["docs"][0]["error"]["type"] == "security_exception"
                    assert "_source" not in result["docs"][0] and "fields" not in result["docs"][0]
        for path in ("/.local/s6/validation/batches/" + initial + ".json", "/data/playbook-sampledata.sql", "/playbook-s6-archive/_search"):
            http("GET", path, cookie=cookies[MSFT], status=404)
        report["cases"].append({"name": "old_physical_index_and_file_paths_denied", "status": "passed"})

        # This is a new profile-change snapshot, not a second application of the
        # earlier content rollback package (whose policy generation is now stale).
        rollback["origin"] = {"kind": "synthetic_s6_only", "sql_executed": False, "purpose": "disable_profile_after_rollback"}
        rollback["profiles"][BOTH]["mock_session"]["enabled"] = False
        commit("disabled_native_profile", rollback)
        http("POST", "/api/dev/session", {"profile_id": BOTH}, status=401)
        native(BOTH, "GET", "/_security/_authenticate", status=401)
        report["cases"][-1]["paths"] = paths({MSFT: (99001,), NCE: ()})
        if restart:
            before_restart = canonical_hash(sync.source_values(sync.sources(store)))
            # Reversible restart, explicitly selected by the CLI; never remove volumes.
            process = subprocess.Popen(["docker", "compose", "restart", "es"], cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 180
            saw_failure = False
            restart_statuses = []
            while time.monotonic() < deadline:
                response = call(port, "GET", "/api/menu", cookie=cookies[MSFT])
                report["http_checks"] += 1
                restart_statuses.append(response[0])
                if response[0] in (401, 503, 504):
                    expect(response, response[0])
                    saw_failure = True
                    if response[0] == 401 and process.poll() == 0:
                        session = call(port, "POST", "/api/dev/session", {"profile_id": MSFT})
                        report["http_checks"] += 1
                        if session[0] == 200:
                            headers, _ = expect(session)
                            cookies[MSFT] = headers["set-cookie"].split(";", 1)[0]
                        else:
                            assert session[0] in (401, 503, 504)
                            expect(session, session[0])
                elif response[0] == 200 and saw_failure and process.poll() == 0:
                    expect(response)
                    break
                else:
                    assert response[0] == 200, response[0]
                time.sleep(1)
            else:
                raise RuntimeError("Restart did not complete with observed fail-closed outage")
            process.communicate(timeout=10)
            assert canonical_hash(sync.source_values(sync.sources(store))) == before_restart
            report["container_restart"] = {"status": "passed", "observed_api_failure": saw_failure,
                "api_statuses": restart_statuses, "reauthentication_after_401": True,
                "stored_records_unchanged": True, "paths": paths({MSFT: (99001,), NCE: ()})}
        assert canonical_hash({k: h["_source"] for k, h in current_documents().items()}) == baseline_real
        report.update(status="passed", real_index_unchanged=True, safety_failures=0)
    except BaseException as error:
        report.update(status="failed", failure_type=type(error).__name__, failure=str(error))
        raise
    finally:
        server.shutdown()
        server.server_close()
        logs = [json.loads(line) for line in server.audit_stream.getvalue().splitlines()]
        assert all(not ({"query", "password", "cookie", "raw", "authorization", "body"} & set(row)) for row in logs)
        report.update(completed_at=es.now(), audit_log_rows=len(logs), audit_sensitive_fields_absent=True)
        es.write_json(ROOT / "reports/S6_CHANGES.json", report)
        history("S6_CHANGE_RUNS.json", report)
    print(json.dumps({k: report[k] for k in ("status", "http_checks", "native_es_checks", "real_index_unchanged")}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--restart-es", action="store_true", help="Also restart the existing ES container without deleting its volumes")
    args = parser.parse_args()
    run(args.restart_es)
