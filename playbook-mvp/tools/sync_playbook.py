#!/usr/bin/env python3
"""S6 bounded manual full-snapshot synchronization, with an explicit outage on failure."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import secrets
import sys
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from documents import CLOSED, SQL, document_record, index_definition
from mock_auth import normalize_authentication_info, validate_mock_profile
from runtime import CatalogStore, atomic_json
from tools import es_permissions as permissions
from tools.extract_playbook_sql import canonical_hash, decode_sql, digest, parse_dump, validate_tables

es = permissions.es
TABLES = {"Document", "MenuItem", "MenuItem_Document", "Role", "Role_MenuItem", "Notice"}
NAV_TABLES = ("MenuItem", "Role", "Role_MenuItem")


def read_json(path):
    return json.loads(Path(path).read_text())


def maintenance_request(store, method, path, body=None, expected=200):
    secret = permissions.credential(store.importer, store.credentials) if store.validation else es.password(store.importer)
    return es.request(store.importer, method, path, body, expected, password_value=secret)


def current_catalog(store):
    active = store.current()
    if active:
        return active
    if store.validation:
        return {"generation": "empty", "snapshot": None, "tables": {}, "profiles": store.initial_profiles(), "batch": None}
    source = read_json(ROOT / "fixtures/source_acl_tables.json")
    return store.baseline(source["source"]["sha256"], {k: source["tables"][k] for k in NAV_TABLES})


def export_snapshot(store, sql, complete):
    es.require(complete is True, "Operator must explicitly attest a complete snapshot of all six tables")
    raw = Path(sql).read_bytes()
    current = current_catalog(store)
    previous = read_json(store.directory / "batches" / (current["batch"] + ".json")) if current["batch"] else {}
    return {"schema": "s6-full-snapshot-v1", "complete": True,
            "origin": {"kind": "sql_export", "sql_sha256": digest(raw), "sql_executed": False},
            "tables": parse_dump(decode_sql(raw)[0]), "profiles": current["profiles"],
            "withdrawn_document_ids": previous.get("withdrawn_document_ids", [])}


def validate_snapshot(batch, store):
    es.require(batch.get("schema") == "s6-full-snapshot-v1" and batch.get("complete") is True, "Incomplete/unrecognized snapshot")
    tables, profiles = batch["tables"], batch["profiles"]
    es.require(set(tables) == TABLES and all(isinstance(v, list) for v in tables.values()), "All six source tables are required")
    es.require(0 < len(tables["Document"]) <= 1000, "Local snapshot must contain 1..1000 documents")
    es.require(len(json.dumps(batch, ensure_ascii=False).encode()) <= 10 * 1024 * 1024, "Snapshot exceeds local 10 MiB cap")
    es.require(all(type(row.get("Id")) is int and row["Id"] > 0 for rows in tables.values() for row in rows), "Invalid source ID")
    es.require({r["Id"] for r in tables["Role"]} == {1, 2, 3} and
               all(type(r.get("Status")) is int and r["Status"] in (0, 1) for r in tables["Role"]), "Incomplete/unknown role state")
    for menu in tables["MenuItem"]:
        es.require(all(type(menu.get(k)) is int and menu[k] in values for k, values in
                       (("Type", (0, 1, 2, 3)), ("Active", (0, 1)), ("UnderControl", (0, 1)), ("BusinessCategory", (0, 1))))
                   and (menu["ParentId"] is None or type(menu["ParentId"]) is int and menu["ParentId"] >= 0)
                   and type(menu["Sort"]) is int and all(isinstance(menu[k], str) for k in ("Name", "NameEn")), "Invalid navigation row")
    audit = validate_tables(tables)
    for field in ("documents_without_menu", "broken_document_menu_edges", "broken_role_menu_edges", "duplicate_document_menu_edges",
                  "duplicate_role_menu_edges", "documents_with_multiple_menu_edges", "menus_with_multiple_documents",
                  "invalid_parent_paths", "menu_cycles", "business_category_mismatches", "relations_pointing_to_non_document_menus"):
        es.require(not audit[field], "Source relationship check failed: " + field)
    es.require(set(profiles) == set(store.initial_profiles()), "Profile membership must match this deployment; no added admin identities")
    ids = set()
    for pid, profile in profiles.items():
        es.require(re.fullmatch(r"pb_[a-z0-9_]+", pid) and profile["profile_id"] == pid, "Invalid profile key")
        session = profile["mock_session"]
        es.require(session["authenticated"] is True and session["expired"] is False and type(session["enabled"]) is bool, "Invalid mock profile state")
        principal = normalize_authentication_info(profile["authentication_info"])
        es.require(principal.user_name == pid and principal.user_id not in ids, "Invalid/duplicate native identity")
        ids.add(principal.user_id)
        if session["enabled"]:
            validate_mock_profile(profile)
    withdrawn = batch["withdrawn_document_ids"]
    es.require(isinstance(withdrawn, list) and all(type(x) is int and x > 0 for x in withdrawn)
               and len(withdrawn) == len(set(withdrawn)), "Invalid withdrawal list")
    snapshot = canonical_hash(tables)  # A controlled table snapshot, explicitly distinct from the input SQL-byte hash.
    records = [document_record(doc, tables, snapshot) for doc in sorted(tables["Document"], key=lambda x: x["Id"])]
    es.require(all(r["acl"]["state"] == "resolved" for r in records), "Unresolved ACL; nothing may be applied")
    return records, snapshot


def sources(store):
    if maintenance_request(store, "HEAD", f"/{store.index}", expected=(200, 404))["status"] == 404:
        return {}
    response = maintenance_request(store, "POST", f"/{store.index}/_search?request_cache=false&allow_partial_search_results=false",
                                   {"size": 1000, "track_total_hits": True, "seq_no_primary_term": True, "query": {"match_all": {}}})
    hits = response["hits"]
    es.require(not response["timed_out"] and response["_shards"]["failed"] == 0
               and hits["total"] == {"value": len(hits["hits"]), "relation": "eq"}, "Incomplete maintenance read")
    return {h["_id"]: h for h in hits["hits"]}


def source_values(hits):
    return {key: value["_source"] for key, value in hits.items()}


def prepare(store, batch):
    records, snapshot = validate_snapshot(batch, store)
    with store.read():
        before = source_values(sources(store))
        active = current_catalog(store)
        if batch.get("origin", {}).get("kind") == "content_only_rollback":
            es.require(batch["origin"].get("policy_batch") == active["batch"], "Rollback policy changed; regenerate from the latest batch")
    proposed = {r["document_key"]: r for r in records}
    common = set(before) & set(proposed)
    diff = {"added": sorted(set(proposed) - set(before)), "deleted": sorted(set(before) - set(proposed)),
            "content_changed": sorted(k for k in common if before[k]["content_revision"] != proposed[k]["content_revision"]),
            "acl_changed": sorted(k for k in common if before[k]["acl_revision"] != proposed[k]["acl_revision"]),
            "metadata_changed": sorted(k for k in common if before[k]["upstream_metadata"] != proposed[k]["upstream_metadata"]),
            "navigation_changed": active["tables"] != {k: batch["tables"][k] for k in NAV_TABLES},
            "profiles_changed": sorted(pid for pid in batch["profiles"] if active["profiles"].get(pid) != batch["profiles"][pid]),
            "withdrawn_document_ids": sorted(batch["withdrawn_document_ids"]), "snapshot_changed": active["snapshot"] != snapshot}
    plan = {"site": "validation" if store.validation else "local", "base_generation": active["generation"],
            "base_index_sha256": canonical_hash(before), "snapshot": snapshot, "batch": batch, "records": records,
            "previous_release": {k: v["lab_release"] for k, v in before.items()}, "diff": diff}
    plan_id = canonical_hash(plan)
    atomic_json(store.directory / "plans" / (plan_id + ".json"), plan)
    return plan_id, diff


def ensure_index(store, definition):
    if maintenance_request(store, "HEAD", f"/{store.index}", expected=(200, 404))["status"] == 404:
        maintenance_request(store, "PUT", f"/{store.index}", definition)
    mapping = maintenance_request(store, "GET", f"/{store.index}/_mapping")[store.index]["mappings"]
    es.require(mapping == definition["mappings"], "Mapping drift; explicit migration required")
    settings = maintenance_request(store, "GET", f"/{store.index}/_settings")[store.index]["settings"]["index"]["analysis"]
    def strings(value):
        if isinstance(value, dict):
            return {k: strings(v) for k, v in value.items()}
        if isinstance(value, list):
            return [strings(v) for v in value]
        return str(value).lower() if isinstance(value, bool) else str(value)
    es.require(strings(settings) == strings(definition["settings"]["analysis"]), "Analyzer drift; explicit migration required")


def write_batch(store, current, proposed):
    actions, expected_count = [], 0
    for key in sorted(set(current) | set(proposed)):
        old = current.get(key)
        if old and old["_source"] == proposed.get(key):
            continue
        version = {"if_seq_no": old["_seq_no"], "if_primary_term": old["_primary_term"]} if old else {}
        if key not in proposed:
            actions.append({"delete": {"_id": key, **version}})
        else:
            actions.extend([{("index" if old else "create"): {"_id": key, **version}}, proposed[key]])
        expected_count += 1
    if not actions:
        return 0
    payload = ("\n".join(json.dumps(x, ensure_ascii=False) for x in actions) + "\n").encode()
    es.require(len(payload) <= 10 * 1024 * 1024, "Rendered bulk exceeds 10 MiB cap")
    result = maintenance_request(store, "POST", f"/{store.index}/_bulk?refresh=wait_for", payload)
    es.require(len(result["items"]) == expected_count and result["errors"] is False
               and all(next(iter(item.values()))["status"] < 300 for item in result["items"]), "Bulk incomplete; gate remains closed, retry the same plan")
    return expected_count


def apply(store, plan_id, actor, approve_release=False, allow_delete=False):
    es.require(re.fullmatch(r"[a-f0-9]{64}", plan_id) and actor.strip() == actor and 1 <= len(actor) <= 100, "Invalid plan/actor")
    plan = read_json(store.directory / "plans" / (plan_id + ".json"))
    es.require(canonical_hash(plan) == plan_id and plan["site"] == ("validation" if store.validation else "local"), "Plan changed or wrong deployment")
    records, snapshot = validate_snapshot(plan["batch"], store)
    es.require(records == plan["records"] and snapshot == plan["snapshot"], "Plan records differ from formal source")
    es.require(not plan["diff"]["deleted"] or allow_delete, "Explicit allow-delete required for the reviewed missing IDs")
    started = time.monotonic()
    with store.maintenance(plan_id) as recovering:
        try:
            decision_path = store.directory / "plans" / (plan_id + ".decision.json")
            decision = {"actor": actor, "approve_release": approve_release, "allow_delete": allow_delete}
            if decision_path.exists():
                stored = read_json(decision_path)
                es.require(all(stored[k] == v for k, v in decision.items()), "Retry must retain the original approval decision")
                decision = stored
            else:
                decision["at"] = es.now()
                atomic_json(decision_path, decision)
            proposed = {}
            for record in records:
                old = plan["previous_release"].get(record["document_key"], CLOSED)
                eligible = record["acl"]["menu_active_raw"] == 1 and record["document_id"] not in plan["batch"]["withdrawn_document_ids"]
                if eligible and old["source_snapshot_sha256"] == snapshot:
                    record["lab_release"] = old
                if eligible and approve_release and not record["lab_release"]["approved"]:
                    record["lab_release"] = {"approved": True, "source_snapshot_sha256": snapshot, "approved_by": actor, "approved_at": decision["at"]}
                proposed[record["document_key"]] = record
            active, current = current_catalog(store), source_values(sources(store))
            es.require(active["generation"] in {plan["base_generation"], plan_id}, "Stale plan generation")
            if not recovering and current == proposed and active["profiles"] == plan["batch"]["profiles"] and active["snapshot"] == snapshot and active.get("withdrawn_document_ids", []) == plan["batch"]["withdrawn_document_ids"]:
                return {"status": "unchanged", "generation": active["generation"], "written": 0}
            if not recovering:
                es.require(active["generation"] == plan["base_generation"] and canonical_hash(current) == plan["base_index_sha256"], "Index changed since planning")
            license = maintenance_request(store, "GET", "/_license")["license"]
            es.require(license["status"] == "active" and license["type"] in ("trial", "platinum", "enterprise"), "DLS license unavailable")
        except BaseException:
            # No index/role/user mutation has happened: a rejected new plan must
            # not interrupt the current generation. Failed retries stay closed.
            if not recovering:
                store.gate.unlink()
            raise
        es.request("elastic", "PUT", f"/_security/role/{store.reader}", {"cluster": [], "indices": []})
        if not store.validation:
            es.request("elastic", "PUT", "/_security/role/pb_s2_reader", {"cluster": [], "indices": []})
        try:
            definition = index_definition(read_json(ROOT / "config/synonyms.json"))
            ensure_index(store, definition)
            written = write_batch(store, sources(store), proposed)
            es.require(source_values(sources(store)) == proposed, "Post-write full record verification failed")
            for pid, profile in plan["batch"]["profiles"].items():
                principal = normalize_authentication_info(profile["authentication_info"])
                es.request("elastic", "PUT", f"/_security/user/{pid}", {"roles": [store.reader],
                           "enabled": profile["mock_session"]["enabled"], "metadata": principal.es_metadata_candidate()})
            catalog = {"generation": plan_id, "snapshot": snapshot, "batch": plan_id,
                       "withdrawn_document_ids": plan["batch"]["withdrawn_document_ids"],
                       "tables": {k: plan["batch"]["tables"][k] for k in NAV_TABLES}, "profiles": plan["batch"]["profiles"]}
            es.request("elastic", "PUT", f"/_security/role/{store.reader}", permissions.reader_role(snapshot, store.index, not store.validation))
            if not store.validation:
                es.request("elastic", "PUT", "/_security/role/pb_s2_reader", permissions.reader_role(snapshot, store.index, False))
            from api import resolve_identity
            for pid, profile in catalog["profiles"].items():
                if profile["mock_session"]["enabled"]:
                    resolve_identity(pid, snapshot, catalog, store)
            atomic_json(store.directory / "batches" / (plan_id + ".json"), plan["batch"])
            atomic_json(store.directory / "active.json", catalog)
        except BaseException:
            es.request("elastic", "PUT", f"/_security/role/{store.reader}", {"cluster": [], "indices": []})
            if not store.validation:
                es.request("elastic", "PUT", "/_security/role/pb_s2_reader", {"cluster": [], "indices": []})
            raise
    return {"status": "passed", "generation": plan_id, "snapshot": snapshot, "written": written,
            "maintenance_ms": round((time.monotonic() - started) * 1000, 2), "confirmed_at": es.now(), "diff": plan["diff"]}


def content_rollback(store, previous_batch):
    es.require(re.fullmatch(r"[a-f0-9]{64}", previous_batch), "Invalid historical batch")
    active = current_catalog(store)
    es.require(active["batch"], "No active S6 batch")
    latest = read_json(store.directory / "batches" / (active["batch"] + ".json"))
    previous = read_json(store.directory / "batches" / (previous_batch + ".json"))
    old = {r["Id"]: r for r in previous["tables"]["Document"]}
    for doc in latest["tables"]["Document"]:
        if doc["Id"] in old:
            for field in ("Content", "ContentEn"):
                doc[field] = old[doc["Id"]][field]
    latest["origin"] = {"kind": "content_only_rollback", "content_batch": previous_batch, "policy_batch": active["batch"]}
    return latest


def init_validation(store):
    es.require(store.validation, "Use S1-S3 provisioning for real identities; this init is synthetic-only")
    es.require(not store.gate.exists(), "Recover the failed plan before provisioning validation identities")
    store.credentials.mkdir(parents=True, mode=0o700, exist_ok=True)
    store.credentials.chmod(0o700)
    profiles = current_catalog(store)["profiles"]
    for user in [store.importer, *profiles]:
        path = store.credentials / (user + "_password")
        if not path.exists():
            path.write_text(secrets.token_hex(32) + "\n")
        path.chmod(0o600)
    es.request("elastic", "PUT", f"/_security/role/{store.importer}", {"cluster": ["monitor"],
               "indices": [{"names": [store.index, "playbook-s6-archive"], "privileges": ["manage", "read", "write"]}]})
    es.request("elastic", "PUT", f"/_security/user/{store.importer}", {"password": permissions.credential(store.importer, store.credentials),
               "roles": [store.importer], "enabled": True, "metadata": {"scope": "s6-isolated-maintenance"}})
    active = store.current()
    role = permissions.reader_role(active["snapshot"], store.index, False) if active else {"cluster": [], "indices": []}
    es.request("elastic", "PUT", f"/_security/role/{store.reader}", role)
    for pid, profile in profiles.items():
        principal = normalize_authentication_info(profile["authentication_info"])
        es.request("elastic", "PUT", f"/_security/user/{pid}", {"password": permissions.credential(pid, store.credentials),
                   "roles": [store.reader], "enabled": profile["mock_session"]["enabled"], "metadata": principal.es_metadata_candidate()})


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation", action="store_true", help="Use only the fixed isolated S6 index/users")
    subs = parser.add_subparsers(dest="command", required=True)
    export = subs.add_parser("export")
    export.add_argument("--sql", type=Path, default=SQL)
    export.add_argument("--complete", action="store_true", required=True)
    export.add_argument("--out", type=Path, required=True)
    subs.add_parser("init-validation")
    stage = subs.add_parser("plan")
    stage.add_argument("--source", type=Path, required=True)
    commit = subs.add_parser("apply")
    commit.add_argument("--plan", required=True)
    commit.add_argument("--approved-by", required=True)
    commit.add_argument("--approve-release", action="store_true")
    commit.add_argument("--allow-delete", action="store_true")
    rollback = subs.add_parser("rollback-content")
    rollback.add_argument("--from-batch", required=True)
    rollback.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    es.require(os.environ.get("APP_ENV") in ("development", "test") and os.environ.get("MOCK_AUTH_ENABLED") == "1", "Local mock environment required")
    store = CatalogStore(args.validation)
    if args.command in ("export", "rollback-content"):
        es.require(not args.out.exists(), "Output exists; use a new path")
        value = export_snapshot(store, args.sql, args.complete) if args.command == "export" else content_rollback(store, args.from_batch)
        atomic_json(args.out, value)
        print(json.dumps({"created": str(args.out), "complete": True}))
    elif args.command == "init-validation":
        init_validation(store)
        print("Isolated validation identities ready; no real source or business account changed.")
    elif args.command == "plan":
        plan_id, diff = prepare(store, read_json(args.source))
        print(json.dumps({"plan_id": plan_id, "diff": diff}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(apply(store, args.plan, args.approved_by, args.approve_release, args.allow_delete), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
