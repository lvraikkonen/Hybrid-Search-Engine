#!/usr/bin/env python3
"""S3 native identity provisioning and independent live DLS acceptance checks."""
import argparse
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import stat
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

import es_local as es
import ingest_playbook as ingest
from documents import INDEX, index_definition, load_records
from mock_auth import AuthError, MockIdentityProvider, load_profiles
from tools.extract_playbook_sql import canonical_hash
from runtime import legacy_setup

READER = "pb_document_reader"
PROBE_INDEX = "playbook-s3-policy-checks"
IMPORTER = "pb_s3_importer"
SYNTHETIC_SNAPSHOT = hashlib.sha256(b"S3 isolated policy probes v1; not SQL content").hexdigest()
CREDENTIALS = es.LOCAL / "s3"


def principals():
    provider = MockIdentityProvider(os.environ.get("APP_ENV", ""), os.environ.get("MOCK_AUTH_ENABLED") == "1")
    result = {pid: provider.get_principal(pid) for pid in load_profiles()}
    es.require(len(result) == 20, "Expected exactly 20 business profiles")
    es.require(len({p.user_id for p in result.values()}) == 20, "Duplicate business user ID")
    for pid, principal in result.items():
        es.require(re.fullmatch(r"pb_[a-z0-9_]+", pid) and principal.user_name == pid, "Unsafe or inconsistent native identity")
    return result


def credential(user, directory=None):
    # Called only after membership and Principal validation; never select an
    # infrastructure password or fall back after a missing/invalid credential.
    directory = directory or CREDENTIALS
    path = directory / f"{user}_password"
    es.require(not directory.is_symlink() and stat.S_IMODE(directory.stat().st_mode) == 0o700, "Unsafe credential directory")
    es.require(not path.is_symlink() and stat.S_IMODE(path.stat().st_mode) == 0o600, "Unsafe credential file")
    value = path.read_text().strip()
    es.require(re.fullmatch(r"[0-9a-f]{64}", value), "Invalid native credential")
    return value


def request_as(profile_id, method, path, body=None, expected=200):
    users = principals()
    if profile_id not in users:
        raise AuthError("unknown_mock_profile")
    user = users[profile_id].user_name
    return es.request(user, method, path, body, expected, password_value=credential(user))


def reader_role(snapshot, index=INDEX, include_probes=True):
    return {"cluster": [], "run_as": [], "indices": [
        {"names": [name], "privileges": ["read"], "query": es.dls_query(sha, approved_by=None)}
        for name, sha in ([(index, snapshot)] + ([(PROBE_INDEX, SYNTHETIC_SNAPSHOT)] if include_probes else []))
    ]}


def policy_probes():
    # Reuse S1 gates; synthetic records live only in their own index/namespace.
    docs = es.synthetic_documents()
    for doc in docs.values():
        doc["title"] = doc.pop("text")
        doc["source"]["instance_key"] = "s3-synthetic"
        for obj, field in ((doc["source"], "snapshot_sha256"), (doc["lab_release"], "source_snapshot_sha256")):
            if obj[field] == es.SNAPSHOT:
                obj[field] = SYNTHETIC_SNAPSHOT
        if doc["lab_release"]["approved_by"] is not None:
            doc["lab_release"]["approved_by"] = "codex-local:S3-synthetic"
        doc["lab_release"]["approved_at"] = "2026-09-29T00:00:00Z" if doc["lab_release"]["approved_at"] else None
    for role in (1, 2, 3):
        for business in (0, 1):
            doc = copy.deepcopy(docs["nce"])
            doc["acl"].update(allowed_role_ids=[role], active_role_ids=[role], required_business_category=business)
            docs[f"role_{role}_{business}"] = doc
    docs["empty_actor"] = copy.deepcopy(docs["public_nce"])
    docs["empty_actor"]["lab_release"]["approved_by"] = ""
    docs["crossed_roles"] = copy.deepcopy(docs["nce"])
    docs["crossed_roles"]["acl"].update(allowed_role_ids=[1], active_role_ids=[2])
    return docs


# Hand-written expectations, not evaluated by production ACL/DLS code.
PROBE_ALLOWED = {
    (1, (0, 1)): {"public_nce", "public_ospa", "other_role", "role_1_0", "role_1_1"},
    (3, (0, 1)): {"public_nce", "public_ospa", "role_3_0", "role_3_1"},
    (2, (0,)): {"public_nce", "public_ospa", "nce", "role_2_0"},
    (2, (1,)): {"public_nce", "public_ospa", "ospa", "role_2_1"},
    (2, (0, 1)): {"public_nce", "public_ospa", "nce", "ospa", "role_2_0", "role_2_1"},
}


@legacy_setup()
def init(users, records, audit, definition):
    ingest.valid_batch(records, audit)
    ingest.license_check("elastic")
    ingest.ensure_index(definition)
    CREDENTIALS.mkdir(mode=0o700, exist_ok=True)
    es.require(not CREDENTIALS.is_symlink(), "Unsafe credential directory")
    CREDENTIALS.chmod(0o700)
    for user in [IMPORTER, *users]:
        path = (es.LOCAL if user == IMPORTER else CREDENTIALS) / f"{user}_password"
        es.require(not path.is_symlink(), "Unsafe credential file")
        if not path.exists():
            with path.open("x") as stream:
                stream.write(secrets.token_hex(32) + "\n")
        path.chmod(0o600)
    es.request("elastic", "PUT", f"/_security/role/{IMPORTER}", {
        "cluster": ["monitor"], "indices": [{"names": [PROBE_INDEX], "privileges": ["manage", "read", "write"]}]})
    es.request("elastic", "PUT", f"/_security/user/{IMPORTER}", {
        "password": es.password(IMPORTER), "roles": [IMPORTER], "enabled": True, "metadata": {"scope": "s3-synthetic-only"}})
    role = reader_role(audit["source_snapshot_sha256"])
    es.request("elastic", "PUT", f"/_security/role/{READER}", role)
    for pid, principal in users.items():
        es.request("elastic", "PUT", f"/_security/user/{pid}", {
            "password": credential(pid), "roles": [READER], "enabled": True,
            "metadata": principal.es_metadata_candidate()})
    if es.request(IMPORTER, "HEAD", f"/{PROBE_INDEX}", expected=(200, 404))["status"] == 404:
        es.request(IMPORTER, "PUT", f"/{PROBE_INDEX}", definition)
    actual_mapping = es.request(IMPORTER, "GET", f"/{PROBE_INDEX}/_mapping")[PROBE_INDEX]["mappings"]
    es.require(actual_mapping == definition["mappings"], "Synthetic mapping changed; explicit review required")
    for key, doc in policy_probes().items():
        es.request(IMPORTER, "PUT", f"/{PROBE_INDEX}/_doc/{key}", doc, expected=(200, 201))
    es.request(IMPORTER, "POST", f"/{PROBE_INDEX}/_refresh")
    result = {"at": es.now(), "status": "passed", "profiles": sorted(users), "reader_role": role,
              "synthetic_index": PROBE_INDEX, "synthetic_documents": len(policy_probes()), "business_profiles": len(users)}
    ingest.history("S3_INIT.json", result)
    return result


def check_item(item, key, expected_source):
    es.require("error" not in item and item["_id"] == key, f"Invalid read response for {key}")
    if expected_source is None:
        es.require(item.get("found") is False and "_source" not in item and "fields" not in item,
                   f"Denied document leaked through read: {key}")
    else:
        es.require(item.get("found") is True and item.get("_source") == expected_source, f"Source differs for {key}")


def check_paths(pid, index, sources, allowed):
    keys = sorted(sources)
    result = request_as(pid, "POST", f"/{index}/_search", {
        "size": 1000, "track_total_hits": True, "query": {"match_all": {}}})
    es.require(not result["timed_out"] and result["_shards"]["failed"] == 0, "Incomplete search")
    hits = result["hits"]
    actual = [h["_id"] for h in hits["hits"]]
    es.require(len(actual) == len(allowed) and set(actual) == allowed and hits["total"] == {"value": len(allowed), "relation": "eq"}, f"{pid}: full search ID set differs")
    es.require(all(h["_source"] == sources[h["_id"]] for h in hits["hits"]), "Search content/metadata differs")
    # Each ID has its own query. This is ID selection, not a repeated ACL filter.
    ndjson = []
    for key in keys:
        ndjson.extend([{}, {"size": 1, "track_total_hits": True, "query": {"ids": {"values": [key]}}}])
    payload = ("\n".join(json.dumps(row) for row in ndjson) + "\n").encode()
    searches = request_as(pid, "POST", f"/{index}/_msearch", payload)["responses"]
    es.require(len(searches) == len(keys), "Incomplete msearch")
    for key, response in zip(keys, searches):
        es.require("error" not in response and response["status"] == 200 and not response["timed_out"] and response["_shards"]["failed"] == 0, f"Search failed for {key}")
        expected = [key] if key in allowed else []
        es.require([h["_id"] for h in response["hits"]["hits"]] == expected and response["hits"]["total"] == {"value": len(expected), "relation": "eq"}, f"ID search differs for {key}")
        es.require(all(h["_source"] == sources[key] for h in response["hits"]["hits"]), "ID search metadata differs")
    batch = request_as(pid, "POST", f"/{index}/_mget", {"ids": keys})["docs"]
    es.require(len(batch) == len(keys), "Incomplete mget")
    for key, item in zip(keys, batch):
        check_item(item, key, sources[key] if key in allowed else None)
    for key in keys:
        item = request_as(pid, "GET", f"/{index}/_doc/{quote(key, safe='')}", expected=200 if key in allowed else 404)
        check_item(item, key, sources[key] if key in allowed else None)
    missing = request_as(pid, "GET", f"/{index}/_doc/s3-not-in-source", expected=404)
    check_item(missing, "s3-not-in-source", None)
    return {"allowed_keys": sorted(allowed), "denied_count": len(keys) - len(allowed),
            "match_all_verified": True, "id_search_cases": len(keys), "get_cases": len(keys), "mget_cases": len(keys),
            "source_and_metadata_verified": True, "missing_http": 404}


def check_user(pid, principal, sources, expected, probes):
    identity = request_as(pid, "GET", "/_security/_authenticate")
    es.require(identity["username"] == pid and identity["roles"] == [READER] and identity["authentication_realm"]["type"] == "native", f"{pid}: wrong native identity")
    es.require(identity["enabled"] and identity["metadata"] == principal.es_metadata_candidate(), f"{pid}: wrong trusted metadata")
    privileges = request_as(pid, "GET", "/_security/user/_privileges")
    es.require(not privileges["cluster"] and not privileges["run_as"] and not privileges["applications"], "Reader has extra privileges")
    es.require(len(privileges["indices"]) == 2 and {tuple(p["names"]) for p in privileges["indices"]} == {(INDEX,), (PROBE_INDEX,)}, "Unexpected reader index scope")
    es.require(all(p["privileges"] == ["read"] and p.get("query") for p in privileges["indices"]), "Unrestricted reader or write privilege")
    real = check_paths(pid, INDEX, sources, {f"playbook-test:Document:{did}" for did in expected["allowed_document_ids"]})
    synthetic = check_paths(pid, PROBE_INDEX, probes, PROBE_ALLOWED[(principal.role_id, principal.effective_business_categories)])
    denied = [
        ("write_body", "PUT", f"/{INDEX}/_create/s3-forbidden", {"title": "must be denied"}),
        ("write_acl", "POST", f"/{PROBE_INDEX}/_update/public_nce", {"doc": {"acl": {"is_public": False}}}),
        ("write_role", "PUT", f"/_security/role/{READER}", {"cluster": ["all"]}),
        ("self_escalation", "PUT", f"/_security/user/{pid}", {"roles": ["superuser"]}),
        ("self_metadata", "PUT", f"/_security/user/{pid}", {"roles": [READER], "metadata": {"playbook_business_categories": [0, 1]}}),
        ("other_index", "GET", f"/{es.INDEX}/_search", None),
    ]
    for _, method, path, body in denied:
        request_as(pid, method, path, body, expected=403)
    return {"username": pid, "roles": identity["roles"], "metadata": identity["metadata"],
            "principal": principal.to_dict(), "real": real, "synthetic": synthetic, "denied_http": {name: 403 for name, *_ in denied}}


def check(users, records, audit, definition):
    report = {"stage": "S3", "started_at": es.now(), "status": "running", "users": {}}
    path = ROOT / "reports/S3_CHECKS.json"
    es.write_json(path, report)
    try:
        ingest.valid_batch(records, audit)
        report["license"] = ingest.license_check(ingest.IMPORTER)
        report["tls"] = es.tls_evidence()
        ingest.ensure_index(definition)
        current = ingest.current_documents()
        ingest.verify_sources(records, current)
        approval = json.loads((ROOT / "reports/S2_RELEASES.json").read_text())[-1]["approval"]
        es.require(approval["approved"] and approval["source_snapshot_sha256"] == audit["source_snapshot_sha256"], "Wrong open snapshot")
        es.require(all(h["_source"]["lab_release"] == approval for h in current.values()), "S2 release changed")
        sources = {key: hit["_source"] for key, hit in current.items()}
        report.update(source_snapshot_sha256=audit["source_snapshot_sha256"], documents_sha256=canonical_hash(sources), lab_release=approval)
        oracle = json.loads((ROOT / "fixtures/expected_access.json").read_text())
        es.require(oracle["source"]["sha256"] == audit["source_snapshot_sha256"], "Oracle snapshot mismatch")
        expectations = {e["profile_id"]: e for e in oracle["expectations"]}
        es.require(set(expectations) == set(users), "Oracle/profile set differs")
        universe = {r["document_id"] for r in records}
        for pid, e in expectations.items():
            allow, deny = set(e["allowed_document_ids"]), set(e["denied_document_ids"])
            es.require(allow | deny == universe and not allow & deny and len(allow) == e["allow_count"] and len(deny) == e["deny_count"], "Invalid independent oracle")
            es.require(e["role_id"] == users[pid].role_id and e["effective_business_categories"] == list(users[pid].effective_business_categories), "Oracle identity differs")
        probes = policy_probes()
        es.request(None, "GET", f"/{INDEX}/_search", expected=401)
        # ponytail: four bounded workers exercise request-local credentials for
        # this 20-user lab; API/session concurrency is a separate S4/S5 check.
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = {pool.submit(check_user, pid, p, sources, expectations[pid], probes): pid for pid, p in users.items()}
            for future in as_completed(jobs):
                pid = jobs[future]
                report["users"][pid] = future.result()
                es.write_json(path, report)
                print(f"{pid}: 130 real + {len(probes)} synthetic IDs passed on search/GET/mget", flush=True)
        after = {key: hit["_source"] for key, hit in ingest.current_documents().items()}
        es.require(canonical_hash(after) == report["documents_sha256"], "Real documents changed during permission checks")
        probe_hits = es.request(IMPORTER, "POST", f"/{PROBE_INDEX}/_search", {"size": 1000, "query": {"match_all": {}}})["hits"]["hits"]
        es.require({h["_id"]: h["_source"] for h in probe_hits} == probes, "Synthetic data changed during denied writes")
        report.update(status="passed", completed_at=es.now(), real_pairs=20 * len(records),
                      synthetic_pairs=20 * len(probes), paths=["id_search", "GET", "mget"],
                      denied_operations=20 * 6, anonymous_http=401, workers=4, real_data_unchanged=True)
    except Exception as error:
        report.update(status="failed", completed_at=es.now(), error_type=type(error).__name__, message=str(error))
        raise
    finally:
        es.write_json(path, report)
    return {k: v for k, v in report.items() if k != "users"}


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("init", "check"))
    args = parser.parse_args()
    users = principals()  # Validate all 20 before any credential or ES operation.
    records, audit = load_records()
    definition = index_definition(json.loads((ROOT / "config/synonyms.json").read_text()))
    result = init(users, records, audit, definition) if args.command == "init" else check(users, records, audit, definition)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        ingest.history("S3_FAILURES.json", {"at": es.now(), "command": sys.argv[1] if len(sys.argv) > 1 else None,
                                          "error_type": type(error).__name__, "message": str(error)})
        raise
