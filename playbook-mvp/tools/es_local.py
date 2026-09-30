#!/usr/bin/env python3
"""S1 infrastructure only: prepare, explicit bootstrap, and live capability checks.

No business fixtures are imported. Do not import this admin utility into the API.
Run with uv run python -B tools/es_local.py --help.
"""

import argparse
import base64
import copy
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import socket
import ssl
import subprocess
import time
from datetime import datetime, timezone
from urllib.error import HTTPError
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener


ROOT = Path(__file__).resolve().parents[1]
LOCAL = ROOT / ".local"
REPORT = ROOT / "reports/S1_CHECKS.json"
URL = "https://localhost:19200"
INDEX = "playbook-s1-capability"
IMPORTER = "pb_s1_importer"
READERS = {"pb_s1_nce": [0], "pb_s1_ospa": [1]}
READER_ROLE = "pb_s1_reader"
SNAPSHOT = hashlib.sha256(b"S1 synthetic capability data, not SQL").hexdigest()
EXPECTED = {
    "pb_s1_nce": ["nce", "public_nce", "public_ospa"],
    "pb_s1_ospa": ["ospa", "public_nce", "public_ospa"],
}


def now():
    return datetime.now(timezone.utc).isoformat()


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def docker(*args):
    return subprocess.check_output(["docker", *args], cwd=ROOT, text=True).strip()


def password(user):
    require(user in {"elastic", IMPORTER, *READERS, "pb_s2_importer", "pb_s2_probe", "pb_s3_importer"}, "Unknown infrastructure identity")
    return (LOCAL / f"{user}_password").read_text().strip()


class ESHTTPError(RuntimeError):
    """Status only; callers must not expose upstream bodies or credentials."""

    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


def request(user, method, path, body=None, expected=200, *, password_value=None):
    # A new request owns its Authorization. There is no shared mutable identity.
    headers = {"Content-Type": "application/x-ndjson" if isinstance(body, bytes) else "application/json"}
    if user:
        secret = password(user) if password_value is None else password_value
        require(isinstance(secret, str) and bool(secret), "Missing request credential")
        token = base64.b64encode(f"{user}:{secret}".encode()).decode()
        headers["Authorization"] = f"Basic {token}"
    context = ssl.create_default_context(cafile=LOCAL / "ca.crt")
    # Local requests never send credentials through an environment HTTP proxy.
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=context))
    data = body if isinstance(body, bytes) else None if body is None else json.dumps(body).encode()
    req = Request(URL + path, data=data, headers=headers, method=method)
    try:
        response = opener.open(req, timeout=30)
    except HTTPError as error:
        response = error
    with response:
        status = response.code
        raw = response.read()
    # Never echo request/response bodies or headers on an error: they may contain secrets.
    accepted = expected if isinstance(expected, tuple) else (expected,)
    if status not in accepted:
        raise ESHTTPError(status, f"{user or 'anonymous'} {method} {path}: HTTP {status}, expected {expected}")
    return json.loads(raw) if raw else {"status": status}


def prepare():
    LOCAL.mkdir(mode=0o700, exist_ok=True)
    LOCAL.chmod(0o700)
    for user in ("elastic", IMPORTER, *READERS):
        path = LOCAL / f"{user}_password"
        if not path.exists():
            with path.open("x") as stream:
                stream.write(secrets.token_hex(32) + "\n")
        # The root init container copies only the bootstrap secret into its
        # named volume as root:root 0640 for ES uid 1000, gid 0 to read.
        path.chmod(0o600)
    print("Prepared local credentials; existing passwords preserved, none printed.")


def dls_query(snapshot=SNAPSHOT, approved_by="s1-capability-check"):
    # Defaults belong to S1 synthetic probes. S2 supplies the actual SQL hash
    # and derives active roles strictly from the formal source relationships.
    query = {"bool": {"filter": [
        {"term": {"lab_release.approved": True}},
        {"term": {"source.snapshot_sha256": snapshot}},
        {"term": {"lab_release.source_snapshot_sha256": snapshot}},
        {"term": {"lab_release.approved_by": approved_by}} if approved_by is not None else {
            "bool": {"filter": [{"exists": {"field": "lab_release.approved_by"}}],
                     "must_not": [{"term": {"lab_release.approved_by": ""}}]}},
        {"exists": {"field": "lab_release.approved_at"}},
        {"term": {"acl.state": "resolved"}},
        {"term": {"acl.menu_active_raw": 1}},
        {"bool": {"minimum_should_match": 1, "should": [
            {"term": {"acl.is_public": True}},
            {"bool": {"filter": [
                {"terms": {"acl.allowed_role_ids": "ROLE_IDS"}},
                {"terms": {"acl.active_role_ids": "ROLE_IDS"}},
                {"terms": {"acl.required_business_category": "BUSINESSES"}},
            ]}},
        ]}},
    ]}}
    source = json.dumps(query)
    for marker, field in (("ROLE_IDS", "playbook_role_ids"), ("BUSINESSES", "playbook_business_categories")):
        source = source.replace(f'"{marker}"', "{{#toJson}}_user.metadata." + field + "{{/toJson}}")
    return {"template": {"source": source}}


def synthetic_documents():
    base = {
        "text": "合同的生成 Azure 预留实例 NCE OSPA 订阅取消",
        "source": {"snapshot_sha256": SNAPSHOT},
        "publication_state": "unverified",
        "lab_release": {"approved": True, "source_snapshot_sha256": SNAPSHOT,
                        "approved_by": "s1-capability-check", "approved_at": now()},
        "acl": {"state": "resolved", "menu_active_raw": 1, "is_public": False,
                "allowed_role_ids": [2], "active_role_ids": [2], "required_business_category": 0},
    }
    docs = {}
    cases = {
        "nce": {}, "ospa": {"acl.required_business_category": 1},
        "public_nce": {"acl.is_public": True, "acl.allowed_role_ids": [], "acl.active_role_ids": []},
        "public_ospa": {"acl.is_public": True, "acl.required_business_category": 1},
        "other_role": {"acl.allowed_role_ids": [1], "acl.active_role_ids": [1]},
        "no_role": {"acl.allowed_role_ids": []},
        "inactive_role": {"acl.active_role_ids": []},
        "closed": {"acl.is_public": True, "lab_release.approved": False},
        "unresolved": {"acl.is_public": True, "acl.state": "unresolved"},
        "inactive_menu": {"acl.is_public": True, "acl.menu_active_raw": 0},
        "stale_approval": {"acl.is_public": True, "lab_release.source_snapshot_sha256": "old"},
        "changed_source": {"acl.is_public": True, "source.snapshot_sha256": "new"},
        "missing_actor": {"acl.is_public": True, "lab_release.approved_by": None},
        "missing_time": {"acl.is_public": True, "lab_release.approved_at": None},
    }
    for key, changes in cases.items():
        document = copy.deepcopy(base)
        for dotted, value in changes.items():
            parent, field = dotted.split(".")
            document[parent][field] = value
        docs[key] = document
    return docs


def bootstrap(start_trial):
    docker("compose", "cp", "es:/usr/share/elasticsearch/config/certs/ca/ca.crt", str(LOCAL / "ca.crt"))
    state_path = LOCAL / "bootstrap.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"started_at": now()}
    license_info = request("elastic", "GET", "/_license")["license"]
    state.setdefault("license_before", license_info)
    if license_info["type"] == "basic":
        require(start_trial, "DLS needs a suitable license. Re-run bootstrap --start-trial to explicitly activate the trial.")
        trial = request("elastic", "POST", "/_license/start_trial?acknowledge=true")
        require(trial.get("trial_was_started") is True, "Trial was not activated; stop before DLS.")
        state["trial_activation"] = {"at": now(), "response": trial}
        write_json(state_path, state)
        license_info = request("elastic", "GET", "/_license")["license"]
    require(license_info["status"] == "active" and license_info["type"] in {"trial", "platinum", "enterprise"}, "No active DLS-capable license")
    state["license_after"] = license_info
    request("elastic", "PUT", f"/_security/role/{IMPORTER}", {
        "cluster": ["monitor"], "indices": [{"names": [INDEX], "privileges": ["manage", "write", "read"]}],
    })
    request("elastic", "PUT", f"/_security/role/{READER_ROLE}", {
        "cluster": [], "indices": [{"names": [INDEX], "privileges": ["read"], "query": dls_query()}],
    })
    for user in (IMPORTER, *READERS):
        metadata = {"scope": "synthetic-s1-only"}
        if user in READERS:
            metadata.update(playbook_role_ids=[2], playbook_business_categories=READERS[user])
        request("elastic", "PUT", f"/_security/user/{user}", {
            "password": password(user), "roles": [IMPORTER if user == IMPORTER else READER_ROLE],
            "enabled": True, "metadata": metadata,
        })
    # No destructive reset. Re-runs update only this fixed synthetic dataset.
    existing = request(IMPORTER, "HEAD", f"/{INDEX}", expected=(200, 404))
    if existing["status"] == 404:
        properties = {
            "text": {"type": "text", "analyzer": "smartcn"},
            "source": {"properties": {"snapshot_sha256": {"type": "keyword"}}},
            "publication_state": {"type": "keyword"},
            "lab_release": {"properties": {
                "approved": {"type": "boolean"}, "source_snapshot_sha256": {"type": "keyword"},
                "approved_by": {"type": "keyword"}, "approved_at": {"type": "date"},
            }},
            "acl": {"properties": {name: {"type": kind} for name, kind in {
                "state": "keyword", "menu_active_raw": "integer", "is_public": "boolean",
                "allowed_role_ids": "integer", "active_role_ids": "integer", "required_business_category": "integer",
            }.items()}},
        }
        request(IMPORTER, "PUT", f"/{INDEX}", {
            "settings": {"number_of_shards": 1, "number_of_replicas": 0},
            "mappings": {"dynamic": "strict", "properties": properties},
        })
        state["index_created"] = True
        write_json(state_path, state)
    for key, document in synthetic_documents().items():
        request(IMPORTER, "PUT", f"/{INDEX}/_doc/{key}", document, expected=(200, 201))
    request(IMPORTER, "POST", f"/{INDEX}/_refresh")
    state.update(documents_written=True, completed_at=now())
    write_json(state_path, state)
    print(f"Bootstrapped {INDEX}; native importer + 2 synthetic readers; license={license_info['type']}.")


def tls_evidence():
    result = {}
    ca = ssl.create_default_context(cafile=LOCAL / "ca.crt")
    for label, context, hostname, should_pass in (
        ("trusted_ca_localhost", ca, "localhost", True),
        ("trusted_ca_loopback", ca, "127.0.0.1", True),
        ("untrusted_ca", ssl.create_default_context(), "localhost", False),
        ("wrong_hostname", ca, "wrong.invalid", False),
    ):
        try:
            with socket.create_connection(("127.0.0.1", 19200), timeout=10) as connection:
                with context.wrap_socket(connection, server_hostname=hostname) as secure:
                    require(should_pass, f"TLS unexpectedly accepted {label}")
                    result[label] = {"protocol": secure.version(), "not_after": secure.getpeercert()["notAfter"],
                                     "certificate_sha256": hashlib.sha256(secure.getpeercert(binary_form=True)).hexdigest()}
        except ssl.SSLCertVerificationError:
            require(not should_pass, f"TLS unexpectedly rejected {label}")
            result[label] = "certificate_verification_rejected"
    return result


def check_readers():
    docs = sorted(synthetic_documents())
    result = {}
    request(None, "GET", f"/{INDEX}/_search", expected=401)
    for user, allowed in EXPECTED.items():
        identity = request(user, "GET", "/_security/_authenticate")
        require(identity["username"] == user and identity["roles"] == [READER_ROLE], "Unexpected reader identity/roles")
        require(identity["authentication_realm"]["type"] == "native", "Reader is not native")
        require(identity["metadata"]["playbook_role_ids"] == [2] and identity["metadata"]["playbook_business_categories"] == READERS[user], "Unexpected reader metadata")
        # No application ACL filter is sent. ES DLS must produce the exact set.
        hits = request(user, "POST", f"/{INDEX}/_search", {"size": 100, "track_total_hits": True, "query": {"match_all": {}}})["hits"]
        require(sorted(hit["_id"] for hit in hits["hits"]) == allowed and hits["total"]["value"] == len(allowed), "DLS search mismatch")
        batch = request(user, "POST", f"/{INDEX}/_mget", {"ids": docs})["docs"]
        require([hit["_id"] for hit in batch if hit.get("found")] == allowed, "DLS mget mismatch")
        require(all("error" not in hit for hit in batch), "mget contains errors")
        for key in docs:
            item = request(user, "GET", f"/{INDEX}/_doc/{key}", expected=200 if key in allowed else 404)
            require(item["found"] == (key in allowed), "DLS get mismatch")
        request(user, "PUT", f"/{INDEX}/_doc/forbidden-write", {"text": "forbidden"}, expected=403)
        request(user, "PUT", f"/_security/role/{READER_ROLE}", {"cluster": ["all"]}, expected=403)
        request(user, "PUT", f"/_security/user/{user}", {"roles": [IMPORTER]}, expected=403)
        result[user] = {"roles": identity["roles"], "metadata": identity["metadata"], "match_all_ids": allowed,
                        "get_cases": len(docs), "mget_cases": len(docs), "write_http": 403, "role_write_http": 403, "self_escalation_http": 403}
    return result


def persistence_snapshot():
    all_docs = request(IMPORTER, "POST", f"/{INDEX}/_search", {"size": 100, "query": {"match_all": {}}})["hits"]["hits"]
    require(sorted(hit["_id"] for hit in all_docs) == sorted(synthetic_documents()), "Synthetic index has unexpected documents")
    sources = {hit["_id"]: hit["_source"] for hit in all_docs}
    return {
        "cluster_uuid": request(IMPORTER, "GET", "/")["cluster_uuid"],
        "index_uuid": request(IMPORTER, "GET", f"/{INDEX}/_settings")[INDEX]["settings"]["index"]["uuid"],
        "documents_sha256": hashlib.sha256(json.dumps(sources, sort_keys=True).encode()).hexdigest(),
        "users": request("elastic", "GET", f"/_security/user/{IMPORTER}," + ",".join(READERS)),
        "roles": request("elastic", "GET", f"/_security/role/{IMPORTER},{READER_ROLE}"),
        "tls": tls_evidence(),
    }


def check(restart):
    report = {"stage": "S1", "started_at": now(), "status": "running", "scope": "synthetic only; no SQL documents/profiles imported"}
    write_json(REPORT, report)
    try:
        report["bootstrap"] = json.loads((LOCAL / "bootstrap.json").read_text())
        report["license"] = request("elastic", "GET", "/_license")["license"]
        require(report["license"]["status"] == "active", "License is not active")
        report["server"] = request(IMPORTER, "GET", "/")
        require(report["server"]["version"]["number"] == "9.5.4", "Unexpected ES version")
        identity = request(IMPORTER, "GET", "/_security/_authenticate")
        require(identity["username"] == IMPORTER and identity["roles"] == [IMPORTER] and identity["authentication_realm"]["type"] == "native", "Unexpected importer identity")
        report["importer"] = {"username": identity["username"], "roles": identity["roles"]}
        report["health"] = request("elastic", "GET", "/_cluster/health?wait_for_status=green&timeout=30s")
        require(report["health"]["status"] == "green" and not report["health"]["timed_out"], "Cluster is not green")
        plugins = request("elastic", "GET", "/_nodes/plugins?filter_path=nodes.*.plugins")
        require(all(any(p["name"] == "analysis-smartcn" and p["version"] == "9.5.4" for p in n["plugins"]) for n in plugins["nodes"].values()), "Missing matching smartcn plugin")
        report["plugins"] = plugins
        report["analyze"] = {}
        for text in ("合同的生成", "Azure 预留实例", "NCE OSPA 订阅取消"):
            tokens = request(IMPORTER, "POST", f"/{INDEX}/_analyze", {"analyzer": "smartcn", "text": text})["tokens"]
            require(bool(tokens), "Analyzer returned no tokens")
            report["analyze"][text] = [t["token"] for t in tokens]
        report["readers_before"] = check_readers()
        request(IMPORTER, "PUT", "/_security/role/pb_s1_out_of_scope", {"cluster": []}, expected=403)
        request(IMPORTER, "PUT", "/playbook-s1-out-of-scope", {}, expected=403)
        report["importer_scope_denied"] = {"security_role": 403, "other_index": 403}
        report["anonymous_http"] = 401
        before = persistence_snapshot()
        report["tls"] = before["tls"]
        container_id = docker("compose", "ps", "-q", "es")
        container = json.loads(docker("inspect", container_id))[0]
        require(container["NetworkSettings"]["Ports"]["9200/tcp"] == [{"HostIp": "127.0.0.1", "HostPort": "19200"}], "ES port is not loopback-only")
        report["environment"] = {"host_architecture": platform.machine(), "python": platform.python_version(),
            "docker": json.loads(docker("info", "--format", '{"cpus":{{.NCPU}},"memory_bytes":{{.MemTotal}},"root":"{{.DockerRootDir}}","version":"{{.ServerVersion}}"}')),
            "image_id": container["Image"], "memory_limit": container["HostConfig"]["Memory"],
            "ports": container["NetworkSettings"]["Ports"], "mounts": container["Mounts"]}
        report["restart"] = {"executed": restart}
        if restart:
            docker("compose", "restart", "es")
            deadline = time.monotonic() + 180
            while docker("inspect", "--format", "{{.State.Health.Status}}", container_id) != "healthy":
                require(time.monotonic() < deadline, "ES did not become healthy after restart")
                time.sleep(2)
            after = persistence_snapshot()
            require(before == after, "Index/data/users/roles/certificates changed across restart")
            report["readers_after"] = check_readers()
            restarted = json.loads(docker("inspect", container_id))[0]
            require(restarted["State"]["StartedAt"] != container["State"]["StartedAt"], "Container did not restart")
            report["restart"].update(persisted=True, cluster_uuid=before["cluster_uuid"], index_uuid=before["index_uuid"],
                documents_sha256=before["documents_sha256"], before_started_at=container["State"]["StartedAt"], after_started_at=restarted["State"]["StartedAt"])
        report.update(status="passed", completed_at=now())
    except Exception as error:
        report.update(status="failed", failed_at=now(), error_type=type(error).__name__)
        write_json(REPORT, report)
        raise
    write_json(REPORT, report)
    print(f"S1 checks passed; restart={restart}; report={REPORT.relative_to(ROOT)}")


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare", help="Create ignored local secrets; never print or rotate existing ones")
    bootstrap_parser = commands.add_parser("bootstrap", help="Create only S1 synthetic index/users/role; admin maintenance")
    bootstrap_parser.add_argument("--start-trial", action="store_true", help="Explicitly acknowledge and activate trial if license is basic")
    check_parser = commands.add_parser("check", help="Run real TLS/DLS/analyzer/identity checks, without re-bootstrap")
    check_parser.add_argument("--restart", action="store_true", help="Also restart ES and compare persisted state")
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    elif args.command == "bootstrap":
        bootstrap(args.start_trial)
    else:
        check(args.restart)


if __name__ == "__main__":
    main()
