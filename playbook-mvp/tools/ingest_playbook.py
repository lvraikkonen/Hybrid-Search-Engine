#!/usr/bin/env python3
"""S2 local ingestion commands. Admin is used only by init; no user-query fallback."""
import argparse
import copy
import json
import os
from pathlib import Path
import secrets
import sys
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tools import es_local as es
from documents import CLOSED, IMPORTER, INDEX, PROBE, READER_ROLE, SQL, index_definition, load_records
from tools.extract_playbook_sql import canonical_hash
from runtime import legacy_setup


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    es.write_json(path, data)


def history(name, event):
    path = ROOT / "reports" / name
    events = json.loads(path.read_text()) if path.exists() else []
    events.append(event)
    save(path, events)


def prepare_dataset():
    records, audit = load_records()
    snapshot = audit["source_snapshot_sha256"]
    target = es.LOCAL / "s2" / snapshot
    target.mkdir(parents=True, exist_ok=True)
    for record in records:
        directory = target / "documents" / str(record["document_id"])
        directory.mkdir(parents=True, exist_ok=True)
        for lang, value in record["languages"].items():
            if value["raw"] is None:
                continue
            path = directory / f"{lang}.md"
            raw = value["raw"].encode("utf-8")
            if path.exists():
                es.require(path.read_bytes() == raw, f"Original export changed: {record['document_id']}/{lang}")
            else:
                path.write_bytes(raw)
    (target / "records.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))
    synonyms = json.loads((ROOT / "config/synonyms.json").read_text())
    definition = index_definition(synonyms)
    save(target / "mapping.json", definition)
    save(target / "quarantine.json", audit["quarantine"])
    manifest = {
        "stage": "S2", "mode": "dry_run_no_es_writes", "source_snapshot_sha256": snapshot,
        "source_audit": audit["source_audit"], "records": len(records),
        "raw_fields": sum(len(r["languages"]) for r in records),
        "quarantine": [{k:v for k,v in q.items() if k != "source_row"} for q in audit["quarantine"]],
        "mapping_sha256": canonical_hash(definition), "synonyms_sha256": canonical_hash(synonyms),
        "artifacts_directory": str(target.relative_to(ROOT)),
        "documents": [{"document_id": r["document_id"], "document_key": r["document_key"],
            "content_revision": r["content_revision"], "acl_revision": r["acl_revision"],
            "raw_sha256": {lang:v["raw_utf8_sha256"] for lang,v in r["languages"].items()},
            "same_content": r["same_content"], "acl_state": r["acl"]["state"],
            "table_warnings": {lang:v["table_warnings"] for lang,v in r["languages"].items() if v["table_warnings"]}}
            for r in records],
    }
    save(ROOT / "reports/S2_DRY_RUN.json", manifest)
    samples = ["# S2 原文、检索文本与表格抽样", "", f"源快照：`{snapshot}`。原文在本机忽略目录逐字节保留；此处只展示检索文本抽样。", ""]
    for did in (25, 63, 99, 148):
        record = next((r for r in records if r["document_id"] == did), None)
        if record is None:
            continue
        value = record["languages"]["zh"]
        lines = value["search_text"].splitlines()
        selected = [line for line in lines if "Title[3]：5.5 |" in line] if did == 148 else lines[:3]
        samples.extend([f"## Document {did}：{record['title']}", "",
            f"业务 {record['business_category']}；正式菜单 {record['acl']['menu_id']}；SQL 行 {record['source']['sql_lines']}；中英源字段相同：{record['same_content']}。", "",
            "```text", "\n".join(selected), "```", "",
            f"表格警告：`{json.dumps(value['table_warnings'], ensure_ascii=False)}`。", ""])
    samples.extend(["合并表格按正式单元格展开：列标题、目录项及行值保持关联。空单元格显示为〔空白〕，不推导为允许/拒绝。Document 148 的末段 rowspan=8 超出实际 7 行，仅重复现有行的标签；原始 HTML 的 rowspan 保持不变。", "",
                    "Document 148 正文虽是权限说明表，应用 ACL 仍只来自 SQL 正式菜单关系；不从这些表格内容生成授权。英文源字段混有中文时仍作为 ContentEn 保留；相同语言字段不删除或翻译。", ""])
    (ROOT / "reports/S2_CONTENT_SAMPLES.md").write_text("\n".join(samples))
    return records, audit, definition, synonyms


def valid_batch(records, audit):
    es.require(not audit["quarantine"], "Quarantine is nonempty; inspect S2_DRY_RUN and local source evidence before import")
    es.require(records and len(records) == audit["source_audit"]["table_counts"]["Document"], "Incomplete source batch")
    # ponytail: one bounded local batch (<=1000 records, <=10 MiB NDJSON).
    # Add paginated reads/chunked writes only when the dataset exceeds this cap.
    es.require(len(records) <= 1000, "Dataset exceeds S2 local batch limit")


def license_check(user):
    license = es.request(user, "GET", "/_license")["license"]
    es.require(license["status"] == "active" and license["type"] in ("trial", "platinum", "enterprise"), "DLS license unavailable; stop")
    return {k: license[k] for k in ("type", "status", "expiry_date") if k in license}


@legacy_setup()
def init(records, audit):
    valid_batch(records, audit)
    license_check("elastic")
    for user in (IMPORTER, PROBE):
        path = es.LOCAL / f"{user}_password"
        if not path.exists():
            with path.open("x") as stream:
                stream.write(secrets.token_hex(32) + "\n")
        path.chmod(0o600)
    es.request("elastic", "PUT", f"/_security/role/{IMPORTER}", {
        "cluster": ["monitor"], "indices": [{"names": [INDEX], "privileges": ["manage", "write", "read"]}]})
    es.request("elastic", "PUT", f"/_security/role/{READER_ROLE}", {
        "cluster": [], "indices": [{"names": [INDEX], "privileges": ["read"],
            "query": es.dls_query(audit["source_snapshot_sha256"], approved_by=None)}]})
    for user, role, metadata in (
        (IMPORTER, IMPORTER, {"scope": "s2-ingestion"}),
        (PROBE, READER_ROLE, {"scope": "s2-release-probe-not-business-profile", "playbook_role_ids": [2], "playbook_business_categories": [0]}),
    ):
        es.request("elastic", "PUT", f"/_security/user/{user}", {
            "password": es.password(user), "roles": [role], "enabled": True, "metadata": metadata})
    return {"importer": IMPORTER, "probe": PROBE, "index": INDEX, "business_profiles_created": 0}


def ensure_index(definition, create=False):
    existing = es.request(IMPORTER, "HEAD", f"/{INDEX}", expected=(200, 404))
    if existing["status"] == 404:
        es.require(create, "Index is missing; run import explicitly")
        es.request(IMPORTER, "PUT", f"/{INDEX}", definition)
    mapping = es.request(IMPORTER, "GET", f"/{INDEX}/_mapping")[INDEX]["mappings"]
    es.require(mapping == definition["mappings"], "Mapping differs; stop for explicit review/reindex, never delete the index")
    actual = es.request(IMPORTER, "GET", f"/{INDEX}/_settings")[INDEX]["settings"]["index"]["analysis"]
    def setting_strings(value):
        if isinstance(value, dict):
            return {k:setting_strings(v) for k,v in value.items()}
        if isinstance(value, list):
            return [setting_strings(v) for v in value]
        return str(value).lower() if isinstance(value, bool) else str(value)
    es.require(setting_strings(actual) == setting_strings(definition["settings"]["analysis"]), "Analyzer settings differ; explicit maintenance required")


def current_documents():
    hits = es.request(IMPORTER, "POST", f"/{INDEX}/_search", {
        "size": 1000, "track_total_hits": True, "seq_no_primary_term": True, "query": {"match_all": {}}})["hits"]
    es.require(hits["total"]["relation"] == "eq" and hits["total"]["value"] == len(hits["hits"]), "Incomplete index read")
    return {h["_id"]: h for h in hits["hits"]}


def without_release(record):
    return {k: v for k, v in record.items() if k != "lab_release"}


def import_actions(records, current):
    es.require(set(current) <= {r["document_key"] for r in records}, "Source removed documents; explicit S6 deletion workflow required")
    actions, unchanged = [], 0
    for record in records:
        key = record["document_key"]
        previous = current.get(key)
        if previous and without_release(previous["_source"]) == without_release(record):
            unchanged += 1  # An identical import preserves an explicit release.
            continue
        candidate = copy.deepcopy(record)
        candidate["lab_release"] = copy.deepcopy(CLOSED)
        if previous:
            action = {"index": {"_id": key, "if_seq_no": previous["_seq_no"], "if_primary_term": previous["_primary_term"]}}
        else:
            action = {"create": {"_id": key}}
        actions.extend([action, candidate])
    return actions, unchanged


def bulk(actions):
    if not actions:
        return {"written": 0, "errors": []}
    payload = ("\n".join(json.dumps(x, ensure_ascii=False) for x in actions) + "\n").encode()
    es.require(len(payload) <= 10 * 1024 * 1024, "Bulk request exceeds S2 size limit")
    response = es.request(IMPORTER, "POST", f"/{INDEX}/_bulk?refresh=wait_for", payload)
    es.require(len(response["items"]) == len(actions) // 2, "Bulk response item count differs")
    errors = []
    for item in response["items"]:
        operation, result = next(iter(item.items()))
        if result["status"] >= 300 or "error" in result:
            errors.append({"operation": operation, "document_key": result["_id"], "status": result["status"],
                           "error_type": result.get("error", {}).get("type", "unknown_error")})
    es.require(bool(errors) == response["errors"], "Bulk error summary differs")
    return {"written": len(response["items"]) - len(errors), "errors": errors}


@legacy_setup()
def apply_import(records, audit, definition):
    valid_batch(records, audit)
    license_check(IMPORTER)
    ensure_index(definition, create=True)
    before = current_documents()
    actions, unchanged = import_actions(records, before)
    result = bulk(actions)
    result.update(at=es.now(), source_snapshot_sha256=audit["source_snapshot_sha256"], unchanged=unchanged,
                  count_before=len(before), count_after=len(current_documents()))
    result["status"] = "failed" if result["errors"] or result["count_after"] != len(records) else "passed"
    history("S2_IMPORTS.json", result)
    es.require(not result["errors"], "Bulk import failed; inspect S2_IMPORTS; no release was performed")
    es.require(result["count_after"] == len(records), "Import count differs")
    return result


def verify_sources(records, current):
    es.require(set(current) == {r["document_key"] for r in records}, "Indexed key set differs from SQL")
    for record in records:
        es.require(without_release(current[record["document_key"]]["_source"]) == without_release(record), f"Source/derived record mismatch: {record['document_id']}")


@legacy_setup()
def release(records, audit, snapshot, actor, definition):
    valid_batch(records, audit)
    es.require(snapshot == audit["source_snapshot_sha256"], "Approval snapshot differs from actual SQL; nothing released")
    es.require(actor and actor.strip() == actor and len(actor) <= 100, "Explicit nonempty actor required")
    license_check(IMPORTER)
    ensure_index(definition)
    current = current_documents()
    verify_sources(records, current)
    es.require(all(r["acl"]["state"] == "resolved" and r["acl"]["menu_active_raw"] == 1 for r in records), "Inactive/unresolved documents cannot be released")
    approval = {"approved": True, "source_snapshot_sha256": snapshot, "approved_by": actor, "approved_at": es.now()}
    journal = es.LOCAL / "s2/releases.jsonl"
    event = {"approval": approval, "index": INDEX, "document_keys": sorted(current), "status": "intent"}
    with journal.open("a") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    actions = []
    for key, previous in current.items():
        actions.extend([{"update": {"_id": key, "if_seq_no": previous["_seq_no"], "if_primary_term": previous["_primary_term"]}}, {"doc": {"lab_release": approval}}])
    result = bulk(actions)
    event.update(result, status="failed" if result["errors"] else "passed")
    with journal.open("a") as stream:
        stream.write(json.dumps(event, ensure_ascii=False) + "\n")
    history("S2_RELEASES.json", event)
    es.require(not result["errors"], "Release partially failed; inspect audit before further use")
    return {"approval": approval, "written": result["written"], "errors": result["errors"]}


def check(records, audit, definition, synonyms, expect):
    valid_batch(records, audit)
    license = license_check(IMPORTER)
    importer_identity = es.request(IMPORTER, "GET", "/_security/_authenticate")
    es.require(importer_identity["username"] == IMPORTER and importer_identity["roles"] == [IMPORTER], "Wrong importer identity")
    ensure_index(definition)
    current = current_documents()
    verify_sources(records, current)  # Includes 260 raw strings + hashes, all metadata and ACL provenance.
    opened = expect == "open"
    for hit in current.values():
        approval = hit["_source"]["lab_release"]
        if opened:
            es.require(approval["approved"] is True and approval["source_snapshot_sha256"] == audit["source_snapshot_sha256"] and approval["approved_by"] and approval["approved_at"], "Incomplete approval")
        else:
            es.require(approval == CLOSED, "Document was released during import")
    identity = es.request(PROBE, "GET", "/_security/_authenticate")
    es.require(identity["username"] == PROBE and identity["roles"] == [READER_ROLE] and identity["authentication_realm"]["type"] == "native", "Wrong probe identity")
    es.require(identity["metadata"] == {"scope": "s2-release-probe-not-business-profile", "playbook_role_ids": [2], "playbook_business_categories": [0]}, "Wrong probe metadata")
    hits = es.request(PROBE, "POST", f"/{INDEX}/_search", {"size": 1000, "track_total_hits": True, "query": {"match_all": {}}})["hits"]
    es.require(hits["total"]["value"] == (102 if opened else 0), "Release gate returned an unexpected count")
    get_status = {}
    for did in (3, 25, 63, 81, 99, 148):
        status = 200 if opened and did in (3, 25, 63, 81) else 404
        es.request(PROBE, "GET", f"/{INDEX}/_doc/" + quote(f"playbook-test:Document:{did}"), expected=status)
        get_status[str(did)] = status
    es.request(PROBE, "PUT", f"/{INDEX}/_doc/forbidden", {"x": 1}, expected=403)
    es.request(None, "GET", f"/{INDEX}/_search", expected=401)
    caps = es.request(IMPORTER, "GET", f"/{INDEX}/_field_caps?fields=languages.zh.raw,languages.zh.html,languages.zh.search_text,upstream_metadata.Status")["fields"]
    es.require(not caps["languages.zh.raw"]["text"]["searchable"] and not caps["languages.zh.html"]["text"]["searchable"], "Raw/HTML accidentally indexed")
    es.require("upstream_metadata.Status" not in caps and caps["languages.zh.search_text"]["text"]["searchable"], "Metadata/search mapping differs")
    analyze = {}
    for value in synonyms["analyze_cases"]:
        tokens = es.request(IMPORTER, "POST", f"/{INDEX}/_analyze", {"analyzer": "playbook_search", "text": value})["tokens"]
        analyze[value] = [{k:t[k] for k in ("token", "position", "positionLength") if k in t} for t in tokens]
    for rule in synonyms["approved_rules"]:
        left, right = rule.split(", ")
        es.require({t["token"] for t in analyze[left]} == {t["token"] for t in analyze[right]}, "Synonym token sets differ")
    controls = {}
    for value in ("Azure Microsoft running", "合同的生成", "NCE OSPA 订阅取消"):
        controls[value] = {}
        for analyzer in ("smartcn", "playbook_search"):
            tokens = es.request(IMPORTER, "POST", f"/{INDEX}/_analyze", {"analyzer": analyzer, "text": value})["tokens"]
            controls[value][analyzer] = [t["token"] for t in tokens]
        es.require(controls[value]["smartcn"] == controls[value]["playbook_search"], "Index/search base tokenization differs")
    record_map = {r["document_id"]: r for r in records}
    for did, menu, category in ((25, 31, 0), (99, 114, 1), (148, 174, 1)):
        es.require((record_map[did]["acl"]["menu_id"], record_map[did]["business_category"]) == (menu, category), "Source mapping differs")
    # Maintenance check of stored bilingual fields; S4 must use the same
    # dis_max/tie_breaker=0 rule between language queries, not sum duplicates.
    es.require(record_map[148]["same_content"], "Bilingual control source changed")
    language_queries = [{"match": {f"languages.{lang}.search_text": "OSPA"}} for lang in ("zh", "en")]
    scores = {}
    for name, query in zip(("zh", "en", "both"), language_queries + [{"dis_max": {"queries": language_queries, "tie_breaker": 0}}]):
        scored = es.request(IMPORTER, "POST", f"/{INDEX}/_search", {"size": 1, "_source": False,
            "query": {"bool": {"filter": {"term": {"document_id": 148}}, "must": query}}})["hits"]["hits"]
        es.require(len(scored) == 1, "Bilingual scoring control not found")
        scores[name] = scored[0]["_score"]
    es.require(scores["both"] == max(scores["zh"], scores["en"]), "Duplicate language scores were summed")
    result = {"at": es.now(), "status": "passed", "expected_release": expect, "index": INDEX, "license": license,
        "documents": len(current), "raw_fields_verified": 2 * len(records), "source_snapshot_sha256": audit["source_snapshot_sha256"],
        "raw_and_html_indexed": False, "probe": PROBE, "probe_roles": identity["roles"], "probe_hits": hits["total"]["value"],
        "probe_get_http": get_status, "probe_write_http": 403, "anonymous_http": 401,
        "synonyms_version": synonyms["version"], "analyze": analyze, "index_search_controls": controls,
        "bilingual_scoring": {"document_id": 148, "query": "OSPA", "tie_breaker": 0, "scores": scores},
        "table_warnings": {str(r["document_id"]): {lang:v["table_warnings"] for lang,v in r["languages"].items() if v["table_warnings"]}
                           for r in records if any(v["table_warnings"] for v in r["languages"].values())}}
    history("S2_CHECKS.json", result)
    return result


def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("dry-run", "init", "import"):
        sub.add_parser(name)
    approve = sub.add_parser("release")
    approve.add_argument("--snapshot", required=True)
    approve.add_argument("--approved-by", required=True)
    verify = sub.add_parser("check")
    verify.add_argument("--expect", choices=("closed", "open"), required=True)
    args = parser.parse_args()
    records, audit, definition, synonyms = prepare_dataset()
    if args.command == "dry-run":
        result = {"records": len(records), "quarantined": len(audit["quarantine"]), "es_writes": 0}
    elif args.command == "init":
        result = init(records, audit)
    elif args.command == "import":
        result = apply_import(records, audit, definition)
    elif args.command == "release":
        result = release(records, audit, args.snapshot, args.approved_by, definition)
    else:
        result = check(records, audit, definition, synonyms, args.expect)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        history("S2_FAILURES.json", {"at": es.now(), "command": sys.argv[1] if len(sys.argv) > 1 else None,
                                    "status": "failed", "error_type": type(error).__name__, "message": str(error)})
        raise
