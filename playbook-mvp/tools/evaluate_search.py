#!/usr/bin/env python3
"""S6 reviewed-reference evaluation; no relevance labels are learned from rankings."""
import argparse
import json
import io
import math
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from documents import load_records
from tools.extract_playbook_sql import canonical_hash
from tools import es_permissions as permissions

GOLDEN = ROOT / "evaluation/golden_v1.json"
PROFILES = ["pb_msft_01", "pb_21v_01", "pb_partner_nce_01", "pb_partner_ospa_01", "pb_partner_both_01"]


def prepare_review():
    if GOLDEN.exists():
        raise ValueError("Golden draft exists; review it instead of overwriting labels")
    questions = json.loads((ROOT / "evaluation/candidate_questions.json").read_text())
    records, audit = load_records()
    docs = {r["document_id"]: r for r in records}
    oracle = json.loads((ROOT / "fixtures/expected_access.json").read_text())
    allowed = {x["profile_id"]: set(x["allowed_document_ids"]) for x in oracle["expectations"]}
    assert questions["review_status"] == "approved"
    assert audit["source_snapshot_sha256"] == oracle["source"]["sha256"] == questions["source_snapshot_sha256"]
    cases = []
    lines = ["# S6 V1 相关性与无答案集审核稿", "", "状态：待用户确认，不能作为已通过的质量验收。", "",
             "32条问题和参考ID沿用已批准V1；以下标注依据源文档，未使用本轮搜索排名。五类代表身份分别为MSFT、21V、Partner-NCE、Partner-OSPA、Partner双业务。",
             "", "等级2=源文档直接相关；等级1=只提供相关外链/背景。未标注文档为未知，不默认判为不相关。核心命中评价采用等级≥1；目标为每类身份可访问核心问题Hit@5≥90%，同时记录MRR@5和已标注参考集Recall@5；不宣称穷尽全库相关文档或总体召回率。",
             "", "无可访问参考的身份/问题不计入核心命中分母，单列为无可访问已标注参考，不能直接称为全库无答案。", "",
             "| 问题 | 参考文档 / 等级 | 具有可访问参考的身份 | 源依据（节选） |", "|---|---|---|---|"]
    for q in questions["questions"]:
        grades = {str(did): 1 if q["id"] == "Q23" else 2 for did in q["reference_document_ids"]}
        visible = {pid: {did: grade for did, grade in grades.items() if int(did) in allowed[pid]} for pid in PROFILES}
        row = {**q, "relevance": grades, "by_profile": visible, "judgments_exhaustive": False}
        cases.append(row)
        record = docs[q["reference_document_ids"][0]]
        excerpt = record["languages"][q["language"]]["search_text"][:230].replace("\n", " / ").replace("|", "／")
        labels = ", ".join(pid.removeprefix("pb_").removesuffix("_01") for pid, refs in visible.items() if refs)
        lines.append(f"| {q['id']} {q['query']} | {record['document_id']}《{record['title']}》 / {next(iter(grades.values()))} | {labels} | {excerpt} |")
    negatives = [
        {"id": "NA01", "query": "员工年假有多少天，如何申请休假？", "language": "zh", "business_scope": "all", "reason": "本快照没有人事休假制度。"},
        {"id": "NA02", "query": "What will the weather be in Shanghai tomorrow?", "language": "en", "business_scope": "all", "reason": "本快照没有实时天气数据。"},
        {"id": "NA03", "query": "2027年12月中国北部3 Standard_D4s_v5 虚拟机实时每小时价格是多少？", "language": "zh", "business_scope": "all", "reason": "现有文档可能介绍价目表入口，但没有指定未来时点的实时机型价格。"},
    ]
    draft = {"version": "s6-v1", "review_status": "pending", "approved_by": None, "approval_evidence": None,
             "source_snapshot_sha256": audit["source_snapshot_sha256"], "question_set_sha256": canonical_hash(questions),
             "profiles": PROFILES, "relevance_scale": {"1": "related reference/link", "2": "directly relevant source"},
             "judgments_exhaustive": False, "hit_at_5_target_per_profile": 0.9, "questions": cases, "no_answer_questions": negatives}
    GOLDEN.write_text(json.dumps(draft, ensure_ascii=False, indent=2) + "\n")
    lines += ["", "## 待确认的无答案问题", "", *[f"- **{q['id']}** {q['query']} — {q['reason']}" for q in negatives], "",
              "无答案样例单独记录返回率、返回ID及摘录；不设置未经标定的BM25分数阈值。返回相关文档不等于这些文档包含所问答案，界面需明确这一点。无答案返回率本轮只测量，不设臆定达标值。", "",
              "Q23只有外部创建订阅文档链接，按等级1记录，不声称本库包含完整步骤。人工确认后冻结本文件对应JSON指纹；后续改标注须新版本，不能为得到PASS修改金标。", ""]
    (ROOT / "evaluation/S6_GOLDEN_REVIEW.md").write_text("\n".join(lines))
    print("Prepared evaluation/golden_v1.json and S6_GOLDEN_REVIEW.md; review pending.")


def freeze(actor, evidence):
    golden = json.loads(GOLDEN.read_text())
    if golden["review_status"] != "pending" or not actor or not evidence:
        raise ValueError("Only a pending review with explicit human approval can be frozen")
    golden.update(review_status="approved", approved_by=actor, approval_evidence=evidence,
                  approval_recorded_at=permissions.es.now())
    golden["frozen_sha256"] = canonical_hash(golden)
    GOLDEN.write_text(json.dumps(golden, ensure_ascii=False, indent=2) + "\n")
    path = ROOT / "evaluation/S6_GOLDEN_REVIEW.md"
    path.write_text(path.read_text().replace("状态：待用户确认，不能作为已通过的质量验收。",
                    "状态：用户已确认草案及每类身份 Hit@5 ≥ 90% 目标；标注冻结，检索结果另行实测。")
                    + f"\n批准原文：{evidence}\n\n记录时间：{golden['approval_recorded_at']}；冻结指纹：`{golden['frozen_sha256']}`。\n")
    print(json.dumps({"review_status": "approved", "frozen_sha256": golden["frozen_sha256"]}))


def reviewed_golden():
    golden = json.loads(GOLDEN.read_text())
    payload = {k: v for k, v in golden.items() if k != "frozen_sha256"}
    if golden["review_status"] != "approved" or canonical_hash(payload) != golden["frozen_sha256"]:
        raise ValueError("Golden labels require an unchanged, approved freeze")
    questions = json.loads((ROOT / "evaluation/candidate_questions.json").read_text())
    if canonical_hash(questions) != golden["question_set_sha256"]:
        raise ValueError("Original approved question set changed")
    return golden


def score(ids, references):
    """Only judged, accessible references enter the denominator; unknown != irrelevant."""
    relevant = {int(did) for did, grade in references.items() if grade >= 1}
    if not relevant:
        return None
    ranks = [rank for rank, did in enumerate(ids[:5], 1) if did in relevant]
    return {"hit_at_5": int(bool(ranks)), "mrr_at_5": 1 / min(ranks) if ranks else 0,
            "judged_recall_at_5": len(set(ids[:5]) & relevant) / len(relevant)}


def latency(values):
    ordered = sorted(values)
    return {"count": len(ordered), **{key: round(ordered[max(0, math.ceil(len(ordered) * p) - 1)], 3)
                                     for key, p in (("p50_ms", .5), ("p95_ms", .95))}}


def environment():
    result = {"os": platform.platform(), "python": platform.python_version(), "machine": platform.machine(),
              "logical_cpus": __import__("os").cpu_count()}
    for key, command in {
        "cpu": ["sysctl", "-n", "machdep.cpu.brand_string"], "host_memory_bytes": ["sysctl", "-n", "hw.memsize"],
        "docker_limits": ["docker", "inspect", "playbook-mvp-es-1", "--format",
                          "memory={{.HostConfig.Memory}} nano_cpus={{.HostConfig.NanoCpus}}"]}.items():
        try:
            result[key] = subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL, timeout=10).strip()
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
            result[key] = "unavailable"
    return result


def evaluate():
    from api import APIServer
    from tools.check_api import call, expect, login
    from tools.ingest_playbook import current_documents, verify_sources
    golden = reviewed_golden()
    records, audit = load_records()
    es = permissions.es
    es.require(audit["source_snapshot_sha256"] == golden["source_snapshot_sha256"], "Golden snapshot changed")
    current = current_documents()  # Maintenance verifies bytes; business searches below use only user sessions.
    verify_sources(records, current)
    es.require(all(h["_source"]["lab_release"]["approved"] is True and
                   h["_source"]["lab_release"]["source_snapshot_sha256"] == golden["source_snapshot_sha256"]
                   for h in current.values()), "Golden snapshot is not fully open")
    oracle = json.loads((ROOT / "fixtures/expected_access.json").read_text())
    allowed = {x["profile_id"]: set(x["allowed_document_ids"]) for x in oracle["expectations"]}
    report = {"stage": "S6", "started_at": es.now(), "status": "running", "golden_sha256": golden["frozen_sha256"],
              "source_snapshot_sha256": golden["source_snapshot_sha256"], "environment": environment(),
              "method": {"top_k": 5, "rounds": 3, "warmup_per_profile": 5, "concurrency": 1,
                         "es_latency": "Elasticsearch took; excludes HTTP/auth", "api_latency": "loopback HTTP wall time including native identity and privilege verification",
                         "percentile": "nearest-rank", "cache": "request_cache=false; OS/Lucene warmed, not flushed",
                         "relevance": "partial judged reference recall; unjudged documents unknown; not exhaustive corpus recall",
                         "no_answer_target": "descriptive only, no uncalibrated BM25 score cutoff"},
              "profiles": {}, "cases": [], "no_answer_cases": []}
    server = APIServer(0)
    server.audit_stream = io.StringIO()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    api_times, es_times = [], []
    try:
        for pid in golden["profiles"]:
            cookie, _ = login(server.server_port, pid)
            measured = []
            def search(q):
                body = {key: q[key] for key in ("query", "language", "business_scope")}
                start = time.perf_counter()
                _, response = expect(call(server.server_port, "POST", "/api/search", {**body, "top_k": 5}, cookie))
                elapsed = (time.perf_counter() - start) * 1000
                ids = [item["document_id"] for item in response["items"]]
                es.require(set(ids) <= allowed[pid], "DLS violation in golden run")
                return ids, response, elapsed
            for q in golden["questions"][:5]:
                search(q)
            for round_number in range(3):
                for q in golden["questions"]:
                    ids, response, elapsed = search(q)
                    api_times.append(elapsed)
                    es_times.append(response["took_ms"])
                    if round_number == 0:
                        metrics = score(ids, q["by_profile"][pid])
                        if metrics is not None:
                            measured.append(metrics)
                        report["cases"].append({"profile_id": pid, "question_id": q["id"], "top_5": ids,
                                                "metrics": metrics, "excluded_reason": None if metrics else "no_accessible_judged_reference"})
            for q in golden["no_answer_questions"]:
                ids, response, _ = search(q)
                report["no_answer_cases"].append({"profile_id": pid, "question_id": q["id"], "top_5": ids,
                    "snippets": [{"document_id": d["document_id"], "snippets": d["snippets"]} for d in response["items"]]})
            aggregate = {key: sum(m[key] for m in measured) / len(measured) for key in measured[0]}
            report["profiles"][pid] = {"answerable_core_questions": len(measured), "excluded": len(golden["questions"]) - len(measured),
                                        **aggregate, "target": golden["hit_at_5_target_per_profile"],
                                        "target_met": aggregate["hit_at_5"] >= golden["hit_at_5_target_per_profile"]}
            print(json.dumps({"profile": pid, **report["profiles"][pid]}), flush=True)
    finally:
        server.shutdown()
        server.server_close()
    report.update(completed_at=es.now(), performance={"es_retrieval": latency(es_times), "api": latency(api_times)},
                  no_answer_return_rate=sum(bool(q["top_5"]) for q in report["no_answer_cases"]) / len(report["no_answer_cases"]),
                  permission_violations=0, measured_queries=len(api_times), warmup_queries=25, no_answer_queries=15)
    report["status"] = "passed" if all(p["target_met"] for p in report["profiles"].values()) else "quality_target_not_met"
    es.write_json(ROOT / "reports/S6_EVALUATION.json", report)
    print(json.dumps({"status": report["status"], "performance": report["performance"], "no_answer_return_rate": report["no_answer_return_rate"]}))
    es.require(report["status"] == "passed", "Frozen quality target not met; see unchanged labels and actual misses in S6_EVALUATION")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare-review", "freeze", "run"])
    parser.add_argument("--approved-by")
    parser.add_argument("--evidence")
    args = parser.parse_args()
    if args.command == "prepare-review":
        prepare_review()
    elif args.command == "freeze":
        freeze(args.approved_by, args.evidence)
    else:
        evaluate()
