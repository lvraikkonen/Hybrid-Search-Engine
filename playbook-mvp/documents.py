"""SQL-derived S2 records and explicit index mapping; no ES calls or publication."""
import copy
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString

from build import render
from tools.extract_playbook_sql import (
    canonical_hash, decode_sql, derive_acl, digest, parse_dump, semantic_acl, validate_tables,
)

ROOT = Path(__file__).resolve().parent
SQL = ROOT / "data/playbook-sampledata.sql"
INDEX = "playbook-documents-v1"
IMPORTER = "pb_s2_importer"
PROBE = "pb_s2_probe"
READER_ROLE = "pb_s2_reader"
TRANSFORM_VERSION = "s2-v1"
CLOSED = {"approved": False, "source_snapshot_sha256": None, "approved_by": None, "approved_at": None}


def table_text(table):
    """Expand actual rowspan/colspan cells, then label every data cell by header.

    Blank is recorded as blank, never interpreted as an ACL denial. Nested or
    malformed tables are quarantined instead of inventing a relationship.
    """
    if table.find("table"):
        raise ValueError("nested_table_requires_review")
    rows = table.find_all("tr")
    grid, headers, warnings = {}, [], []
    for row_index, row in enumerate(rows):
        cells = row.find_all(["th", "td"], recursive=False)
        if not cells:
            continue
        is_header = all(cell.name == "th" for cell in cells)
        if is_header:
            headers.append(row_index)
        column = 0
        for cell in cells:
            while (row_index, column) in grid:
                column += 1
            rowspan, colspan = int(cell.get("rowspan", 1)), int(cell.get("colspan", 1))
            if not (1 <= rowspan <= 1000 and 1 <= colspan <= 100):
                raise ValueError("invalid_table_span")
            group = row.find_parent(["thead", "tbody", "tfoot"])
            end = row_index + 1
            while end < len(rows) and rows[end].find_parent(["thead", "tbody", "tfoot"]) is group:
                end += 1
            if rowspan > end - row_index:
                warnings.append(f"row_{row_index + 1}:rowspan_{rowspan}_exceeds_{end - row_index}_existing_group_rows")
                # Preserve original HTML; repeat labels only on actual rows.
                # Never create a missing source row to satisfy a bad rowspan.
                rowspan = end - row_index
            text = " ".join(cell.get_text(" ", strip=True).split())
            for r in range(row_index, row_index + rowspan):
                for c in range(column, column + colspan):
                    if (r, c) in grid:
                        raise ValueError("overlapping_table_cells")
                    grid[r, c] = text
            column += colspan
    if not grid:
        return "", warnings
    width = max(c for _, c in grid) + 1
    names = []
    for c in range(width):
        labels = list(dict.fromkeys(grid.get((r, c), "") for r in headers))
        names.append(" / ".join(x for x in labels if x) or f"列{c + 1}")
    lines = []
    for r in range(len(rows)):
        if r in headers:
            continue
        values = []
        for c, label in enumerate(names):
            if (r, c) not in grid:
                raise ValueError("ragged_table_requires_review")
            # Repeated header labels (e.g. Catalogue spanning 2 columns) remain
            # distinct columns, with merged row labels repeated on each row.
            values.append(f"{label}[{c + 1}]：{grid[r, c] or '〔空白〕'}")
        lines.append(" | ".join(values))
    return "\n".join(lines), warnings


def language_record(raw, field, prefix):
    if raw is not None and not isinstance(raw, str):
        raise ValueError("non_string_source_body")
    view = render(raw, prefix)  # Reuse the existing Markdown + HTML allowlist.
    soup = BeautifulSoup(view["html"], "html.parser")
    for decoration in soup.select(".asset-placeholder"):
        decoration.decompose()
    table_count = len(soup.find_all("table"))
    merged_count = len(soup.select("table [rowspan], table [colspan]"))
    table_warnings = []
    for table in list(soup.find_all("table")):
        semantic_text, warnings = table_text(table)
        table_warnings.extend(warnings)
        table.replace_with(NavigableString("\n" + semantic_text + "\n"))
    # Keep ordered steps in search text; their numbers need not be guessed from
    # the browser's CSS counter. Nested lists retain their own numbering.
    for listing in soup.find_all("ol"):
        items = listing.find_all("li", recursive=False)
        number = int(listing.get("start", len(items) if listing.has_attr("reversed") else 1))
        for item in items:
            number = int(item.get("value", number))
            item.insert(0, f"{number}. ")
            number += -1 if listing.has_attr("reversed") else 1
    for element in soup.find_all(["p", "div", "li", "blockquote", "pre", "br", "hr", "h1", "h2", "h3", "h4", "h5", "h6"]):
        element.insert_before(NavigableString("\n"))
        element.insert_after(NavigableString("\n"))
    text = "\n".join(line.strip() for line in soup.get_text().splitlines() if line.strip())
    return {
        "source_field": field, "raw": raw,
        "raw_utf8_sha256": digest(raw.encode()) if raw is not None else None,
        "search_text": text, "headings": [h["text"] for h in view["toc"]],
        "html": view["html"], "assets": view["assets"],
        "table_count": table_count, "merged_cell_count": merged_count,
        "table_warnings": table_warnings,
        "omitted_decorations": view["omitted_decorations"],
    }


def document_record(doc, tables, snapshot):
    if type(doc["Id"]) is not int or doc["Id"] < 1:
        raise ValueError("invalid_document_id")
    if type(doc["BusinessCategory"]) is not int or doc["BusinessCategory"] not in (0, 1):
        raise ValueError("invalid_document_business")
    original = derive_acl(doc, tables)
    reasons = list(original["reasons"])
    statuses = {}
    for role in original["provenance"]["Role"]:
        rid, status = role["Id"], role.get("Status")
        if type(rid) is not int or rid not in (1, 2, 3) or type(status) is not int or status not in (0, 1):
            reasons.append("unknown_role_or_status")
        if str(rid) in statuses and statuses[str(rid)] != status:
            reasons.append("conflicting_role_status")
        statuses[str(rid)] = status
    for menu in original["provenance"]["MenuItem"]:
        for key in ("Active", "UnderControl", "BusinessCategory", "Type"):
            if type(menu.get(key)) is not int or menu[key] not in (0, 1):
                reasons.append(f"invalid_menu_{key}")
    grants = original["provenance"]["Role_MenuItem"]
    pairs = [(g["RoleId"], g["MenuItemId"]) for g in grants]
    if len(pairs) != len(set(pairs)):
        reasons.append("duplicate_role_grant")
    if any(type(rid) is not int or rid not in (1, 2, 3) or str(rid) not in statuses for rid in original["allowed_role_ids"]):
        reasons.append("unverifiable_granted_role")
    acl = {k: original[k] for k in ("state", "menu_id", "menu_active_raw", "is_public", "allowed_role_ids", "required_business_category")}
    acl.update(role_status_raw=statuses, reasons=sorted(set(reasons)))
    if reasons:
        acl["state"] = "unresolved"
    acl["active_role_ids"] = [rid for rid in acl["allowed_role_ids"] if statuses.get(str(rid)) == 1] if not reasons else []
    try:
        revision = canonical_hash(semantic_acl(original))
    except KeyError:
        revision = canonical_hash({"unresolved_source_acl": original})
    return {
        "schema_version": "1.0", "transform_version": TRANSFORM_VERSION,
        "document_key": f"playbook-test:Document:{doc['Id']}", "document_id": doc["Id"], "kb_id": "playbook",
        "title": doc["Title"], "title_en": doc["TitleEn"], "slug": doc["Slug"], "business_category": doc["BusinessCategory"],
        "source": {"instance_key": "playbook-test", "table": "dbo.Document", "primary_key": doc["Id"],
                   "snapshot_sha256": snapshot, "sql_lines": doc["_sql_lines"]},
        "upstream_metadata": {k: v for k, v in doc.items() if k not in ("Content", "ContentEn", "_sql_lines")},
        "timestamp_timezone": None, "publication_state": "unverified", "lab_release": copy.deepcopy(CLOSED),
        "languages": {lang: language_record(doc[field], field, f"d{doc['Id']}-{lang}") for lang, field in (("zh", "Content"), ("en", "ContentEn"))},
        "same_content": doc["Content"] == doc["ContentEn"],
        "acl": acl, "acl_provenance": original["provenance"],
        "content_revision": canonical_hash({"Content": doc["Content"], "ContentEn": doc["ContentEn"]}), "acl_revision": revision,
    }


def load_records(sql=SQL):
    raw = Path(sql).read_bytes()
    tables = parse_dump(decode_sql(raw)[0])
    if any(not tables[k] for k in ("Document", "MenuItem", "MenuItem_Document", "Role", "Role_MenuItem")):
        raise ValueError("required_source_table_empty")
    audit = validate_tables(tables)  # Duplicate primary keys fail the whole batch.
    snapshot = digest(raw)
    records, quarantine = [], []
    for doc in sorted(tables["Document"], key=lambda d: d["Id"]):
        try:
            record = document_record(doc, tables, snapshot)
        except (ValueError, KeyError) as error:
            quarantine.append({"document_id": doc["Id"], "sql_lines": doc["_sql_lines"], "reason": str(error), "source_row": doc})
            continue
        records.append(record)
        if record["acl"]["state"] != "resolved":
            quarantine.append({"document_id": doc["Id"], "sql_lines": doc["_sql_lines"], "reason": record["acl"]["reasons"]})
    return records, {"source_snapshot_sha256": snapshot, "source_audit": audit, "quarantine": quarantine}


def index_definition(synonyms):
    if synonyms["approved_rules"] and (synonyms["review_status"] != "approved" or not synonyms.get("approved_by") or not synonyms.get("approved_at")):
        raise ValueError("synonyms_require_recorded_business_review")
    keyword = {"type": "keyword"}
    text = {"type": "text", "analyzer": "smartcn", "search_analyzer": "playbook_search"}
    stored = {"type": "object", "enabled": False}
    language = {"properties": {
        "source_field": keyword, "raw": {"type": "text", "index": False}, "raw_utf8_sha256": keyword,
        "search_text": text, "headings": text, "html": {"type": "text", "index": False}, "assets": stored, "table_warnings": keyword,
        **{k: {"type": "integer"} for k in ("table_count", "merged_cell_count", "omitted_decorations")},
    }}
    properties = {
        **{k: keyword for k in ("schema_version", "transform_version", "document_key", "kb_id", "slug", "publication_state", "content_revision", "acl_revision", "timestamp_timezone")},
        "document_id": {"type": "integer"}, "business_category": {"type": "integer"}, "title": text, "title_en": text,
        "same_content": {"type": "boolean"}, "upstream_metadata": stored, "acl_provenance": stored,
        "source": {"properties": {**{k: keyword for k in ("instance_key", "table", "snapshot_sha256")},
                                   "primary_key": {"type": "integer"}, "sql_lines": {"type": "integer", "index": False}}},
        "languages": {"properties": {"zh": language, "en": language}},
        "lab_release": {"properties": {"approved": {"type": "boolean"}, "source_snapshot_sha256": keyword,
                                       "approved_by": keyword, "approved_at": {"type": "date"}}},
        "acl": {"properties": {
            "state": keyword, "is_public": {"type": "boolean"}, "reasons": keyword,
            **{k: {"type": "integer"} for k in ("menu_id", "menu_active_raw", "allowed_role_ids", "active_role_ids", "required_business_category")},
            "role_status_raw": {"properties": {str(rid): {"type": "integer"} for rid in (1, 2, 3)}},
        }},
    }
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0, "analysis": {
            "filter": {"playbook_synonyms": {"type": "synonym_graph", "lenient": False, "synonyms": synonyms["approved_rules"]}},
            "analyzer": {"playbook_search": {"type": "custom", "tokenizer": "smartcn_tokenizer", "filter": ["porter_stem", "smartcn_stop", "playbook_synonyms"]}},
        }},
        "mappings": {"dynamic": "strict", "_meta": {"transform_version": TRANSFORM_VERSION, "synonyms_sha256": canonical_hash(synonyms)}, "properties": properties},
    }
