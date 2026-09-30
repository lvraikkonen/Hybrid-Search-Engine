#!/usr/bin/env python3
"""S4 loopback-only development API. Run with APP_ENV and MOCK_AUTH_ENABLED."""
import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
import hashlib
from html import unescape
from http.cookies import CookieError, SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import secrets
import socket
import sys
import threading
import time
from urllib.error import URLError
from urllib.parse import parse_qsl, quote, unquote, urlsplit

from documents import INDEX, TRANSFORM_VERSION
from mock_auth import POLICY_VERSION, load_profiles, validate_mock_profile
from runtime import CatalogStore
from tools import es_permissions as permissions
from tools.extract_playbook_sql import canonical_hash

ROOT = Path(__file__).resolve().parent
WEB = ROOT / "web"
STATIC_FILES = {"/": "index.html", "/index.html": "index.html", "/style.css": "style.css", "/app.js": "app.js"}
APP_CSP = "default-src 'none'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
es = permissions.es
COOKIE = "playbook_session"
SESSION_SECONDS = 3600
DOCUMENT_KEY = r"playbook-test:Document:[1-9][0-9]{0,9}"
DISPLAY_FIELDS = ["document_key", "document_id", "title", "title_en", "business_category",
                  "kb_id", "source", "acl.is_public", "content_revision", "acl_revision"]
MESSAGES = {400: "请求参数无效。", 401: "会话或身份已失效，请重新选择用户。",
            403: "请求来源不被允许。", 404: "资源不存在或不可访问。",
            503: "检索服务或权限能力暂不可用。", 504: "检索超时，请重试。"}
CODES = {400: "invalid_request", 401: "invalid_session", 403: "origin_denied",
         404: "not_found", 503: "search_unavailable", 504: "search_timeout"}


class APIError(Exception):
    def __init__(self, status):
        self.status = status


def require(condition, status=400):
    if not condition:
        raise APIError(status)


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


@dataclass(frozen=True)
class ESIdentity:
    profile_id: str
    principal: object
    password: str = field(repr=False)
    index: str = INDEX

    def request(self, method, path, body=None, expected=200):
        try:
            return es.request(self.profile_id, method, path, body, expected, password_value=self.password)
        except es.ESHTTPError as error:
            raise APIError(401 if error.status == 401 else 503) from None
        except (TimeoutError, socket.timeout):
            raise APIError(504) from None
        except URLError as error:
            raise APIError(504 if isinstance(error.reason, TimeoutError) else 503) from None
        except (OSError, ValueError):
            raise APIError(503) from None


def resolve_identity(profile_id, snapshot, catalog=None, store=None):
    """Resolve only trusted profiles; each request owns its native credential."""
    try:
        principal = permissions.principals()[profile_id] if catalog is None else validate_mock_profile(catalog["profiles"][profile_id])
        require(principal.user_name == profile_id, 401)
        secret = permissions.credential(profile_id) if store is None else permissions.credential(profile_id, store.credentials)
        identity = ESIdentity(profile_id, principal, secret, store.index if store else INDEX)
    except (KeyError, ValueError, RuntimeError, OSError):
        raise APIError(401) from None
    actual = identity.request("GET", "/_security/_authenticate")
    require(actual.get("username") == profile_id and actual.get("roles") == [store.reader if store else permissions.READER]
            and actual.get("enabled") is True and actual.get("authentication_realm", {}).get("type") == "native"
            and actual.get("metadata") == principal.es_metadata_candidate(), 401)
    privileges = identity.request("GET", "/_security/user/_privileges")
    require(not any(privileges.get(k) for k in ("cluster", "global", "applications", "run_as", "remote_indices", "remote_cluster")), 503)
    indices = privileges.get("indices", [])
    expected = permissions.reader_role(snapshot, identity.index, not store or not store.validation)["indices"]
    require(len(indices) == len(expected) and {tuple(p["names"]) for p in indices} == {tuple(p["names"]) for p in expected}, 503)
    for item in indices:
        rule = next(p for p in expected if p["names"] == item["names"])
        require(item["privileges"] == ["read"] and not item.get("allow_restricted_indices")
                and not item.get("field_security")
                and [json.loads(q) for q in item.get("query", [])] == [rule["query"]], 503)
    return identity


def search_hits(identity, body):
    result = identity.request("POST", f"/{identity.index}/_search?request_cache=false&allow_partial_search_results=false", body)
    require(result.get("timed_out") is False, 504)
    require(result.get("_shards", {}).get("failed") == 0, 503)
    return result


def display_document(doc):
    result = {k: doc[k] for k in ("document_key", "document_id", "title", "title_en", "business_category", "content_revision", "acl_revision")}
    result["is_public"] = doc["acl"]["is_public"]
    result["source"] = {"kb_id": doc["kb_id"], **{k: doc["source"][k] for k in ("instance_key", "table", "primary_key", "snapshot_sha256")}}
    return result


def metadata_document(doc):
    return {**display_document(doc), "metadata": {k: doc["upstream_metadata"].get(k) for k in
            ("Author", "Status", "TextFormatType", "CreatedTime", "UpdatedTime")},
            "publication_state": doc["publication_state"], "timestamp_timezone": doc["timestamp_timezone"],
            "same_content": doc["same_content"]}


def highlight_parts(fragment, limit=300):
    """Decode ES's HTML-escaped text; only its fixed mark tags carry match flags."""
    parts, highlighted = [], False
    for chunk in re.split(r"(<mark>|</mark>)", fragment):
        if chunk in ("<mark>", "</mark>"):
            highlighted = chunk == "<mark>"
            continue
        text = unescape(chunk)[:limit]
        if text:
            parts.append({"text": text, "highlighted": highlighted})
            limit -= len(text)
    return parts


def bm25_search(identity, args):
    require(isinstance(args, dict) and not set(args) - {"query", "top_k", "language", "business_scope"})
    query, top_k = args.get("query"), args.get("top_k", 5)
    lang, scope = args.get("language", "both"), args.get("business_scope", "all")
    require(isinstance(query, str) and 1 <= len(query.strip()) <= 1000 and not any(0xD800 <= ord(c) <= 0xDFFF for c in query))
    require(type(top_k) is int and 1 <= top_k <= 20)
    require(isinstance(lang, str) and lang in ("zh", "en", "both"))
    require(scope == "all" or type(scope) is int and scope in (0, 1))
    languages = ("zh", "en") if lang == "both" else (lang,)
    clauses = [{"multi_match": {"query": query.strip(), "_name": language, "type": "best_fields",
                "fields": [f"{'title' if language == 'zh' else 'title_en'}^3", f"languages.{language}.headings^2", f"languages.{language}.search_text"]}}
               for language in languages]
    body = {"size": top_k, "track_total_hits": False, "timeout": "5s", "_source": DISPLAY_FIELDS,
            "query": {"bool": {"must": [{"dis_max": {"queries": clauses, "tie_breaker": 0}}],
                               "filter": [] if scope == "all" else [{"term": {"business_category": scope}}]}},
            "sort": [{"_score": "desc"}, {"document_id": "asc"}],
            "highlight": {"encoder": "html", "pre_tags": ["<mark>"], "post_tags": ["</mark>"], "number_of_fragments": 1,
                          "fragment_size": 180, "no_match_size": 180, "order": "score",
                          "fields": {**{f"languages.{language}.search_text": {} for language in languages},
                                     **{("title" if language == "zh" else "title_en"): {"number_of_fragments": 0}
                                        for language in languages}}}}
    result = search_hits(identity, body)
    items = []
    for hit in result["hits"]["hits"]:
        matched = [language for language in languages if language in hit.get("matched_queries", [])]
        require(bool(matched), 503)
        score = hit.get("_score")
        require(type(score) in (int, float) and math.isfinite(score) and score >= 0, 503)
        highlights = hit.get("highlight", {})
        snippets = []
        for language in matched:
            for fragment in highlights.get(f"languages.{language}.search_text", [])[:1]:
                parts = highlight_parts(fragment)
                snippets.append({"source_field": "Content" if language == "zh" else "ContentEn",
                                 "text": "".join(p["text"] for p in parts), "parts": parts})
        title_highlights = {key: highlight_parts(highlights[key][0], len(hit["_source"][key] or ""))
                            for key in ("title", "title_en") if highlights.get(key)}
        items.append({**display_document(hit["_source"]), "score": score, "title_highlights": title_highlights,
                      "matched_language": "both" if len(matched) == 2 else matched[0], "snippets": snippets})
    require(len({d["document_key"] for d in items}) == len(items) <= top_k, 503)
    return {"retrieval_mode": "bm25", "took_ms": result["took"], "items": items}


def menu_nodes(identity, tables, snapshot):
    # The only document collection comes from this user's DLS search, never a local oracle/body dump.
    result = search_hits(identity, {"size": 1000, "track_total_hits": True, "timeout": "5s", "query": {"match_all": {}},
                                    "_source": ["document_key", "acl.menu_id", "source.snapshot_sha256"]})
    hits = result["hits"]["hits"]
    require(result["hits"]["total"] == {"value": len(hits), "relation": "eq"}, 503)
    menus = {m["Id"]: m for m in tables["MenuItem"]}
    principal = identity.principal
    grants = {r["MenuItemId"] for r in tables["Role_MenuItem"] if r["RoleId"] == principal.role_id}
    active = any(r["Id"] == principal.role_id and r["Status"] == 1 for r in tables["Role"])

    def visible(menu):
        return active and menu["Active"] == 1 and (menu["UnderControl"] == 0 or
                menu["UnderControl"] == 1 and menu["Id"] in grants and menu["BusinessCategory"] in principal.effective_business_categories)

    own, counts, keep = {}, {}, set()

    def retain(mid, count):
        seen = set()
        while mid:
            require(mid in menus and mid not in seen, 503)
            seen.add(mid)
            keep.add(mid)
            counts[mid] = counts.get(mid, 0) + count
            mid = menus[mid]["ParentId"]

    for hit in hits:
        doc = hit["_source"]
        require(doc["source"]["snapshot_sha256"] == snapshot, 503)
        mid = doc["acl"]["menu_id"]
        own.setdefault(mid, []).append(doc["document_key"])
        retain(mid, 1)
    for menu in menus.values():
        if menu["Type"] in (2, 3) and visible(menu):
            retain(menu["Id"], 0)
    return [{"menu_id": m["Id"], "parent_id": m["ParentId"],
             "label": m["Name"] if visible(m) else "上级目录", "label_en": m["NameEn"] if visible(m) else "上级目录",
             "type": m["Type"] if visible(m) else None, "placeholder": not visible(m),
             "business_category": m["BusinessCategory"] if visible(m) else None,
             "is_public": m["UnderControl"] == 0 if visible(m) else None,
             "document_keys": sorted(own.get(m["Id"], [])), "visible_document_count": counts.get(m["Id"], 0)}
            for m in sorted(menus.values(), key=lambda m: (m["Sort"], m["Id"])) if m["Id"] in keep]


class APIServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, port=18080, store=None):
        super().__init__(("127.0.0.1", port), APIHandler)
        self.origins = {f"http://{host}:{self.server_port}" for host in ("127.0.0.1", "localhost")}
        self.enabled = os.environ.get("APP_ENV") in ("development", "test") and os.environ.get("MOCK_AUTH_ENABLED") == "1"
        self.sessions, self.lock = {}, threading.Lock()
        self.store = store or CatalogStore()
        self.audit_stream = sys.stdout
        source = json.loads((ROOT / "fixtures/source_acl_tables.json").read_text())
        self.snapshot = source["source"]["sha256"]
        require(hashlib.sha256((ROOT / "data/playbook-sampledata.sql").read_bytes()).hexdigest() == self.snapshot, 503)
        self.tables = {key: source["tables"][key] for key in ("MenuItem", "Role", "Role_MenuItem")}
        self.synonyms = json.loads((ROOT / "config/synonyms.json").read_text())

    def session(self, token, generation=None):
        # ponytail: one in-memory session store for this single local process;
        # restart logs out everyone. Distributed sessions belong with real login.
        with self.lock:
            value = self.sessions.get(token)
            if value and value["expires"] > time.monotonic() and (generation is None or value.get("generation") == generation):
                return dict(value)
            self.sessions.pop(token, None)
        raise APIError(401)


class APIHandler(BaseHTTPRequestHandler):
    server_version = "PlaybookLocal"
    sys_version = ""

    def setup(self):
        super().setup()
        self.connection.settimeout(10)

    def log_message(self, *_):
        pass  # No raw request lines, cookies, query text, or exception bodies.

    def token(self):
        cookie = SimpleCookie()
        try:
            require(len(self.headers.get_all("Cookie", [])) <= 1)
            cookie.load(self.headers.get("Cookie", ""))
        except CookieError:
            raise APIError(400) from None
        return cookie[COOKIE].value if COOKIE in cookie else None

    def origin(self):
        require(len(self.headers.get_all("Origin", [])) == 1 and self.headers["Origin"] == f"http://{self.headers['Host']}", 403)
        require(self.headers.get("Sec-Fetch-Site") in (None, "same-origin"), 403)

    def body(self):
        require(self.headers.get("Content-Type", "").lower() in ("application/json", "application/json; charset=utf-8"))
        length = self.headers.get("Content-Length", "")
        require(length.isascii() and length.isdigit() and 0 < int(length) <= 16384)

        def unique(pairs):
            obj = {}
            for key, value in pairs:
                require(key not in obj)
                obj[key] = value
            return obj

        try:
            raw = self.rfile.read(int(length))
            require(len(raw) == int(length))
            value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                               parse_constant=lambda _: (_ for _ in ()).throw(APIError(400)))
        except (UnicodeError, ValueError, RecursionError):
            raise APIError(400) from None
        require(isinstance(value, dict))
        return value

    def dispatch(self):
        require(len(self.headers.get_all("Host", [])) == 1 and f"http://{self.headers['Host']}" in self.server.origins, 403)
        require(not self.headers.get_all("Transfer-Encoding") and len(self.headers.get_all("Content-Length", [])) <= 1)
        if self.command != "POST":
            require(self.headers.get("Content-Length", "0") == "0")
        try:
            parsed = urlsplit(self.path)
            require(not parsed.scheme and not parsed.netloc and not parsed.fragment)
            path = unquote(parsed.path, errors="strict")
            pairs = parse_qsl(parsed.query, keep_blank_values=True, strict_parsing=True, max_num_fields=8)
        except (ValueError, UnicodeError):
            raise APIError(400) from None
        params = dict(pairs)
        require(len(params) == len(pairs))
        if path in STATIC_FILES and self.command in ("GET", "HEAD"):
            require(not params)
            asset = WEB / STATIC_FILES[path]
            require(not WEB.is_symlink() and not asset.is_symlink() and asset.is_file()
                    and asset.resolve().parent == WEB.resolve(), 404)
            self.route = "app_shell"
            self.content_type = {".html": "text/html", ".css": "text/css", ".js": "text/javascript"}[asset.suffix] + "; charset=utf-8"
            self.extra_headers["Content-Security-Policy"] = APP_CSP
            return asset.read_bytes()
        self.route = path if path in ("/api/dev/profiles", "/api/dev/session", "/api/session", "/api/me", "/api/menu", "/api/search") else "document_or_unknown"
        if path.startswith("/api/dev/"):
            require(self.server.enabled, 404)
        if self.command in ("POST", "DELETE"):
            self.origin()
        if path == "/api/dev/profiles" and self.command == "GET":
            require(not params)
            return {"profiles": [{"profile_id": pid, "label": p["description"]} for pid, p in self.catalog["profiles"].items()]}
        self.session_token = self.token()
        if path == "/api/dev/session" and self.command == "POST":
            require(not params)
            args = self.body()
            require(set(args) == {"profile_id"} and isinstance(args["profile_id"], str) and args["profile_id"] in self.catalog["profiles"])
            if self.session_token:
                self.server.session(self.session_token, self.catalog["generation"])
                with self.server.lock:
                    require(self.server.sessions.pop(self.session_token, None) is not None, 401)
            identity = resolve_identity(args["profile_id"], self.catalog["snapshot"], self.catalog, self.server.store)
            self.profile_id = identity.profile_id
            self.session_token = secrets.token_urlsafe(32)
            self.revision = secrets.token_urlsafe(24)
            with self.server.lock:
                now = time.monotonic()
                self.server.sessions = {k: v for k, v in self.server.sessions.items() if v["expires"] > now}
                require(len(self.server.sessions) < 1000, 503)
                self.server.sessions[self.session_token] = {"profile_id": self.profile_id, "revision": self.revision,
                                                           "principal": identity.principal.to_dict(), "expires": now + SESSION_SECONDS,
                                                           "generation": self.catalog["generation"]}
            self.extra_headers["Set-Cookie"] = f"{COOKIE}={self.session_token}; HttpOnly; SameSite=Strict; Path=/; Max-Age={SESSION_SECONDS}"
            return {"principal": {"profile_id": self.profile_id, **identity.principal.to_dict()}}
        if path == "/api/session" and self.command == "DELETE":
            require(not params)
            with self.server.lock:
                self.server.sessions.pop(self.session_token, None)
            self.extra_headers["Set-Cookie"] = f"{COOKIE}=; HttpOnly; SameSite=Strict; Path=/; Max-Age=0"
            self.session_token, self.status = None, 204
            return b""
        document = re.fullmatch(r"/api/documents/(" + DOCUMENT_KEY + r")(/raw|/download)?", path)
        if path.startswith("/api/documents/") and not document:
            raise APIError(400)
        require(document and self.command == "GET" or path in ("/api/me", "/api/menu") and self.command == "GET"
                or path == "/api/search" and self.command == "POST", 404)
        require(self.server.enabled, 401)
        session = self.server.session(self.session_token, self.catalog["generation"])
        self.profile_id, self.revision = session["profile_id"], session["revision"]
        identity = resolve_identity(self.profile_id, self.catalog["snapshot"], self.catalog, self.server.store)
        require(session["principal"] == identity.principal.to_dict(), 401)
        if not document:
            require(not params)
            if path == "/api/me":
                return {"principal": {"profile_id": self.profile_id, **identity.principal.to_dict()}}
            if path == "/api/menu":
                nodes = menu_nodes(identity, self.catalog["tables"], self.catalog["snapshot"])
                self.result_ids = sorted({k for node in nodes for k in node["document_keys"]})
                return {"nodes": nodes}
            result = bm25_search(identity, self.body())
            self.result_ids = [item["document_key"] for item in result["items"]]
            return result
        key, action = document.groups()
        require(not set(params) - ({"language", "kind"} if action == "/download" else {"language"} if action == "/raw" else set()))
        language, kind = params.get("language", "zh"), params.get("kind", "markdown")
        require(language in ("zh", "en") and kind in ("markdown", "metadata"))
        result = identity.request("GET", f"/{identity.index}/_doc/{quote(key, safe='')}", expected=(200, 404))
        require("error" not in result, 503)
        require(result.get("found") is True, 404)
        doc = result["_source"]
        require(doc["document_key"] == key and doc["source"]["snapshot_sha256"] == self.catalog["snapshot"], 503)
        self.result_ids = [key]
        if not action:
            return {"document": {**metadata_document(doc), "languages": {lang: {**{k: value[k] for k in
                    ("source_field", "html", "headings", "assets", "raw_utf8_sha256", "table_warnings", "omitted_decorations")},
                    "raw_is_null": value["raw"] is None, "raw_is_empty": value["raw"] == ""} for lang, value in doc["languages"].items()}}}
        if action == "/download":
            extension = "json" if kind == "metadata" else "md"
            self.extra_headers["Content-Disposition"] = f'attachment; filename="document-{doc["document_id"]}-{language}.{extension}"'
            if kind == "metadata":
                return metadata_document(doc)
        self.content_type = "text/plain; charset=utf-8"
        raw = doc["languages"][language]["raw"]
        if raw is None:
            self.status, self.extra_headers["X-Source-Field-Null"] = 204, "true"
            return b""
        return raw.encode("utf-8")

    def handle_api(self):
        with ExitStack() as stack:
            self.handle_locked_api(stack)

    def handle_locked_api(self, stack):
        started = time.monotonic()
        self.request_id = secrets.token_hex(16)
        self.status, self.content_type = 200, "application/json; charset=utf-8"
        self.extra_headers, self.result_ids = {}, []
        self.profile_id, self.revision, self.session_token, self.route = None, None, None, "unparsed"
        try:
            self.catalog = stack.enter_context(self.server.store.read()) or self.server.store.baseline(self.server.snapshot, self.server.tables)
            value = self.dispatch()
            if self.revision:
                require(self.server.session(self.session_token, self.catalog["generation"])["revision"] == self.revision, 401)
            if isinstance(value, dict):
                value = {"request_id": self.request_id, **({"session_revision": self.revision} if self.revision else {}), **value}
            payload = value if isinstance(value, bytes) else json_bytes(value)
        except Exception as error:
            self.status = error.status if isinstance(error, APIError) else 503
            self.result_ids, self.extra_headers = [], {}
            self.content_type = "application/json; charset=utf-8"
            payload = json_bytes({"request_id": self.request_id, "error": {"code": CODES[self.status], "message": MESSAGES[self.status]}})
            if self.status == 401:
                with self.server.lock:
                    self.server.sessions.pop(self.session_token, None)
        self.send_response(self.status)
        for key, value in {"Content-Type": self.content_type, "Content-Length": str(len(payload)), "Cache-Control": "no-store",
                           "X-Request-ID": self.request_id, "X-Content-Type-Options": "nosniff",
                           "Content-Security-Policy": "default-src 'none'; frame-ancestors 'none'; sandbox",
                           **({"X-Session-Revision": self.revision} if self.revision else {}), **self.extra_headers}.items():
            self.send_header(key, value)
        self.end_headers()
        try:
            if self.command != "HEAD":
                self.wfile.write(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        print(json.dumps({"request_id": self.request_id, "profile_id": self.profile_id, "route": self.route, "status": self.status,
                          "index": self.server.store.index, "policy_version": POLICY_VERSION, "transform_version": TRANSFORM_VERSION,
                          "analyzer": "playbook_search", "synonyms_version": self.server.synonyms["version"],
                          "synonyms_sha256": canonical_hash(self.server.synonyms), "result_ids": self.result_ids,
                          "elapsed_ms": round((time.monotonic() - started) * 1000, 2)}, ensure_ascii=False), file=self.server.audit_stream, flush=True)

    do_GET = do_POST = do_DELETE = do_PUT = do_PATCH = do_OPTIONS = do_HEAD = handle_api


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=18080)
    parser.add_argument("--validation", action="store_true", help="Serve only the isolated S6 validation catalog/users")
    args = parser.parse_args()
    with APIServer(args.port, CatalogStore(args.validation)) as server:
        print(f"Playbook API: http://127.0.0.1:{server.server_port} (mock {'enabled' if server.enabled else 'disabled'})", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
