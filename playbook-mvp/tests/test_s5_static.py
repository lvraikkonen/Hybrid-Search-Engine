"""Exercise the real static HTTP boundary, independent of ES availability."""
import io
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import api
from tools.check_api import call, expect


class S5StaticChecks(unittest.TestCase):
    def test_only_shell_assets_are_served_without_auth_or_es(self):
        with api.APIServer(0) as server, patch.object(api.es, "request", side_effect=AssertionError("Static must not query ES")):
            server.audit_stream = io.StringIO()
            worker = threading.Thread(target=server.serve_forever, daemon=True)
            worker.start()
            try:
                for path, name in api.STATIC_FILES.items():
                    headers, body = expect(call(server.server_port, "GET", path))
                    self.assertEqual(body, (api.WEB / name).read_bytes())
                    self.assertEqual(headers["cache-control"], "no-store")
                    self.assertIn("script-src 'self'", headers["content-security-policy"])
                    self.assertNotIn("set-cookie", headers)
                shell = (api.WEB / "index.html").read_text()
                self.assertNotIn("portal-data", shell)
                self.assertNotIn("INLINE_", shell)
                self.assertIn('src="/app.js"', shell)
                self.assertEqual(expect(call(server.server_port, "HEAD", "/"))[1], b"")
                paths = ["/src/app.js", "/web/index.html", "/data/portal_data.json", "/data/playbook-sampledata.sql",
                         "/fixtures/expected_access.json", "/fixtures/authorization_cases.jsonl", "/.local/s3/",
                         "/reports/S4_CHECKS.json", "/raw/document-25.md", "/markdown/000025.zh.md", "/source.zip",
                         "/Playbook_知识库_20用户演示.html", "/%2e%2e/data/playbook-sampledata.sql", "/..%2fdata/portal_data.json",
                         "/%252e%252e/data/portal_data.json", "/app.js/../data/portal_data.json", "/app.js%00"]
                for path in paths:
                    expect(call(server.server_port, "GET", quote(path, safe="/%")), 404)
                expect(call(server.server_port, "GET", "/?file=data/portal_data.json"), 400)
                expect(call(server.server_port, "GET", "/", headers={"Host": "elsewhere.invalid"}), 403)
                with tempfile.TemporaryDirectory() as directory, patch.object(api, "WEB", Path(directory)):
                    (Path(directory) / "index.html").symlink_to(ROOT / "data/portal_data.json")
                    expect(call(server.server_port, "GET", "/"), 404)
                    expect(call(server.server_port, "GET", "/app.js"), 404)
                with tempfile.TemporaryDirectory() as directory:
                    link = Path(directory) / "web"
                    link.symlink_to(api.WEB, target_is_directory=True)
                    with patch.object(api, "WEB", link):
                        expect(call(server.server_port, "GET", "/"), 404)
            finally:
                server.shutdown()
                worker.join()


if __name__ == "__main__":
    unittest.main()
