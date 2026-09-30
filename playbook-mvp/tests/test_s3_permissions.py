"""S3 trust-boundary checks; live permission matrices run via es_permissions CLI."""
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from mock_auth import AuthError
from tools import es_permissions as permissions


class S3PermissionsTests(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"APP_ENV": "test", "MOCK_AUTH_ENABLED": "1"})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_twenty_normalized_identities_match_independent_candidates(self):
        users = permissions.principals()
        candidates = json.loads((permissions.ROOT / "fixtures/es_metadata_candidates.json").read_text())
        self.assertEqual(len(users), 20)
        self.assertEqual({pid: p.es_metadata_candidate() for pid, p in users.items()},
                         {u["username"]: u["metadata"] for u in candidates["users"]})

    def test_invalid_identity_batch_rejected_before_credentials_or_es(self):
        fixture = json.loads((permissions.ROOT / "fixtures/mock_users.json").read_text())
        for roles in ([], [{"Id": 1, "Name": "MSFT"}] * 2,
                      [{"Id": 1, "Name": "MSFT"}, {"Id": 2, "Name": "Partner"}],
                      [{"Id": True, "Name": "MSFT"}]):
            invalid = copy.deepcopy(fixture)
            invalid["profiles"][0]["authentication_info"]["Roles"] = roles
            with patch("mock_auth.read_json", return_value=invalid), patch.object(permissions, "credential") as password, patch.object(permissions.es, "request") as request:
                with self.assertRaises(AuthError):
                    permissions.request_as("pb_msft_01", "GET", "/_security/_authenticate")
                password.assert_not_called()
                request.assert_not_called()
        for field, value in (("authenticated", False), ("enabled", False), ("expired", True)):
            invalid = copy.deepcopy(fixture)
            invalid["profiles"][0]["mock_session"][field] = value
            with patch("mock_auth.read_json", return_value=invalid), self.assertRaises(AuthError):
                permissions.principals()

    def test_production_disabled_and_nonprofile_requests_never_reach_es(self):
        with patch.object(permissions.es, "request") as request:
            for environment, enabled in (("production", "1"), ("test", "0"), ("", "1")):
                with patch.dict(os.environ, {"APP_ENV": environment, "MOCK_AUTH_ENABLED": enabled}), self.assertRaisesRegex(AuthError, "mock_disabled"):
                    permissions.request_as("pb_msft_01", "GET", "/_security/_authenticate")
            for unknown in ("elastic", "pb_s2_importer", "../elastic", "pb_unknown"):
                with self.assertRaisesRegex(AuthError, "unknown_mock_profile"):
                    permissions.request_as(unknown, "GET", "/_security/_authenticate")
            request.assert_not_called()

    def test_missing_unsafe_credentials_never_fall_back_and_valid_are_bound(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(permissions, "CREDENTIALS", Path(temp)), patch.object(permissions.es, "request", return_value={}) as request:
            Path(temp).chmod(0o700)
            with self.assertRaises(FileNotFoundError):
                permissions.request_as("pb_msft_01", "GET", "/_security/_authenticate")
            path = Path(temp) / "pb_msft_01_password"
            path.write_text("a" * 64 + "\n")
            path.chmod(0o644)
            with self.assertRaisesRegex(RuntimeError, "Unsafe credential"):
                permissions.request_as("pb_msft_01", "GET", "/_security/_authenticate")
            request.assert_not_called()
            path.chmod(0o600)
            permissions.request_as("pb_msft_01", "GET", "/_security/_authenticate")
            request.assert_called_once_with("pb_msft_01", "GET", "/_security/_authenticate", None, 200, password_value="a" * 64)

    def test_permission_assertions_reject_leaked_or_incomplete_responses(self):
        permissions.check_item({"_id": "denied", "found": False}, "denied", None)
        for item in ({"_id": "denied", "found": False, "_source": {"title": "secret"}},
                     {"_id": "denied", "error": {"type": "some_failure"}},
                     {"_id": "wrong", "found": False}):
            with self.assertRaises(RuntimeError):
                permissions.check_item(item, "denied", None)
        with self.assertRaises(RuntimeError):
            permissions.check_item({"_id": "allowed", "found": True, "_source": {}}, "allowed", {"title": "expected"})


if __name__ == "__main__":
    unittest.main()
