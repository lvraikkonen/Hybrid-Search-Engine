#!/usr/bin/env bash
# Existing, explicitly initialized/open S1-S3 environment is required.
# Add --restart-es to include the reversible container restart check.
set -euo pipefail
cd "$(dirname "$0")/.."
export APP_ENV=test MOCK_AUTH_ENABLED=1
uv run python -B build.py
uv run python -B -m unittest discover -s tests -p 'test_*.py' -v
node --test tests/test_s5_client.mjs
uv run python -B tools/es_permissions.py check
uv run python -B tools/check_api.py
uv run python -B tools/evaluate_search.py run
uv run python -B tools/check_s6.py "$@"
printf '%s\n' 'Automated checks passed. Browser interaction evidence is recorded separately in reports/S6_BROWSER_CHECKS.json.'
