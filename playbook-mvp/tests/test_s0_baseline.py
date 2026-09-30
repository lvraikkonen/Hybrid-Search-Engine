"""Reproduce the frozen S0 SQL/Mock baseline without changing project data."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from mock_auth import read_json


ROOT = Path(__file__).resolve().parents[1]
SQL_SHA256 = 'cf738a308166a03adcafb9f7a8fdb38e88488fecf928dbf37063c284b5a64821'


class S0BaselineTests(unittest.TestCase):
    def test_sql_fixtures_and_original_bodies_reproduce(self):
        sql = ROOT / 'data/playbook-sampledata.sql'
        self.assertEqual(hashlib.sha256(sql.read_bytes()).hexdigest(), SQL_SHA256)
        for name in ('mock_users.json', 'mock_users_effective.json', 'expected_access.json'):
            self.assertEqual((ROOT / 'data' / name).read_bytes(), (ROOT / 'fixtures' / name).read_bytes(), name)
        for entry in read_json(ROOT / 'MANIFEST.json')['files']:
            if entry['path'].startswith('fixtures/'):
                self.assertEqual(hashlib.sha256((ROOT / entry['path']).read_bytes()).hexdigest(), entry['sha256'], entry['path'])

        portal = {d['id']: d for d in read_json(ROOT / 'data/portal_data.json')['documents']}
        fixture = {d['document_id']: d for d in read_json(ROOT / 'fixtures/document_acl_snapshot.json')['documents']}
        with tempfile.TemporaryDirectory(prefix='playbook-s0-') as temporary:
            output = Path(temporary)
            commands = [
                ['tools/build_dataset.py', '--sql', str(sql), '--out', str(output / 'mock')],
                ['tools/extract_playbook_sql.py', str(sql), '--out', str(output / 'extract')],
            ]
            for args in commands:
                result = subprocess.run([sys.executable, '-B', *args], cwd=ROOT, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)

            rebuilt = [p for p in (output / 'mock').rglob('*') if p.is_file()]
            self.assertEqual(len(rebuilt), 9)
            for path in rebuilt:
                relative = path.relative_to(output / 'mock')
                self.assertEqual(path.read_bytes(), (ROOT / relative).read_bytes(), str(relative))

            extracted = output / 'extract'
            manifest = read_json(extracted / 'manifest.json')
            self.assertEqual(manifest['source']['sha256'], SQL_SHA256)
            self.assertFalse(manifest['source']['sql_executed'])
            self.assertEqual(manifest['counts'], {
                'Document': 130, 'MenuItem': 164, 'MenuItem_Document': 130,
                'Notice': 4, 'Role': 3, 'Role_MenuItem': 195,
            })
            self.assertEqual(manifest['markdown_files'], 260)
            self.assertEqual(len(manifest['entries']), 130)
            self.assertEqual({e['document_id'] for e in manifest['entries']}, set(portal))
            self.assertEqual(set(portal), set(fixture))
            roundtrip = read_json(extracted / 'reports/body_roundtrip_verification.json')
            self.assertEqual(roundtrip['checked'], 260)
            self.assertTrue(roundtrip['all_equal'])

            for entry in manifest['entries']:
                did = entry['document_id']
                with self.subTest(document=did):
                    metadata = read_json(extracted / entry['metadata_file'])
                    acl = read_json(extracted / entry['acl_file'])
                    self.assertEqual(entry['document_key'], f'playbook-test:Document:{did}')
                    self.assertEqual(entry['document_key'], fixture[did]['document_key'])
                    self.assertEqual(metadata['publication_state'], 'unverified')
                    self.assertIsNone(metadata['timestamp_timezone'])
                    self.assertEqual(metadata['source']['sql_lines'], fixture[did]['document_sql_lines'])
                    for key in ('state', 'menu_id', 'menu_active_raw', 'is_public', 'allowed_role_ids', 'required_business_category'):
                        self.assertEqual(acl[key], fixture[did]['acl'][key], key)
                        self.assertEqual(acl[key], portal[did]['acl'][key], key)
                    statuses = {str(r['Id']): r['Status'] for r in acl['provenance']['Role']}
                    self.assertEqual(statuses, fixture[did]['acl']['role_status_raw'])
                    self.assertEqual(statuses, {str(r['Id']): r['Status'] for r in portal[did]['acl_provenance']['Role']})
                    for field, language in [('Content', 'zh'), ('ContentEn', 'en')]:
                        source = metadata['files'][field]
                        body = (extracted / source['file']).read_bytes()
                        self.assertEqual(body.decode('utf-8'), portal[did]['languages'][language]['raw'])
                        self.assertEqual(hashlib.sha256(body).hexdigest(), source['utf8_sha256'])
                    if did in (25, 99):
                        menu, business, relation, grants = {
                            25: (31, 0, 87, [146, 173, 200]),
                            99: (114, 1, 685, [845, 878, 911]),
                        }[did]
                        self.assertEqual((acl['menu_id'], acl['required_business_category']), (menu, business))
                        self.assertEqual(acl['allowed_role_ids'], [1, 2, 3])
                        self.assertEqual([r['Id'] for r in acl['provenance']['MenuItem_Document']], [relation])
                        self.assertEqual([r['Id'] for r in acl['provenance']['Role_MenuItem']], grants)

    def test_seven_mock_cli_checks(self):
        cases = read_json(ROOT / 'reports/CLI_CHECKS.json')
        self.assertEqual(len(cases), 7)
        users = {u['profile_id']: u for u in read_json(ROOT / 'fixtures/mock_users_effective.json')['users']}
        expectations = {e['profile_id']: e for e in read_json(ROOT / 'fixtures/expected_access.json')['expectations']}
        candidates = {u['username']: u['metadata'] for u in read_json(ROOT / 'fixtures/es_metadata_candidates.json')['users']}
        for case in cases:
            args = case['command'][1:]
            with self.subTest(command=args, environment=case['environment']):
                result = subprocess.run(
                    [sys.executable, '-B', *args], cwd=ROOT,
                    env={**os.environ, **case['environment']}, capture_output=True, text=True,
                )
                self.assertEqual(result.returncode, case['expected_return_code'], result.stderr)
                data = json.loads(result.stdout if result.returncode == 0 else result.stderr)
                if result.returncode != 0:
                    self.assertEqual(data['error'], 'mock_disabled')
                elif args[1] == 'list':
                    self.assertEqual(len(data), 20)
                    self.assertEqual({u['profile_id'] for u in data}, set(users))
                elif args[1] == 'whoami':
                    self.assertEqual(data, {k: v for k, v in users[args[-1]].items() if k != 'profile_id'})
                elif args[1] == 'visible':
                    self.assertEqual(data['visible_document_ids'], expectations[args[-1]]['allowed_document_ids'])
                    self.assertEqual(data['visible_count'], expectations[args[-1]]['allow_count'])
                elif args[1] == 'es-metadata':
                    self.assertEqual(data, candidates[args[-1]])
                else:
                    self.fail(f'Unrecognized CLI check: {args}')


if __name__ == '__main__':
    unittest.main()
