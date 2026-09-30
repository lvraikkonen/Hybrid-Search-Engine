#!/usr/bin/env python3
"""Build twenty exclusive-role mock identities and SQL-derived ACL expectations.

Reads an SSMS dump lexically without executing it. Uses an in-memory SQLite JOIN
of the parsed metadata/relations as an independent oracle for expected access.
No passwords, native ES users or production permissions are created.
"""
from __future__ import annotations
import argparse
from collections import Counter
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mock_auth import validate_mock_profile
from extract_playbook_sql import decode_sql, parse_dump, validate_tables, derive_acl, digest

GROUPS = [
    ('msft', 1, [], [0, 1], 'MSFT · 全业务', 130),
    ('21v', 3, [], [0, 1], '21V · 全业务', 130),
    ('partner_nce', 2, [0], [0], 'Partner · 仅 NCE', 102),
    ('partner_ospa', 2, [1], [1], 'Partner · 仅 OSPA', 104),
    ('partner_both', 2, [0, 1], [0, 1], 'Partner · 双业务', 130),
]


def dump(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def build(sql: Path, out: Path):
    if (out / 'fixtures/mock_users.json').exists():
        raise ValueError('Output fixture already exists; choose a new --out directory.')
    raw = sql.read_bytes()
    text, encoding = decode_sql(raw)
    tables = parse_dump(text)
    checks = validate_tables(tables)
    if any(v for k, v in checks.items() if k != 'table_counts'):
        raise ValueError('Source relationships are incomplete or ambiguous; refuse to invent permissions.')
    roles = {r['Id']: r['Name'] for r in tables['Role']}
    if roles != {1: 'MSFT', 2: 'Partner', 3: '21V'}:
        raise ValueError('Role dictionary changed; review the fixture contract first.')
    source = {'file_name': sql.name, 'sha256': digest(raw), 'encoding': encoding,
              'instance_key': 'playbook-test', 'evaluation_scope': 'acl_only_not_publication_or_search',
              'policy': 'direct_single_menu_with_exclusive_user_role_v2',
              'publication_inference': 'none',
              'test_gates': 'MenuItem.Active=1 and granting Role.Status=1; confirm final upstream semantics.'}
    documents = []
    for d in sorted(tables['Document'], key=lambda d: d['Id']):
        a = derive_acl(d, tables)
        if a['state'] != 'resolved':
            raise ValueError(f'Unresolved ACL for Document {d["Id"]}')
        documents.append({
            'document_key': f'playbook-test:Document:{d["Id"]}', 'document_id': d['Id'],
            'title': d['Title'], 'title_en': d['TitleEn'], 'slug': d['Slug'],
            'business_category': d['BusinessCategory'], 'upstream_status': d['Status'],
            'publication_state': 'unverified', 'document_sql_lines': d['_sql_lines'],
            'acl': {k: a[k] for k in ('state', 'menu_id', 'menu_active_raw', 'is_public',
                                      'allowed_role_ids', 'required_business_category')} | {
                'role_status_raw': {str(r['Id']): r['Status'] for r in a['provenance']['Role']},
                'source_relation_ids': [r['Id'] for r in a['provenance']['MenuItem_Document']],
                'source_grant_ids': [r['Id'] for r in a['provenance']['Role_MenuItem']],
            },
        })
    profiles, principals, reference_identity = [], [], {}
    for group, rid, raw_bc, effective_bc, label, _ in GROUPS:
        for i in range(1, 5):
            name = f'pb_{group}_{i:02d}'
            uid = f'mock-{len(profiles)+1:03d}'
            p = {'profile_id': name, 'classification': 'valid_mock_identity', 'group': group,
                 'description': f'{label} · 模拟用户 {i:02d}',
                 'mock_session': {'authenticated': True, 'enabled': True, 'expired': False},
                 'authentication_info': {'UserId': uid, 'UserName': name, 'IsPartner': rid == 2,
                    'Roles': [{'Id': rid, 'Name': roles[rid]}], 'BusinessCategories': raw_bc.copy()}}
            profiles.append(p)
            normalized = validate_mock_profile(p).to_dict()
            principals.append({'profile_id': name, **normalized})
            # Oracle uses design-declared role/business, not normalization output.
            reference_identity[name] = (rid, effective_bc)
    f = out / 'fixtures'
    dump(f / 'mock_users.json', {'schema_version': '2.0', 'auth_mode': 'mock',
        'mutually_exclusive_roles': True, 'roles_per_user': 1, 'role_dictionary': roles,
        'business_dictionary': {0: 'NCE', 1: 'OSPA'},
        'business_categories_are_local_enum': True, 'profiles': profiles})
    dump(f / 'mock_users_effective.json', {'schema_version': '2.0',
        'description': 'Server-side normalized fixture; not a token or browser-supplied grant.', 'users': principals})
    dump(f / 'es_metadata_candidates.json', {'schema_version': '2.0',
        'provisioned': False, 'description': 'Metadata only. No ES accounts, roles, credentials or DLS configuration created.',
        'users': [{'username': p['profile_id'],
            'metadata': validate_mock_profile(p).es_metadata_candidate()} for p in profiles]})
    dump(f / 'document_acl_snapshot.json', {'source': source, 'documents': documents})
    subset = {t: rows for t, rows in tables.items() if t not in ('Document', 'Notice')}
    subset['Document'] = [{k: r[k] for k in ('Id', 'Title', 'TitleEn', 'Slug', 'BusinessCategory', 'Status', '_sql_lines')}
                          for r in tables['Document']]
    dump(f / 'source_acl_tables.json', {'source': source, 'tables': subset})
    c = sqlite3.connect(':memory:')
    c.executescript('''
       CREATE TABLE docs(id INTEGER, business INTEGER);
       CREATE TABLE menus(id INTEGER, public INTEGER, business INTEGER, active INTEGER, type INTEGER);
       CREATE TABLE bindings(doc_id INTEGER, menu_id INTEGER);
       CREATE TABLE grants(role_id INTEGER, menu_id INTEGER);
       CREATE TABLE roles(id INTEGER, status INTEGER);
    ''')
    c.executemany('INSERT INTO docs VALUES (?,?)', [(d['Id'], d['BusinessCategory']) for d in tables['Document']])
    c.executemany('INSERT INTO menus VALUES (?,?,?,?,?)', [(m['Id'], 1-m['UnderControl'], m['BusinessCategory'], m['Active'], m['Type']) for m in tables['MenuItem']])
    c.executemany('INSERT INTO bindings VALUES (?,?)', [(e['DocumentId'], e['MenuItemId']) for e in tables['MenuItem_Document']])
    c.executemany('INSERT INTO grants VALUES (?,?)', [(g['RoleId'], g['MenuItemId']) for g in tables['Role_MenuItem']])
    c.executemany('INSERT INTO roles VALUES (?,?)', [(r['Id'], r['Status']) for r in tables['Role']])
    all_ids = {d['document_id'] for d in documents}
    expected, cases = [], []
    doc_by_id = {d['document_id']: d for d in documents}
    for p in profiles:
        rid, bc = reference_identity[p['profile_id']]
        sql_query = '''SELECT DISTINCT d.id FROM docs d
            JOIN bindings b ON b.doc_id=d.id JOIN menus m ON m.id=b.menu_id
            WHERE m.type=0 AND m.active=1 AND d.business=m.business
            AND (m.public=1 OR (m.public=0 AND m.business IN (''' + ','.join('?' for _ in bc) + ''')
            AND EXISTS (SELECT 1 FROM grants g JOIN roles r ON r.id=g.role_id
                        WHERE g.menu_id=m.id AND g.role_id=? AND r.status=1))) ORDER BY d.id'''
        allow = [r[0] for r in c.execute(sql_query, [*bc, rid])]
        deny = sorted(all_ids - set(allow))
        expected.append({'profile_id': p['profile_id'], 'role_id': rid, 'effective_business_categories': bc,
            'allow_count': len(allow), 'deny_count': len(deny), 'allowed_document_ids': allow,
            'denied_document_ids': deny, 'group': p['group']})
        for did in sorted(all_ids):
            d = doc_by_id[did]
            cases.append({'profile_id': p['profile_id'], 'document_id': did,
                'expected_acl_allow': did in set(allow), 'evaluation_scope': 'acl_only',
                'menu_id': d['acl']['menu_id'], 'business_category': d['business_category']})
    c.close()
    dump(f / 'expected_access.json', {'source': source,
        'oracle': 'Independent SQLite JOIN over parsed source ACL relations; no original T-SQL executed.',
        'expectations': expected})
    (f / 'authorization_cases.jsonl').write_text(''.join(json.dumps(x, ensure_ascii=False) + '\n' for x in cases), encoding='utf-8')
    profiles_by_id = {p['profile_id']: p for p in profiles}
    lines = ['# 20 个模拟用户（角色严格互斥）', '',
        '以下仅为当前 SQL 的 ACL 可见集合；不是已发布数量、检索命中数或 DLS 测试结果。', '',
        '|用户ID|用户名|唯一角色|有效业务权限|ACL允许|ACL拒绝|', '|---|---|---|---|---:|---:|']
    for e in expected:
        p = profiles_by_id[e['profile_id']]
        bc_label = '+'.join({0:'NCE', 1:'OSPA'}[b] for b in e['effective_business_categories'])
        lines.append(f'|{p["authentication_info"]["UserId"]}|{e["profile_id"]}|{roles[e["role_id"]]}|{bc_label}|{e["allow_count"]}|{e["deny_count"]}|')
    lines += ['', '20 个不同身份、5 种角色/业务组合、3 种实际 ACL 文档集合。同类用户拥有相同 ACL，未引入个人或组织级限制。']
    (out / 'reports').mkdir(exist_ok=True)
    (out / 'reports/USERS_OVERVIEW.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    summary = {'source': source, 'table_validation': checks, 'user_count': len(profiles),
        'all_users_single_role': all(len(p['authentication_info']['Roles']) == 1 for p in profiles),
        'all_users_valid': True, 'user_groups': dict(Counter(p['group'] for p in profiles)),
        'document_count': len(documents), 'public_count': sum(d['acl']['is_public'] for d in documents),
        'controlled_by_business': {str(b): sum(not d['acl']['is_public'] and d['business_category']==b for d in documents) for b in (0,1)},
        'unique_acl_sets': len({tuple(x['allowed_document_ids']) for x in expected}),
        'offline_authorization_cases': len(cases),
        'acl_counts_by_group': {g: sorted({x['allow_count'] for x in expected if x['group']==g}) for g,*_ in GROUPS},
        'es_integration_executed': False, 'production_publication_inferred': False}
    dump(out / 'reports/SOURCE_VALIDATION.json', summary)
    return summary


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sql', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    try:
        print(json.dumps(build(args.sql, args.out), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError) as exc:
        p.exit(2, f'Build failed: {exc}\n')
