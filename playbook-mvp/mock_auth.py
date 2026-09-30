"""Exclusive-role Playbook mock identities and an OFFLINE ACL reference predicate.

No HTTP calls, passwords, ES accounts or real tokens. The CLI can only resolve
server-maintained profile IDs when mock mode is explicitly enabled locally.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parent
ROLE_NAMES = {1: 'MSFT', 2: 'Partner', 3: '21V'}
BUSINESS_CATEGORIES = {0, 1}  # LOCAL enum, not the external BusinessTypes keys.
POLICY_VERSION = 'mock-auth-v2-exclusive-role'


class AuthError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class Principal:
    user_id: str
    user_name: str
    role_id: int
    raw_business_categories: tuple[int, ...]
    effective_business_categories: tuple[int, ...]
    business_permission_source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            'user_id': self.user_id,
            'user_name': self.user_name,
            'is_partner': self.role_id == 2,
            'role_ids': [self.role_id],
            'role_names': [ROLE_NAMES[self.role_id]],
            'raw_business_categories': list(self.raw_business_categories),
            'effective_business_categories': list(self.effective_business_categories),
            'business_permission_source': self.business_permission_source,
            'authz_mode': 'mock',
            'policy_version': POLICY_VERSION,
        }

    def es_metadata_candidate(self) -> dict[str, Any]:
        """Metadata projection only; does not provision or authenticate an ES user."""
        return {
            'playbook_role_ids': [self.role_id],
            'playbook_business_categories': list(self.effective_business_categories),
            'authz_mode': 'mock',
            'authz_policy_version': POLICY_VERSION,
        }


def normalize_authentication_info(raw: Mapping[str, Any]) -> Principal:
    """Normalize trusted server-side mock data; never trust browser role overrides.

    Three roles are mutually exclusive. Even a duplicate pair [1, 1] is invalid;
    do not silently deduplicate multi-entry input into one valid role.
    """
    if not isinstance(raw, Mapping):
        raise AuthError('invalid_authentication_info')
    uid, name = raw.get('UserId'), raw.get('UserName')
    if not isinstance(uid, str) or not uid.strip() or not isinstance(name, str) or not name.strip():
        raise AuthError('identity_missing')
    roles = raw.get('Roles')
    if not isinstance(roles, list) or len(roles) != 1:
        raise AuthError('exactly_one_role_required')
    role = roles[0]
    if not isinstance(role, Mapping) or type(role.get('Id')) is not int:
        raise AuthError('invalid_role')
    rid = role['Id']
    if rid not in ROLE_NAMES:
        raise AuthError('unknown_role')
    if role.get('Name') != ROLE_NAMES[rid]:
        raise AuthError('role_name_mismatch')
    if type(raw.get('IsPartner')) is not bool or raw['IsPartner'] != (rid == 2):
        raise AuthError('partner_flag_mismatch')
    categories = raw.get('BusinessCategories')
    if not isinstance(categories, list) or any(type(x) is not int for x in categories):
        raise AuthError('invalid_business_categories')
    if not set(categories).issubset(BUSINESS_CATEGORIES):
        raise AuthError('unknown_business')
    if rid in (1, 3):
        effective, source = (0, 1), 'internal_default'
    else:
        if not categories:
            raise AuthError('partner_business_missing')
        effective, source = tuple(sorted(set(categories))), 'partner_explicit'
    return Principal(uid, name, rid, tuple(categories), effective, source)


def validate_mock_profile(profile: Mapping[str, Any]) -> Principal:
    if not isinstance(profile, Mapping):
        raise AuthError('invalid_profile')
    session = profile.get('mock_session')
    if not isinstance(session, Mapping) or session.get('authenticated') is not True:
        raise AuthError('not_authenticated')
    if session.get('enabled') is not True:
        raise AuthError('identity_disabled')
    if session.get('expired') is not False:
        raise AuthError('session_expired')
    return normalize_authentication_info(profile.get('authentication_info'))


def can_read_acl(principal: Principal, document: Mapping[str, Any]) -> bool:
    """ACL-only OFFLINE predicate, not a production search filter or DLS test.

    Direct single-menu snapshot policy. Active/Role.Status=1 are conservative
    test gates; Document.Status is not interpreted as a publication flag.
    """
    if not isinstance(document, Mapping):
        return False
    acl = document.get('acl')
    if not isinstance(acl, Mapping) or acl.get('state') != 'resolved':
        return False
    if type(acl.get('menu_active_raw')) is not int or acl['menu_active_raw'] != 1:
        return False
    if type(acl.get('is_public')) is not bool:
        return False
    if acl['is_public']:
        return True
    roles, business = acl.get('allowed_role_ids'), acl.get('required_business_category')
    statuses = acl.get('role_status_raw')
    if not isinstance(roles, list) or not roles or any(type(r) is not int or r not in ROLE_NAMES for r in roles):
        return False
    if type(business) is not int or business not in BUSINESS_CATEGORIES:
        return False
    if not isinstance(statuses, dict) or statuses.get(str(principal.role_id)) != 1:
        return False
    return principal.role_id in roles and business in principal.effective_business_categories


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def load_profiles(path: Path = ROOT / 'fixtures' / 'mock_users.json') -> dict[str, Any]:
    items = read_json(path)['profiles']
    result = {item['profile_id']: item for item in items}
    if len(result) != len(items):
        raise AuthError('duplicate_profile_id')
    return result


class MockIdentityProvider:
    def __init__(self, environment: str, explicitly_enabled: bool):
        if environment not in {'development', 'test'} or explicitly_enabled is not True:
            raise AuthError('mock_disabled')
        self._profiles = load_profiles()

    def get_principal(self, profile_id: str) -> Principal:
        if profile_id not in self._profiles:
            raise AuthError('unknown_mock_profile')
        return validate_mock_profile(self._profiles[profile_id])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['list', 'whoami', 'visible', 'es-metadata'])
    parser.add_argument('--profile')
    args = parser.parse_args()
    try:
        provider = MockIdentityProvider(os.environ.get('APP_ENV', ''), os.environ.get('MOCK_AUTH_ENABLED') == '1')
        if args.command == 'list':
            result = [{'profile_id': k, 'description': v['description']} for k, v in load_profiles().items()]
        else:
            if not args.profile:
                parser.error('--profile is required')
            principal = provider.get_principal(args.profile)
            if args.command == 'whoami':
                result = principal.to_dict()
            elif args.command == 'es-metadata':
                result = principal.es_metadata_candidate()
            else:
                docs = read_json(ROOT / 'fixtures' / 'document_acl_snapshot.json')['documents']
                visible = sorted(d['document_id'] for d in docs if can_read_acl(principal, d))
                result = {'profile_id': args.profile, 'evaluation_scope': 'acl_only_not_publication_or_search',
                          'visible_count': len(visible), 'visible_document_ids': visible}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (AuthError, OSError, json.JSONDecodeError, KeyError) as exc:
        print(json.dumps({'error': getattr(exc, 'code', type(exc).__name__), 'detail': str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
