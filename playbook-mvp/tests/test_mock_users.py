from __future__ import annotations
from copy import deepcopy
import itertools
import json
from pathlib import Path
import unittest

from mock_auth import (AuthError, MockIdentityProvider, ROLE_NAMES, can_read_acl,
                       load_profiles, normalize_authentication_info, read_json,
                       validate_mock_profile)

ROOT = Path(__file__).resolve().parents[1]


class MockUsersTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.profiles = load_profiles()
        cls.docs = read_json(ROOT / 'fixtures/document_acl_snapshot.json')['documents']
        cls.doc = {d['document_id']: d for d in cls.docs}
        cls.expected = read_json(ROOT / 'fixtures/expected_access.json')['expectations']
        cls.summary = read_json(ROOT / 'reports/SOURCE_VALIDATION.json')

    def raw(self, name='pb_partner_nce_01'):
        return deepcopy(self.profiles[name]['authentication_info'])

    def principal(self, name):
        return validate_mock_profile(self.profiles[name])

    def assert_invalid(self, raw, code):
        with self.assertRaises(AuthError) as cm:
            normalize_authentication_info(raw)
        self.assertEqual(cm.exception.code, code)

    def test_01_exactly_twenty_valid_users(self):
        self.assertEqual(len(self.profiles), 20)
        for p in self.profiles.values():
            self.assertIsNotNone(validate_mock_profile(p))
            self.assertEqual(p['classification'], 'valid_mock_identity')

    def test_02_unique_user_ids(self):
        self.assertEqual(len({p['authentication_info']['UserId'] for p in self.profiles.values()}), 20)

    def test_03_unique_user_names(self):
        self.assertEqual(len({p['authentication_info']['UserName'] for p in self.profiles.values()}), 20)

    def test_04_every_user_has_one_role(self):
        self.assertTrue(all(len(p['authentication_info']['Roles']) == 1 for p in self.profiles.values()))

    def test_05_dictionary_matches_sql(self):
        tables = read_json(ROOT/'fixtures/source_acl_tables.json')['tables']
        self.assertEqual({r['Id']:r['Name'] for r in tables['Role']}, ROLE_NAMES)

    def test_06_five_groups_four_each(self):
        self.assertEqual(self.summary['user_groups'], {'msft':4,'21v':4,'partner_nce':4,'partner_ospa':4,'partner_both':4})

    def test_07_internal_empty_raw_defaults_all_business(self):
        for group in ('msft','21v'):
            for i in range(1,5):
                p=self.principal(f'pb_{group}_{i:02d}')
                self.assertEqual(p.raw_business_categories,())
                self.assertEqual(p.effective_business_categories,(0,1))

    def test_08_partner_preserves_explicit_business(self):
        for suffix, bc in [('nce',(0,)),('ospa',(1,)),('both',(0,1))]:
            for i in range(1,5):
                self.assertEqual(self.principal(f'pb_partner_{suffix}_{i:02d}').effective_business_categories,bc)

    def test_09_partner_flag_is_exact(self):
        for p in self.profiles.values():
            raw=p['authentication_info']
            self.assertEqual(raw['IsPartner'],raw['Roles'][0]['Id']==2)

    def test_10_saved_effective_fixtures_match_normalizer(self):
        expected = read_json(ROOT/'fixtures/mock_users_effective.json')['users']
        for p in expected:
            self.assertEqual({k:v for k,v in p.items() if k!='profile_id'}, self.principal(p['profile_id']).to_dict())

    def test_11_es_metadata_is_candidate_only_and_normalized(self):
        metadata=read_json(ROOT/'fixtures/es_metadata_candidates.json')
        self.assertFalse(metadata['provisioned'])
        for user in metadata['users']:
            self.assertEqual(user['metadata'],self.principal(user['username']).es_metadata_candidate())
            self.assertNotIn('password',user)

    def test_12_missing_role_rejected(self):
        r=self.raw();r['Roles']=[]
        self.assert_invalid(r,'exactly_one_role_required')

    def test_13_all_mixed_role_pairs_rejected(self):
        for a,b in itertools.permutations(ROLE_NAMES,2):
            with self.subTest(roles=(a,b)):
                r=self.raw();r['Roles']=[{'Id':x,'Name':ROLE_NAMES[x]} for x in (a,b)]
                self.assert_invalid(r,'exactly_one_role_required')

    def test_14_duplicate_role_is_not_silently_deduplicated(self):
        r=self.raw();r['Roles']*=2
        self.assert_invalid(r,'exactly_one_role_required')

    def test_15_three_roles_rejected(self):
        r=self.raw();r['Roles']=[{'Id':x,'Name':ROLE_NAMES[x]} for x in ROLE_NAMES]
        self.assert_invalid(r,'exactly_one_role_required')

    def test_16_partner_flag_mismatch_rejected(self):
        r=self.raw();r['IsPartner']=False
        self.assert_invalid(r,'partner_flag_mismatch')

    def test_17_unknown_role_rejected(self):
        r=self.raw();r['Roles']=[{'Id':99,'Name':'Admin'}]
        self.assert_invalid(r,'unknown_role')

    def test_18_unknown_business_rejected(self):
        r=self.raw();r['BusinessCategories']=[9]
        self.assert_invalid(r,'unknown_business')

    def test_19_empty_partner_business_rejected(self):
        r=self.raw();r['BusinessCategories']=[]
        self.assert_invalid(r,'partner_business_missing')

    def test_20_boolean_role_id_rejected(self):
        r=self.raw();r['Roles']=[{'Id':True,'Name':'MSFT'}]
        self.assert_invalid(r,'invalid_role')

    def test_21_identity_missing_rejected(self):
        r=self.raw();r['UserId']=''
        self.assert_invalid(r,'identity_missing')

    def test_22_untrusted_data_shape_rejected(self):
        self.assert_invalid(None,'invalid_authentication_info')

    def test_23_disabled_and_expired_rejected(self):
        for key,value,code in [('authenticated',False,'not_authenticated'),('enabled',False,'identity_disabled'),('expired',True,'session_expired')]:
            p=deepcopy(self.profiles['pb_partner_ospa_01']);p['mock_session'][key]=value
            with self.subTest(key=key), self.assertRaises(AuthError) as cm:
                validate_mock_profile(p)
            self.assertEqual(cm.exception.code,code)

    def test_24_production_always_rejects_mock(self):
        with self.assertRaises(AuthError):
            MockIdentityProvider('production',True)

    def test_25_explicit_mock_switch_required(self):
        for env, flag in [('development',False),('',True),('test',False)]:
            with self.subTest(env=env,flag=flag),self.assertRaises(AuthError):
                MockIdentityProvider(env,flag)

    def test_26_unknown_profile_rejected(self):
        with self.assertRaises(AuthError):
            MockIdentityProvider('test',True).get_principal('not-a-predefined-user')

    def test_27_document_counts_from_sql(self):
        self.assertEqual(len(self.docs),130)
        self.assertEqual(sum(d['acl']['is_public'] for d in self.docs),76)
        for bc, count in [(0,26),(1,28)]:
            self.assertEqual(sum(not d['acl']['is_public'] and d['business_category']==bc for d in self.docs),count)

    def test_28_all_2600_pairs_match_independent_relational_oracle(self):
        cases=[json.loads(s) for s in (ROOT/'fixtures/authorization_cases.jsonl').read_text().splitlines()]
        self.assertEqual(len(cases),2600)
        for case in cases:
            with self.subTest(user=case['profile_id'],document=case['document_id']):
                self.assertEqual(can_read_acl(self.principal(case['profile_id']),self.doc[case['document_id']]),case['expected_acl_allow'])

    def test_29_expected_group_counts_and_three_sets(self):
        counts={'msft':130,'21v':130,'partner_nce':102,'partner_ospa':104,'partner_both':130}
        for e in self.expected:
            self.assertEqual(e['allow_count'],counts[e['group']])
        self.assertEqual(len({tuple(e['allowed_document_ids']) for e in self.expected}),3)

    def test_30_public_is_cross_business(self):
        for profile in self.profiles:
            for did in (3,81):
                self.assertTrue(can_read_acl(self.principal(profile),self.doc[did]))

    def test_31_real_same_title_contract_pair(self):
        self.assertEqual(self.doc[25]['title'],self.doc[99]['title'])
        nce=self.principal('pb_partner_nce_01');ospa=self.principal('pb_partner_ospa_01')
        self.assertTrue(can_read_acl(nce,self.doc[25]));self.assertFalse(can_read_acl(nce,self.doc[99]))
        self.assertFalse(can_read_acl(ospa,self.doc[25]));self.assertTrue(can_read_acl(ospa,self.doc[99]))

    def test_32_internal_is_not_superuser_synthetic_unit_test_only(self):
        d=deepcopy(self.doc[99]);d['document_id']=-999  # Synthetic in-memory mutation, not source data.
        d['acl']['allowed_role_ids']=[2]
        for profile in ('pb_msft_01','pb_21v_01'):
            self.assertFalse(can_read_acl(self.principal(profile),d))
        self.assertTrue(can_read_acl(self.principal('pb_partner_ospa_01'),d))

    def test_33_unresolved_and_inactive_deny(self):
        for key,value in [('state','unresolved'),('menu_active_raw',0)]:
            d=deepcopy(self.doc[99]);d['acl'][key]=value
            self.assertFalse(can_read_acl(self.principal('pb_partner_ospa_01'),d))

    def test_34_no_publication_inference(self):
        self.assertTrue(all(d['publication_state']=='unverified' for d in self.docs))
        self.assertFalse(self.summary['production_publication_inferred'])
        self.assertFalse(self.summary['es_integration_executed'])

    def test_35_normalization_does_not_mutate_source(self):
        r=self.raw('pb_msft_01');original=deepcopy(r)
        normalize_authentication_info(r)
        self.assertEqual(r,original)

    def test_36_allow_and_deny_cover_exact_universe(self):
        universe=set(self.doc)
        for e in self.expected:
            allow=set(e['allowed_document_ids']);deny=set(e['denied_document_ids'])
            self.assertEqual(allow|deny,universe);self.assertFalse(allow&deny)
            self.assertEqual(e['allow_count']+e['deny_count'],130)

    def test_37_source_role_menu_sets_equal_not_artificially_differentiated(self):
        tables=read_json(ROOT/'fixtures/source_acl_tables.json')['tables']
        sets={rid:{g['MenuItemId'] for g in tables['Role_MenuItem'] if g['RoleId']==rid} for rid in ROLE_NAMES}
        self.assertEqual({len(v) for v in sets.values()},{65})
        self.assertEqual(sets[1],sets[2]);self.assertEqual(sets[2],sets[3])


if __name__=='__main__':
    unittest.main()
