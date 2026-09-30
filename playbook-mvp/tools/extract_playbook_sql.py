#!/usr/bin/env python3
"""Extract an SSMS-style SQL INSERT snapshot without executing any SQL.

Supported values: SQL string literals (including N'', doubled quotes and newlines),
integers, NULL, and CAST(N'...' AS datetime/datetime2). Unsupported syntax fails.
Only [dbo].[Document/MenuItem/MenuItem_Document/Notice/Role/Role_MenuItem] is read.
This is a dump reader, not a general T-SQL engine.
"""
from __future__ import annotations
import argparse
import bisect
import collections
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

TABLES = {'Document','MenuItem','MenuItem_Document','Notice','Role','Role_MenuItem'}
HEAD = re.compile(r'INSERT\s+(?:INTO\s+)?\[dbo\]\.\[([^\]]+)\]\s*\(([^)]+)\)\s*VALUES\s*\(', re.I)
STR = re.compile(r"N?'((?:[^']|'')*)'", re.I | re.S)
CAST = re.compile(r"CAST\(\s*(N?'(?:[^']|'')*')\s+AS\s+(DateTime(?:2)?(?:\(\d+\))?)\s*\)", re.I | re.S)

def digest(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def decode_sql(raw: bytes) -> tuple[str,str]:
    if raw.startswith((b'\xff\xfe',b'\xfe\xff')):
        return raw.decode('utf-16'),'utf-16'
    return raw.decode('utf-8-sig'),'utf-8-sig'

def skip_string(s: str, i: int) -> int:
    assert s[i] == "'"
    j=i+1
    while j < len(s):
        if s[j] == "'":
            if j+1 < len(s) and s[j+1] == "'":
                j += 2
                continue
            return j+1
        j += 1
    raise ValueError(f'Unterminated string at offset {i}')

def value(token: str) -> Any:
    token=token.strip()
    if token.upper()=='NULL': return None
    if re.fullmatch(r'[+-]?\d+',token): return int(token)
    m=STR.fullmatch(token)
    if m: return m.group(1).replace("''", "'")
    m=CAST.fullmatch(token)
    if m: return value(m.group(1))
    raise ValueError(f'Unsupported SQL value: {token[:150]!r}')

def parse_dump(s: str) -> dict[str,list[dict[str,Any]]]:
    """Lexically skip strings/comments; a fake INSERT inside a body is never parsed."""
    out={t:[] for t in sorted(TABLES)}
    nl=[i for i,c in enumerate(s) if c=='\n']
    i=0
    while i<len(s):
        if s.startswith('--',i):
            j=s.find('\n',i+2); i=len(s) if j<0 else j+1; continue
        if s.startswith('/*',i):
            depth=1;j=i+2
            while j<len(s) and depth:
                if s.startswith('/*',j): depth+=1;j+=2
                elif s.startswith('*/',j): depth-=1;j+=2
                else:j+=1
            if depth: raise ValueError('Unterminated block comment')
            i=j;continue
        if s[i]=="'": i=skip_string(s,i);continue
        if s[i]=='[':
            j=i+1
            while j<len(s):
                if s.startswith(']]',j):j+=2
                elif s[j]==']':j+=1;break
                else:j+=1
            i=j;continue
        # Matching only outside quoted literals and identifiers is essential.
        if s[i:i+6].upper()=='INSERT' and (i==0 or not(s[i-1].isalnum() or s[i-1]=='_')):
            m=HEAD.match(s,i)
            if not m: raise ValueError(f'Unsupported INSERT at line {bisect.bisect_left(nl,i)+1}')
            table=m.group(1)
            if table not in TABLES: raise ValueError(f'Unexpected table {table}')
            cols=re.findall(r'\[([^\]]+)\]',m.group(2))
            j=m.end();start=j;depth=1;tokens=[]
            while j<len(s):
                c=s[j]
                if c=="'":j=skip_string(s,j);continue
                if c=='(':depth+=1
                elif c==')':
                    depth-=1
                    if depth==0:
                        tokens.append(s[start:j]);break
                elif c==',' and depth==1:
                    tokens.append(s[start:j]);start=j+1
                j+=1
            if depth: raise ValueError(f'Unterminated VALUES for {table}')
            vals=[value(x) for x in tokens]
            if len(cols)!=len(vals) or len(set(cols))!=len(cols):
                raise ValueError(f'Column/value mismatch for {table}')
            tail=j+1
            while tail<len(s) and s[tail].isspace():tail+=1
            if tail<len(s) and s[tail]==',':
                raise ValueError('Multi-row VALUES is unsupported; export one INSERT per row')
            row=dict(zip(cols,vals))
            row['_sql_lines']=[bisect.bisect_left(nl,i)+1,bisect.bisect_left(nl,j)+1]
            out[table].append(row)
            i=j+1;continue
        i+=1
    return out

def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def strip_provenance(row: dict[str,Any]) -> dict[str,Any]:
    return {k:v for k,v in row.items() if k!='_sql_lines'}


POLICY = 'direct_single_menu_v1_provisional'
BUSINESS = {0:'NCE',1:'OSPA'}

def validate_tables(tables: dict[str,list[dict[str,Any]]]) -> dict[str,Any]:
    for table,rows in tables.items():
        ids=[r['Id'] for r in rows]
        if len(ids)!=len(set(ids)): raise ValueError(f'Duplicate Id in {table}')
    maps={t:{r['Id']:r for r in rr} for t,rr in tables.items()}
    edges=tables['MenuItem_Document'];grants=tables['Role_MenuItem']
    def duplicate_pairs(rows,fields):
        c=collections.Counter(tuple(r[f] for f in fields) for r in rows)
        return [list(k) for k,v in c.items() if v>1]
    doc_ids={r['DocumentId'] for r in edges}
    broken_md=[r for r in edges if r['DocumentId'] not in maps['Document'] or r['MenuItemId'] not in maps['MenuItem']]
    broken_rm=[r for r in grants if r['RoleId'] not in maps['Role'] or r['MenuItemId'] not in maps['MenuItem']]
    by_doc=collections.Counter(r['DocumentId'] for r in edges)
    by_menu=collections.Counter(r['MenuItemId'] for r in edges)
    bad_parents=[];cycles=[]
    for mid,m in maps['MenuItem'].items():
        seen=set();cur=mid
        while cur not in (None,0):
            if cur in seen:cycles.append(mid);break
            if cur not in maps['MenuItem']:bad_parents.append(mid);break
            seen.add(cur);cur=maps['MenuItem'][cur]['ParentId']
    return {
        'table_counts':{k:len(v) for k,v in tables.items()},
        'documents_without_menu':sorted(set(maps['Document'])-doc_ids),
        'broken_document_menu_edges':broken_md,'broken_role_menu_edges':broken_rm,
        'duplicate_document_menu_edges':duplicate_pairs(edges,('DocumentId','MenuItemId')),
        'duplicate_role_menu_edges':duplicate_pairs(grants,('RoleId','MenuItemId')),
        'documents_with_multiple_menu_edges':sorted(k for k,n in by_doc.items() if n>1),
        'menus_with_multiple_documents':sorted(k for k,n in by_menu.items() if n>1),
        'invalid_parent_paths':bad_parents,'menu_cycles':cycles,
        'business_category_mismatches':[
            {'document_id':r['DocumentId'],'menu_id':r['MenuItemId']}
            for r in edges if r not in broken_md and maps['Document'][r['DocumentId']]['BusinessCategory']!=maps['MenuItem'][r['MenuItemId']]['BusinessCategory']
        ],
        'relations_pointing_to_non_document_menus':[
            r for r in edges if r not in broken_md and maps['MenuItem'][r['MenuItemId']]['Type']!=0
        ],
        'controlled_menus_without_role_grants':[
            mid for mid,m in maps['MenuItem'].items() if m['UnderControl']==1 and not any(r['MenuItemId']==mid for r in grants)
        ]
    }

def derive_acl(doc:dict[str,Any],tables:dict[str,list[dict[str,Any]]]) -> dict[str,Any]:
    menus={r['Id']:r for r in tables['MenuItem']}
    roles={r['Id']:r for r in tables['Role']}
    links=[r for r in tables['MenuItem_Document'] if r['DocumentId']==doc['Id']]
    base={'policy_id':POLICY,'state':'unresolved','is_public':False,
          'menu_id':None,'menu_active_raw':None,'required_business_category':None,
          'allowed_role_ids':[],'binding_menu_ids':[r['MenuItemId'] for r in links],
          'upstream_acl_version':None,'reasons':[],
          'provenance':{'MenuItem_Document':links,'MenuItem':[],'Role_MenuItem':[],'Role':[]}}
    if len(links)!=1:
        base['reasons'].append('missing_menu_binding' if not links else 'multiple_menu_semantics_unconfirmed')
        return base
    m=menus.get(links[0]['MenuItemId'])
    if m is None:
        base['reasons'].append('menu_missing');return base
    grants=[r for r in tables['Role_MenuItem'] if r['MenuItemId']==m['Id']]
    granted_ids=sorted({r['RoleId'] for r in grants})
    base.update(menu_id=m['Id'],menu_active_raw=m['Active'],required_business_category=m['BusinessCategory'])
    base['provenance'].update(MenuItem=[m],Role_MenuItem=grants,Role=[roles[i] for i in granted_ids if i in roles])
    if m['Type']!=0:base['reasons'].append('not_document_menu')
    if m['BusinessCategory'] not in BUSINESS or doc['BusinessCategory']!=m['BusinessCategory']:
        base['reasons'].append('business_category_mismatch_or_unknown')
    if m['UnderControl'] not in (0,1):base['reasons'].append('invalid_under_control')
    if any(i not in roles for i in granted_ids):base['reasons'].append('unknown_role')
    if m['UnderControl']==1 and not granted_ids:base['reasons'].append('no_formal_role_grants')
    if base['reasons']:return base
    base.update(state='resolved',is_public=(m['UnderControl']==0),allowed_role_ids=granted_ids)
    # Status/Active meanings are not inferred. Keep raw values and gate conservatively at evaluation.
    return base

def acl_allows(acl:dict[str,Any],role_ids:list[int],business_categories:list[int]) -> bool:
    """Offline PROVISIONAL predicate, independent from any publication decision.
    It is NOT a test of Elasticsearch DLS or of the upstream application's actual code.
    """
    if acl['state']!='resolved' or acl['menu_active_raw']!=1:return False
    if acl['is_public']:return True
    enabled={r['Id'] for r in acl['provenance']['Role'] if r['Status']==1}
    return bool(set(role_ids)&set(acl['allowed_role_ids'])&enabled) and acl['required_business_category'] in business_categories

def semantic_acl(acl:dict[str,Any]) -> dict[str,Any]:
    return {k:acl[k] for k in ('policy_id','state','is_public','menu_id','menu_active_raw','required_business_category','allowed_role_ids')} | {
        'role_status_raw':[{ 'id':r['Id'],'status':r['Status']} for r in acl['provenance']['Role']]
    }

def canonical_hash(obj:Any) -> str:
    return digest(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode('utf-8'))

def export_bundle(sql:Path,out:Path,instance_key:str='playbook-test') -> dict[str,Any]:
    if not re.fullmatch(r'[A-Za-z0-9_.-]+',instance_key):raise ValueError('Unsafe instance key')
    if (out/'manifest.json').exists() or (out/'documents').exists():
        raise ValueError('Output already contains a dataset; use a NEW output directory (no implicit overwrite).')
    raw=sql.read_bytes();text,encoding=decode_sql(raw);tables=parse_dump(text)
    if any(not tables[t] for t in ('Document','MenuItem','MenuItem_Document','Role','Role_MenuItem')):
        raise ValueError('A required table is empty or missing')
    audit=validate_tables(tables)
    out.mkdir(parents=True,exist_ok=True)
    source_dir=out/'source';source_dir.mkdir(exist_ok=True)
    target=source_dir/'playbook-sampledata.sql'
    if target.resolve()!=sql.resolve():target.write_bytes(raw)
    snapshot=digest(raw)
    script_date=re.search(r'Script Date:\s*([^*]+?)\s*\*',text)
    source={'filename':sql.name,'instance_key':instance_key,'sha256':snapshot,'snapshot_id':'sha256:'+snapshot,
            'encoding':encoding,'script_date_raw':script_date.group(1).strip() if script_date else None,
            'timezone':None,'sql_executed':False}
    dump_json(source_dir/'source_metadata.json',source)
    for table in ('MenuItem','MenuItem_Document','Role','Role_MenuItem','Notice'):
        dump_json(out/'upstream'/f'{table}.json',tables[table])
    rows=[];candidates=[];entries=[];acls=[]
    for doc in sorted(tables['Document'],key=lambda x:x['Id']):
        key=f"{instance_key}:Document:{doc['Id']}";folder=Path('documents')/f"{doc['Id']:06d}"
        dest=out/folder;dest.mkdir(parents=True)
        acl=derive_acl(doc,tables);acl['fingerprint']=canonical_hash(semantic_acl(acl));acls.append(acl)
        fields={};flags=[]
        for field,short,hint in [('Content','zh','zh-CN'),('ContentEn','en','en')]:
            body=doc[field]
            if body is None:
                fields[field]={'file':None,'source_is_null':True,'language_hint':hint,'language_verified':False};continue
            if not isinstance(body,str):raise ValueError(f'Non-string content at {doc["Id"]}.{field}')
            rel=folder/f'{short}.md';body_bytes=body.encode('utf-8');(out/rel).write_bytes(body_bytes)
            tags=sorted(set(x.lower() for x in re.findall(r'<\s*(table|div|svg|p|h[1-6]|img|a|span|script|style)\b',body,re.I)))
            fields[field]={'file':rel.as_posix(),'source_is_null':False,'source_field':field,
                'language_hint':hint,'language_verified':False,'utf8_sha256':digest(body_bytes),
                'characters':len(body),'bytes':len(body_bytes),'line_endings_preserved':True,
                'contains_html_heuristic':bool(tags),'html_tags_observed':tags,
                'merged_table_cells_observed':bool(re.search(r'\b(?:rowspan|colspan)\s*=',body,re.I)),
                'local_asset_reference_observed':bool(re.search(r'(?:images[\\/]|\$learnfiles[\\/])',body,re.I))}
        if doc['Content']==doc['ContentEn']:flags.append('source_language_fields_identical')
        if doc['TextFormatType'] not in (1,2):flags.append('unseen_text_format_value')
        if doc['TextFormatType']==2:flags.append('text_format_type_2_requires_semantic_confirmation')
        if any(x.get('contains_html_heuristic') for x in fields.values()):flags.append('html_preserved_not_converted')
        if any(x.get('merged_table_cells_observed') for x in fields.values()):flags.append('merged_html_table_preserved')
        if any(x.get('local_asset_reference_observed') for x in fields.values()):flags.append('assets_not_fetched')
        if re.search('更新中|Coming soon',doc['Content'],re.I):flags.append('possible_placeholder_text')
        metadata={
            'schema_version':'1.0','document_key':key,'document_id':doc['Id'],'kb_id':'playbook',
            'source':{'instance_key':instance_key,'table':'dbo.Document','primary_key':doc['Id'],
                      'file':'source/playbook-sampledata.sql','sql_lines':doc['_sql_lines'],'snapshot_sha256':snapshot},
            'upstream_metadata':{k:v for k,v in strip_provenance(doc).items() if k not in ('Content','ContentEn')},
            'files':fields,'acl_file':(folder/'acl.json').as_posix(),
            'content_fingerprint':canonical_hash({'Content':doc['Content'],'ContentEn':doc['ContentEn']}),
            'publication_state':'unverified','timestamp_timezone':None,
            'extraction':{'method':'sql_literal_decode_and_utf8_export','body_rewritten':False,
                          'translated':False,'line_endings_preserved':True},
            'quality_flags':flags}
        dump_json(dest/'metadata.json',metadata);dump_json(dest/'acl.json',acl)
        entries.append({'document_key':key,'document_id':doc['Id'],'metadata_file':(folder/'metadata.json').as_posix(),
                        'acl_file':(folder/'acl.json').as_posix(),'content_files':[v['file'] for v in fields.values() if v['file']]})
        rows.append(metadata)
        # Not a ready-to-publish Bulk payload; content is an unchanged original, not cleaned search text.
        candidate={'document_key':key,'document_id':doc['Id'],'kb_id':'playbook','title':doc['Title'],'title_en':doc['TitleEn'],
          'slug':doc['Slug'],'business_category':doc['BusinessCategory'],'language_mode':doc['LanguageMode'],
          'source_metadata':metadata['upstream_metadata'],'source':metadata['source'],
          'body_markdown_zh':doc['Content'],'body_markdown_en':doc['ContentEn'],
          'publication_state':'unverified','acl':semantic_acl(acl)|{'fingerprint':acl['fingerprint']},
          'content_fingerprint':metadata['content_fingerprint'],'quality_flags':flags}
        candidates.append(candidate)
    for name,objects in [('metadata.jsonl',rows),('index_candidates.jsonl',candidates)]:
        (out/name).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in objects),encoding='utf-8')
    audit.update({
        'policy':POLICY,'policy_confirmed_in_upstream_application':False,
        'document_acl_state_counts':dict(collections.Counter(a['state'] for a in acls)),
        'public_documents':sum(a['state']=='resolved' and a['is_public'] for a in acls),
        'controlled_documents':sum(a['state']=='resolved' and not a['is_public'] for a in acls),
        'document_access_distribution':[{'business_category':cat,'business_name':BUSINESS[cat],
             'public':sum(c['business_category']==cat and c['acl']['is_public'] for c in candidates),
             'controlled':sum(c['business_category']==cat and not c['acl']['is_public'] for c in candidates)} for cat in BUSINESS],
        'markdown_files':sum(len(e['content_files']) for e in entries),
        'content_fields_identical_document_ids':[d['Id'] for d in tables['Document'] if d['Content']==d['ContentEn']],
        'text_format_counts':dict(collections.Counter(d['TextFormatType'] for d in tables['Document'])),
        'text_format_type_2_ids':[d['Id'] for d in tables['Document'] if d['TextFormatType']==2],
        'html_containing_source_fields_heuristic':sum(f.get('contains_html_heuristic',False) for r in rows for f in r['files'].values()),
        'null_source_fields':sum(f.get('source_is_null',False) for r in rows for f in r['files'].values()),
        'public_docs_with_source_business_nce':[c['document_id'] for c in candidates if c['acl']['is_public'] and c['business_category']==0],
        'controlled_menu_role_id_sets':sorted({tuple(sorted({g['RoleId'] for g in tables['Role_MenuItem'] if g['MenuItemId']==m['Id']})) for m in tables['MenuItem'] if m['UnderControl']==1}),
        'native_es_tests_executed':False,'database_restore_executed':False,
        'publication_status_mapped':False,'downloaded_external_assets':False})
    offline_identities=[('partner_nce',[2],[0]),('partner_ospa',[2],[1]),('partner_both',[2],[0,1]),
                         ('role_only',[2],[]),('business_only',[],[1]),('public_only',[],[]),
                         ('msft_ospa',[1],[1]),('21v_ospa',[3],[1])]
    matrix=[]
    docs_sorted=sorted(tables['Document'],key=lambda x:x['Id'])
    for name,r,b in offline_identities:
        visible=[d['Id'] for d,a in zip(docs_sorted,acls) if acl_allows(a,r,b)]
        matrix.append({'test_identity':name,'synthetic_identity':True,'role_ids':r,'business_categories':b,
                      'acl_only_visible_count':len(visible),'document_ids':visible,'publication_gate_applied':False,
                      'executed_against_es':False})
    dump_json(out/'reports'/'audit.json',audit);dump_json(out/'reports'/'acl_test_matrix.json',matrix)
    manifest={'source':source,'entries':entries,'counts':audit['table_counts'],'markdown_files':audit['markdown_files'],
              'extraction_policy':POLICY,'publication_state':'unverified',
              'warning':'This package exports source data and provisional ACLs; it does not grant filesystem permissions or validate ES DLS.'}
    dump_json(out/'manifest.json',manifest)
    checks=[]
    for doc,entry in zip(docs_sorted,entries):
        folder=out/Path(entry['metadata_file']).parent
        for f,lang in [('Content','zh'),('ContentEn','en')]:
            if doc[f] is not None:
                actual=(folder/f'{lang}.md').read_bytes().decode('utf-8')
                if actual!=doc[f]:raise ValueError(f'Exported body differs: {doc["Id"]}.{f}')
                checks.append({'document_id':doc['Id'],'source_field':f,'exact_unicode_and_line_endings_match':True})
    dump_json(out/'reports'/'body_roundtrip_verification.json',{'checked':len(checks),'all_equal':True,'checks':checks})
    return audit

def main()->None:
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('sql',type=Path)
    p.add_argument('--out',type=Path,required=True,help='Fresh output directory; existing datasets are never overwritten')
    p.add_argument('--instance-key',default='playbook-test',help='Locally configured source-instance namespace; not an upstream database name')
    args=p.parse_args();audit=export_bundle(args.sql,args.out,args.instance_key)
    print(json.dumps({k:audit[k] for k in ('table_counts','markdown_files','public_documents','controlled_documents','document_acl_state_counts')},ensure_ascii=False,indent=2))

if __name__=='__main__':
    try:main()
    except (ValueError,OSError,KeyError) as exc:
        print(f'ERROR: {exc}',file=sys.stderr);sys.exit(1)
