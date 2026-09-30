"""Bounded declared closure fixtures, no release authorization."""
from copy import deepcopy
from pathlib import Path
import hashlib
import pytest
from infinity_grid.storage_schema import canonical_bytes,strict_loads
from infinity_grid.storage_collections import ContentDirectory
from infinity_grid.storage_closure import verify_declared_closure,ClosureError,ClosureLimits

FIX=Path(__file__).parent/'fixtures/storage_contract_v1/positive'


def obj(name):return strict_loads((FIX/name).read_bytes())
def put(root,raw):
    r={'sha256':hashlib.sha256(raw).hexdigest(),'size_bytes':str(len(raw))};(root/(r['sha256']+'.blob')).write_bytes(raw);return r


def seed(tmp_path):
    root=tmp_path/'content';root.mkdir();data=put(root,b'engineering-data');dictionary=put(root,b'engineering-dictionary')
    codec=put(root,canonical_bytes({'schema_id':'IG_STORAGE_EXTERNAL_DEPENDENCIES_V1','dependencies':[dictionary]}))
    block=obj('Block_0.json');block.update(stored_content=data,codec_ref=codec);bref=put(root,canonical_bytes(block))
    spec={'schema_id':'IG_STORAGE_DECLARED_CLOSURE_REQUEST_V1','purpose':'SCHEMA_FIXTURE','roots':[bref],
          'interpretations':[{'content_ref':r,'format':f} for r,f in [(bref,'STORAGE_RECORD_V1'),(codec,'JSON_DEPENDENCIES_V1'),(data,'OPAQUE_LEAF_V1'),(dictionary,'OPAQUE_LEAF_V1')]],
          'required_content':[bref,codec,data,dictionary]}
    return root,spec,block


def run(root,spec,**kw):return verify_declared_closure(ContentDirectory(root),put(root,canonical_bytes(spec)),**kw)


def test_recursive_codec_dictionary_inventory_and_decoded_separation(tmp_path):
    root,spec,block=seed(tmp_path);r=run(root,spec)
    assert r['status']=='STRUCTURAL_PASS' and r['scientific_acceptance']=='NOT_GRANTED'
    assert r['verification_scope']=='DECLARED_REQUIRED_CONTENT_CLOSURE'
    assert len(r['required_content'])==4 and len(r['decoded_checks_pending'])==1
    assert len(r['opaque_leaf_declarations'])==2
    assert block['decoded_content'] not in r['required_content']


def test_missing_transitive_inventory_entry_refused(tmp_path):
    root,spec,_=seed(tmp_path);spec['required_content'].pop()
    with pytest.raises(ClosureError,match='INVENTORY_MISMATCH'):run(root,spec)


def test_extra_unreachable_inventory_entry_refused(tmp_path):
    root,spec,_=seed(tmp_path);spec['required_content'].append(put(root,b'extra'))
    with pytest.raises(ClosureError,match='INVENTORY_MISMATCH'):run(root,spec)


def test_unregistered_interpretation_is_not_opaque_by_default(tmp_path):
    root,spec,_=seed(tmp_path);spec['interpretations'].pop()
    with pytest.raises(ClosureError,match='INTERPRETATION_REQUIRED'):run(root,spec)


def test_wrong_external_manifest_format_refused(tmp_path):
    root,spec,_=seed(tmp_path);spec['interpretations'][2]['format']='JSON_DEPENDENCIES_V1'
    with pytest.raises(ValueError):run(root,spec)


def test_corrupt_transitive_dictionary_refused(tmp_path):
    root,spec,_=seed(tmp_path);ref=spec['required_content'][-1]
    (root/(ref['sha256']+'.blob')).write_bytes(b'x'*int(ref['size_bytes']))
    with pytest.raises(ValueError,match='DIGEST_MISMATCH'):run(root,spec)


def test_depth_budget_refuses_recursive_dependency(tmp_path):
    root,spec,_=seed(tmp_path)
    with pytest.raises(ClosureError,match='DEPTH_BUDGET'):run(root,spec,limits=ClosureLimits(max_depth=1))


def test_total_byte_preflight_and_untrusted_provider(tmp_path):
    root,spec,_=seed(tmp_path);req=put(root,canonical_bytes(spec))
    class Bad:
        def read(self,*args):raise AssertionError('SHOULD_NOT_READ')
    with pytest.raises(ClosureError,match='BYTE_BUDGET'):verify_declared_closure(Bad(),req,limits=ClosureLimits(max_total_bytes=1))
    class Wrong:
        def read(self,*args):return b'wrong'
    with pytest.raises(ClosureError,match='PROVIDER_CONTENT_MISMATCH'):verify_declared_closure(Wrong(),req)


def test_shared_dependency_bytes_keep_each_role_edge(tmp_path):
    root,spec,block=seed(tmp_path)
    spec['roots'].append(spec['roots'][0]);r=run(root,spec)
    assert len(r['required_content'])==4 and len([e for e in r['dependency_edges'] if e['role']=='ROOT'])==2
    assert r['records_checked']==1


def test_duplicate_inventory_and_conflicting_lengths_refused(tmp_path):
    root,spec,_=seed(tmp_path);spec['required_content'].append(spec['required_content'][0])
    with pytest.raises(ClosureError,match='DUPLICATE_INVENTORY'):run(root,spec)
    spec['required_content'][-1]=dict(spec['required_content'][0],size_bytes='1')
    with pytest.raises(ClosureError,match='LENGTH_CONFLICT'):run(root,spec)


def native_seed(tmp_path,cycle=False):
    root,spec,block=seed(tmp_path);leaf=block['stored_content']
    profile=obj('IdentityProfile.json');profile.update(source_archive_ref=leaf,interpretation_ref=leaf)
    pref=put(root,canonical_bytes(profile))
    def nr(n):return {'native_id':n,'scope_ref':leaf,'profile_ref':pref}
    o=obj('Object_S.json');o.update(object_ref=nr('S'),canonicalization_ref=leaf,carrier_schema_ref=leaf,equality_contract_ref=leaf,
        reference_record_ref=None,structure_ref={'state':'NOT_APPLICABLE','content_ref':None,'explanation':'fixture'})
    payload=obj('Payload_Whole.json');payload.update(canonicalization_ref=leaf,whole_storage_ref=leaf,whole_codec_ref=block['codec_ref'])
    payloadref=put(root,canonical_bytes(payload));o['canonical_payload_ref']=payloadref
    rows=[]
    for n,other in [('left','right'),('right','left')]:
        row=obj('Occurrence_left.json');row.update(occurrence_ref=nr(n),object_ref=nr('S'),context_ref=leaf,formation_ref=leaf,
            parent_occurrence_ref=nr(other) if cycle else None,position_ref={'state':'PRESENT','content_ref':leaf,'explanation':'fixture'})
        rows.append(row)
    kp={'schema_id':'IG_STORAGE_COLLECTION_KEY_PROFILE_V1','format':'CANONICAL_STORAGE_JSONL_V1','key_encoding':'CANONICAL_STRING_ARRAY_UTF8_HEX','order':'ASCII_BYTEWISE','bounds':'INCLUSIVE_DISJOINT','key_fields':[['occurrence_ref','native_id']]}
    kref=put(root,canonical_bytes(kp));shard=put(root,b''.join(canonical_bytes(r)+b'\n' for r in rows))
    keys=[canonical_bytes([r['occurrence_ref']['native_id']]).hex() for r in rows]
    page={'schema_id':'IG_STORAGE_COLLECTIONPAGE_V1','contract_version':'1.0.0','purpose':'SCHEMA_FIXTURE','extensions':[],
        'collection_kind':'occurrences','record_schema_id':'IG_STORAGE_OCCURRENCE_V1','key_definition_ref':kref,'depth':0,
        'first_key':keys[0],'last_key':keys[-1],'record_count':'2','entries':[{'entry_type':'SHARD','content_ref':shard,'first_key':keys[0],'last_key':keys[-1],'record_count':'2'}]}
    pageref=put(root,canonical_bytes(page));o['evidence']={'root':pageref,'record_schema_id':'IG_STORAGE_OCCURRENCE_V1','row_count':'2','key_definition_ref':kref,'cardinality_semantics':'OCCURRENCE_LOG'}
    oref=put(root,canonical_bytes(o));spec['roots']=[oref]
    spec['interpretations']=[r for r in spec['interpretations'] if r['content_ref']!=spec['required_content'][0]]
    spec['interpretations'] += [{'content_ref':r,'format':'STORAGE_RECORD_V1'} for r in [oref,pref,payloadref]]
    spec['required_content']=spec['required_content'][1:]+[oref,pref,payloadref,kref,shard]
    return root,spec,o,rows,page


def test_collection_closure_and_scoped_native_resolution(tmp_path):
    root,spec,_,_,page=native_seed(tmp_path);r=run(root,spec)
    assert r['native_bindings']==3 and r['resolved_native_references']==2
    assert len(r['structural_content'])==1 and r['structural_content'][0] not in r['required_content']
    assert r['records_checked']==5


def test_native_cycles_do_not_become_storage_hash_recursion(tmp_path):
    root,spec,_,_,_=native_seed(tmp_path,cycle=True);r=run(root,spec)
    assert r['native_bindings']==3 and r['resolved_native_references']==4
    assert r['scientific_acceptance']=='NOT_GRANTED'


def test_known_payload_cannot_be_declared_opaque(tmp_path):
    root,spec,o,_,_=native_seed(tmp_path)
    next(x for x in spec['interpretations'] if x['content_ref']==o['canonical_payload_ref'])['format']='OPAQUE_LEAF_V1'
    with pytest.raises(ClosureError,match='ROLE_CONFLICT'):run(root,spec)


def test_dangling_native_reference_refused(tmp_path):
    root,spec,o,rows,page=native_seed(tmp_path);old=spec['roots'][0];o['object_ref']['native_id']='OTHER';new=put(root,canonical_bytes(o))
    spec['roots']=[new];spec['required_content']=[new if r==old else r for r in spec['required_content']]
    next(x for x in spec['interpretations'] if x['content_ref']==old)['content_ref']=new
    with pytest.raises(ClosureError,match='DANGLING_NATIVE_REFERENCE'):run(root,spec)


def test_unknown_field_blocks_closure(tmp_path):
    root,spec,o,_,_=native_seed(tmp_path);old=spec['roots'][0];o['structure_ref']={'state':'UNKNOWN','content_ref':None,'explanation':'unresolved'}
    new=put(root,canonical_bytes(o));spec['roots']=[new]
    spec['required_content']=[new if r==old else r for r in spec['required_content']]
    next(x for x in spec['interpretations'] if x['content_ref']==old)['content_ref']=new
    with pytest.raises(ClosureError,match='UNRESOLVED_FIELD'):run(root,spec)


def test_purpose_mismatch_refused(tmp_path):
    root,spec,_=seed(tmp_path);spec['purpose']='SCIENCE'
    with pytest.raises(ClosureError,match='PURPOSE_MISMATCH'):run(root,spec)


def test_legacy_record_requires_dependency_adapter(tmp_path):
    root,spec,_=seed(tmp_path);spec['interpretations'][0]['format']='NATIVE_REFERENCE_RECORD_V1'
    with pytest.raises(ClosureError,match='CLOSURE_ADAPTER_REQUIRED'):run(root,spec)


def test_edge_and_object_budgets_refuse(tmp_path):
    root,spec,_=seed(tmp_path)
    with pytest.raises(ClosureError,match='OBJECT_BUDGET'):run(root,spec,limits=ClosureLimits(max_objects=1))
    with pytest.raises(ClosureError,match='EDGE_BUDGET'):run(root,spec,limits=ClosureLimits(max_edges=1))
