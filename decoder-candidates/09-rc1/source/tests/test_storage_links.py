"""Dependency-link recovery is distinct from relation/authority interpretation."""
from copy import deepcopy
import json
import pytest
from test_storage_legacy import fixture,put,run
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_legacy import legacy_dependencies,LegacyBindingError
from infinity_grid.storage_closure import ClosureError


def linked(tmp_path,relation='REQUIRES',required=True):
    root,req,records,raws,bindings=fixture(tmp_path)
    r=deepcopy(records[0]);r['record_id']='IGRD/L0/LINK/L';r['record_type']='DEPENDENCY_LINK'
    r['payload']={'from_record_id':records[0]['record_id'],'to_record_id':records[1]['record_id'],'relation':relation,'required':required}
    r['dependencies']=[records[0]['record_id'],records[1]['record_id']];r=seal_reference_record(r)
    raw=json.dumps(r,indent=2).encode()+b'\n';ref=put(root,raw)
    binding={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1',
        'record_ref':{'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':ref},
        'record_links':[bindings[i]['record_ref'] for i in [0,1]],
        'digest_bindings':[bindings[0]['digest_bindings'][0]],'symbol_bindings':[]}
    bref=put(root,canonical_bytes(binding));req['roots'].append(ref)
    req['interpretations'].append({'content_ref':ref,'format':'NATIVE_REFERENCE_RECORD_V1'})
    req['required_content'] += [ref,bref];req['legacy_bindings'].append({'content_ref':ref,'binding_ref':bref})
    return root,req,r,raw,binding


def change(root,req,binding):
    old=req['legacy_bindings'][-1]['binding_ref'];new=put(root,canonical_bytes(binding))
    req['legacy_bindings'][-1]['binding_ref']=new
    req['required_content']=[new if r==old else r for r in req['required_content']]


def test_link_all_relations_and_required_flags(tmp_path):
    for relation in ['REQUIRES','PRODUCES','AUTHORIZES','COMPARES','QUALIFIES']:
        for required in [True,False]:
            p=tmp_path/(relation+str(required));p.mkdir()
            root,req,r,raw,b=linked(p,relation,required);out=run(root,req)
            assert out['status']=='STRUCTURAL_PASS' and out['scientific_acceptance']=='NOT_GRANTED'
            assert len(out['legacy_records_verified'])==4
            assert (root/(b['record_ref']['content_ref']['sha256']+'.blob')).read_bytes()==raw


def test_optional_link_still_requires_both_endpoints(tmp_path):
    root,req,r,raw,b=linked(tmp_path,required=False);b['record_links'].pop();change(root,req,b)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_link_missing_source_endpoint_refused(tmp_path):
    root,req,r,raw,b=linked(tmp_path);b['record_links'].pop(0);change(root,req,b)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_link_wrong_endpoint_seal_refused(tmp_path):
    root,req,r,raw,b=linked(tmp_path);b['record_links'][1]=dict(b['record_links'][1],record_sha256='0'*64);change(root,req,b)
    with pytest.raises(ClosureError,match='LEGACY_REFERENCE_BINDING'):run(root,req)


def test_link_extra_endpoint_refused(tmp_path):
    root,req,r,raw,b=linked(tmp_path);b['record_links'].append(b['record_ref']);change(root,req,b)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_authorizes_relation_grants_no_authority(tmp_path):
    root,req,r,raw,b=linked(tmp_path,'AUTHORIZES');out=legacy_dependencies(raw,canonical_bytes(b))
    assert out['authority_effect']=='NONE' and out['scientific_acceptance']=='NOT_GRANTED'
    assert out['science_execution']=='NONE' and out['epistemic_status']=='HISTORICAL'
    assert {x['record_id'] for x in out['record_links']}==set(r['dependencies'])


def test_link_provenance_content_required(tmp_path):
    root,req,r,raw,b=linked(tmp_path);b['digest_bindings']=[];change(root,req,b)
    with pytest.raises(LegacyBindingError,match='MISSING_DIGEST'):run(root,req)


def test_link_relation_mutation_invalidates_original_seal(tmp_path):
    root,req,r,raw,b=linked(tmp_path);r['payload']['relation']='AUTHORIZES'
    with pytest.raises(Exception,match='hash mismatch'):legacy_dependencies(canonical_bytes(r),canonical_bytes(b))
