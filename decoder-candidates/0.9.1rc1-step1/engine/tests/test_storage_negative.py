"""Bounded witness storage profiles do not certify a negative assertion."""
from copy import deepcopy
import json
import pytest
from test_storage_legacy import fixture,put,run
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_legacy import legacy_dependencies,LegacyBindingError
from infinity_grid.storage_closure import ClosureError


def negative(tmp_path,witnesses=None):
    root,req,records,raws,bindings=fixture(tmp_path);source=bindings[0]['digest_bindings'][0]['content_ref']
    if witnesses is None:witnesses=[{'source':'original/source','anchors':['fixture anchor']},{'source':'original/source','field':'status','value':'NOT_EARNED'}]
    r=deepcopy(records[0]);r.update(record_type='NEGATIVE_RESULT',record_id='IGRD/L0/NEGATIVE/N')
    r['payload']={'obligation_id':'FIXTURE','tested_scope':{'bounded':True},'negative_statement':'Fixture only','witnesses':witnesses,'does_not_establish':['UNBOUNDED_ABSENCE']};r=seal_reference_record(r)
    raw=json.dumps(r,indent=2).encode()+b'\n';ref=put(root,raw)
    b={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1','record_ref':{'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':ref},'record_links':[],
       'digest_bindings':[bindings[0]['digest_bindings'][0]],'symbol_bindings':[{'path':f'/payload/witnesses/{i}/source','value':w['source'],'content_ref':source} for i,w in enumerate(witnesses) if isinstance(w,dict) and isinstance(w.get('source'),str)]}
    br=put(root,canonical_bytes(b));req['roots'].append(ref);req['required_content'] += [ref,br];req['legacy_bindings'].append({'content_ref':ref,'binding_ref':br});req['interpretations'].append({'content_ref':ref,'format':'NATIVE_REFERENCE_RECORD_V1'})
    return root,req,r,raw,b


def change(root,req,b):
    old=req['legacy_bindings'][-1]['binding_ref'];new=put(root,canonical_bytes(b));req['legacy_bindings'][-1]['binding_ref']=new;req['required_content']=[new if x==old else x for x in req['required_content']]


def test_negative_source_witnesses_recover_original_bytes(tmp_path):
    root,req,r,raw,b=negative(tmp_path);out=run(root,req)
    assert out['status']=='STRUCTURAL_PASS' and out['scientific_acceptance']=='NOT_GRANTED'
    assert (root/(b['record_ref']['content_ref']['sha256']+'.blob')).read_bytes()==raw


def test_negative_empty_witness_list_preserved(tmp_path):
    root,req,r,raw,b=negative(tmp_path,[]);assert run(root,req)['status']=='STRUCTURAL_PASS';assert json.loads(raw)['payload']['witnesses']==[]


def test_negative_inline_text_witnesses_preserved(tmp_path):
    root,req,r,raw,b=negative(tmp_path,['bounded anchor','another anchor']);assert run(root,req)['status']=='STRUCTURAL_PASS';assert b['symbol_bindings']==[]


def test_negative_unpinned_witness_source_refused(tmp_path):
    root,req,r,raw,b=negative(tmp_path,[{'source':'unknown','anchors':['a']}])
    with pytest.raises(LegacyBindingError,match='SOURCE_NOT_PINNED'):run(root,req)


def test_negative_missing_witness_source_binding_refused(tmp_path):
    root,req,r,raw,b=negative(tmp_path);b['symbol_bindings'].pop();change(root,req,b)
    with pytest.raises(LegacyBindingError,match='MISSING_SYMBOL'):run(root,req)


def test_negative_wrong_witness_source_label_refused(tmp_path):
    root,req,r,raw,b=negative(tmp_path);b['symbol_bindings'][0]['value']='other';change(root,req,b)
    with pytest.raises(LegacyBindingError,match='SYMBOL_BINDING_MISMATCH'):run(root,req)


def test_negative_wrong_pinned_source_bytes_refused(tmp_path):
    root,req,r,raw,b=negative(tmp_path);b['symbol_bindings'][0]['content_ref']=put(root,b'wrong');change(root,req,b)
    with pytest.raises(LegacyBindingError,match='SOURCE_DIGEST_MISMATCH'):run(root,req)


def test_negative_unknown_witness_shape_refused(tmp_path):
    root,req,r,raw,b=negative(tmp_path,[{'object_id':'UNINTERPRETED'}])
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_NEGATIVE_WITNESS'):run(root,req)


def test_negative_empty_or_nonstrings_anchors_refused(tmp_path):
    for i,anchors in enumerate([[],[1],['']]):
        p=tmp_path/str(i);p.mkdir();root,req,r,raw,b=negative(p,[{'source':'original/source','anchors':anchors}])
        with pytest.raises(LegacyBindingError,match='UNSUPPORTED_NEGATIVE_WITNESS'):run(root,req)


def test_negative_extra_witness_fields_refused(tmp_path):
    root,req,r,raw,b=negative(tmp_path,[{'source':'original/source','anchors':['a'],'hidden_ref':'UNINTERPRETED'}])
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_NEGATIVE_WITNESS'):run(root,req)


def test_negative_scope_nonclaims_and_statement_seal(tmp_path):
    root,req,r,raw,b=negative(tmp_path)
    for field,value in [('tested_scope',{'unbounded':True}),('negative_statement','UNBOUNDED CLAIM'),('does_not_establish',['DIFFERENT'])]:
        altered=deepcopy(r);altered['payload'][field]=value
        with pytest.raises(Exception,match='hash mismatch'):legacy_dependencies(canonical_bytes(altered),canonical_bytes(b))


def test_negative_duplicate_source_binding_refused(tmp_path):
    root,req,r,raw,b=negative(tmp_path);b['symbol_bindings'].append(b['symbol_bindings'][0]);change(root,req,b)
    with pytest.raises(LegacyBindingError,match='SYMBOL_BINDING_MISMATCH'):run(root,req)


def test_negative_bindings_and_source_in_inventory(tmp_path):
    root,req,r,raw,b=negative(tmp_path);req['required_content'].remove(req['legacy_bindings'][-1]['binding_ref'])
    with pytest.raises(ClosureError,match='INVENTORY_MISMATCH'):run(root,req)


def test_negative_recovery_does_not_claim_witness_truth(tmp_path):
    root,req,r,raw,b=negative(tmp_path);out=legacy_dependencies(raw,canonical_bytes(b))
    assert len(out['stored_content'])==3 and out['scientific_acceptance']=='NOT_GRANTED'
    assert out['epistemic_status']=='HISTORICAL' and out['science_execution']=='NONE'
    assert out['authority_effect']=='NONE';assert json.loads(raw)['payload']['does_not_establish']==['UNBOUNDED_ABSENCE']
