"""Recipe recovery binds dependencies without executing implementation bytes."""
from copy import deepcopy
import json
import pytest
from test_storage_legacy import fixture,put,run
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_legacy import legacy_dependencies,LegacyBindingError
from infinity_grid.storage_closure import ClosureError


def recipe(tmp_path,params=None,determinism='BYTE_IDENTICAL'):
    root,req,records,raws,bindings=fixture(tmp_path)
    impl=put(root,b'raise RuntimeError("THIS IMPLEMENTATION MUST NOT EXECUTE")\n')
    r=deepcopy(records[0]);r.update(record_type='GENERATION_RECIPE',record_id='IGRD/L0/RECIPE/R',dependencies=[x['record_id'] for x in records[:2]])
    r['payload']={'recipe_id':'FIXTURE','implementation_ref':'must_not_import.fixture:run','implementation_sha256':impl['sha256'],'input_record_ids':r['dependencies'],'parameters':params if params is not None else {'assertion_id':'FIXTURE','anchors':['anchor']},'expected_record_types':['SRCF_EVIDENCE','COMPARISON'],'determinism_contract':determinism};r=seal_reference_record(r)
    raw=json.dumps(r,indent=2).encode()+b'\n';ref=put(root,raw)
    b={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1','record_ref':{'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':ref},'record_links':[x['record_ref'] for x in bindings[:2]],'digest_bindings':[bindings[0]['digest_bindings'][0],{'path':'/payload/implementation_sha256','content_ref':impl}],'symbol_bindings':[]}
    br=put(root,canonical_bytes(b));req['roots'].append(ref);req['required_content'] += [ref,br,impl];req['legacy_bindings'].append({'content_ref':ref,'binding_ref':br});req['interpretations'] += [{'content_ref':ref,'format':'NATIVE_REFERENCE_RECORD_V1'},{'content_ref':impl,'format':'OPAQUE_LEAF_V1'}]
    return root,req,r,raw,b


def change(root,req,b):
    old=req['legacy_bindings'][-1]['binding_ref'];new=put(root,canonical_bytes(b));req['legacy_bindings'][-1]['binding_ref']=new;req['required_content']=[new if x==old else x for x in req['required_content']]


def test_recipe_all_inline_parameter_profiles(tmp_path):
    profiles=[{}, {'assertion_id':'A','anchors':['a']},{'assertion_id':'A','anchors':['a'],'execution_class':'SOURCE_INTEGRITY_ONLY'},{'assertion_ids':['A','B'],'execution_class':'HISTORICAL_RESULT_ONLY'}]
    for i,params in enumerate(profiles):
        p=tmp_path/str(i);p.mkdir();root,req,r,raw,b=recipe(p,params)
        assert run(root,req)['status']=='STRUCTURAL_PASS'
        assert (root/(b['record_ref']['content_ref']['sha256']+'.blob')).read_bytes()==raw


def test_recipe_does_not_import_or_execute_implementation(tmp_path):
    root,req,r,raw,b=recipe(tmp_path);out=run(root,req)
    assert out['scientific_acceptance']=='NOT_GRANTED' and len(out['legacy_records_verified'])==4
    out=legacy_dependencies(raw,canonical_bytes(b));assert out['science_execution']=='NONE' and out['authority_effect']=='NONE'


def test_recipe_missing_input_record_refused(tmp_path):
    root,req,r,raw,b=recipe(tmp_path);b['record_links'].pop();change(root,req,b)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_recipe_wrong_input_seal_refused(tmp_path):
    root,req,r,raw,b=recipe(tmp_path);b['record_links'][0]=dict(b['record_links'][0],record_sha256='0'*64);change(root,req,b)
    with pytest.raises(ClosureError,match='LEGACY_REFERENCE_BINDING'):run(root,req)


def test_recipe_missing_implementation_binding_refused(tmp_path):
    root,req,r,raw,b=recipe(tmp_path);b['digest_bindings'].pop();change(root,req,b)
    with pytest.raises(LegacyBindingError,match='MISSING_DIGEST'):run(root,req)


def test_recipe_wrong_implementation_bytes_refused(tmp_path):
    root,req,r,raw,b=recipe(tmp_path);b['digest_bindings'][1]['content_ref']=put(root,b'wrong');change(root,req,b)
    with pytest.raises(LegacyBindingError,match='DIGEST_BINDING_MISMATCH'):run(root,req)


def test_recipe_implementation_inventory_required(tmp_path):
    root,req,r,raw,b=recipe(tmp_path);req['required_content'].remove(b['digest_bindings'][1]['content_ref'])
    with pytest.raises(ClosureError,match='INVENTORY_MISMATCH'):run(root,req)


def test_recipe_unknown_parameter_keys_refused(tmp_path):
    root,req,r,raw,b=recipe(tmp_path,{'external_content_ref':'UNINTERPRETED'})
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_LEGACY_RECIPE_PARAMETERS'):run(root,req)


def test_recipe_nested_parameter_values_refused(tmp_path):
    root,req,r,raw,b=recipe(tmp_path,{'assertion_id':'A','anchors':[{'ref':'UNINTERPRETED'}]})
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_LEGACY_RECIPE_PARAMETERS'):run(root,req)


def test_recipe_wrong_parameter_scalar_type_refused(tmp_path):
    root,req,r,raw,b=recipe(tmp_path,{'assertion_ids':['A'],'execution_class':True})
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_LEGACY_RECIPE_PARAMETERS'):run(root,req)


def test_recipe_determinism_declarations_not_certified(tmp_path):
    for kind in ['BYTE_IDENTICAL','CANONICAL_SEMANTIC_IDENTITY']:
        p=tmp_path/kind;p.mkdir();root,req,r,raw,b=recipe(p,determinism=kind)
        assert run(root,req)['scientific_acceptance']=='NOT_GRANTED'
        assert json.loads(raw)['payload']['determinism_contract']==kind


def test_recipe_parameter_mutation_breaks_original_seal(tmp_path):
    root,req,r,raw,b=recipe(tmp_path);r['payload']['parameters']['assertion_id']='OTHER'
    with pytest.raises(Exception,match='hash mismatch'):legacy_dependencies(canonical_bytes(r),canonical_bytes(b))
