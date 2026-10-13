"""Recover comparison operands without certifying the recorded outcome."""
from copy import deepcopy
import json
import pytest
from test_storage_evidence import evidence
from test_storage_legacy import put,run
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_legacy import legacy_dependencies,LegacyBindingError
from infinity_grid.storage_closure import ClosureError


def comparison(tmp_path,outcome='REPRODUCED',qualifications=None,shared=False,mode='CANONICAL_JSON',certificate=None):
    root,req,first,raw,b=evidence(tmp_path);sides=[b['record_ref']]
    if not shared:
        second=deepcopy(first);second['record_id']='IGRD/L0/EVIDENCE/REPLAY';second['epistemic_status']='REPLAYED';second['payload']['evidence_mode']='VERIFIED_RESTORED_BLOCK';second=seal_reference_record(second)
        ref=put(root,canonical_bytes(second));bb=deepcopy(b);bb['record_ref']={'record_id':second['record_id'],'record_sha256':second['record_sha256'],'content_ref':ref};br=put(root,canonical_bytes(bb))
        req['interpretations'].append({'content_ref':ref,'format':'NATIVE_REFERENCE_RECORD_V1'});req['required_content'] += [ref,br];req['legacy_bindings'].append({'content_ref':ref,'binding_ref':br});sides.append(bb['record_ref'])
    else:sides.append(sides[0])
    r=deepcopy(first);r.update(record_type='COMPARISON',record_id='IGRD/L0/COMPARISON/C',dependencies=list(dict.fromkeys(x['record_id'] for x in sides)))
    eq=deepcopy(first['payload']['equality_contract']);eq.update(mode=mode,certificate_sha256=certificate)
    r['payload']={'obligation_id':'FIXTURE','historical_record_ids':[sides[0]['record_id']],'replay_record_ids':[sides[1]['record_id']],'equality_contract':eq,'outcome':outcome,'qualification_ids':qualifications or []};r=seal_reference_record(r)
    raw=json.dumps(r,indent=2).encode()+b'\n';ref=put(root,raw);binding={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1','record_ref':{'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':ref},'record_links':sides[:1] if shared else sides,'digest_bindings':b['digest_bindings'],'symbol_bindings':[]};br=put(root,canonical_bytes(binding))
    req['roots'][-1]=ref;req['interpretations'].append({'content_ref':ref,'format':'NATIVE_REFERENCE_RECORD_V1'});req['required_content'] += [ref,br];req['legacy_bindings'].append({'content_ref':ref,'binding_ref':br})
    return root,req,r,raw,binding


def change(root,req,b):
    old=req['legacy_bindings'][-1]['binding_ref'];new=put(root,canonical_bytes(b));req['legacy_bindings'][-1]['binding_ref']=new;req['required_content']=[new if x==old else x for x in req['required_content']]


def test_comparison_both_evidence_sides_recovered(tmp_path):
    root,req,r,raw,b=comparison(tmp_path);out=run(root,req)
    assert len(out['legacy_records_verified'])==6 and out['status']=='STRUCTURAL_PASS'
    assert (root/(b['record_ref']['content_ref']['sha256']+'.blob')).read_bytes()==raw


def test_comparison_all_outcomes_preserved_without_acceptance(tmp_path):
    for outcome in ['REPRODUCED','QUALIFIED_REPRODUCTION','MISMATCH','EQUIVALENCE_NOT_CERTIFIED']:
        p=tmp_path/outcome;p.mkdir();root,req,r,raw,b=comparison(p,outcome=outcome)
        assert run(root,req)['scientific_acceptance']=='NOT_GRANTED'
        assert json.loads(raw)['payload']['outcome']==outcome


def test_comparison_missing_historical_side_refused(tmp_path):
    root,req,r,raw,b=comparison(tmp_path);b['record_links'].pop(0);change(root,req,b)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_comparison_missing_replay_side_refused(tmp_path):
    root,req,r,raw,b=comparison(tmp_path);b['record_links'].pop();change(root,req,b)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_comparison_wrong_operand_seal_refused(tmp_path):
    root,req,r,raw,b=comparison(tmp_path);b['record_links'][1]=dict(b['record_links'][1],record_sha256='0'*64);change(root,req,b)
    with pytest.raises(ClosureError,match='LEGACY_REFERENCE_BINDING'):run(root,req)


def test_comparison_qualification_catalogue_not_guessed(tmp_path):
    root,req,r,raw,b=comparison(tmp_path,qualifications=['QUAL/FIXTURE'])
    with pytest.raises(LegacyBindingError,match='QUALIFICATION_CATALOGUE_REQUIRED'):run(root,req)


def test_comparison_certificate_profile_not_guessed(tmp_path):
    root,req,r,raw,b=comparison(tmp_path,mode='CERTIFIED_SEMANTIC_EQUIVALENCE',certificate='0'*64)
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_LEGACY_COMPARISON_PROFILE'):run(root,req)


def test_comparison_shared_operand_keeps_original_roles(tmp_path):
    root,req,r,raw,b=comparison(tmp_path,shared=True);out=run(root,req)
    assert len(out['legacy_records_verified'])==5 and len(b['record_links'])==1
    assert json.loads(raw)['payload']['historical_record_ids']==json.loads(raw)['payload']['replay_record_ids']


def test_comparison_outcome_tampering_breaks_seal(tmp_path):
    root,req,r,raw,b=comparison(tmp_path);r['payload']['outcome']='MISMATCH'
    with pytest.raises(Exception,match='hash mismatch'):legacy_dependencies(canonical_bytes(r),canonical_bytes(b))


def test_comparison_v2_empty_semantics_and_missing_inventory(tmp_path):
    root,req,r,raw,b=comparison(tmp_path);b['schema_id']='IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V2';b['semantic_bindings']=[];change(root,req,b)
    assert run(root,req)['status']=='STRUCTURAL_PASS'
    req['required_content'].remove(req['legacy_bindings'][-1]['binding_ref'])
    with pytest.raises(ClosureError,match='INVENTORY_MISMATCH'):run(root,req)
