"""Audit recovery preserves evidence and historical decision, never grants authority."""
from copy import deepcopy
import json
import pytest
from test_storage_legacy import fixture,put,run
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_legacy import legacy_dependencies,LegacyBindingError
from infinity_grid.storage_closure import ClosureError


def audit(tmp_path,decision='CONTINUE',shared=False):
    root,req,records,raws,bindings=fixture(tmp_path);source=bindings[0]['digest_bindings'][0]['content_ref']
    ev=source if shared else put(root,b'original external audit evidence; engineering fixture')
    r=deepcopy(records[0]);r.update(record_type='AUDIT_AUTHORIZATION',record_id='IGRD/L0/AUDIT/A',epistemic_status='EXTERNAL_DECISION',authority_effect='EXTERNAL_DECISION_RECORDED')
    r['provenance']['classification']='EXTERNAL_AUDIT_DECISION'
    r['payload']={'authorization_id':'FIXTURE_AUTH','obligation_ids':['FIXTURE_OBLIGATION'],'decision':decision,'authorized_scope':{'bounded':True},'limitations':['NO_CURRENT_ACCEPTANCE'],'evidence_hashes':[{'ref':'not-a-local-path/audit','sha256':ev['sha256']}]};r=seal_reference_record(r)
    raw=json.dumps(r,indent=2).encode()+b'\n';ref=put(root,raw)
    b={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1','record_ref':{'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':ref},'record_links':[],
       'digest_bindings':[bindings[0]['digest_bindings'][0],{'path':'/payload/evidence_hashes/0/sha256','content_ref':ev}],'symbol_bindings':[]}
    br=put(root,canonical_bytes(b));req['roots'].append(ref);req['required_content'] += [ref,br];req['legacy_bindings'].append({'content_ref':ref,'binding_ref':br});req['interpretations'].append({'content_ref':ref,'format':'NATIVE_REFERENCE_RECORD_V1'})
    if not shared:req['required_content'].append(ev);req['interpretations'].append({'content_ref':ev,'format':'OPAQUE_LEAF_V1'})
    return root,req,r,raw,b


def change(root,req,b):
    old=req['legacy_bindings'][-1]['binding_ref'];new=put(root,canonical_bytes(b));req['legacy_bindings'][-1]['binding_ref']=new;req['required_content']=[new if x==old else x for x in req['required_content']]


def test_audit_both_decisions_recover_without_current_acceptance(tmp_path):
    for decision in ['CONTINUE','CERTIFY_AND_ADVANCE']:
        p=tmp_path/decision;p.mkdir();root,req,r,raw,b=audit(p,decision);out=run(root,req)
        assert out['status']=='STRUCTURAL_PASS' and out['scientific_acceptance']=='NOT_GRANTED'
        out=legacy_dependencies(raw,canonical_bytes(b));assert out['authority_effect']=='EXTERNAL_DECISION_RECORDED' and out['epistemic_status']=='EXTERNAL_DECISION'
        assert out['science_execution']=='NONE'
        assert (root/(b['record_ref']['content_ref']['sha256']+'.blob')).read_bytes()==raw


def test_audit_missing_evidence_binding_refused(tmp_path):
    root,req,r,raw,b=audit(tmp_path);b['digest_bindings'].pop();change(root,req,b)
    with pytest.raises(LegacyBindingError,match='MISSING_DIGEST'):run(root,req)


def test_audit_missing_provenance_binding_refused(tmp_path):
    root,req,r,raw,b=audit(tmp_path);b['digest_bindings'].pop(0);change(root,req,b)
    with pytest.raises(LegacyBindingError,match='MISSING_DIGEST'):run(root,req)


def test_audit_wrong_evidence_digest_refused(tmp_path):
    root,req,r,raw,b=audit(tmp_path);b['digest_bindings'][1]['content_ref']=b['digest_bindings'][0]['content_ref'];change(root,req,b)
    with pytest.raises(LegacyBindingError,match='DIGEST_BINDING_MISMATCH'):run(root,req)


def test_audit_shared_evidence_and_provenance_keep_roles(tmp_path):
    root,req,r,raw,b=audit(tmp_path,shared=True);out=legacy_dependencies(raw,canonical_bytes(b))
    assert len(out['stored_content'])==2 and out['stored_content'][0]['ref']==out['stored_content'][1]['ref']
    assert out['stored_content'][0]['path']!=out['stored_content'][1]['path']
    assert run(root,req)['status']=='STRUCTURAL_PASS'


def test_audit_evidence_inventory_required(tmp_path):
    root,req,r,raw,b=audit(tmp_path);req['required_content'].remove(b['digest_bindings'][1]['content_ref'])
    with pytest.raises(ClosureError,match='INVENTORY_MISMATCH'):run(root,req)


def test_audit_duplicate_digest_binding_refused(tmp_path):
    root,req,r,raw,b=audit(tmp_path);b['digest_bindings'].append(b['digest_bindings'][1]);change(root,req,b)
    with pytest.raises(LegacyBindingError,match='DIGEST_BINDING_MISMATCH'):run(root,req)


def test_audit_scope_and_limitations_tampering_breaks_seal(tmp_path):
    root,req,r,raw,b=audit(tmp_path)
    for field,value in [('authorized_scope',{'unbounded':True}),('limitations',[]),('decision','CERTIFY_AND_ADVANCE')]:
        altered=deepcopy(r);altered['payload'][field]=value
        with pytest.raises(Exception):legacy_dependencies(canonical_bytes(altered),canonical_bytes(b))


def test_audit_external_provenance_cannot_be_reclassified(tmp_path):
    root,req,r,raw,b=audit(tmp_path);r['provenance']['classification']='FINITE_COMPUTATIONAL_OBSERVATION'
    with pytest.raises(Exception,match='external audit provenance'):seal_reference_record(r)


def test_audit_v2_does_not_accept_invented_semantic_bindings(tmp_path):
    root,req,r,raw,b=audit(tmp_path);b['schema_id']='IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V2';b['semantic_bindings']=[];change(root,req,b)
    assert run(root,req)['status']=='STRUCTURAL_PASS'
    b['semantic_bindings']=[{'path':'/payload/evidence_hashes/0/sha256','profile':'IG_CANONICAL_JSON_V1','content_ref':b['digest_bindings'][1]['content_ref']}]
    with pytest.raises(LegacyBindingError,match='SEMANTIC_BINDING_MISMATCH'):legacy_dependencies(raw,canonical_bytes(b))
