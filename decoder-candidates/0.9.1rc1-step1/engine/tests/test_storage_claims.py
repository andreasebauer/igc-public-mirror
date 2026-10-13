"""Recover declared support without treating recovery as scientific certification."""
from copy import deepcopy
import json
import pytest
from test_storage_legacy import fixture,put,run
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_legacy import legacy_dependencies,LegacyBindingError
from infinity_grid.storage_closure import ClosureError


def claims(tmp_path,decision='CONDITIONAL',strength='CONDITIONAL',authority='NONE'):
    root,req,records,raws,bindings=fixture(tmp_path);added=[]
    for kind,ident,payload in [
        ('GRADUATION','IGRD/L0/GRADUATION/G',{'decision':decision,'graduated_scope':{'bounded':True},'authorizes':['FIXTURE_ONLY'],'evidence_record_ids':[records[0]['record_id'],records[1]['record_id']]}),
        ('EARNED_ALGEBRA','IGRD/L0/ALGEBRA/A',{'statement_id':'FIXTURE','statement':'Synthetic claim only','strength':strength,'statement_scope':{'bounded':True},'supporting_record_ids':['IGRD/L0/GRADUATION/G'],'nonclaims':['NO_THEOREM_PROVEN']})]:
        r=deepcopy(records[0]);r.update(record_type=kind,record_id=ident,payload=payload,authority_effect=authority)
        r['dependencies']=payload['evidence_record_ids' if kind=='GRADUATION' else 'supporting_record_ids'];r=seal_reference_record(r)
        raw=json.dumps(r,indent=2).encode()+b'\n';ref=put(root,raw)
        known={b['record_ref']['record_id']:b['record_ref'] for b in bindings}
        b={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1','record_ref':{'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':ref},
            'record_links':[known[x] for x in r['dependencies']], 'digest_bindings':[bindings[0]['digest_bindings'][0]],'symbol_bindings':[]}
        bref=put(root,canonical_bytes(b));req['interpretations'].append({'content_ref':ref,'format':'NATIVE_REFERENCE_RECORD_V1'})
        req['required_content'] += [ref,bref];req['legacy_bindings'].append({'content_ref':ref,'binding_ref':bref})
        bindings.append(b);added.append((r,raw,b))
    req['roots'].append(added[-1][2]['record_ref']['content_ref'])
    return root,req,added


def change(root,req,b,index):
    old=req['legacy_bindings'][index]['binding_ref'];new=put(root,canonical_bytes(b));req['legacy_bindings'][index]['binding_ref']=new
    req['required_content']=[new if r==old else r for r in req['required_content']]


def test_claim_recursive_support_chain_and_raw_bytes(tmp_path):
    root,req,added=claims(tmp_path);out=run(root,req)
    assert out['status']=='STRUCTURAL_PASS' and len(out['legacy_records_verified'])==5
    for r,raw,b in added:
        assert (root/(b['record_ref']['content_ref']['sha256']+'.blob')).read_bytes()==raw
        assert b['record_ref']['content_ref']['sha256']!=r['record_sha256']


def test_claim_all_decisions_and_strengths_no_acceptance(tmp_path):
    for decision in ['GRADUATED','NOT_GRADUATED','CONDITIONAL']:
        for strength in ['EXACT','THEOREM_BACKED','BOUNDED_EMPIRICAL','CONDITIONAL']:
            p=tmp_path/(decision+strength);p.mkdir();root,req,added=claims(p,decision,strength)
            assert run(root,req)['scientific_acceptance']=='NOT_GRANTED'
            assert added[0][0]['payload']['decision']==decision and added[1][0]['payload']['strength']==strength


def test_graduation_missing_evidence_binding_refused(tmp_path):
    root,req,added=claims(tmp_path);b=added[0][2];b['record_links'].pop();change(root,req,b,3)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_earned_missing_support_binding_refused(tmp_path):
    root,req,added=claims(tmp_path);b=added[1][2];b['record_links']=[];change(root,req,b,4)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_claim_extra_support_binding_refused(tmp_path):
    root,req,added=claims(tmp_path);b=added[1][2];b['record_links'].append(added[1][2]['record_ref']);change(root,req,b,4)
    with pytest.raises(LegacyBindingError,match='LINK_CLOSURE'):run(root,req)


def test_claim_wrong_support_semantic_seal_refused(tmp_path):
    root,req,added=claims(tmp_path);b=added[1][2];b['record_links'][0]=dict(b['record_links'][0],record_sha256='0'*64);change(root,req,b,4)
    with pytest.raises(ClosureError,match='LEGACY_REFERENCE_BINDING'):run(root,req)


def test_claim_missing_provenance_binding_refused(tmp_path):
    root,req,added=claims(tmp_path)
    for r,raw,b in added:
        b=deepcopy(b);b['digest_bindings']=[]
        with pytest.raises(LegacyBindingError,match='MISSING_DIGEST'):legacy_dependencies(raw,canonical_bytes(b))


def test_claim_historical_authority_is_only_recorded(tmp_path):
    root,req,added=claims(tmp_path,authority='HISTORICAL_AUTHORITY_RECORDED')
    for r,raw,b in added:
        out=legacy_dependencies(raw,canonical_bytes(b));assert out['authority_effect']=='HISTORICAL_AUTHORITY_RECORDED'
        assert out['epistemic_status']=='HISTORICAL' and out['science_execution']=='NONE'
        assert out['scientific_acceptance']=='NOT_GRANTED'
    assert run(root,req)['scientific_acceptance']=='NOT_GRANTED'


def test_claim_mutated_decision_or_strength_breaks_seal(tmp_path):
    root,req,added=claims(tmp_path)
    for (r,raw,b),field,value in [(added[0],'decision','GRADUATED'),(added[1],'strength','EXACT')]:
        r=deepcopy(r);r['payload'][field]=value
        with pytest.raises(Exception,match='hash mismatch'):legacy_dependencies(canonical_bytes(r),canonical_bytes(b))


def test_claim_scope_and_nonclaims_remain_inline_metadata(tmp_path):
    root,req,added=claims(tmp_path)
    for r,raw,b in added:
        out=legacy_dependencies(raw,canonical_bytes(b))
        assert len(out['stored_content'])==1 and out['stored_content'][0]['role']=='LEGACY_RAW_DIGEST'
        assert json.loads(raw)['scope']==r['scope'] and json.loads(raw)['nonclaims']==r['nonclaims']
        assert json.loads(raw)['payload']==r['payload']
    # A made-up interpretation for an authorizes label cannot be smuggled in.
    r,raw,b=added[0];b=deepcopy(b);b['symbol_bindings']=[{'path':'/payload/authorizes/0','value':'FIXTURE_ONLY','content_ref':b['digest_bindings'][0]['content_ref']}]
    with pytest.raises(LegacyBindingError,match='SYMBOL_BINDING'):legacy_dependencies(raw,canonical_bytes(b))


def test_claim_support_requires_adapter_registration(tmp_path):
    root,req,added=claims(tmp_path);req['legacy_bindings'].pop(3)
    with pytest.raises(ClosureError,match='ADAPTER_REQUIRED'):run(root,req)


def test_claim_binding_itself_is_required_inventory(tmp_path):
    root,req,added=claims(tmp_path);req['required_content'].remove(req['legacy_bindings'][4]['binding_ref'])
    with pytest.raises(ClosureError,match='INVENTORY_MISMATCH'):run(root,req)
