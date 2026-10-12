"""Evidence observation bytes and canonical identity are separate obligations."""
from copy import deepcopy
import json
import pytest
from test_storage_legacy import fixture,put,run
from infinity_grid.canon import canonical_sha256
from infinity_grid.storage_schema import canonical_bytes,StorageSchemaError
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_legacy import legacy_dependencies,LegacyBindingError
from infinity_grid.storage_closure import ClosureError,ClosureLimits,verify_declared_closure
from infinity_grid.storage_collections import ContentDirectory


def evidence(tmp_path,mode='CANONICAL_JSON',compression='EXACT',certificate=None):
    root,req,records,raws,bindings=fixture(tmp_path)
    observation={'assertion_id':'FIXTURE','expected_outcome':'ZERO_MISMATCH','count':2}
    result_raw=json.dumps(observation,indent=3).encode()+b'\n';result_ref=put(root,result_raw)
    r=deepcopy(records[0]);r.update(record_type='SRCF_EVIDENCE',record_id='IGRD/L0/EVIDENCE/E')
    r['payload']={'series':'S','obligation_id':'FIXTURE','result_identity':canonical_sha256(observation),
        'equality_contract':{'mode':mode,'cardinality_semantics':'ORDERED','compression':compression,'observer':'FIXTURE_OBSERVER','certificate_sha256':certificate},
        'evidence_mode':'HISTORICAL_RESULT_ONLY','outcome':'REPRODUCED'};r=seal_reference_record(r)
    raw=json.dumps(r,indent=2).encode()+b'\n';ref=put(root,raw)
    b={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V2','record_ref':{'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':ref},
        'record_links':[],'digest_bindings':[bindings[0]['digest_bindings'][0]],'symbol_bindings':[],
        'semantic_bindings':[{'path':'/payload/result_identity','profile':'IG_CANONICAL_JSON_V1','content_ref':result_ref}]}
    bref=put(root,canonical_bytes(b));req['roots'].append(ref)
    req['interpretations'] += [{'content_ref':ref,'format':'NATIVE_REFERENCE_RECORD_V1'},{'content_ref':result_ref,'format':'OPAQUE_LEAF_V1'}]
    req['required_content'] += [ref,result_ref,bref];req['legacy_bindings'].append({'content_ref':ref,'binding_ref':bref})
    return root,req,r,raw,b


def change(root,req,b):
    old=req['legacy_bindings'][-1]['binding_ref'];new=put(root,canonical_bytes(b));req['legacy_bindings'][-1]['binding_ref']=new
    req['required_content']=[new if x==old else x for x in req['required_content']]


def test_evidence_pretty_bytes_and_semantic_identity_distinct(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);ref=b['semantic_bindings'][0]['content_ref'];before=(root/(ref['sha256']+'.blob')).read_bytes()
    assert ref['sha256']!=r['payload']['result_identity']
    out=run(root,req);assert out['status']=='STRUCTURAL_PASS' and out['scientific_acceptance']=='NOT_GRANTED'
    assert (root/(ref['sha256']+'.blob')).read_bytes()==before


def test_graduation_recovers_supported_evidence_chain(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);g=deepcopy(r);g.update(record_id='IGRD/L0/GRADUATION/G',record_type='GRADUATION',dependencies=[r['record_id']])
    g['payload']={'decision':'GRADUATED','graduated_scope':{'fixture':True},'authorizes':['FIXTURE'],'evidence_record_ids':[r['record_id']]};g=seal_reference_record(g)
    ref=put(root,canonical_bytes(g));binding={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1','record_ref':{'record_id':g['record_id'],'record_sha256':g['record_sha256'],'content_ref':ref},'record_links':[b['record_ref']],'digest_bindings':b['digest_bindings'],'symbol_bindings':[]};br=put(root,canonical_bytes(binding))
    req['roots'][-1]=ref;req['interpretations'].append({'content_ref':ref,'format':'NATIVE_REFERENCE_RECORD_V1'});req['required_content'] += [ref,br];req['legacy_bindings'].append({'content_ref':ref,'binding_ref':br})
    out=run(root,req);assert len(out['legacy_records_verified'])==5 and out['scientific_acceptance']=='NOT_GRANTED'


def test_evidence_wrong_observation_identity_refused(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);b['semantic_bindings'][0]['content_ref']=put(root,b'{"wrong":true}');change(root,req,b)
    with pytest.raises(ClosureError,match='RESULT_IDENTITY'):run(root,req)


def test_evidence_v1_cannot_omit_semantic_binding(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);b['schema_id']='IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V1';b.pop('semantic_bindings')
    with pytest.raises(LegacyBindingError,match='SEMANTIC_BINDING_REQUIRED'):legacy_dependencies(raw,canonical_bytes(b))


def test_evidence_missing_semantic_binding_refused(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);b['semantic_bindings']=[]
    with pytest.raises(LegacyBindingError,match='MISSING_SEMANTIC'):legacy_dependencies(raw,canonical_bytes(b))


def test_evidence_unknown_profile_refused(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);b['semantic_bindings'][0]['profile']='RAW_SHA256'
    with pytest.raises(LegacyBindingError,match='SEMANTIC_BINDING_MISMATCH'):legacy_dependencies(raw,canonical_bytes(b))


def test_evidence_duplicate_semantic_binding_refused(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);b['semantic_bindings']*=2
    with pytest.raises(LegacyBindingError,match='SEMANTIC_BINDING_MISMATCH'):legacy_dependencies(raw,canonical_bytes(b))


def test_evidence_wrong_semantic_path_refused(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);b['semantic_bindings'][0]['path']='/payload/outcome'
    with pytest.raises(LegacyBindingError,match='SEMANTIC_BINDING_MISMATCH'):legacy_dependencies(raw,canonical_bytes(b))


def test_evidence_certificate_mode_remains_blocked(tmp_path):
    root,req,r,raw,b=evidence(tmp_path,mode='CERTIFIED_SEMANTIC_EQUIVALENCE',certificate='0'*64)
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_LEGACY_EVIDENCE_PROFILE'):run(root,req)


def test_evidence_compressed_observation_remains_blocked(tmp_path):
    root,req,r,raw,b=evidence(tmp_path,compression='LOSSLESS_COMPRESSED')
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_LEGACY_EVIDENCE_PROFILE'):run(root,req)


def test_evidence_raw_byte_equality_not_guessed(tmp_path):
    root,req,r,raw,b=evidence(tmp_path,mode='EXACT_BYTES')
    with pytest.raises(LegacyBindingError,match='UNSUPPORTED_LEGACY_EVIDENCE_PROFILE'):run(root,req)


def test_evidence_observation_requires_inventory(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);req['required_content'].remove(b['semantic_bindings'][0]['content_ref'])
    with pytest.raises(ClosureError,match='INVENTORY_MISMATCH'):run(root,req)


def test_evidence_duplicate_json_keys_refused(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);b['semantic_bindings'][0]['content_ref']=put(root,b'{"x":1,"x":2}');change(root,req,b)
    with pytest.raises(StorageSchemaError,match='DUPLICATE_JSON_KEY'):run(root,req)


def test_evidence_semantic_reads_obey_byte_budget(tmp_path):
    root,req,r,raw,b=evidence(tmp_path);rr=put(root,canonical_bytes(req))
    with pytest.raises(ClosureError,match='BYTE_BUDGET'):verify_declared_closure(ContentDirectory(root),rr,limits=ClosureLimits(max_total_bytes=int(rr['size_bytes'])+1))
