import json
import pytest
from test_storage_snapshot_results import fixture as snapshot_fixture
from test_storage_legacy import put
from infinity_grid.canon import canonical_sha256,canonical_bytes
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_collections import ContentDirectory
from infinity_grid.storage_result_closure import verify_snapshot_result_closure
from infinity_grid.storage_closure import ClosureError
from infinity_grid.storage_schema import canonical_bytes as storage_bytes


def fixture(tmp_path):
    s=snapshot_fixture(tmp_path);root=tmp_path/'content';root.mkdir()
    source=put(root,b'original provenance bytes');observation={'count':2,'scope':'fixture'}
    obs=put(root,json.dumps(observation,indent=2).encode())
    old=next(iter(s['bindings']));b=s['bindings'][old];records=[]
    for raw in b['records']:
        r=json.loads(raw);r['provenance']['source_hashes']=[{'ref':'fixture','sha256':source['sha256']}]
        if r['record_type']=='SRCF_EVIDENCE':r['payload']['result_identity']=canonical_sha256(observation)
        records.append(seal_reference_record(r))
    raws=[json.dumps(r,indent=2).encode() for r in records];refs=[put(root,raw) for raw in raws]
    triples={r['record_id']:{'record_id':r['record_id'],'record_sha256':r['record_sha256'],'content_ref':ref} for r,ref in zip(records,refs)}
    b['records']=raws;cp=json.loads(s['checkpoints'][0]);nr=cp['acceptance']['node_result']
    nr['result_sha256']=records[-1]['record_sha256'];nr['evidence_sha256']=records[1]['record_sha256']
    cp.pop('checkpoint_sha256');cp['checkpoint_sha256']=canonical_sha256(cp);new=cp['checkpoint_sha256']
    s['checkpoints']=[canonical_bytes(cp)];s['bindings']={new:b}
    st=json.loads(s['state_raw']);st['accepted_checkpoint_sha256_by_node'][cp['node_id']]=new;st.pop('state_sha256');st['state_sha256']=canonical_sha256(st);s['state_raw']=canonical_bytes(st)
    bindings=[]
    for r in records:
        row={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V2','record_ref':triples[r['record_id']],
            'record_links':[triples[rid] for rid in r['dependencies']],
            'digest_bindings':[{'path':'/provenance/source_hashes/0/sha256','content_ref':source}],
            'symbol_bindings':[],'semantic_bindings':[]}
        if r['record_type']=='SRCF_EVIDENCE':row['semantic_bindings']=[{'path':'/payload/result_identity','profile':'IG_CANONICAL_JSON_V1','content_ref':obs}]
        bindings.append(put(root,storage_bytes(row)))
    req={'schema_id':'IG_STORAGE_DECLARED_CLOSURE_REQUEST_V2','purpose':'SCHEMA_FIXTURE','roots':refs,
        'interpretations':[{'content_ref':r,'format':'NATIVE_REFERENCE_RECORD_V1'} for r in refs]+[{'content_ref':r,'format':'OPAQUE_LEAF_V1'} for r in [source,obs]],
        'required_content':refs+bindings+[source,obs],
        'legacy_bindings':[{'content_ref':r,'binding_ref':b} for r,b in zip(refs,bindings)]}
    return root,s,req,source,obs


def run(root,s,req,**kw):return verify_snapshot_result_closure(ContentDirectory(root),put(root,storage_bytes(req)),snapshot_inputs=s,**kw)


def test_joined_snapshot_result_observation_provenance_closure(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);out=run(root,s,req)
    assert out['status']=='SNAPSHOT_RESULT_DECLARED_CLOSURE_VERIFIED' and out['result_roots_verified']==3
    assert out['full_frontier_closure_verified'] is False and out['populated_dataset_verified'] is False
    assert out['scientific_acceptance']=='NOT_GRANTED' and out['execution_authorized'] is False


def test_joined_raw_record_identity_is_exact(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);ref=req['roots'][0];raw=(root/(ref['sha256']+'.blob')).read_bytes()
    assert ref['sha256']!=json.loads(raw)['record_sha256']
    out=run(root,s,req);assert out['declared_closure']['legacy_records_verified']


def test_joined_missing_observation_bytes_refused(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);(root/(obs['sha256']+'.blob')).unlink()
    with pytest.raises(Exception):run(root,s,req)


def test_joined_wrong_observation_identity_refused(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);wrong=put(root,b'{"count":3,"scope":"fixture"}')
    for link in req['legacy_bindings'][:2]:
        old=link['binding_ref'];r=json.loads((root/(old['sha256']+'.blob')).read_bytes());r['semantic_bindings'][0]['content_ref']=wrong
        new=put(root,storage_bytes(r));link['binding_ref']=new;req['required_content']=[new if x==old else x for x in req['required_content']]
    with pytest.raises(ClosureError,match='LEGACY_RESULT_IDENTITY_MISMATCH'):run(root,s,req)


def test_joined_missing_provenance_bytes_refused(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);(root/(src['sha256']+'.blob')).unlink()
    with pytest.raises(Exception):run(root,s,req)


def test_joined_native_records_cannot_be_opaque_roots(tmp_path):
    root,s,req,src,obs=fixture(tmp_path)
    for row in req['interpretations'][:3]:row['format']='OPAQUE_LEAF_V1'
    req['required_content']=req['roots'];req['legacy_bindings']=[]
    with pytest.raises(ClosureError,match='RESULT_CLOSURE_NATIVE_BINDING_MISMATCH'):run(root,s,req)


def test_joined_missing_result_root_refused(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);req['roots'].pop()
    with pytest.raises(ClosureError,match='RESULT_CLOSURE_ROOT_MISMATCH'):run(root,s,req)


def test_joined_extra_result_root_refused(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);req['roots'].append(obs)
    with pytest.raises(ClosureError,match='RESULT_CLOSURE_ROOT_MISMATCH'):run(root,s,req)


def test_joined_missing_native_binding_refused(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);req['legacy_bindings'].pop()
    with pytest.raises(ClosureError,match='NATIVE_REFERENCE_CLOSURE_ADAPTER_REQUIRED'):run(root,s,req)


def test_joined_provider_cannot_lie_about_request_bytes(tmp_path):
    root,s,req,src,obs=fixture(tmp_path)
    class Liar:
        def read(self,ref,bound):return b'{}'
    with pytest.raises(ClosureError,match='PROVIDER_CONTENT_MISMATCH'):
        verify_snapshot_result_closure(Liar(),put(root,storage_bytes(req)),snapshot_inputs=s)


def test_joined_shared_read_budget(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);out=run(root,s,req)
    assert out['bytes_checked']>out['snapshot_result_verification']['bytes_checked']
    with pytest.raises(ClosureError,match='JOINED_BYTE_BUDGET'):run(root,s,req,max_total_bytes=out['bytes_checked']-1)


def test_joined_tampered_snapshot_still_refused(tmp_path):
    root,s,req,src,obs=fixture(tmp_path);st=json.loads(s['state_raw']);st['root_run_id']='OTHER';s['state_raw']=canonical_bytes(st)
    with pytest.raises(Exception,match='state hash mismatch'):run(root,s,req)
