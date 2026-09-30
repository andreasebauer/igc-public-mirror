"""Actual preserved P9 closure; no scientific handler or new replay execution."""
import json,zipfile,hashlib,collections
from pathlib import Path
import pytest
from test_storage_legacy import put
from test_storage_auxiliary_closure import change
from infinity_grid.canon import canonical_sha256
from infinity_grid.replay_reference_data import empty_manifest
from infinity_grid.storage_schema import canonical_bytes
from infinity_grid.storage_collections import ContentDirectory
from infinity_grid.storage_result_closure import verify_snapshot_result_closure


def fixture(tmp_path):
    S=Path(__file__).resolve().parents[1];F=S/'tests/fixtures/storage_observations'
    archive=(S/'tests/fixtures/dev79_index_reproduction/P9_CAPTURED_STATE_OBJECT_V1.zip').read_bytes()
    assert hashlib.sha256(archive).hexdigest()=='b360397d2593cfee3086ac4b23560c6ab8de2418d6d607d89844463a00766cfc'
    import io
    with zipfile.ZipFile(io.BytesIO(archive)) as z:objs={n:z.read(n) for n in z.namelist() if n.endswith('.json')}
    records={json.loads(b)['record_id']:(b,json.loads(b)) for n,b in objs.items() if '/replay_reference_data/records/' in n}
    seals={r['record_sha256']:rid for rid,(b,r) in records.items()}
    checkpoints=[b for n,b in objs.items() if '/replay_runner/checkpoints/' in n];bindings={};used=set()
    for raw in checkpoints:
        cp=json.loads(raw);nr=cp['acceptance'].get('node_result')
        if nr is None:continue
        rid=seals[nr['result_sha256']];p=records[rid][1]['payload'];h=p['historical_record_ids'];r=p['replay_record_ids']
        rs=[records[k][1]['record_sha256'] for k in r]
        paired=[records[k][1]['record_sha256'] for pair in zip(h,r) for k in pair]
        matches=[]
        if len(rs)==1 and nr['evidence_sha256']==rs[0]:matches.append('SINGLE_REPLAY_RECORD_V1')
        if nr['evidence_sha256']==canonical_sha256(rs):matches.append('ORDERED_REPLAY_SEALS_V1')
        if nr['evidence_sha256']==canonical_sha256(paired):matches.append('ORDERED_PAIRED_SEALS_V1')
        assert len(matches)==1
        ids={rid,*h,*r}
        if nr.get('audit_authorization_record_sha256'):ids.add(seals[nr['audit_authorization_record_sha256']])
        used.update(ids);bindings[cp['checkpoint_sha256']]={'profile':matches[0],'records':[records[k][0] for k in sorted(ids)]}
    snapshot={'manifest_raw':(S/'infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_DAG_V9.json').read_bytes(),
        'state_raw':next(b for n,b in objs.items() if n.endswith('replay_runner/runner_state.json')),
        'dataset_raw':canonical_bytes(empty_manifest()),'bindings':bindings,'checkpoints':checkpoints,
        'capsules':[b for n,b in objs.items() if '/replay_runner/audit_capsules/' in n],
        'decisions':[b for n,b in objs.items() if '/replay_runner/external_decisions/' in n]}
    root=tmp_path/'content';root.mkdir();refs={};formats={}
    def add(raw,fmt='OPAQUE_LEAF_V1'):
        ref=put(root,raw);refs[ref['sha256']]=ref;formats[ref['sha256']]={'content_ref':ref,'format':fmt};return ref
    triples={rid:{'record_id':rid,'record_sha256':records[rid][1]['record_sha256'],'content_ref':add(records[rid][0],'NATIVE_REFERENCE_RECORD_V1')} for rid in sorted(used)}
    def pinned(row):
        if row['ref']=='infinity_grid/replay_node_in_audit.py' and row['sha256']=='cc8be94113f20e36b2befd8788fc1acb265d4b93d6505f1a30c0829af30647aa':raw=(F/'ORIGINAL_replay_node_in_audit.py').read_bytes()
        else:raw=(S/row['ref']).read_bytes()
        ref=add(raw);assert ref['sha256']==row['sha256'];return ref
    maps=[];bs={}
    for rid in sorted(used):
        raw,r=records[rid];digests=[{'path':f'/provenance/source_hashes/{i}/sha256','content_ref':pinned(row)} for i,row in enumerate(r['provenance']['source_hashes'])]
        if r['record_type']=='AUDIT_AUTHORIZATION':digests += [{'path':f'/payload/evidence_hashes/{i}/sha256','content_ref':pinned(row)} for i,row in enumerate(r['payload']['evidence_hashes'])]
        b={'schema_id':'IG_STORAGE_LEGACY_DEPENDENCY_BINDING_V2','record_ref':triples[rid],'record_links':[triples[k] for k in r['dependencies']],'digest_bindings':digests,'symbol_bindings':[],'semantic_bindings':[]}
        if r['record_type']=='SRCF_EVIDENCE':
            identity=r['payload']['result_identity'];obs=(F/'payloads'/(identity+'.json')).read_bytes();assert hashlib.sha256(obs).hexdigest()==identity
            b['semantic_bindings']=[{'path':'/payload/result_identity','profile':'IG_CANONICAL_JSON_V1','content_ref':add(obs)}]
        br=put(root,canonical_bytes(b));refs[br['sha256']]=br;maps.append({'content_ref':triples[rid]['content_ref'],'binding_ref':br});bs[rid]=b
    req={'schema_id':'IG_STORAGE_DECLARED_CLOSURE_REQUEST_V2','purpose':'SCIENCE','roots':[t['content_ref'] for t in triples.values()],'interpretations':list(formats.values()),'required_content':list(refs.values()),'legacy_bindings':maps}
    return root,snapshot,req,bs


def run(root,s,req,**kw):return verify_snapshot_result_closure(ContentDirectory(root),put(root,canonical_bytes(req)),snapshot_inputs=s,**kw)


def test_actual_p9_snapshot_and_all_result_dependencies_close(tmp_path):
    root,s,req,bs=fixture(tmp_path);out=run(root,s,req)
    assert out['status']=='SNAPSHOT_RESULT_DECLARED_CLOSURE_VERIFIED'
    assert out['result_roots_verified']==107 and len(out['declared_closure']['legacy_records_verified'])==107
    assert collections.Counter(b['profile'] for b in s['bindings'].values())=={'SINGLE_REPLAY_RECORD_V1':16,'ORDERED_REPLAY_SEALS_V1':4,'ORDERED_PAIRED_SEALS_V1':4}
    observations=[b['semantic_bindings'][0]['content_ref']['sha256'] for b in bs.values() if b['semantic_bindings']]
    assert len(observations)==80 and len(set(observations))==49
    assert not out['execution_authorized'] and not out['populated_dataset_verified'] and not out['full_frontier_closure_verified']
    assert out['scientific_acceptance']=='NOT_GRANTED'


def test_p9_missing_reconstructed_observation_refused(tmp_path):
    root,s,req,bs=fixture(tmp_path);b=next(b for b in bs.values() if b['semantic_bindings']);ref=b['semantic_bindings'][0]['content_ref'];(root/(ref['sha256']+'.blob')).unlink()
    with pytest.raises(Exception):run(root,s,req)


def test_p9_wrong_observation_identity_refused(tmp_path):
    root,s,req,bs=fixture(tmp_path);b=next(b for b in bs.values() if b['semantic_bindings']);b['semantic_bindings'][0]['content_ref']=put(root,b'{}');change(root,req,b)
    with pytest.raises(ValueError,match='LEGACY_RESULT_IDENTITY_MISMATCH'):run(root,s,req)


def test_p9_later_node_audit_cannot_replace_historical_source(tmp_path):
    root,s,req,bs=fixture(tmp_path);old='cc8be94113f20e36b2befd8788fc1acb265d4b93d6505f1a30c0829af30647aa'
    b=next(b for b in bs.values() if any(x['content_ref']['sha256']==old for x in b['digest_bindings']))
    row=next(x for x in b['digest_bindings'] if x['content_ref']['sha256']==old);row['content_ref']=put(root,(Path(__file__).resolve().parents[1]/'infinity_grid/replay_node_in_audit.py').read_bytes());change(root,req,b)
    with pytest.raises(ValueError,match='DIGEST_BINDING_MISMATCH'):run(root,s,req)


def test_p9_altered_historical_source_provider_refused(tmp_path):
    root,s,req,bs=fixture(tmp_path);(root/'cc8be94113f20e36b2befd8788fc1acb265d4b93d6505f1a30c0829af30647aa.blob').write_bytes(b'changed')
    with pytest.raises(Exception):run(root,s,req)


def test_p9_result_root_omission_refused(tmp_path):
    root,s,req,bs=fixture(tmp_path);req['roots'].pop()
    with pytest.raises(ValueError,match='RESULT_CLOSURE_ROOT_MISMATCH'):run(root,s,req)


def test_p9_observation_inventory_omission_refused(tmp_path):
    root,s,req,bs=fixture(tmp_path);b=next(b for b in bs.values() if b['semantic_bindings']);req['required_content'].remove(b['semantic_bindings'][0]['content_ref'])
    with pytest.raises(ValueError,match='INVENTORY_MISMATCH'):run(root,s,req)


def test_p9_combined_read_budget_enforced(tmp_path):
    root,s,req,bs=fixture(tmp_path);out=run(root,s,req)
    with pytest.raises(ValueError,match='BUDGET'):run(root,s,req,max_total_bytes=out['bytes_checked']-1)
