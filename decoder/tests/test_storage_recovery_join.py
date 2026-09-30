import hashlib,json
from pathlib import Path
import pytest
from test_storage_result_closure import fixture as joined_fixture
from test_storage_legacy import put
from infinity_grid.canon import canonical_bytes,canonical_sha256
from infinity_grid.storage_schema import canonical_bytes as storage_bytes
from infinity_grid.replay_obligation_compiler import compile_obligations
from infinity_grid.replay_l0_executor import seal_l0_frontier
from infinity_grid.replay_reference_data import ReplayReferenceDataStore
from infinity_grid.storage_collections import ContentDirectory
from infinity_grid.storage_recovery_join import verify_supported_recovery
from infinity_grid.storage_closure import ClosureError


def dataset(root,raws):
    store=ReplayReferenceDataStore.initialize(root)
    for raw in raws:store.put(json.loads(raw))
    return {'manifest_raw':(root/'MANIFEST.json').read_bytes(),'records':raws,
        **{key:[p.read_bytes() for p in (root/key).glob('*.json')] for key in ['transactions','commits']}}


def fixture(tmp_path):
    root,s,req,src,obs=joined_fixture(tmp_path)
    cat=json.loads((Path(__file__).resolve().parents[1]/'infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json').read_text())
    sources={};raw=b'synthetic pinned source bytes';digest=hashlib.sha256(raw).hexdigest()
    for group,fields in [('obligations',['source_hashes','evidence_hashes']),('historical_assertion_mappings',['source_hashes']),('historical_audit_authorizations',['historical_source_hashes','accepted_equivalence_certificates','audit_provenance','audit_evidence']),('known_replay_qualifications',['evidence_pins'])]:
        for row in cat.get(group,[]):
            for field in fields:
                for pin in row.get(field,[]):pin['sha256']=digest;sources[pin['ref']]=raw
            for field in ['decision_hash','record_hash']:
                if field in row:row.pop(field);row[field]=canonical_sha256(row)
    for field in ['audit_provenance_ledger','known_replay_qualification_ledger','execution_class_ledger','cost_budget_ledger']:
        if cat.get(field):cat[field]['sha256']=digest;sources[cat[field]['ref']]=raw
    m=compile_obligations(cat);s['manifest_raw']=canonical_bytes(m)
    cp=json.loads(s['checkpoints'][0]);old=cp.pop('checkpoint_sha256');cp['manifest_sha256']=m['dag_sha256'];cp['checkpoint_sha256']=canonical_sha256(cp);new=cp['checkpoint_sha256']
    s['checkpoints']=[canonical_bytes(cp)];s['bindings']={new:s['bindings'][old]}
    state=json.loads(s['state_raw']);state['manifest_sha256']=m['dag_sha256'];state['accepted_checkpoint_sha256_by_node'][cp['node_id']]=new;state.pop('state_sha256');state['state_sha256']=canonical_sha256(state);s['state_raw']=canonical_bytes(state)
    next_id=next(n for n in m['topological_order'] if n not in state['completed_node_ids'])
    f=canonical_bytes(seal_l0_frontier(runner_state=state,manifest=m,next_node_id=next_id,waiting_for_audit=False))
    raws=list(next(iter(s['bindings'].values()))['records'])+[f]
    args={'snapshot_inputs':s,'dataset_inputs':dataset(tmp_path/'full_dataset',raws),'frontier_raw':f,'catalogue_raw':canonical_bytes(cat),'sources':sources}
    return root,req,args,obs


def run(root,req,args,**kw):return verify_supported_recovery(ContentDirectory(root),put(root,storage_bytes(req)),**args,**kw)


def test_recovery_join_all_components_bound(tmp_path):
    root,req,args,obs=fixture(tmp_path);out=run(root,req,args)
    assert out['status']=='SUPPORTED_RECOVERY_BUNDLE_VERIFIED'
    assert len(out['dataset_snapshot']['record_ids'])==4


def test_recovery_join_does_not_grant_authority_or_freshness(tmp_path):
    root,req,args,obs=fixture(tmp_path);out=run(root,req,args)
    assert out['execution_authorized'] is False and out['scientific_acceptance']=='NOT_GRANTED'
    assert out['accepted_head_freshness_verified'] is False and out['production_release_verified'] is False


def test_recovery_join_other_catalogue_refused(tmp_path):
    root,req,args,obs=fixture(tmp_path);c=json.loads(args['catalogue_raw']);c['status']='OTHER';args['catalogue_raw']=canonical_bytes(c)
    with pytest.raises(Exception,match='CATALOGUE_SEMANTIC_HASH_MISMATCH'):run(root,req,args)


def test_recovery_join_missing_manifest_source_refused(tmp_path):
    root,req,args,obs=fixture(tmp_path);args['sources'].pop(next(iter(args['sources'])))
    with pytest.raises(Exception,match='SOURCE_INVENTORY_MISMATCH'):run(root,req,args)


def test_recovery_join_wrong_frontier_refused(tmp_path):
    from infinity_grid.replay_reference_data import seal_reference_record
    root,req,args,obs=fixture(tmp_path);f=json.loads(args['frontier_raw']);f['payload']['root_run_id']='OTHER';args['frontier_raw']=canonical_bytes(seal_reference_record(f))
    with pytest.raises(Exception,match='FRONTIER_SNAPSHOT_MISMATCH'):run(root,req,args)


def test_recovery_join_dataset_without_frontier_refused(tmp_path):
    root,req,args,obs=fixture(tmp_path);args['dataset_inputs']=dataset(tmp_path/'other',args['dataset_inputs']['records'][:-1])
    with pytest.raises(ClosureError,match='DATASET_RECORD_BINDING_MISMATCH'):run(root,req,args)


def test_recovery_join_extra_valid_dataset_record_refused(tmp_path):
    from infinity_grid.replay_reference_data import seal_reference_record
    root,req,args,obs=fixture(tmp_path);raws=args['dataset_inputs']['records'];extra=json.loads(raws[0]);extra['record_id']+='EXTRA'
    args['dataset_inputs']=dataset(tmp_path/'other',raws+[canonical_bytes(seal_reference_record(extra))])
    with pytest.raises(ClosureError,match='DATASET_RECORD_BINDING_MISMATCH'):run(root,req,args)


def test_recovery_join_reformatted_dataset_record_is_not_same_raw_binding(tmp_path):
    root,req,args,obs=fixture(tmp_path);args['dataset_inputs']['records'][0]=json.dumps(json.loads(args['dataset_inputs']['records'][0]),indent=4).encode()
    with pytest.raises(ClosureError,match='DATASET_RECORD_BINDING_MISMATCH'):run(root,req,args)


def test_recovery_join_missing_observation_refused(tmp_path):
    root,req,args,obs=fixture(tmp_path);(root/(obs['sha256']+'.blob')).unlink()
    with pytest.raises(Exception):run(root,req,args)


def test_recovery_join_pending_dataset_transaction_refused(tmp_path):
    root,req,args,obs=fixture(tmp_path);args['dataset_inputs']['commits'].pop()
    with pytest.raises(Exception,match='DATASET_PENDING_OR_DUPLICATE_COMMIT'):run(root,req,args)


def test_recovery_join_total_budget_covers_all_components(tmp_path):
    root,req,args,obs=fixture(tmp_path);out=run(root,req,args)
    with pytest.raises(Exception,match='BYTE_BUDGET'):run(root,req,args,max_total_bytes=out['bytes_checked']-1)


def test_recovery_join_frontier_must_be_final_dataset_record(tmp_path):
    root,req,args,obs=fixture(tmp_path);raws=args['dataset_inputs']['records'];args['dataset_inputs']=dataset(tmp_path/'other',[raws[-1]]+raws[:-1])
    with pytest.raises(ClosureError,match='FRONTIER_NOT_FINAL_RECORD'):run(root,req,args)
