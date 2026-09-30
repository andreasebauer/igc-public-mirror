import json
import pytest
from test_storage_runner_snapshot import snapshot
from test_p2a_replay_dag_runner import create, result, manifest
from infinity_grid.canon import canonical_bytes
from infinity_grid.replay_l0_executor import seal_l0_frontier
from infinity_grid.replay_reference_data import seal_reference_record
from infinity_grid.storage_frontier_snapshot import verify_frontier_snapshot
from infinity_grid.storage_runner_snapshot import RunnerSnapshotError


def frontier(s):
    state=json.loads(s[1]);m=json.loads(s[0])
    next_id=next(n for n in m['topological_order'] if n not in state['completed_node_ids'])
    return seal_l0_frontier(runner_state=state,manifest=m,next_node_id=next_id,
        waiting_for_audit=state['active_audit_capsule_sha256'] is not None)


def check(s,f=None,**kw):
    return verify_frontier_snapshot(canonical_bytes(frontier(s) if f is None else f),s[0],s[1],**s[2],**kw)


def changed(s,key,value):
    f=frontier(s);f['payload'][key]=value;return seal_reference_record(f)


def test_frontier_empty_snapshot_bound_without_scheduling(tmp_path):
    s=snapshot(tmp_path);before=s[1];out=check(s)
    assert out['status']=='FRONTIER_SNAPSHOT_BOUND' and s[1]==before
    assert out['execution_authorized'] is False and out['dependency_closure_verified'] is False


def test_frontier_accepted_snapshot_distinguishes_raw_and_semantic(tmp_path):
    s=snapshot(tmp_path,'accepted');out=check(s)
    assert out['snapshot']['inventory'][0]['content_ref']['sha256']!=out['snapshot']['manifest_sha256']
    assert out['records_checked']==4 and out['scientific_acceptance']=='NOT_GRANTED'


def test_frontier_active_audit_binding(tmp_path):
    s=snapshot(tmp_path,'stopped');out=check(s)
    assert out['frontier_status']=='WAITING_FOR_EXTERNAL_AUDIT'


def test_frontier_resolved_audit_retains_decision(tmp_path):
    s=snapshot(tmp_path,'resolved');out=check(s)
    assert out['records_checked']==5 and out['frontier_status']=='READY'


def test_frontier_executor_wait_is_metadata_not_authority(tmp_path):
    s=snapshot(tmp_path);out=check(s,changed(s,'frontier_status','WAITING_FOR_NODE_EXECUTOR_BINDING'))
    assert out['execution_authorized'] is False


def test_frontier_other_run_refused(tmp_path):
    s=snapshot(tmp_path)
    with pytest.raises(RunnerSnapshotError,match='MISMATCH:root_run_id'):check(s,changed(s,'root_run_id','OTHER'))


def test_frontier_manifest_and_state_seals_refused(tmp_path):
    s=snapshot(tmp_path)
    for key in ['manifest_dag_sha256','runner_state_sha256']:
        with pytest.raises(RunnerSnapshotError,match='MISMATCH:'+key):check(s,changed(s,key,'0'*64))


def test_frontier_stale_completed_index_refused(tmp_path):
    s=snapshot(tmp_path,'accepted');f=frontier(s);f['payload']['completed_node_ids']=[];f['payload']['checkpoint_sha256_by_node']={}
    with pytest.raises(RunnerSnapshotError,match='MISMATCH:completed_node_ids'):check(s,seal_reference_record(f))


def test_frontier_wrong_checkpoint_seal_refused(tmp_path):
    s=snapshot(tmp_path,'accepted');f=frontier(s);key=f['payload']['completed_node_ids'][0];f['payload']['checkpoint_sha256_by_node'][key]='0'*64
    with pytest.raises(RunnerSnapshotError,match='MISMATCH:checkpoint_sha256_by_node'):check(s,seal_reference_record(f))


def test_frontier_wrong_next_node_refused(tmp_path):
    s=snapshot(tmp_path)
    with pytest.raises(RunnerSnapshotError,match='NEXT_NODE_MISMATCH'):check(s,changed(s,'next_node_id',manifest()['topological_order'][1]))


def test_frontier_status_cannot_hide_stop_or_invent_completion(tmp_path):
    s=snapshot(tmp_path,'stopped')
    with pytest.raises(RunnerSnapshotError,match='AUDIT_STATUS_MISMATCH'):check(s,changed(s,'frontier_status','READY'))
    p=tmp_path/'ready';p.mkdir();s=snapshot(p)
    for status in ['WAITING_FOR_EXTERNAL_AUDIT','COMPLETE']:
        with pytest.raises(RunnerSnapshotError,match='READY_STATUS_MISMATCH'):check(s,changed(s,'frontier_status',status))


def test_frontier_unknown_or_raw_manifest_provenance_refused(tmp_path):
    import hashlib
    s=snapshot(tmp_path)
    for ref,digest in [('other',json.loads(s[0])['dag_sha256']),('compiled_manifest.dag_sha256',hashlib.sha256(s[0]).hexdigest())]:
        f=frontier(s);f['provenance']['source_hashes']=[{'ref':ref,'sha256':digest}]
        with pytest.raises(RunnerSnapshotError,match='UNSUPPORTED_FRONTIER_PROVENANCE'):check(s,seal_reference_record(f))


def test_frontier_pending_gate_refused_without_transition(tmp_path):
    runner=create(tmp_path)
    for _ in range(4):runner.record_node_result(result(runner.next_action()))
    root=tmp_path/'run';before=(root/'runner_state.json').read_bytes()
    s=[canonical_bytes(manifest()),before,{'checkpoints':[p.read_bytes() for p in (root/'checkpoints').glob('*.json')]}]
    with pytest.raises(RunnerSnapshotError,match='PENDING_GATE_TRANSITION'):check(s)
    assert (root/'runner_state.json').read_bytes()==before


def test_frontier_budgets_include_frontier_record(tmp_path):
    s=snapshot(tmp_path);out=check(s)
    with pytest.raises(RunnerSnapshotError,match='BYTE_BUDGET'):check(s,max_total_bytes=out['bytes_checked']-1)
    p=tmp_path/'accepted';p.mkdir();s=snapshot(p,'accepted')
    with pytest.raises(RunnerSnapshotError,match='RECORD_BUDGET'):check(s,max_records=3)
