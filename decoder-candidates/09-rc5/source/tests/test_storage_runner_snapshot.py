"""Reuse native rollback/audit checks on bounded transported snapshots."""
import json
import pytest
from test_p2a_replay_dag_runner import create,result,manifest,ZERO,ONE
from infinity_grid.canon import canonical_bytes
from infinity_grid.replay_dag_runner import ReplayDagRunnerError,seal_external_decision,DECISION_SCHEMA
from infinity_grid.storage_runner_snapshot import verify_runner_snapshot,RunnerSnapshotError


def snapshot(tmp_path,mode='empty'):
    runner=create(tmp_path)
    if mode=='accepted':runner.record_node_result(result(runner.next_action()))
    if mode in {'stopped','resolved'}:
        action=runner.next_action();capsule=runner.record_node_result(result(action,'RESULT_MISMATCH'))['audit_capsule']
        if mode=='resolved':
            runner.apply_external_decision(seal_external_decision({'schema_id':DECISION_SCHEMA,'decision':'REPEAT','audit_capsule_sha256':capsule['audit_capsule_sha256'],'root_run_id':runner.state['root_run_id'],'stopped_node_id':action['node_id'],'bound_result_sha256':ZERO,'bound_evidence_sha256':ONE}))
    root=tmp_path/'run'
    return [json.dumps(manifest(),indent=2).encode()+b'\n',(root/'runner_state.json').read_bytes(),
        {key:[p.read_bytes() for p in (root/directory).glob('*.json')] for key,directory in [('checkpoints','checkpoints'),('capsules','audit_capsules'),('decisions','external_decisions')]}]


def run(s,**kw):return verify_runner_snapshot(s[0],s[1],**s[2],**kw)


def test_snapshot_empty_state_no_scheduling(tmp_path):
    s=snapshot(tmp_path);before=s[1];out=run(s)
    assert out['status']=='RUNNER_SNAPSHOT_VERIFIED' and out['completed_node_ids']==[]
    assert s[1]==before and out['execution_authorized'] is False


def test_snapshot_accepted_checkpoint_and_raw_identity(tmp_path):
    s=snapshot(tmp_path,'accepted');out=run(s)
    assert len(out['completed_node_ids'])==1 and out['records_checked']==3
    assert out['inventory'][1]['content_ref']['sha256']!=out['state_sha256']


def test_snapshot_stopped_audit_is_preserved(tmp_path):
    s=snapshot(tmp_path,'stopped');out=run(s)
    assert out['scientific_acceptance']=='NOT_GRANTED' and out['completed_node_ids']==[]
    assert json.loads(s[1])['runner_status']=='WAITING_FOR_EXTERNAL_AUDIT'


def test_snapshot_resolved_decision_history_is_retained(tmp_path):
    s=snapshot(tmp_path,'resolved');out=run(s)
    assert len(s[2]['decisions'])==1 and out['execution_authorized'] is False
    assert out['records_checked']==4


def test_snapshot_missing_checkpoint_refused(tmp_path):
    s=snapshot(tmp_path,'accepted');s[2]['checkpoints']=[]
    with pytest.raises(ReplayDagRunnerError):run(s)


def test_snapshot_unindexed_checkpoint_refused(tmp_path):
    s=snapshot(tmp_path,'accepted');p=tmp_path/'empty';p.mkdir();empty=snapshot(p);empty[2]['checkpoints']=s[2]['checkpoints']
    with pytest.raises(ReplayDagRunnerError,match='ROLLBACK_OR_INTERRUPTED_ACCEPT'):run(empty)


def test_snapshot_checkpoint_tampering_refused(tmp_path):
    s=snapshot(tmp_path,'accepted');cp=json.loads(s[2]['checkpoints'][0]);cp['acceptance']['result_language']='CONFIRMED';s[2]['checkpoints'][0]=canonical_bytes(cp)
    with pytest.raises(ReplayDagRunnerError,match='content hash mismatch'):run(s)


def test_snapshot_missing_audit_capsule_refused(tmp_path):
    s=snapshot(tmp_path,'stopped');s[2]['capsules']=[]
    with pytest.raises(ReplayDagRunnerError):run(s)


def test_snapshot_missing_external_decision_refused(tmp_path):
    s=snapshot(tmp_path,'resolved');s[2]['decisions']=[]
    with pytest.raises(ReplayDagRunnerError,match='DECISION_INVENTORY_MISMATCH'):run(s)


def test_snapshot_duplicate_semantic_identity_refused(tmp_path):
    s=snapshot(tmp_path,'accepted');s[2]['checkpoints']*=2
    with pytest.raises(RunnerSnapshotError,match='DUPLICATE_SNAPSHOT'):run(s)


def test_snapshot_byte_and_record_budgets(tmp_path):
    s=snapshot(tmp_path,'accepted')
    with pytest.raises(RunnerSnapshotError,match='BYTE_BUDGET'):run(s,max_total_bytes=1)
    with pytest.raises(RunnerSnapshotError,match='RECORD_BUDGET'):run(s,max_records=2)


def test_snapshot_member_identity_cannot_be_path_or_wrong_schema(tmp_path):
    s=snapshot(tmp_path,'accepted');original=json.loads(s[2]['checkpoints'][0])
    for field,value in [('checkpoint_sha256','../escape'),('schema_id','UNKNOWN')]:
        cp=dict(original);cp[field]=value;s[2]['checkpoints'][0]=canonical_bytes(cp)
        with pytest.raises(RunnerSnapshotError,match='INVALID_SNAPSHOT_RECORD_IDENTITY'):run(s)
