import json
from copy import deepcopy
import pytest
from test_p2a_replay_dag_runner import manifest,result
from infinity_grid.canon import canonical_bytes,canonical_sha256
from infinity_grid.replay_dag_runner import ReplayDagRunner
from infinity_grid.replay_l0_executor import _record
from infinity_grid.replay_reference_data import empty_manifest,seal_reference_record
from infinity_grid.storage_snapshot_results import verify_snapshot_results
from infinity_grid.storage_runner_snapshot import RunnerSnapshotError


def record(rid,typ,payload,dependencies=()):
    return _record(rid,typ,payload,dependencies=tuple(dependencies),
        provenance_class='FINITE_COMPUTATIONAL_OBSERVATION',source_hashes=[{'ref':'fixture','sha256':'0'*64}])


def fixture(tmp_path,profile='SINGLE_REPLAY_RECORD_V1',stop=False,empty=False):
    m=manifest();root=tmp_path/'runner';d=empty_manifest()
    runner=ReplayDagRunner.create(m,root,root_run_id='RESULT-FIXTURE',dataset_root={'state':'EMPTY','sha256':d['manifest_sha256']})
    bindings={}
    if not empty:
        action=runner.next_action();node=action['node_id'];eq={'mode':'CANONICAL_JSON','cardinality_semantics':'ORDERED','compression':'EXACT','observer':'FIXTURE','certificate_sha256':None}
        historical=[];replay=[];records=[]
        for i in range(1 if profile=='SINGLE_REPLAY_RECORD_V1' else 2):
            for role,ids in [('H',historical),('R',replay)]:
                rid=f'IGRD/L0/EVIDENCE/{role}{i}';ids.append(rid)
                records.append(record(rid,'SRCF_EVIDENCE',{'series':'S','obligation_id':node,'result_identity':'1'*64,'equality_contract':eq,'evidence_mode':'SOURCE_INTEGRITY_ONLY','outcome':'REPRODUCED'}))
        cp=record('IGRD/L0/COMPARISON/FIXTURE','COMPARISON',{'obligation_id':node,'historical_record_ids':historical,'replay_record_ids':replay,'equality_contract':eq,'outcome':'MISMATCH' if stop else 'REPRODUCED','qualification_ids':[]},historical+replay)
        seals={r['record_id']:r['record_sha256'] for r in records}
        ids=replay if profile=='ORDERED_REPLAY_SEALS_V1' else [rid for pair in zip(historical,replay) for rid in pair]
        digest=seals[replay[0]] if profile=='SINGLE_REPLAY_RECORD_V1' else canonical_sha256([seals[rid] for rid in ids])
        records.append(cp);res=result(action,'RESULT_MISMATCH' if stop else 'EXACT_HISTORICAL_REPLAY_AUTHORIZED');res.update(result_sha256=cp['record_sha256'],evidence_sha256=digest)
        saved=runner.record_node_result(res);owner=saved['audit_capsule']['audit_capsule_sha256'] if stop else saved['checkpoint']['checkpoint_sha256']
        bindings[owner]={'profile':profile,'records':[json.dumps(r,indent=2).encode() for r in records]}
    return {'manifest_raw':canonical_bytes(m),'state_raw':(root/'runner_state.json').read_bytes(),'dataset_raw':json.dumps(d,indent=2).encode(),'bindings':bindings,
        'checkpoints':[p.read_bytes() for p in (root/'checkpoints').glob('*.json')],
        'capsules':[p.read_bytes() for p in (root/'audit_capsules').glob('*.json')]}


def binding(s):return next(iter(s['bindings'].values()))


def test_result_single_record_and_empty_root_bound(tmp_path):
    out=verify_snapshot_results(**fixture(tmp_path));assert out['status']=='SNAPSHOT_RESULT_IDENTITIES_VERIFIED'
    assert out['execution_authorized'] is False and out['dependency_closure_verified'] is False


def test_result_ordered_replay_seals(tmp_path):
    assert len(verify_snapshot_results(**fixture(tmp_path,'ORDERED_REPLAY_SEALS_V1'))['bindings_verified'])==1


def test_result_ordered_historical_replay_pairs(tmp_path):
    assert len(verify_snapshot_results(**fixture(tmp_path,'ORDERED_PAIRED_SEALS_V1'))['bindings_verified'])==1


def test_result_active_audit_capsule_bound(tmp_path):
    s=fixture(tmp_path,stop=True);out=verify_snapshot_results(**s)
    assert out['snapshot']['completed_node_ids']==[] and len(out['bindings_verified'])==1


def test_result_empty_snapshot_needs_no_result_bindings(tmp_path):
    out=verify_snapshot_results(**fixture(tmp_path,empty=True));assert out['bindings_verified']==[]


def test_result_missing_owner_binding_refused(tmp_path):
    s=fixture(tmp_path);s['bindings']={}
    with pytest.raises(RunnerSnapshotError,match='BINDING_INVENTORY'):verify_snapshot_results(**s)


def test_result_extra_owner_binding_refused(tmp_path):
    s=fixture(tmp_path);s['bindings']['0'*64]=deepcopy(binding(s))
    with pytest.raises(RunnerSnapshotError,match='BINDING_INVENTORY'):verify_snapshot_results(**s)


def test_result_missing_comparison_refused(tmp_path):
    s=fixture(tmp_path);binding(s)['records'].pop()
    with pytest.raises(RunnerSnapshotError,match='COMPARISON_BINDING'):verify_snapshot_results(**s)


def test_result_missing_operand_refused(tmp_path):
    s=fixture(tmp_path);binding(s)['records'].pop(0)
    with pytest.raises(RunnerSnapshotError,match='EVIDENCE_BINDING'):verify_snapshot_results(**s)


def test_result_altered_record_seal_refused(tmp_path):
    s=fixture(tmp_path);b=binding(s);r=json.loads(b['records'][0]);r['payload']['outcome']='OPEN';b['records'][0]=canonical_bytes(r)
    with pytest.raises(Exception,match='hash mismatch'):verify_snapshot_results(**s)


def test_result_other_obligation_operand_refused(tmp_path):
    s=fixture(tmp_path);b=binding(s);r=json.loads(b['records'][0]);r['payload']['obligation_id']='OTHER';b['records'][0]=canonical_bytes(seal_reference_record(r))
    with pytest.raises(RunnerSnapshotError,match='EVIDENCE_BINDING'):verify_snapshot_results(**s)


def test_result_duplicate_record_refused(tmp_path):
    s=fixture(tmp_path);b=binding(s);b['records'].append(b['records'][0])
    with pytest.raises(RunnerSnapshotError,match='DUPLICATE_RESULT_RECORD'):verify_snapshot_results(**s)


def test_result_wrong_evidence_profile_refused(tmp_path):
    s=fixture(tmp_path);binding(s)['profile']='ORDERED_REPLAY_SEALS_V1'
    with pytest.raises(RunnerSnapshotError,match='EVIDENCE_SEAL_MISMATCH'):verify_snapshot_results(**s)


def test_result_unknown_profile_refused(tmp_path):
    s=fixture(tmp_path);binding(s)['profile']='RAW_SHA256'
    with pytest.raises(RunnerSnapshotError,match='UNSUPPORTED_RESULT_BINDING_PROFILE'):verify_snapshot_results(**s)


def test_result_initial_dataset_requires_exact_frozen_empty_manifest(tmp_path):
    s=fixture(tmp_path);d=json.loads(s['dataset_raw']);d['status']='POPULATED';s['dataset_raw']=canonical_bytes(d)
    with pytest.raises(RunnerSnapshotError,match='EMPTY_INITIAL_DATASET'):verify_snapshot_results(**s)


def test_result_initial_dataset_state_binding_refused(tmp_path):
    s=fixture(tmp_path);state=json.loads(s['state_raw']);state['dataset_root']['sha256']='0'*64;state.pop('state_sha256');state['state_sha256']=canonical_sha256(state);s['state_raw']=canonical_bytes(state)
    with pytest.raises(RunnerSnapshotError,match='EMPTY_INITIAL_DATASET'):verify_snapshot_results(**s)


def test_result_shared_record_and_byte_budgets(tmp_path):
    s=fixture(tmp_path);out=verify_snapshot_results(**s)
    with pytest.raises(RunnerSnapshotError,match='BYTE_BUDGET'):verify_snapshot_results(**s,max_total_bytes=out['bytes_checked']-1)
    with pytest.raises(RunnerSnapshotError,match='RECORD_BUDGET'):verify_snapshot_results(**s,max_records=out['records_checked']-1)


def test_result_raw_identity_distinct_from_record_seal(tmp_path):
    out=verify_snapshot_results(**fixture(tmp_path));row=out['bindings_verified'][0]['records'][0]
    assert row['content_ref']['sha256']!=row['record_sha256']
    assert out['scientific_acceptance']=='NOT_GRANTED'
