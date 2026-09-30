import json,zipfile,io,hashlib
from pathlib import Path
import pytest
from test_storage_history_assembly import originals
from infinity_grid.storage_frontier_history import verify_frontier_history,FrontierHistoryError


def fixture():
    root=Path(__file__).resolve().parents[1];pins=json.loads((root/'infinity_grid/storage_history_pins.json').read_text())
    snapshots={};frontiers=[]
    for name,raw in sorted(originals().items()):
        assert hashlib.sha256(raw).hexdigest()==pins[name]
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            objs={n:z.read(n) for n in z.namelist() if n.endswith('.json')}
            state=next(b for n,b in objs.items() if n.endswith('replay_runner/runner_state.json'));seal=json.loads(state)['state_sha256']
            for n,b in objs.items():
                if '/replay_reference_data/records/' not in n:continue
                r=json.loads(b)
                if r['record_type']=='RESUME_FRONTIER' and r['payload']['runner_state_sha256']==seal:
                    frontiers.append(b);snapshots[r['record_id']]={'state_raw':state,**{k:[v for n,v in objs.items() if '/replay_runner/'+d+'/' in n] for k,d in [('checkpoints','checkpoints'),('capsules','audit_capsules'),('decisions','external_decisions')]}}
    return {'manifest_raw':(root/'infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_DAG_V9.json').read_bytes(),'frontiers':frontiers,'snapshots':snapshots,'current_frontier_id':'IGRD/O1_O3/FRONTIER/AUTOMATIC_PASS'}


def test_real_six_frontiers_bound_to_original_snapshots():
    s=fixture();out=verify_frontier_history(**s)
    assert out['status']=='FRONTIER_HISTORY_BOUND' and len(out['historical_frontier_ids'])==5
    assert sorted(len(v['snapshot']['completed_node_ids']) for v in out['frontiers'].values())==[5,10,15,20,25,30]
    assert not out['execution_authorized'] and not out['production_recovery_verified'] and not out['accepted_head_freshness_verified']
    assert out['scientific_acceptance']=='NOT_GRANTED'


def test_missing_historical_snapshot_refused():
    s=fixture();s['snapshots'].pop(next(iter(s['snapshots'])))
    with pytest.raises(FrontierHistoryError,match='INVENTORY_MISMATCH'):verify_frontier_history(**s)


def test_duplicate_frontier_refused():
    s=fixture();s['frontiers'].append(s['frontiers'][0])
    with pytest.raises(FrontierHistoryError,match='DUPLICATE_FRONTIER'):verify_frontier_history(**s)


def test_cross_stage_snapshot_substitution_refused():
    s=fixture();keys=list(s['snapshots']);s['snapshots'][keys[0]]=s['snapshots'][keys[1]]
    with pytest.raises(Exception,match='FRONTIER_SNAPSHOT_MISMATCH'):verify_frontier_history(**s)


def test_extra_snapshot_refused():
    s=fixture();s['snapshots']['OTHER']=next(iter(s['snapshots'].values()))
    with pytest.raises(FrontierHistoryError,match='INVENTORY_MISMATCH'):verify_frontier_history(**s)


def test_historical_frontier_cannot_label_later_states_as_history():
    s=fixture();s['current_frontier_id']=next(iter(s['snapshots']))
    with pytest.raises(FrontierHistoryError,match='PREFIX_MISMATCH|NOT_EARLIER'):verify_frontier_history(**s)


def test_frontier_history_original_inputs_unchanged():
    import copy
    s=fixture();before=copy.deepcopy(s);verify_frontier_history(**s);assert s==before


def test_frontier_history_shared_budget_enforced():
    s=fixture();out=verify_frontier_history(**s)
    with pytest.raises(ValueError,match='BUDGET'):verify_frontier_history(**s,max_total_bytes=out['bytes_checked']-1)
    with pytest.raises(FrontierHistoryError,match='INVALID_HISTORY_BUDGET'):verify_frontier_history(**s,max_total_bytes=True)
