from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from infinity_grid.replay_o1_o3_executor import execute_o1_o3, NODE_IDS
from infinity_grid.replay_execution_accounting import attach_execution_accounting
from infinity_grid.v05_stage_runtime import ReplayRootPaused, StageRuntimeError
from test_p2b_registered_replay_root import fake_runtime, parameters, MANIFEST
from test_p4_l0_pilot import parameters as p4_parameters

ROOT=Path(__file__).resolve().parents[1]
CATALOGUE=ROOT/'infinity_grid/resources/replay/L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json'


def test_all_nine_o1_o3_semantic_hashes_and_requested_observed_accounting():
    manifest=json.loads(MANIFEST.read_text());catalogue=json.loads(CATALOGUE.read_text());total=0
    for node in manifest['nodes']:
        if node['canonical_id'] not in NODE_IDS:continue
        result=execute_o1_o3(node=node,catalogue=catalogue,repository_root=ROOT,attempt=1)
        historical=[r for r in result['reference_records'] if r['record_type']=='SRCF_EVIDENCE' and r['epistemic_status']=='HISTORICAL']
        observed=[r for r in result['reference_records'] if r['record_type']=='SRCF_EVIDENCE' and r['epistemic_status']=='REPLAYED']
        assert len(historical)==len(observed)
        for a,b in zip(historical,observed):assert a['payload']['result_identity']==b['payload']['result_identity'];total+=1
        rows=result['assertion_execution_accounting']
        assert len(rows)==len(node['assertion_mapping_ids'])
        assert all(not r['counts_toward_full_fresh_generation'] for r in rows)
        assert all(r['observed_execution_class']!='FRESH_RECOMPUTE' for r in rows)
    assert total==9


@pytest.mark.parametrize('field,value,error',[
    ('root_run_id','WRONG_ROOT','REPLAY_RESUME_ROOT_RUN_ID_MISMATCH'),
    ('dataset_root_sha256','a'*64,'REPLAY_RESUME_DATASET_ROOT_MISMATCH')])
def test_resume_rejects_different_request_before_mutating_state(tmp_path,monkeypatch,field,value,error):
    monkeypatch.setattr('infinity_grid.v05_origin_guard.require_registered_output',lambda *a,**k:None)
    runtime,pubs=fake_runtime(tmp_path);params=parameters()
    with pytest.raises(ReplayRootPaused):runtime.run_registered_replay_root(params,{'compiled_replay_manifest':str(MANIFEST)})
    state=runtime._root/'replay_runner/runner_state.json';before=state.read_bytes();count=len(pubs)
    params[field]=value
    with pytest.raises(StageRuntimeError,match=error):runtime.run_registered_replay_root(params,{'compiled_replay_manifest':str(MANIFEST)})
    assert state.read_bytes()==before and len(pubs)==count


def test_catalogue_dag_mismatch_even_when_its_file_hash_is_valid(tmp_path,monkeypatch):
    monkeypatch.setattr('infinity_grid.v05_origin_guard.require_registered_output',lambda *a,**k:None)
    runtime,pubs=fake_runtime(tmp_path);params=p4_parameters()
    catalogue=json.loads(CATALOGUE.read_text());catalogue['historical_assertion_mappings'][0]['expected_outcome']='ALTERED'
    changed=tmp_path/'catalogue.json';changed.write_text(json.dumps(catalogue))
    params['reference_catalogue_file_sha256']=hashlib.sha256(changed.read_bytes()).hexdigest()
    with pytest.raises(StageRuntimeError,match='REPLAY_REFERENCE_CATALOGUE_DAG_BINDING'):
        runtime.run_registered_replay_root(params,{'compiled_replay_manifest':str(MANIFEST),'replay_reference_catalogue':str(changed)})
    assert not (runtime._root/'replay_runner/runner_state.json').exists() and not pubs
