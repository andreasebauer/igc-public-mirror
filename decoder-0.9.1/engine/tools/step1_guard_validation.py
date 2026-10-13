#!/usr/bin/env python3
"""Engineering root integration check; native admission/origin is not bypassed in production.

This tool explicitly substitutes only the output-origin and execution-authority
checks to exercise controller-owned root logic on a new temporary state directory.
It executes source-integrity checks and reads historical results, not science.
"""


def _require_integrity(condition, message=None):
    """Acceptance invariant retained under optimized Python."""
    if not condition:
        if message is None:
            raise AssertionError()
        raise AssertionError(message())

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from infinity_grid.canon import canonical_sha256
from infinity_grid.replay_o1_o3_executor import execute_o1_o3,NODE_IDS
from infinity_grid.replay_l0_executor import NODE_EXECUTOR_BINDINGS
from infinity_grid.replay_reference_data import empty_manifest
from infinity_grid.v05_stage_runtime import StageScienceRuntime,StageRuntimeError,ReplayRootPaused


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    root=Path(__file__).resolve().parents[1];res=root/'infinity_grid/resources/replay';mp=res/'L0_UPWARD_SRCF_ASSERTION_DAG_V9.json';cp=res/'L0_UPWARD_SRCF_ASSERTION_SOURCE_V9.json';manifest=json.loads(mp.read_text());catalogue=json.loads(cp.read_text())
    parameters={'operation':'EXECUTE_L0_PILOT_P4','manifest_input':'compiled_replay_manifest','manifest_file_sha256':hashlib.sha256(mp.read_bytes()).hexdigest(),'manifest_dag_sha256':manifest['dag_sha256'],'reference_catalogue_input':'replay_reference_catalogue','reference_catalogue_file_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'dataset_root_sha256':empty_manifest()['manifest_sha256'],'root_run_id':'STEP1-NEW-GUARD-ROOT','authorized_through_layer':'G8','next_wait_layer':'GLOBAL','node_executor_bindings':NODE_EXECUTOR_BINDINGS,'pilot_fault_injection':'NONE'}
    artifacts={'compiled_replay_manifest':str(mp),'replay_reference_catalogue':str(cp)};guards=[]
    with TemporaryDirectory(prefix='step1_guards_') as temp, patch('infinity_grid.v05_origin_guard.require_registered_output',lambda *a,**k:None):
        runtime=object.__new__(StageScienceRuntime);runtime._root=Path(temp)/'output';runtime._root.mkdir();runtime._require_execution=lambda:None;pubs=[]
        runtime.publish_json=lambda name,obj:pubs.append(obj) or {'logical_name':name,'sha256':canonical_sha256(obj)}
        bad=deepcopy(catalogue);bad['historical_assertion_mappings'][0]['expected_outcome']='ALTERED';bp=Path(temp)/'changed.json';bp.write_text(json.dumps(bad));params=dict(parameters,reference_catalogue_file_sha256=hashlib.sha256(bp.read_bytes()).hexdigest())
        try:runtime.run_registered_replay_root(params,dict(artifacts,replay_reference_catalogue=str(bp)))
        except StageRuntimeError as e:guards.append({'probe':'valid-file-hash wrong catalogue/DAG pair','rejection':str(e),'before_state_write':not (runtime._root/'replay_runner').exists()})
        else:raise AssertionError('wrong catalogue accepted')
        first=runtime.run_registered_replay_root(parameters,artifacts);sp=runtime._root/'replay_runner/runner_state.json';before=sp.read_bytes();pubcount=len(pubs)
        try:runtime.run_registered_replay_root(dict(parameters,root_run_id='DIFFERENT-ROOT'),artifacts)
        except StageRuntimeError as e:guards.append({'probe':'stored/requested root mismatch','rejection':str(e),'state_unchanged':sp.read_bytes()==before,'publication_unchanged':len(pubs)==pubcount})
        else:raise AssertionError('wrong root accepted')
        resumed=runtime.run_registered_replay_root(parameters,artifacts);_require_integrity(sp.read_bytes()==before)
        old=first['assertion_execution_accounting'];coverage=old
    semantic=[]
    for node in manifest['nodes']:
        if node['canonical_id'] not in NODE_IDS:continue
        result=execute_o1_o3(node=node,catalogue=catalogue,repository_root=root,attempt=1)
        rows=result['reference_records'];hs=[r for r in rows if r['record_type']=='SRCF_EVIDENCE' and r['epistemic_status']=='HISTORICAL'];rs=[r for r in rows if r['record_type']=='SRCF_EVIDENCE' and r['epistemic_status']=='REPLAYED']
        for h,r in zip(hs,rs):semantic.append({'historical_record_id':h['record_id'],'replay_record_id':r['record_id'],'historical_identity':h['payload']['result_identity'],'observed_identity':r['payload']['result_identity'],'equal':h['payload']['result_identity']==r['payload']['result_identity']})
    _require_integrity(len(semantic)==9 and all(r['equal'] for r in semantic))
    data={'schema_id':'IG_STEP1_ROOT_GUARD_VALIDATION_V1','status':'PASS','guards':guards,'same_root_resume_byte_exact':True,'o1_o3_semantic_pairs':semantic,'assertion_execution_accounting':coverage,'science_jobs_executed':False,'scope':'ROOT_ENGINEERING_HARNESS; PRODUCTION_ORIGIN_AND_ADMISSION_NOT_EXERCISED','historical_attempts_resumed':False}
    (a.output/'ROOT_GUARD_VALIDATION.json').write_text(json.dumps(data,indent=2)+'\n');print('Root engineering guards PASS; nine semantic comparisons PASS.')

if __name__=='__main__':main()
