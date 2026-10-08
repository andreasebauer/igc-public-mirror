"""Restore-only native comparison from independently audited terminal193-root DAG."""
import hashlib
from pathlib import Path
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound

def handler(stage,runtime):
 inputs=stage['input_artifacts'];config=stage['execution']['parameters'];bindings=config['bindings']
 parent=read_bound(inputs['bootstrap'],bindings['bootstrap'])
 if parent['level']!=100 or parent['candidate_count']!=193 or parent['selected_count']!=193 or len(parent['dag']['roots'])!=193 or parent['dag']['science_sha256']!=config['bootstrap_science_sha256']:raise ValueError('TERMINAL_BOOTSTRAP_SCOPE')
 payload={'mode':'RESTORE','level':100,'parent_path':inputs['bootstrap'],'parent_sha256':bindings['bootstrap'],'reference_path':inputs['reference'],'reference_sha256':bindings['reference']}
 name='g1_terminal_interface_restore';task=TaskSpec(task_id=name,task_kind='G1_EXACT_TERMINAL_RESTORE',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
 runtime.run_content_indexed_generation(phase_id=name,tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=1,max_task_occurrences=1,max_result_bytes=config['result_transport_budget_bytes'],max_work_seconds=config['task_work_budget_seconds'])
 rows=list(runtime.iter_generated_states(phase_id=name))
 expected={'level':100,'restored_roots':193,'interfaces_checked':193,'dag_science_sha256':parent['dag']['science_sha256']}
 if len(rows)!=1 or rows[0]['state']!=expected:raise ValueError('RESTORE_SCOPE')
 p=runtime.publish_json('g1_terminal_interface_comparison',expected);comparison={'path':p['path'],'sha256':hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()}
 return ChainExecutionResult(result={'outcome':'PASS_TERMINAL_INTERFACE_COMPARISON','completed_depth':100,'roots':193,'interfaces_checked':193,'master_scientific_slices':151,'new_admissions':0,'Q2_payload_generated':False,'terminal_comparison':'PASS','candidate_build_calls':0,'interface_comparison':comparison})
