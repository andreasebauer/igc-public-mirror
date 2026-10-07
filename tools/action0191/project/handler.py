"""Native resumable G1 exact ancestry, followed by independent restore comparison."""
import hashlib,json
from pathlib import Path
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound

def handler(stage,runtime):
 inputs=stage['input_artifacts'];config=stage['execution']['parameters'];bindings=config['bindings']
 plan=read_bound(inputs['recipe_plan'],bindings['recipe_plan'])
 if plan['terminal_level']!=100 or plan['candidate_template_count']!=193:raise ValueError('SCOPE')
 common={n+'_path':inputs[n] for n in ('pairs','recipes','reference')}
 common.update({n+'_sha256':bindings[n] for n in ('pairs','recipes','reference')})
 parent=None
 for level in range(7,101):
  payload=dict(common,mode='SEED' if level==7 else 'LEVEL',level=level)
  if parent:payload.update(parent_path=parent['path'],parent_sha256=parent['raw_sha256'])
  phase='g1_exact_depth_'+str(level)
  task=TaskSpec(task_id=phase,binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
  runtime.run_content_indexed_generation(phase_id=phase,tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=1,max_task_occurrences=1,max_result_bytes=8*1024*1024,max_work_seconds=300.0)
  rows=list(runtime.iter_generated_states(phase_id=phase))
  if len(rows)!=1:raise ValueError('PHASE_STATE_COUNT')
  state=rows[0]['state'];expected=193 if level==100 else 24
  if state['level']!=level or len(state['dag']['roots'])!=expected or state['selected_count']!=expected:raise ValueError('PHASE_CENSUS')
  parent=runtime.publish_json(phase,state)
  parent['raw_sha256']=hashlib.sha256(Path(parent['path']).read_bytes()).hexdigest()
 payload=dict(common,mode='RESTORE',level=100,parent_path=parent['path'],parent_sha256=parent['raw_sha256'])
 task=TaskSpec(task_id='cold_restore',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
 runtime.run_content_indexed_generation(phase_id='g1_exact_cold_restore',tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=1,max_task_occurrences=1,max_result_bytes=8*1024*1024,max_work_seconds=300.0)
 rows=list(runtime.iter_generated_states(phase_id='g1_exact_cold_restore'))
 if len(rows)!=1 or rows[0]['state']['interfaces_checked']!=193:raise ValueError('RESTORE_COMPARISON')
 return ChainExecutionResult(result={'outcome':'PASS','terminal_roots':193,'interfaces_checked':193,'cold_restore':True,'master_scientific_slices':151,'new_admissions':0,'Q2_payload_generated':False,'exact_terminal_artifact':parent,'scope':'BOUND_HISTORICAL_SEED_G1_EXACT_RECONSTRUCTION'})
