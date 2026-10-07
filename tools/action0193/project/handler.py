"""Bounded native continuation from saved depth24, no earlier generation."""
import hashlib
from pathlib import Path
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound
from .partitions import load_parent,read

def ref(p):return {'path':p['path'],'sha256':hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()}

def handler(stage,runtime):
 inputs=stage['input_artifacts'];config=stage['execution']['parameters'];bindings=config['bindings']
 plan=read_bound(inputs['recipe_plan'],bindings['recipe_plan'])
 if plan['terminal_level']!=100 or plan['candidate_template_count']!=193:raise ValueError('SCOPE')
 base={'path':inputs['bootstrap'],'sha256':bindings['bootstrap']};parent=base;parts=[]
 initial=load_parent(base['path'],base['sha256'])
 if initial['level']!=24 or initial['dag']['science_sha256']!=config['bootstrap_science_sha256']:raise ValueError('BOOTSTRAP_SCOPE')
 common={n+'_path':inputs[n] for n in ('pairs','recipes','reference')};common.update({n+'_sha256':bindings[n] for n in ('pairs','recipes','reference')})
 for level in range(25,31):
  payload=dict(common,mode='LEVEL',level=level,parent_path=parent['path'],parent_sha256=parent['sha256'])
  phase='g1_partition_depth_'+str(level)
  task=TaskSpec(task_id=phase,task_kind='G1_EXACT_DELTA',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
  runtime.run_content_indexed_generation(phase_id=phase,tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=1,max_task_occurrences=1,max_result_bytes=8*1024*1024,max_work_seconds=300.0)
  rows=list(runtime.iter_generated_states(phase_id=phase))
  if len(rows)!=1:raise ValueError('PHASE_COUNT')
  data=rows[0]['state'];prev=load_parent(parent['path'],parent['sha256'])
  if data['level']!=level or data['selected_count']!=24 or len(data['roots'])!=24 or data['parent_science_sha256']!=prev['dag']['science_sha256']:raise ValueError('PHASE_SCOPE')
  part=ref(runtime.publish_json(phase,data));parts.append(part)
  manifest={'schema_id':'IG_G1_PARTITION_MANIFEST_V1','base':base,'partitions':list(parts),'level':level,'roots':data['roots'],'science_sha256':data['science_sha256'],'selected_count':24,'candidate_count':193}
  parent=ref(runtime.publish_json(phase+'_manifest',manifest));load_parent(parent['path'],parent['sha256'])
 return ChainExecutionResult(result={'outcome':'PASS_BOUNDED_CONTINUATION','completed_depth':30,'roots':24,'master_scientific_slices':151,'new_admissions':0,'Q2_payload_generated':False,'terminal_comparison':'NOT_RUN','exact_manifest':parent})
