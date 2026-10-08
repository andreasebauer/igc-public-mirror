"""Terminal native generation and independent193-root interface restore."""
import hashlib
from pathlib import Path
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound
from .partitions import load_parent

def ref(p):return {'path':p['path'],'sha256':hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()}

def phase(runtime,config,name,payload):
 task=TaskSpec(task_id=name,task_kind='G1_EXACT_TERMINAL',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
 runtime.run_content_indexed_generation(phase_id=name,tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=1,max_task_occurrences=1,max_result_bytes=config['result_transport_budget_bytes'],max_work_seconds=config['task_work_budget_seconds'])
 rows=list(runtime.iter_generated_states(phase_id=name))
 if len(rows)!=1:raise ValueError('PHASE_COUNT')
 return rows[0]['state']

def handler(stage,runtime):
 inputs=stage['input_artifacts'];config=stage['execution']['parameters'];bindings=config['bindings']
 plan=read_bound(inputs['recipe_plan'],bindings['recipe_plan'])
 if plan['terminal_level']!=100 or plan['candidate_template_count']!=193 or config['start_depth']!=100 or config['stop_depth']!=100:raise ValueError('TERMINAL_SCOPE')
 base={'path':inputs['bootstrap'],'sha256':bindings['bootstrap']};initial=load_parent(base['path'],base['sha256'])
 if initial['level']!=99 or initial['selected_count']!=24 or initial['dag']['science_sha256']!=config['bootstrap_science_sha256']:raise ValueError('BOOTSTRAP_SCOPE')
 common={n+'_path':inputs[n] for n in ('pairs','recipes','reference')};common.update({n+'_sha256':bindings[n] for n in ('pairs','recipes','reference')})
 payload=dict(common,mode='LEVEL',level=100,parent_path=base['path'],parent_sha256=base['sha256'])
 data=phase(runtime,config,'g1_partition_depth_100',payload)
 if data['level']!=100 or data['candidate_count']!=193 or data['selected_count']!=193 or len(data['roots'])!=193 or data['parent_science_sha256']!=initial['dag']['science_sha256']:raise ValueError('TERMINAL_CENSUS')
 part=ref(runtime.publish_json('g1_partition_depth_100',data))
 manifest={'schema_id':'IG_G1_PARTITION_MANIFEST_V1','base':base,'partitions':[part],'level':100,'roots':data['roots'],'science_sha256':data['science_sha256'],'selected_count':193,'candidate_count':193}
 parent=ref(runtime.publish_json('g1_partition_depth_100_manifest',manifest));load_parent(parent['path'],parent['sha256'])
 restored=phase(runtime,config,'g1_terminal_interface_restore',dict(common,mode='RESTORE',level=100,parent_path=parent['path'],parent_sha256=parent['sha256']))
 if restored!={'level':100,'restored_roots':193,'interfaces_checked':193,'dag_science_sha256':data['science_sha256']}:raise ValueError('RESTORE_SCOPE')
 comparison=ref(runtime.publish_json('g1_terminal_interface_comparison',restored))
 return ChainExecutionResult(result={'outcome':'PASS_TERMINAL_INTERFACE_COMPARISON','completed_depth':100,'roots':193,'interfaces_checked':193,'master_scientific_slices':151,'new_admissions':0,'Q2_payload_generated':False,'terminal_comparison':'PASS','exact_manifest':parent,'interface_comparison':comparison})
