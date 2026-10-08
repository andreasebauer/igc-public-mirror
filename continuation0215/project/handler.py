"""Native bounded historical seed/8–14qualification; no prior replay state reuse."""
import hashlib
from pathlib import Path
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound
from .partitions import load_parent

def ref(p):return {'path':p['path'],'sha256':hashlib.sha256(Path(p['path']).read_bytes()).hexdigest()}
def run(runtime,c,name,payload):
 task=TaskSpec(task_id=name,task_kind='HISTORICAL_G1_QUALIFICATION',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
 runtime.run_content_indexed_generation(phase_id=name,tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=1,max_task_occurrences=1,max_result_bytes=c['result_transport_budget_bytes'],max_work_seconds=c['task_work_budget_seconds'])
 rows=list(runtime.iter_generated_states(phase_id=name))
 if len(rows)!=1:raise ValueError('PHASE_COUNT')
 return rows[0]['state']
def handler(stage,runtime):
 i=stage['input_artifacts'];c=stage['execution']['parameters'];h=c['bindings']
 for key in ('recipe_plan','pairs','recipes','reference','anchor','bootstrap'):read_bound(i[key],h[key])
 base={'path':i['bootstrap'],'sha256':h['bootstrap']};parent=base;parts=[]
 load_parent(base['path'],base['sha256'])
 common={k+'_path':i[k] for k in ('pairs','recipes','reference','anchor')};common.update({k+'_sha256':h[k] for k in ('pairs','recipes','reference','anchor')})
 for level in range(21,27):
  data=run(runtime,c,'historical_g1_depth_'+str(level),dict(common,mode='LEVEL',level=level,parent_path=parent['path'],parent_sha256=parent['sha256']))
  if data['level']!=level or data['candidate_count']!=193 or data['selected_count']!=24 or len(data['roots'])!=24:raise ValueError('PHASE_SCOPE')
  part=ref(runtime.publish_json('historical_g1_depth_'+str(level),data));parts.append(part)
  manifest={'schema_id':'IG_G1_PARTITION_MANIFEST_V1','base':base,'partitions':list(parts),'level':level,'roots':data['roots'],'science_sha256':data['science_sha256'],'selected_count':24,'candidate_count':193};parent=ref(runtime.publish_json('historical_g1_depth_'+str(level)+'_manifest',manifest));load_parent(parent['path'],parent['sha256'])
 if data['historical_depth26_anchor']!='PASS24_EXACT_ROOT_SKIN_CAPS':raise ValueError('ANCHOR_RESULT')
 return ChainExecutionResult(result={'outcome':'PASS_HISTORICAL_DEPTH26_CONTINUATION','completed_depth':26,'roots':24,'historical_depth26_anchor':'PASS24_EXACT_ROOT_SKIN_CAPS','candidate_build_calls':1158,'master_scientific_slices':151,'new_admissions':0,'Q2_payload_generated':False,'terminal_comparison':'NOT_RUN','exact_manifest':parent})
