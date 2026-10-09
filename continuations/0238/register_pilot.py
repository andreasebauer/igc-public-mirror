"""Freeze bounded complete-class Q2 probe; no full S2 observer claim or execution."""
from pathlib import Path
import json,hashlib,shutil,sys
B=Path(__file__).resolve().parent;W=B.parent;E=Path('/tmp/ig_engine0237');sys.path.insert(0,str(E))
from infinity_grid.canon import canonical_sha256
from infinity_grid.workflow_guard import preflight
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 r=json.loads((B/'REVIEW.json').read_text());assert r['pilot']['cases']==62 and r['pilot_complete_classes']==31
 project=B/'project';shutil.copytree(W/'continuation0235/project',project,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
 (project/'handler.py').write_text('''"""Fixed complete-class Q2 pilot; full historical public observer remains unbound."""
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound
def handler(stage,runtime):
    params=stage['execution']['parameters'];h=params['bindings'];i=stage['input_artifacts'];payload={k+'_path':i[k] for k in h};payload.update({k+'_sha256':v for k,v in h.items()});payload['source_binding']=params['source_binding']
    cases=read_bound(i['cases'],h['cases']);source={x['ordinal']:x for x in cases['cases']}
    task=TaskSpec(task_id='complete_class_Q2_pilot',task_kind='HISTORICAL_S1_COMPLETE_CLASS_Q2_PILOT',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
    runtime.run_content_indexed_generation(phase_id='complete_class_Q2_pilot',tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=62,max_task_occurrences=62,max_result_bytes=16777216,max_work_seconds=1800)
    data=[row['state'] for row in runtime.iter_generated_states(phase_id='complete_class_Q2_pilot')]
    if len(data)!=62 or {x['ordinal'] for x in data}!=set(source):raise ValueError('PILOT_CENSUS')
    groups={}
    for row in data:
        key=source[row['ordinal']]['record']['outcome_science_sha256'];q=row['public_projection']
        if key in groups and groups[key]!=q:raise ValueError('FULL_Q2_PAYLOAD_CLASS_SPLIT')
        groups[key]=q
    if set(groups)!=set(cases['class_keys']) or len(groups)!=31:raise ValueError('COMPLETE_CLASS_CENSUS')
    runtime.publish_json('complete_class_Q2_pilot',dict(cases=data,Q2_classes_checked=31,full_historical_public_observer_compared=False,G2_promotion=False))
    return ChainExecutionResult(result=dict(outcome='PASS_COMPLETE_CLASS_Q2_PILOT_V1',cases_checked=62,complete_Q2_classes_checked=31,G1_candidate_generation=0,G2_primary_realizations=62,G2_swap_checks=62,full_historical_public_observer_compared=False,master_slices=152,new_admissions=0,G2_promotion=False))
''')
 pins={str(p.relative_to(B)):sha(p) for p in sorted(project.rglob('*.py'))};pins['engine/uplift_structural.py']=sha(E/'infinity_grid/uplift_structural.py');pins['engine/workflow_guard.py']=sha(E/'infinity_grid/workflow_guard.py');binding=canonical_sha256(pins)
 s=json.loads((W/'continuation0237/SPEC.json').read_text());s['job_id']='MASTER.G2.COMPLETE.CLASS.Q2.PILOT.0239';s['question']=dict(stage_id=s['job_id'],description='Fixed62 complete31 stored collision classes: historical deterministic realization and exact full Q2 payload equality, excluding unbound full S2 observer',outcomes=['PASS_COMPLETE_CLASS_Q2_PILOT_V1'],stopping_rule='Stop first source/input/DAG/projected/D4/bridge witness/capacity/Q2/repaired/swap/class-payload or resource mismatch; no admission or G2 promotion')
 s['project_source']=str(project.resolve());s['execution']['parameters']['bindings']['cases']=sha(B/'PILOT_CASES.json');s['execution']['parameters']['source_binding']=binding
 for x in s['inputs']:
  if x['logical_name']=='cases':x.update(path=str((B/'PILOT_CASES.json').resolve()),sha256=sha(B/'PILOT_CASES.json'))
 s['output_contract']['claim']='Bounded complete-class strict-public Q2 comparison only; no full historical public observer, admission or G2 promotion'
 s['output_contract']['result_checks']=[dict(pointer='/'+k,equals=v) for k,v in dict(outcome='PASS_COMPLETE_CLASS_Q2_PILOT_V1',cases_checked=62,complete_Q2_classes_checked=31,G1_candidate_generation=0,G2_primary_realizations=62,G2_swap_checks=62,full_historical_public_observer_compared=False,master_slices=152,new_admissions=0,G2_promotion=False).items()]
 (B/'NEXT_SPEC.json').write_text(json.dumps(s,indent=2)+'\n')
 pilot=json.loads((B/'PILOT_CASES.json').read_text());first=pilot['class_keys'][0];dry=[x['ordinal'] for x in pilot['cases'] if x['record']['outcome_science_sha256']==first]
 prereg=dict(schema_id='IG_COMPLETE_CLASS_Q2_PILOT_PREREG_V1',status='FROZEN_BEFORE_DRY_AND_OFFICIAL_EXECUTION',source_sha256=pins,source_binding=binding,input_sha256=s['execution']['parameters']['bindings'],case_selection_sha256=sha(B/'PILOT_CASES.json'),case_count=62,complete_classes=31,case_ordinals=[x['ordinal'] for x in pilot['cases']],dry_case_ordinals=dry,dry_scope='First complete lexicographic class, all members; assess resource feasibility using unchanged scientific comparison functions',dry_official_overlap_declared=True,blind_holdout_claim=False,official_cases_executed=0,historical_full_observer_gate='BLOCKED_UNTIL_EXACT_SERIALIZER_SOURCE_BOUND',historical_full_observer_comparison=False,retained_prior_Q2_scope='Existing62 cases cover none of the selected collision classes; no prior-case reuse',memory_budget_bytes=4294967296,workers=1,resource_feasibility_for_diverse62='NOT_YET_ESTABLISHED; require bounded dry gate before official run',master_slices=152,new_admissions=0,G2_promotion=False,stopping_rule=s['question']['stopping_rule'])
 (B/'NEXT_PREREGISTRATION.json').write_text(json.dumps(prereg,indent=2)+'\n')
 with __import__('tempfile').TemporaryDirectory(prefix='ig_preflight0238_') as t:
  T=Path(t);shutil.copytree(E/'infinity_grid',T/'infinity_grid');shutil.copytree(project,T/'project');gate=preflight(T,[T/'project/handler.py',T/'project/worker.py'])
 assert len(gate['modules'])==22
 (B/'NEXT_PREFLIGHT.json').write_text(json.dumps(dict(status='PASS_COMPLETE_CAPTURED_SOURCE_PREFLIGHT',modules_checked=22,gate=gate,official_cases_executed=0,resource_feasibility='PENDING_DRY_GATE',historical_full_observer='BLOCKED_EXACT_SOURCE_BINDING'),indent=2)+'\n');print('PASS next62 complete31-class Q2 registration and native preflight; resource dry and full historical observer gates remain explicit')
if __name__=='__main__':main()
