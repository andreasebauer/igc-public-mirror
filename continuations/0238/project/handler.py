"""Fixed complete-class Q2 pilot; full historical public observer remains unbound."""
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
