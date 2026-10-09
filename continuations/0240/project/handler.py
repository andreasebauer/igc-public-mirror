"""One frozen whole-class chunk per native capture; full S2 observer unbound."""
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound
def handler(stage,runtime):
    params=stage['execution']['parameters'];h=params['bindings'];i=stage['input_artifacts'];payload={k+'_path':i[k] for k in h};payload.update({k+'_sha256':v for k,v in h.items()});payload['source_binding']=params['source_binding']
    cases=read_bound(i['cases'],h['cases']);source={x['ordinal']:x for x in cases['cases']}
    count=params['case_count']; class_count=params['class_count']; chunk=params['chunk_id']
    if not 1<=count<=62 or len(source)!=count or len(cases['class_keys'])!=class_count:raise ValueError('REGISTERED_CHUNK_CENSUS')
    task=TaskSpec(task_id=chunk,task_kind='HISTORICAL_S1_COMPLETE_CLASS_Q2_PILOT',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
    runtime.run_content_indexed_generation(phase_id=chunk,tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=count,max_task_occurrences=count,max_result_bytes=16777216,max_work_seconds=1800)
    data=[row['state'] for row in runtime.iter_generated_states(phase_id=chunk)]
    if len(data)!=count or {x['ordinal'] for x in data}!=set(source):raise ValueError('PILOT_CENSUS')
    groups={}
    for row in data:
        key=source[row['ordinal']]['record']['outcome_science_sha256'];q=row['public_projection']
        if key in groups and groups[key]!=q:raise ValueError('FULL_Q2_PAYLOAD_CLASS_SPLIT')
        groups[key]=q
    if set(groups)!=set(cases['class_keys']) or len(groups)!=class_count:raise ValueError('COMPLETE_CLASS_CENSUS')
    runtime.publish_json(chunk,dict(cases=sorted(data,key=lambda r:r['ordinal']),chunk_id=chunk,Q2_classes_checked=class_count,full_historical_public_observer_compared=False,G2_promotion=False))
    return ChainExecutionResult(result=dict(outcome='PASS_COMPLETE_CLASS_Q2_CHUNK_V1',chunk_id=chunk,cases_checked=count,complete_Q2_classes_checked=class_count,G1_candidate_generation=0,G2_primary_realizations=count,G2_swap_checks=count,full_historical_public_observer_compared=False,master_slices=152,new_admissions=0,G2_promotion=False))
