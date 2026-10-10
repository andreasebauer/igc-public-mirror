"""New registered observer task; does not mutate or resume earned Q2 captures."""
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound

def handler(stage,runtime):
    p=stage['execution']['parameters'];h=p['bindings'];i=stage['input_artifacts']
    payload={k+'_path':i[k] for k in h};payload.update({k+'_sha256':v for k,v in h.items()})
    payload['source_binding']=p['source_binding'];chunk=p['chunk_id'];count=p['case_count']
    cases=read_bound(i['cases'],h['cases']);source={x['ordinal']:x for x in cases['cases']}
    if not 1<=count<=62 or len(source)!=count:raise ValueError('REGISTERED_CENSUS')
    task=TaskSpec(task_id=chunk,task_kind='HISTORICAL_S1_PUBLIC_OBSERVER_COMPARE',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
    runtime.run_content_indexed_generation(phase_id=chunk,tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=count,max_task_occurrences=count,max_result_bytes=16777216,max_work_seconds=1800)
    data=[row['state'] for row in runtime.iter_generated_states(phase_id=chunk)]
    if len(data)!=count or {x['ordinal'] for x in data}!=set(source):raise ValueError('OBSERVER_CENSUS')
    groups={}
    for row in data:
        key=source[row['ordinal']]['record']['outcome_science_sha256'];observer=row['public_observer']
        if key in groups and groups[key]!=observer:raise ValueError('FULL_OBSERVER_CLASS_SPLIT')
        groups[key]=observer
    if set(groups)!=set(cases['class_keys']):raise ValueError('OBSERVER_CLASS_CENSUS')
    runtime.publish_json(chunk,dict(cases=sorted(data,key=lambda x:x['ordinal']),chunk_id=chunk,historical_observer_hashes_compared=count,full_historical_population_compared=False,G2_promotion=False))
    return ChainExecutionResult(result=dict(outcome='PASS_BOUND_HISTORICAL_OBSERVER_CHUNK_V1',chunk_id=chunk,cases_checked=count,complete_observer_classes_checked=len(groups),historical_observer_hashes_compared=count,full_historical_public_observer_compared=False,master_slices=152,new_admissions=0,G2_promotion=False))
