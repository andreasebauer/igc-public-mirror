"""One warm decoder task over a fixed62-case probe; no G1 regeneration."""
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
def handler(stage,runtime):
    h=stage['execution']['parameters']['bindings'];i=stage['input_artifacts'];payload={k+'_path':i[k] for k in h};payload.update({k+'_sha256':v for k,v in h.items()});payload['source_binding']=stage['execution']['parameters']['source_binding']
    task=TaskSpec(task_id='bounded62_s1_q2_probe',task_kind='HISTORICAL_S1_BINARY_REALIZATION_PROBE',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
    runtime.run_content_indexed_generation(phase_id='bounded62_s1_q2_probe',tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=62,max_task_occurrences=62,max_result_bytes=16777216,max_work_seconds=1800)
    data=[row['state'] for row in runtime.iter_generated_states(phase_id='bounded62_s1_q2_probe')]
    if len(data)!=62:raise ValueError('BOUNDED_CENSUS')
    runtime.publish_json('bounded_q2_probe',dict(scope='BOUNDED_HISTORICAL_S1_DETERMINISTIC_Q2_PROBE',cases=data,G2_promotion=False))
    return ChainExecutionResult(result=dict(outcome='PASS_BOUNDED62_HISTORICAL_S1_Q2_PROBE',cases_checked=62,G1_candidate_generation=0,G2_primary_realizations=62,G2_swap_checks=62,master_slices=152,new_admissions=0,G2_promotion=False))
