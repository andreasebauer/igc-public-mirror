"""Native completion recovery of frozen historical evidence under a new budget."""
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult

def handler(stage,runtime):
 c=stage['execution']['parameters'];inputs=stage['input_artifacts']
 payload={k+'_path':inputs[k] for k in c['bindings']}
 payload.update({k+'_sha256':v for k,v in c['bindings'].items()})
 payload.update(prior_capture_id=c['prior_capture_id'],final_science_sha256=c['final_science_sha256'])
 task=TaskSpec(task_id='saved_depth80_verification',task_kind='HISTORICAL_COMPLETION_RECOVERY',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
 runtime.run_content_indexed_generation(phase_id='saved_depth80_verification',tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=1,max_task_occurrences=1,max_result_bytes=1048576,max_work_seconds=1800)
 rows=list(runtime.iter_generated_states(phase_id='saved_depth80_verification'))
 if len(rows)!=1:raise ValueError('RECOVERY_COUNT')
 result=rows[0]['state'];runtime.publish_json('DEPTH80_RECOVERY_EVIDENCE',result)
 return ChainExecutionResult(result=result)
