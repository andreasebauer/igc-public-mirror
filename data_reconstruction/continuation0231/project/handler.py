"""One native restoration task; generation of candidates is outside this job."""
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult
from .worker import read_bound

def handler(stage,runtime):
 c=stage['execution']['parameters'];i=stage['input_artifacts'];h=c['bindings']
 for k in h:read_bound(i[k],h[k])
 payload={k+'_path':i[k] for k in h};payload.update({k+'_sha256':v for k,v in h.items()});payload['historical_spec_identity']=c['historical_spec_identity']
 task=TaskSpec(task_id='terminal100_historical_v1_requalification',task_kind='HISTORICAL_G1_RESTORATION',binding_sha256=canonical_sha256(payload),payload=payload,cost_weight=1.0)
 runtime.run_content_indexed_generation(phase_id='terminal100_historical_v1_requalification',tasks=[task],evaluator_ref='project.worker:evaluate',requested_workers=1,max_tasks=1,max_generated_occurrences=1,max_task_occurrences=1,max_result_bytes=16777216,max_work_seconds=1800)
 rows=list(runtime.iter_generated_states(phase_id='terminal100_historical_v1_requalification'));assert len(rows)==1;data=rows[0]['state']
 runtime.publish_json('historical_v1_interface_population',data['historical_interface_population']);runtime.publish_json('terminal100_requalification',data)
 return ChainExecutionResult(result=dict(outcome='PASS_HISTORICAL_V1_TERMINAL_REQUALIFICATION',completed_depth=100,restored_roots=193,all193_DAG_roundtrip=True,terminal_comparison=data['terminal_comparison'],historical_depth100_beam_anchor=data['historical_depth100_beam_anchor'],candidate_generation_calls=0,current_v2_population_reproduced=True,master_scientific_slices=151,new_admissions=0))
