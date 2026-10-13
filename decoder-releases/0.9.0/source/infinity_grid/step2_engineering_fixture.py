"""Pure bounded task recipes for explicit Step 2 engineering registrations."""
import os
from .canon import canonical_sha256
from .execution import TaskSpec
from .v05_chain import ChainExecutionResult


def evaluate(payload):
    value=payload['value'];checksum=value+1
    for i in range(payload['rounds']):checksum=(checksum*1664525+1013904223)&0xffffffff
    return {'states':[{'identity':{'kind':'shared','value':value//2},'state':{'value':value,'slot':0}},
                      {'identity':{'kind':'unique','value':value},'state':{'value':value,'slot':1}}],
            'metrics':{'checksum':checksum,'rounds':payload['rounds'],
                       'engineering_worker_pid':os.getpid(),'engineering_controller_pid':os.getppid()}}


def handler(stage,runtime):
    p=stage['execution']['parameters'];tasks=[]
    for value in range(p['tasks']):
        payload={'value':value,'rounds':p['rounds']}
        tasks.append(TaskSpec(f'T{value:05d}','STEP2_ENGINEERING',canonical_sha256(payload),payload))
    shard=runtime.publish_bulk_shard('ENGINEERING_INPUTS',{'values':list(range(p['shard_records'])),
        'scope':'ENGINEERING_FIXTURE_NO_MASTER_DATA'})
    result=runtime.run_content_indexed_generation(phase_id='BOUNDED_GENERATION',tasks=tasks,
        evaluator_ref='infinity_grid.step2_engineering_fixture:evaluate',max_tasks=p['tasks'],
        max_generated_occurrences=2*p['tasks'],max_task_occurrences=2,
        max_result_bytes=65536,max_work_seconds=p['max_work_seconds'])
    return ChainExecutionResult(result={'outcome':'PASS','generation':result.summary,'shard':shard,
        'execution':result.execution_metadata,'scope':'ENGINEERING_FIXTURE_NO_MASTER_DATA'})
