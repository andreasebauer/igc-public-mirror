"""Eight deterministic engineering values; no scientific acceptance claim."""
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.v05_chain import ChainExecutionResult


def evaluate(payload):
    value = int(payload['value'])
    return {'signature': {'parity': value % 2}, 'outcome_count': 1,
            'metrics': {'fixture_value': value}}


def handler(stage, runtime):
    tasks = []
    for value in range(8):
        payload = {'value': value}
        tasks.append(TaskSpec(task_id=f'v{value:02d}', task_kind='P0_PAUSED_ENGINEERING',
            binding_sha256=canonical_sha256({'question': stage['question_sha256'], 'payload': payload}),
            payload=payload, cost_weight=1.0))
    part = runtime.run_structural_partition(phase_id='PAUSED_PARTITION', tasks=tasks,
        evaluator_ref='project.fixture:evaluate', requested_workers=1, max_tasks=8,
        stream_shard_task_limit=1)
    return ChainExecutionResult(result={'outcome': 'PASS', 'partition': part.summary,
        'execution': part.execution_metadata})
