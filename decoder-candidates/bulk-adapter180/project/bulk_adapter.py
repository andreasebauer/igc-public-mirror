"""Decoder-owned HET4 cutover candidate; scheduling and commits belong to runtime."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
from infinity_grid.canon import canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.o7_edge_canonicalization import state_identity, verify_relabeling_witness
from infinity_grid.v05_chain import ChainExecutionResult

_CONTEXT = None

def context():
    global _CONTEXT
    if _CONTEXT is None:
        obj = json.loads(Path(__file__).with_name('context.json').read_bytes())
        if obj['schema'] != 'IG_HET4_FROZEN_CONTEXT_V1':
            raise ValueError('CONTEXT_SCHEMA')
        _CONTEXT = SimpleNamespace(n=len(obj['parents']), parents=tuple(
            SimpleNamespace(exact_color=p['exact_color'], h6=p['h6']) for p in obj['parents']))
    return _CONTEXT

def evaluate(payload):
    ctx = context()
    output = []
    for descriptor in payload['descriptors']:
        ordinal, edge = descriptor['ordinal'], descriptor['edge']
        result = state_identity(ctx, (tuple(edge),))
        if not verify_relabeling_witness(ctx, (tuple(edge),), result['edges'], result['witness']):
            raise ValueError('WITNESS_VERIFICATION')
        output.append({'ordinal': ordinal, 'raw_edge': edge,
                       'V3_identity': result['object_identity'],
                       'V2_context_alias': result['legacy_context_identity'],
                       'corrected_edges': result['edges'], 'relabeling_witness': result['witness'],
                       'relabeling_witness_sha256': canonical_sha256(result['witness'])})
    return {'signature': {'schema': 'IG_HET4_FULL_WITNESS_BATCH_V1', 'rows': output},
            'outcome_count': len(output), 'metrics': {'worker_pid': os.getpid()}}

def handler(stage, runtime):
    params = stage['execution']['parameters']
    descriptors = json.loads(Path(stage['input_artifacts']['descriptor_sample']).read_bytes())
    ordinals = [r['ordinal'] for r in descriptors]
    if ordinals != sorted(set(ordinals)) or len(descriptors) != params['registered_rows']:
        raise ValueError('DESCRIPTOR_SCOPE')
    batch_size = params['batch_size']
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError('BATCH_SIZE')
    tasks = []
    for start in range(0, len(descriptors), batch_size):
        payload = {'descriptors': descriptors[start:start+batch_size]}
        tasks.append(TaskSpec(task_id=f'batch{start:09d}', task_kind='HET4_FULL_WITNESS',
            binding_sha256=canonical_sha256({'question': stage['question_sha256'], 'payload': payload}),
            payload=payload, cost_weight=len(payload['descriptors'])))
    partition = runtime.run_structural_partition(phase_id='HET4_MAPPING', tasks=tasks,
        evaluator_ref='project.bulk_adapter:evaluate', requested_workers=params['workers'],
        stream_shard_task_limit=1)
    references = []
    observed = []
    for row in runtime.iter_partition_signatures('HET4_MAPPING'):
        observed.extend(x['ordinal'] for x in row['signature']['rows'])
        references.append(runtime.publish_bulk_shard(row['task_id'], row['signature']))
    if observed != ordinals:
        raise ValueError('ORDERED_COVERAGE_MISMATCH')
    return ChainExecutionResult(result={'outcome': 'PASS', 'rows': len(observed),
        'partition': partition.summary, 'execution_metadata': partition.execution_metadata,
        'full_witness_shards': references, 'scientific_adoption': False,
        'scope': 'Bounded adapter qualification; full census and adoption remain open'})
