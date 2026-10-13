"""Decoder-owned HET4 cutover candidate; scheduling and commits belong to runtime."""
from pathlib import Path
from types import SimpleNamespace
import hashlib
import json
import os
from infinity_grid.canon import canonical_sha256
from infinity_grid.structural_encoding import structural_canonical_sha256
from infinity_grid.execution import TaskSpec
from infinity_grid.o7_edge_canonicalization import state_identity, verify_relabeling_witness, hierarchy_cache_statistics, bind_immutable_context
from infinity_grid.v05_chain import ChainExecutionResult
from .historical_alias import old_edge_alias, context as alias_context

_CONTEXT = None

def context():
    global _CONTEXT
    if _CONTEXT is None:
        obj = json.loads(Path(__file__).with_name('context.json').read_bytes())
        if obj['schema'] != 'IG_HET4_FROZEN_CONTEXT_V1':
            raise ValueError('CONTEXT_SCHEMA')
        candidate=SimpleNamespace(n=len(obj['parents']), parents=tuple(
            SimpleNamespace(exact_color=p['exact_color'], h6=p['h6']) for p in obj['parents']))
        historical=alias_context()
        from infinity_grid.structural_encoding import structural_canonical_bytes as encode
        if encode([(p.exact_color,p.h6) for p in candidate.parents])!=encode([(p.exact_color,p.h6) for p in historical.parents]):
            raise ValueError('ALIAS_CONTEXT_BINDING')
        _CONTEXT=bind_immutable_context(candidate)
    return _CONTEXT

def evaluate(payload):
    ctx = context()
    output = [];previous_witness=None;previous_digest=None
    for descriptor in payload['descriptors']:
        ordinal, edge = descriptor['ordinal'], descriptor['edge']
        result = state_identity(ctx, (tuple(edge),))
        if not verify_relabeling_witness(ctx, (tuple(edge),), result['edges'], result['witness']):
            raise ValueError('WITNESS_VERIFICATION')
        witness=result['witness']
        if previous_witness!=witness:
            previous_digest=structural_canonical_sha256(witness);previous_witness=witness
        output.append({'ordinal': ordinal, 'raw_edge': edge,
                       'old_edge_alias': old_edge_alias((tuple(edge),)),
                       'literal_descriptor_sha256': descriptor['literal_descriptor_sha256'],
                       'V3_identity': result['object_identity'],
                       'V2_context_alias': result['legacy_context_identity'],
                       'corrected_edges': result['edges'], 'relabeling_witness': result['witness'],
                       'relabeling_witness_sha256': previous_digest})
    return {'signature': {'schema': 'IG_HET4_FULL_WITNESS_BATCH_V1', 'rows': output},
            'outcome_count': len(output), 'metrics': {'worker_pid': os.getpid(), 'kernel_cache': hierarchy_cache_statistics()}}

def handler(stage, runtime):
    params = stage['execution']['parameters']
    descriptors = json.loads(Path(stage['input_artifacts']['descriptor_sample']).read_bytes())
    from infinity_grid.bulk_descriptor_plan import DOMAIN
    if any(type(r.get('ordinal')) is not int or not 0<=r['ordinal']<DOMAIN for r in descriptors):
        raise ValueError('QUALIFICATION_DESCRIPTOR_SCOPE')
    from infinity_grid.bulk_descriptor_plan import build_plan,indexed_descriptors,admit_frozen_plan
    from infinity_grid.structural_encoding import structural_canonical_bytes as encode
    saved_plan = json.loads(Path(stage['input_artifacts']['full_domain_plan']).read_bytes())
    if params['window_rows']==16384:
        plan=admit_frozen_plan(stage['input_artifacts']['raw_descriptor_archive'],stage['input_artifacts']['preserved_prefix_archive'],saved_plan)
    else:
        plan=build_plan(stage['input_artifacts']['raw_descriptor_archive'],stage['input_artifacts']['preserved_prefix_archive'],window_rows=params['window_rows'])
    if encode(plan)!=encode(saved_plan):
        raise ValueError('FULL_DOMAIN_PLAN_BINDING')
    index = params.get('window_index')
    if index is not None:
        if type(index) is not int or not 0 <= index < len(plan['windows']):
            raise ValueError('WINDOW_INDEX')
        window = plan['windows'][index]
        if [r['ordinal'] for r in descriptors]!=list(range(window['start'],window['stop'])):
            raise ValueError('WINDOW_DESCRIPTOR_COVERAGE')
        if hashlib.sha256(b''.join(encode(r)+b'\n' for r in descriptors)).hexdigest()!=window['descriptor_stream_sha256']:
            raise ValueError('WINDOW_DESCRIPTOR_HASH')
    else:
        if any(type(r.get('ordinal')) is not int or not 0 <= r['ordinal'] < plan['domain_rows'] for r in descriptors):
            raise ValueError('QUALIFICATION_DESCRIPTOR_SCOPE')
        requested={r['ordinal']:r for r in descriptors}
        matched=set()
        for row in indexed_descriptors(stage['input_artifacts']['raw_descriptor_archive'],min(requested,default=0),max(requested,default=-1)+1):
            if row['ordinal'] in requested:
                if encode(row)!=encode(requested[row['ordinal']]):
                    raise ValueError('QUALIFICATION_DESCRIPTOR_BINDING')
                matched.add(row['ordinal'])
        if matched != set(requested):
            raise ValueError('QUALIFICATION_DESCRIPTOR_MISSING')
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
    partition = runtime.run_exact_bulk_mapping(phase_id='HET4_MAPPING', tasks=tasks,
        evaluator_ref='project.bulk_adapter:evaluate', requested_workers=params['workers'])
    partition['summary']['plan_sha256']=plan['plan_sha256']
    partition['summary']['descriptor_stream_sha256']=hashlib.sha256(b''.join(encode(r)+b'\n' for r in descriptors)).hexdigest()
    partition['summary']['preserved_prefix_rows']=plan['preserved_rows']
    published = runtime.publish_json('mapping-index', partition['summary'])
    return ChainExecutionResult(result={'outcome':'PASS','rows':partition['summary']['rows'],
        'mapping':partition['summary'],'execution_metadata':partition['execution_metadata'],
        'mapping_index':published,'scientific_adoption':False,
        'scope':'Exact compressed full-witness mapping window; no census adoption'})
