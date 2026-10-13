from __future__ import annotations

"""Registered, bounded engineering checks for the shared exact-relation kernel.

No pool, durable storage, checkpoint, private cache, reducer or telemetry here.
All execution uses StageScienceRuntime. These are NOT G6 scientific stages.
"""
from importlib.resources import files
import json
from typing import Any, Mapping

from .adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter
from .canon import canonical_sha256
from .execution import TaskSpec
from .exact_tree_relation_kernel import ExactTreeRelationKernelError, tree_from_record
from .g6_controller_evaluators import exact_one_step_relation_evaluator
from .v05_chain import ChainExecutionResult


def load_engineering_fixture() -> dict[str, Any]:
    p = files('infinity_grid').joinpath('resources/engineering/EXACT_TREE_KERNEL_E1_FIXTURE.json')
    d = json.loads(p.read_text(encoding='utf-8'))
    if d.get('fixture_sha256') != canonical_sha256({k:v for k,v in d.items() if k!='fixture_sha256'}):
        raise ExactTreeRelationKernelError('E1 fixture integrity mismatch')
    return d


def historical_reference_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Science-only extraction of 0.30.86's direct owner-pair enumeration.

    The formula/body is intentionally slow and keeps public_read +
    unrooted_canon calls. It never invokes a historical stage executor or pool.
    Regression tests compare it directly with the unchanged historical function.
    """
    state, probe = tree_from_record(payload['state_tree']), tree_from_record(payload['probe_tree'])
    pos = str(payload.get('position','LEFT')).upper()
    if pos not in {'LEFT','RIGHT'}:
        raise ExactTreeRelationKernelError('unsupported relation position')
    left,right = (state,probe) if pos=='LEFT' else (probe,state)
    op = tuple(payload['operator']); ad = G4AcceptedAdapter()
    if op not in ad.operator_basis():
        raise ExactTreeRelationKernelError('reference operator outside frozen basis')
    canons = set(); legal=0
    for lu in range(left.n):
        for rv in range(right.n):
            off=left.n
            child=DecoratedG4Tree(left.n+right.n,
                left.edges+tuple((a+off,b+off) for a,b in right.edges)+((lu,rv+off),),
                left.H_classes+right.H_classes,
                left.edge_operators+right.edge_operators+(op,))
            if not ad.public_read(child).get('legal'):
                continue
            legal+=1; canons.add(ad.unrooted_canon(child))
    return {'signature':tuple(sorted(canons,key=repr)), 'outcome_count':len(canons),
            'metrics':{'reference_attempted_owner_pairs':left.n*right.n,
                       'reference_legal_owner_pairs':legal}}


def exact_equivalence_evaluator(payload: Mapping[str, Any]) -> dict[str, Any]:
    old=historical_reference_evaluator(payload)
    new=exact_one_step_relation_evaluator(payload)
    if old['signature'] != new['signature']:
        raise ExactTreeRelationKernelError('E1 exact old/new structural relation mismatch')
    if old['metrics']['reference_legal_owner_pairs'] != new['metrics']['legal_owner_pairs']:
        raise ExactTreeRelationKernelError('E1 legal-owner count mismatch')
    return {'signature':new['signature'],'outcome_count':new['outcome_count'],
            'metrics':dict(new['metrics'],exact_equivalence_checks=1,
                           reference_attempted_owner_pairs=old['metrics']['reference_attempted_owner_pairs'])}


def engineering_stage_handler(stage: Mapping[str, Any], runtime) -> ChainExecutionResult:
    params=stage['execution']['parameters']; fixture=load_engineering_fixture()
    if params['fixture_sha256']!=fixture['fixture_sha256']:
        raise ExactTreeRelationKernelError('frozen E1 task fixture mismatch')
    refs={
        'reference':'infinity_grid.engine_kernel_acceptance:historical_reference_evaluator',
        'optimized':'infinity_grid.g6_controller_evaluators:exact_one_step_relation_evaluator',
        'equivalence':'infinity_grid.engine_kernel_acceptance:exact_equivalence_evaluator',
    }
    mode=str(params['mode'])
    if mode not in refs:
        raise ExactTreeRelationKernelError('unregistered engineering measurement mode')
    tasks=[]
    for row in fixture['states']:
        payload={'state_tree':row['tree'],'probe_tree':fixture['probe_tree'],
                 'operator':fixture['operator'],'position':fixture['position']}
        tasks.append(TaskSpec(task_id=row['task_id'],task_kind='E1_ENGINEERING_RELATION',
            binding_sha256=canonical_sha256({'question_sha256':stage['question_sha256'],'payload':payload}),
            payload=payload,cost_weight=row['tree']['n']))
    part=runtime.run_structural_partition(phase_id='P',tasks=tasks,evaluator_ref=refs[mode],
        requested_workers=runtime.default_workers,max_tasks=fixture['selected_count'])
    return ChainExecutionResult(result={
        'outcome':'COMPLETE_ENGINEERING_MEASUREMENT','partition':part.summary,
        'fixture_sha256':fixture['fixture_sha256'],'mode':mode,
        'scope':'BOUNDED_ENGINEERING_ONLY_NOT_FULL_S3R_ACCEPTANCE',
        'promotion_effect':'NONE',
    })
