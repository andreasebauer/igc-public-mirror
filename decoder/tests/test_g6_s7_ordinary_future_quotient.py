from __future__ import annotations

import inspect


def test_s7_registered_and_marker_free():
    import infinity_grid.v05_controller_event_loop as loop
    from pathlib import Path
    reg=loop._registry(Path(loop.__file__).resolve().parent.parent)
    assert 'G6_S7_ORDINARY_FUTURE_QUOTIENT' in reg[loop.SCIENCE_JOB]['allowed_operations']
    src=inspect.getsource(__import__('infinity_grid.g6_s7_evaluators',fromlist=['*']))
    assert 'MARKER_Q' not in src
    assert 'MARKER_DECODE' not in src
    assert 'OBSERVER_DECODE' not in src


def test_s7_evaluator_specs_are_public_service_only():
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import PUBLIC_KERNEL_SERVICES
    for ref in (
        'infinity_grid.g6_s7_evaluators:s7_public_state_evaluator',
        'infinity_grid.g6_s7_evaluators:s7_ordinary_branch_count_component_evaluator',
        'infinity_grid.g6_s7_evaluators:s7_ordinary_branch_relation_evaluator',
    ):
        spec=get_evaluator_spec(ref)
        assert set(spec.allowed_kernel_services) <= PUBLIC_KERNEL_SERVICES
        assert not ({'MARKER_Q','MARKER_DECODE','MARKER_WRITE','OBSERVER_DECODE','OBSERVER_WRITE'} & set(spec.allowed_kernel_services))


def test_public_read_service_matches_inherited_adapter_on_basis():
    from infinity_grid.v05_kernel_service_providers import _basis_records, _public_read_service
    from infinity_grid.exact_tree_relation_kernel import tree_from_record
    from infinity_grid.adapters.g4_accepted import G4AcceptedAdapter
    ad=G4AcceptedAdapter()
    for rec in _basis_records().values():
        assert _public_read_service(rec) == ad.public_read(tree_from_record(rec))


def test_s7_multiset_signature_preserves_successor_block_multiplicity_without_exact_ids():
    from infinity_grid.g6_s7_evaluators import _multiset_signature
    a=("D",(1,2,3),(("C",2),))
    b=("D",(1,2,4),(("C",2),))
    out=_multiset_signature([a,b,a])
    rows={repr(sig):count for sig,count in out}
    assert rows[repr(a)]==2
    assert rows[repr(b)]==1


def test_s7_depth2_is_declared_as_refinement_not_scalar_count_only():
    import inspect
    import infinity_grid.g6_s7_evaluators as ev
    src=inspect.getsource(ev.s7_ordinary_branch_relation_evaluator)
    assert 'previous = fast_depth1(tree_record)' in src
    assert '_multiset_signature(vals)' in src
    assert 'EXACT_IDENTITY' not in src


def test_s7_profile_service_matches_full_relation_count_on_basis_contexts():
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel, tree_from_record
    from infinity_grid.v05_kernel_service_providers import _basis_records
    configure_relation_kernel(scope_identity='S7:PROFILE:EQUIV')
    k=get_relation_kernel()
    basis=_basis_records(); refs=sorted(basis)
    for left_ref,right_ref,op in ((refs[0],refs[1],(0,0)),(refs[-1],refs[0],(0,1)),(refs[1],refs[-1],(1,0))):
        left=tree_from_record(basis[left_ref]); right=tree_from_record(basis[right_ref])
        prof=k.relation_profile(left,right,op); full=k.relation(left,right,op)
        assert prof.exact_outcome_count==len(full.children)==len(full.canons)
        assert prof.legal_owner_pairs==full.legal_owner_pairs
        assert prof.rooted_owner_pair_candidates==full.rooted_owner_pair_candidates


def test_s7_depth1_fast_path_preserves_full_public_multiset_formula():
    from infinity_grid.g6_s7_evaluators import _compose_public_signature
    left=('D',(2,1,0,0,0,0,0),(('C',2),))
    right=('D',(1,2,0,0,0,0,0),(('C',1),('D',1)))
    assert _compose_public_signature(left,right,(0,1)) == ('D',(2,2,0,0,0,0,0),(('C',3),('D',1)))


def test_s7_lazy_component_and_heavy_shard_controls_present():
    import inspect
    import infinity_grid.g6_s7_ordinary_future_quotient as s7
    src=inspect.getsource(s7)
    assert 'MONOTONE_LAZY_COMPONENT_REFINEMENT' in src
    assert 'partition_equivalent_to_full_248_component_observer' in src
    assert 'stream_shard_task_limit=1' in src
    assert 's7_ordinary_branch_count_component_evaluator' in src
    assert 'component_batch_size' in src


def test_s7_fast_depth1_signature_is_exactly_reference_materialization_on_frozen_seed():
    import infinity_grid.g6_s7_evaluators as ev
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import bind_kernel_view
    from infinity_grid.v05_kernel_service_providers import (
        _basis_records, build_kernel_service_providers, _exact_relation_service, _public_read_service,
    )
    configure_relation_kernel(scope_identity='S7:FAST:REFERENCE:EQUIV',max_cache_entries=512,max_cache_bytes=128*1024*1024,max_relation_cache_entries=512,max_relation_cache_bytes=128*1024*1024)
    spec=get_evaluator_spec('infinity_grid.g6_s7_evaluators:s7_ordinary_branch_relation_evaluator')
    bind_kernel_view(spec,build_kernel_service_providers(spec))
    ev._BASIS_RECORD_CACHE=None; ev._BASIS_PUBLIC_CACHE=None
    basis=_basis_records(); refs=tuple(sorted(basis)); ops=tuple(get_relation_kernel().authority.operators)
    state=basis[refs[0]]
    payload={'state_tree':state,'future_depth':1,'basis_refs':list(refs),'operator_basis':[list(x) for x in ops]}
    fast=ev.s7_ordinary_branch_relation_evaluator(payload)['signature']
    root=ev._public_signature(_public_read_service(state)); blocks=[]
    for ref in refs:
        seed=basis[ref]
        for op in ops:
            for pos in ('LEFT','RIGHT'):
                left,right=(state,seed) if pos=='LEFT' else (seed,state)
                full=_exact_relation_service(left,right,op)
                vals=[ev._public_signature(_public_read_service(child)) for child in full['children']]
                blocks.append(ev._multiset_signature(vals))
    reference=('ORDINARY_BRANCH_MULTISET',1,root,tuple(blocks))
    assert fast==reference


def test_s7_depth1_component_evaluator_needs_no_public_read_service():
    import infinity_grid.g6_s7_evaluators as ev
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import bind_kernel_view
    from infinity_grid.v05_kernel_service_providers import _basis_records, build_kernel_service_providers
    configure_relation_kernel(scope_identity='S7:COMPONENT:NO:PUBLIC:READ')
    spec=get_evaluator_spec('infinity_grid.g6_s7_evaluators:s7_ordinary_branch_count_component_evaluator')
    assert set(spec.allowed_kernel_services)=={'AUTHORITY_BASIS','EXACT_RELATION_PROFILE'}
    bind_kernel_view(spec,build_kernel_service_providers(spec))
    ev._BASIS_RECORD_CACHE=None; ev._BASIS_PUBLIC_CACHE=None
    basis=_basis_records(); refs=tuple(sorted(basis))
    payload={
        'state_tree':basis[refs[0]],
        'basis_refs':list(refs),
        'contexts':[{'basis_ref':refs[1],'operator':[0,0],'position':'LEFT'}],
    }
    out=ev.s7_ordinary_branch_count_component_evaluator(payload)
    assert out['signature'][0]=='S7_D1_BRANCH_MULTIPLICITY_COMPONENT_BATCH'
    assert len(out['signature'][1])==1
    assert out['metrics']['exact_relation_profile_call_count']==1
    assert ev._BASIS_PUBLIC_CACHE is None
