from __future__ import annotations

import inspect
from pathlib import Path


def test_s7d2_operation_registered_and_reuse_only():
    import infinity_grid.v05_controller_event_loop as loop
    import infinity_grid.g6_s7_depth2_completion as d2
    src=Path(loop.__file__).resolve().parent.parent
    reg=loop._registry(src)
    assert 'G6_S7_DEPTH2_COMPLETION' in reg[loop.SCIENCE_JOB]['allowed_operations']
    text=inspect.getsource(d2)
    assert 'no_s1_regeneration' in text
    assert 'no_higher_regeneration' in text
    assert 'no_depth1_recomputation' in text
    assert 'MARKER_Q' not in text and 'OBSERVER_DECODE' not in text
    assert d2.INNER_PREFIX_SCHEDULE[-1] == 124


def test_s7d2_evaluator_uses_public_services_only_and_no_public_read():
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import PUBLIC_KERNEL_SERVICES
    ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator'
    spec=get_evaluator_spec(ref)
    assert set(spec.allowed_kernel_services)=={'AUTHORITY_BASIS','EXACT_RELATION','EXACT_RELATION_PROFILE','EXACT_RELATION_PROFILE_BATCH','EXACT_RELATION_PROFILE_FAMILY'}
    assert set(spec.allowed_kernel_services) <= PUBLIC_KERNEL_SERVICES
    assert 'PUBLIC_READ' not in spec.allowed_kernel_services


def test_s7d2_outer_prefix_component_matches_direct_exact_formula_on_frozen_seed():
    import infinity_grid.g6_s7_evaluators as ev
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import bind_kernel_view
    from infinity_grid.v05_kernel_service_providers import (
        _basis_records, build_kernel_service_providers,
        _exact_relation_service, _exact_relation_profile_service,
    )
    configure_relation_kernel(
        scope_identity='S7D2:COMPONENT:EQUIV',
        max_cache_entries=512,max_cache_bytes=128*1024*1024,
        max_relation_cache_entries=512,max_relation_cache_bytes=128*1024*1024,
    )
    spec=get_evaluator_spec('infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator')
    bind_kernel_view(spec,build_kernel_service_providers(spec))
    ev._BASIS_RECORD_CACHE=None
    ev._D2_CHILD_PREFIX_CACHE.clear()
    basis=_basis_records(); refs=tuple(sorted(basis)); ops=tuple(get_relation_kernel().authority.operators)
    payload={
        'state_tree':basis[refs[0]],
        'basis_refs':list(refs),
        'operator_basis':[list(x) for x in ops],
        'outer_context_index':0,
        'inner_prefix_context_count':8,
    }
    got=ev.s7_depth2_outer_prefix_evaluator(payload)['signature']
    contexts=tuple((ref,op,pos) for ref in refs for op in ops for pos in ('LEFT','RIGHT'))
    oref,oop,opos=contexts[0]; seed=basis[oref]
    left,right=(basis[refs[0]],seed) if opos=='LEFT' else (seed,basis[refs[0]])
    rel=_exact_relation_service(left,right,oop)
    vals=[]
    for child in rel['children']:
        counts=[]
        for ref,op,pos in contexts[:8]:
            seed2=basis[ref]; l,r=(child,seed2) if pos=='LEFT' else (seed2,child)
            counts.append(int(_exact_relation_profile_service(l,r,op)['exact_outcome_count']))
        vals.append(('S7_CHILD_D1_BRANCH_COUNT_PREFIX',tuple(counts)))
    expected=('S7_D2_OUTER_SUCCESSOR_SIG1_PREFIX_MULTISET',8,ev._multiset_signature(vals))
    assert got==expected


def test_s7d2_exact_child_identity_never_enters_signature_source():
    import infinity_grid.g6_s7_evaluators as ev
    src=inspect.getsource(ev.s7_depth2_outer_prefix_evaluator)
    assert '_multiset_signature(child_prefixes)' in src
    assert 'S7_D2_OUTER_SUCCESSOR_SIG1_PREFIX_MULTISET' in src


def _bind_s7d2_evaluator(ref):
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import bind_kernel_view
    from infinity_grid.v05_kernel_service_providers import build_kernel_service_providers
    spec=get_evaluator_spec(ref)
    bind_kernel_view(spec,build_kernel_service_providers(spec))
    return spec


def test_s7d2_global_dedup_evaluators_use_existing_kernel_services_only():
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import PUBLIC_KERNEL_SERVICES
    expected={
      'infinity_grid.g6_s7_evaluators:s7_depth2_outer_children_generation_evaluator':{'AUTHORITY_BASIS','EXACT_RELATION'},
      'infinity_grid.g6_s7_evaluators:s7_depth2_child_profile_generation_evaluator':{'AUTHORITY_BASIS','EXACT_RELATION_PROFILE'},
    }
    for ref,services in expected.items():
        spec=get_evaluator_spec(ref)
        assert set(spec.allowed_kernel_services)==services
        assert set(spec.allowed_kernel_services)<=PUBLIC_KERNEL_SERVICES
    # Parent assembly is intentionally kernel-free. It is authorized by the
    # controller execution permit, not granted a dummy KernelView permission.
    from infinity_grid.v05_stage_registry import EVALUATOR_SPEC_BY_REF
    assembly_ref='infinity_grid.g6_s7_evaluators:s7_depth2_parent_profile_multiset_evaluator'
    assert assembly_ref not in EVALUATOR_SPEC_BY_REF
    import inspect, infinity_grid.g6_s7_evaluators as ev
    assembly_src=inspect.getsource(ev.s7_depth2_parent_profile_multiset_evaluator)
    assert 'current_kernel_view' not in assembly_src
    assert '.call(' not in assembly_src


def test_s7d2_active_handler_uses_parent_prefix_monotone_not_global_child_universe():
    import infinity_grid.g6_s7_depth2_completion as d2
    run_src=inspect.getsource(d2.run_g6_s7_depth2_completion)
    assert 'run_structural_partition' in run_src
    assert 'D2_COMPONENT_EVALUATOR' in run_src
    assert 'FULL_INNER_OUTER_MONOTONE_FACTOR_SWAP_NORMALIZED_V2' in inspect.getsource(d2)
    assert 'run_content_indexed_generation' not in run_src
    assert 'OUTER_GENERATION_EVALUATOR' not in run_src
    assert 'CHILD_PROFILE_GENERATION_EVALUATOR' not in run_src
    assert 'PARENT_ASSEMBLY_EVALUATOR' not in run_src
    assert 'exact_tree_relation_kernel' not in run_src
    assert 'v05_kernel_cache' not in run_src
    assert d2.INNER_PREFIX_SCHEDULE == (124,)


def test_s7d2_split_global_dedup_signature_matches_reference_at_prefix8_and32():
    import infinity_grid.g6_s7_evaluators as ev
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_kernel_service_providers import _basis_records
    configure_relation_kernel(
        scope_identity='S7D2:GLOBAL:DEDUP:EQUIV',
        max_cache_entries=512,max_cache_bytes=128*1024*1024,
        max_relation_cache_entries=512,max_relation_cache_bytes=128*1024*1024,
    )
    basis=_basis_records(); refs=tuple(sorted(basis)); ops=tuple(get_relation_kernel().authority.operators)
    parent=basis[refs[0]]

    direct_ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator'
    outer_ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_children_generation_evaluator'
    profile_ref='infinity_grid.g6_s7_evaluators:s7_depth2_child_profile_generation_evaluator'
    assembly_ref='infinity_grid.g6_s7_evaluators:s7_depth2_parent_profile_multiset_evaluator'

    _bind_s7d2_evaluator(outer_ref)
    outer=ev.s7_depth2_outer_children_generation_evaluator({
        'state_tree':parent,'basis_refs':list(refs),'operator_basis':[list(x) for x in ops],
        'outer_context_index':0,
    })
    assert len(outer['states'])>0

    profiles8=[]; profiles32=[]
    for row in outer['states']:
        child=row['state']['tree']; ident=row['identity']
        _bind_s7d2_evaluator(profile_ref)
        r8=ev.s7_depth2_child_profile_generation_evaluator({
            'child_tree':child,'child_identity':ident,'basis_refs':list(refs),'operator_basis':[list(x) for x in ops],
            'start_context_index':0,'end_context_index':8,'previous_counts':[],
        })
        c8=tuple(r8['states'][0]['state']['profile_counts']); assert len(c8)==8
        profiles8.append(c8)
        _bind_s7d2_evaluator(profile_ref)
        r32=ev.s7_depth2_child_profile_generation_evaluator({
            'child_tree':child,'child_identity':ident,'basis_refs':list(refs),'operator_basis':[list(x) for x in ops],
            'start_context_index':8,'end_context_index':32,'previous_counts':list(c8),
        })
        c32=tuple(r32['states'][0]['state']['profile_counts']); assert len(c32)==32 and c32[:8]==c8
        profiles32.append(c32)

    # Parent assembly is intentionally kernel-free; emulate the StageRuntime
    # worker path by clearing any previously bound KernelView rather than
    # registering or granting it a dummy kernel service.
    from infinity_grid.v05_kernel_services import clear_kernel_view
    clear_kernel_view()
    split8=ev.s7_depth2_parent_profile_multiset_evaluator({
        'inner_prefix_context_count':8,'child_profile_counts':[list(x) for x in profiles8],
    })['signature']
    clear_kernel_view()
    split32=ev.s7_depth2_parent_profile_multiset_evaluator({
        'inner_prefix_context_count':32,'child_profile_counts':[list(x) for x in profiles32],
    })['signature']

    _bind_s7d2_evaluator(direct_ref)
    direct8=ev.s7_depth2_outer_prefix_evaluator({
        'state_tree':parent,'basis_refs':list(refs),'operator_basis':[list(x) for x in ops],
        'outer_context_index':0,'inner_prefix_context_count':8,
    })['signature']
    _bind_s7d2_evaluator(direct_ref)
    direct32=ev.s7_depth2_outer_prefix_evaluator({
        'state_tree':parent,'basis_refs':list(refs),'operator_basis':[list(x) for x in ops],
        'outer_context_index':0,'inner_prefix_context_count':32,
    })['signature']
    assert split8==direct8
    assert split32==direct32


def test_s7d2_profile_extension_reuses_prior_prefix_without_recomputing_it():
    import infinity_grid.g6_s7_evaluators as ev
    src=inspect.getsource(ev.s7_depth2_child_profile_generation_evaluator)
    assert 'contexts[start:end]' in src
    assert 'previous_counts' in src
    assert 'profile_contexts_reused' in src


def test_s7d2_prefix1_prefilter_is_exact_prefix_of_reference_prefix8():
    import infinity_grid.g6_s7_evaluators as ev
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_kernel_service_providers import _basis_records
    configure_relation_kernel(
        scope_identity='S7D2:PREFILTER:EXACT',
        max_cache_entries=512,max_cache_bytes=128*1024*1024,
        max_relation_cache_entries=512,max_relation_cache_bytes=128*1024*1024,
    )
    basis=_basis_records(); refs=tuple(sorted(basis)); ops=tuple(get_relation_kernel().authority.operators)
    parent=basis[refs[0]]
    ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator'
    _bind_s7d2_evaluator(ref)
    p1=ev.s7_depth2_outer_prefix_evaluator({
        'state_tree':parent,'basis_refs':list(refs),'operator_basis':[list(x) for x in ops],
        'outer_context_index':0,'inner_prefix_context_count':1,
    })['signature']
    _bind_s7d2_evaluator(ref)
    p8=ev.s7_depth2_outer_prefix_evaluator({
        'state_tree':parent,'basis_refs':list(refs),'operator_basis':[list(x) for x in ops],
        'outer_context_index':0,'inner_prefix_context_count':8,
    })['signature']
    assert p1[0]==p8[0]=='S7_D2_OUTER_SUCCESSOR_SIG1_PREFIX_MULTISET'
    assert p1[1]==1 and p8[1]==8
    # Prefix-1 is exactly the structural projection of the prefix-8 child bag.
    projected=[]
    for value,multiplicity in p8[2]:
        tag,counts=value
        projected.extend([(tag,tuple(counts[:1]))]*int(multiplicity))
    assert p1[2]==ev._multiset_signature(projected)


def test_s7d2_parent_prefix_monotone_path_never_touches_kernel_files():
    import inspect
    import infinity_grid.g6_s7_depth2_completion as d2
    run_src=inspect.getsource(d2.run_g6_s7_depth2_completion)
    assert 'D2_COMPONENT_EVALUATOR' in run_src
    assert '_parent_prefix_task' in run_src
    assert '_refine_parent_prefix' in run_src
    assert 'run_content_indexed_generation' not in run_src
    assert 'exact_tree_relation_kernel' not in run_src
    assert 'v05_kernel_cache' not in run_src


def test_s7d2_controller_permit_admits_existing_reference_prefilter_evaluator():
    import inspect
    import infinity_grid.v05_controller_event_loop as loop
    src=inspect.getsource(loop._handle_g6_science)
    block=src.split("elif op=='G6_S7_DEPTH2_COMPLETION':",1)[1].split("elif op=='G6_R0_POST_GRADUATION_FIBER':",1)[0]
    assert 'infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator' in block
    assert 'infinity_grid.g6_s7_evaluators:s7_depth2_outer_children_generation_evaluator' in block
    assert 'infinity_grid.g6_s7_evaluators:s7_depth2_child_profile_generation_evaluator' in block
    assert 'infinity_grid.g6_s7_evaluators:s7_depth2_parent_profile_multiset_evaluator' in block


def test_s7d2_compact_profile_task_payload_avoids_eager_identity_and_tree_expansion():
    import json
    import infinity_grid.g6_s7_depth2_completion as d2
    identity={"kind":"TEST_CHILD","ports":[0,1,2],"nested":{"x":[1,2,3]}}
    identity_json=json.dumps(identity,sort_keys=True,separators=(",",":"))
    state_json=json.dumps({"tree":{"n":7,"payload":[1,2,3]}},sort_keys=True,separators=(",",":"))
    row={
        "state_token":"tok-1",
        "identity_canonical_bytes":identity_json.encode("utf-8"),
        "canonical_size_bytes":len(identity_json),
        "state_json":state_json,
    }
    refs=("A","B","C","D")
    ops=tuple((i,i+1) for i in range(31))
    task=d2._profile_task("s1",row,outer_context_index=0,start_context_index=0,end_context_index=8,previous_counts=tuple(),basis_refs=refs,operators=ops)
    p=task.payload
    assert "child_identity" not in p and "child_tree" not in p
    assert p["child_identity_json"]==identity_json
    assert p["child_state_json"]==state_json
    assert p["child_token"]=="tok-1"
    assert p["basis_refs"] is refs and p["operator_basis"] is ops
    assert p["previous_counts"]==tuple()


def test_s7d2_compact_profile_evaluator_matches_legacy_payload():
    import json
    import infinity_grid.g6_s7_evaluators as ev
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_kernel_service_providers import _basis_records
    configure_relation_kernel(scope_identity='S7D2:COMPACT:PROFILE:EQUIV',max_cache_entries=512,max_cache_bytes=128*1024*1024,max_relation_cache_entries=512,max_relation_cache_bytes=128*1024*1024)
    basis=_basis_records(); refs=tuple(sorted(basis)); ops=tuple(get_relation_kernel().authority.operators)
    outer_ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_children_generation_evaluator'
    profile_ref='infinity_grid.g6_s7_evaluators:s7_depth2_child_profile_generation_evaluator'
    _bind_s7d2_evaluator(outer_ref)
    outer=ev.s7_depth2_outer_children_generation_evaluator({'state_tree':basis[refs[0]],'basis_refs':refs,'operator_basis':ops,'outer_context_index':0})
    row=outer['states'][0]; child=row['state']['tree']; ident=row['identity']
    legacy={'child_tree':child,'child_identity':ident,'basis_refs':refs,'operator_basis':ops,'start_context_index':0,'end_context_index':8,'previous_counts':tuple()}
    compact={'child_token':'tok-exact','child_state_json':json.dumps({'tree':child},sort_keys=True,separators=(",",":")),'child_identity_json':json.dumps(ident,sort_keys=True,separators=(",",":")),'basis_refs':refs,'operator_basis':ops,'start_context_index':0,'end_context_index':8,'previous_counts':tuple()}
    _bind_s7d2_evaluator(profile_ref); a=ev.s7_depth2_child_profile_generation_evaluator(legacy)
    _bind_s7d2_evaluator(profile_ref); b=ev.s7_depth2_child_profile_generation_evaluator(compact)
    from infinity_grid.canon import canonical_text
    assert canonical_text(a['states'][0]['identity'],pretty=False)==canonical_text(b['states'][0]['identity'],pretty=False)
    assert a['states'][0]['state']['profile_counts']==b['states'][0]['state']['profile_counts']
    assert b['states'][0]['state']['child_token']=='tok-exact'


def test_s7d2_compact_parent_assembly_matches_legacy_payload():
    import json
    import infinity_grid.g6_s7_evaluators as ev
    rows=[tuple(range(8)),tuple(reversed(range(8))),tuple(range(8))]
    legacy=ev.s7_depth2_parent_profile_multiset_evaluator({'inner_prefix_context_count':8,'child_profile_counts':[list(x) for x in rows]})
    compact=ev.s7_depth2_parent_profile_multiset_evaluator({'inner_prefix_context_count':8,'child_profile_counts_json':json.dumps(rows,separators=(",",":"))})
    assert legacy['signature']==compact['signature']
    assert legacy['outcome_count']==compact['outcome_count']


def test_s7d2_a25_full_inner_outer_monotone_preserves_chaining_and_bounds_task_universe():
    import inspect
    import infinity_grid.g6_s7_depth2_completion as d2
    run_src=inspect.getsource(d2.run_g6_s7_depth2_completion)
    assert 'for prefix_count in INNER_PREFIX_SCHEDULE' in run_src
    assert 'for outer_idx in range(EXECUTION_CONTEXT_COUNT)' in run_src
    assert 'if len(g["members"]) > 1' in run_src
    assert 'run_structural_partition' in run_src
    assert 'max_tasks=4464 if panel == "s1" else 247' in run_src
    assert 'run_content_indexed_generation' not in run_src
    assert '_refine(' not in run_src
    assert 'exact_tree_relation_kernel' not in run_src and 'v05_kernel_cache' not in run_src


def test_s7d2_parent_prefix_task_is_exact_registered_component_and_task_count_is_parent_bounded():
    import infinity_grid.g6_s7_depth2_completion as d2
    row={"state_token":"tok","state":{"n":7}}
    refs=("A","B","C","D")
    ops=tuple((i%7,(i+1)%7) for i in range(31))
    t=d2._parent_prefix_task("s1",row,outer_context_index=3,prefix_context_count=8,basis_refs=refs,operators=ops)
    assert t.task_id=="S7D2-PARENT-PREFIX-s1-P008-O003-tok"
    assert t.task_kind=="G6_S7_DEPTH2_PARENT_PREFIX_MONOTONE_COMPONENT"
    assert t.payload["inner_prefix_context_count"]==8
    assert t.payload["outer_context_index"]==3
    assert t.payload["execution_context_normalization"]==d2.FACTOR_SWAP_NORMALIZATION_ID


def test_s7d2_factor_swap_normalization_basis_is_exact_and_124_wide():
    import infinity_grid.g6_s7_evaluators as ev
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_kernel_service_providers import _basis_records, _exact_relation_service
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import bind_kernel_view
    from infinity_grid.v05_kernel_service_providers import build_kernel_service_providers

    configure_relation_kernel(
        scope_identity='S7D2:FACTOR:SWAP:EXACT',
        max_cache_entries=512,max_cache_bytes=128*1024*1024,
        max_relation_cache_entries=512,max_relation_cache_bytes=128*1024*1024,
    )
    ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator'
    spec=get_evaluator_spec(ref); bind_kernel_view(spec,build_kernel_service_providers(spec))
    basis=_basis_records(); refs=tuple(sorted(basis)); ops=tuple(get_relation_kernel().authority.operators)
    reps=ev._s7_factor_swap_representative_contexts(refs,ops)
    assert len(reps)==124
    assert all(pos=='LEFT' for _ref,_op,pos in reps)
    assert all((b,a) in set(ops) for a,b in ops)

    # Exact relation theorem on the full frozen seed basis: swapping the factors
    # and transposing the endpoint operator preserves the canonical child set.
    for left_ref in refs:
        for right_ref in refs:
            left,right=basis[left_ref],basis[right_ref]
            for a,b in ops:
                lr=_exact_relation_service(left,right,(a,b))
                rl=_exact_relation_service(right,left,(b,a))
                assert tuple(lr['canons'])==tuple(rl['canons'])


def test_s7d2_factor_swap_normalized_inner_vector_expands_to_literal_248_vector():
    import infinity_grid.g6_s7_evaluators as ev
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_kernel_service_providers import _basis_records, _exact_relation_service
    from infinity_grid.v05_kernel_services import current_kernel_view
    configure_relation_kernel(
        scope_identity='S7D2:FACTOR:SWAP:INNER:EQUIV',
        max_cache_entries=1024,max_cache_bytes=256*1024*1024,
        max_relation_cache_entries=1024,max_relation_cache_bytes=256*1024*1024,
    )
    basis=_basis_records(); refs=tuple(sorted(basis)); ops=tuple(get_relation_kernel().authority.operators)
    op_index={op:i for i,op in enumerate(ops)}
    ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator'
    _bind_s7d2_evaluator(ref)
    # One exact child is enough to verify lossless inner-coordinate expansion.
    rel=_exact_relation_service(basis[refs[0]],basis[refs[1]],ops[0])
    child=rel['children'][0]
    metrics0={'child_prefix_cache_hits':0,'child_prefix_cache_evictions':0,'exact_relation_profile_call_count':0,
              'attempted_owner_pairs':0,'legal_owner_pairs':0,'rooted_owner_pair_candidates':0,'child_canon_constructions':0}
    full=ev._d2_child_count_prefix(current_kernel_view(),child,basis=basis,basis_refs=refs,operators=ops,
        prefix_count=248,metrics=dict(metrics0),execution_context_normalization=None)
    norm=ev._d2_child_count_prefix(current_kernel_view(),child,basis=basis,basis_refs=refs,operators=ops,
        prefix_count=124,metrics=dict(metrics0),execution_context_normalization='FACTOR_SWAP_LEFT_V1')
    expanded=[]
    for ri,_r in enumerate(refs):
        base=ri*31
        for oi,(a,b) in enumerate(ops):
            expanded.append(norm[base+oi])
            expanded.append(norm[base+op_index[(b,a)]])
    assert tuple(expanded)==tuple(full)

def test_s7d2_active_factor_swap_execution_is_124_by_124_and_scientific_observer_stays_248():
    import inspect
    import infinity_grid.g6_s7_depth2_completion as d2
    src=inspect.getsource(d2.run_g6_s7_depth2_completion)
    assert d2.EXECUTION_CONTEXT_COUNT==124
    assert d2.INNER_PREFIX_SCHEDULE==(124,)
    assert 'for outer_idx in range(EXECUTION_CONTEXT_COUNT)' in src
    assert 'ordinary_context_count": 248' in src
    assert 'execution_context_count": EXECUTION_CONTEXT_COUNT' in src
    assert 'FACTOR_SWAP_NORMALIZATION_ID' in inspect.getsource(d2._parent_prefix_task)


def test_s7d2_parent_prefix_worker_child_cache_disabled_after_zero_hit_audit():
    import infinity_grid.g6_s7_evaluators as ev
    assert ev._D2_CHILD_PREFIX_CACHE_MAX == 0
    ev._D2_CHILD_PREFIX_CACHE.clear()
