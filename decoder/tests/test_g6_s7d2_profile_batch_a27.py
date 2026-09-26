from __future__ import annotations


def test_kernel_relation_profile_batch_exactly_matches_scalar_all_31_ops_equal_and_unequal_sizes():
    from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel
    from infinity_grid.g5_capabilities import _g5_s1_carriers
    carriers=_g5_s1_carriers(); refs=tuple(sorted(carriers))
    pairs=((carriers['D2_PATH'],carriers['D2_BROOM']), (carriers['D4_PATH'],carriers['D2_PATH']))
    for i,(left,right) in enumerate(pairs):
        kb=ExactTreeRelationKernel(scope_identity=f'A27:BATCH:{i}')
        ops=tuple(kb.authority.operators)
        batched=kb.relation_profile_batch(left,right,ops)
        ks=ExactTreeRelationKernel(scope_identity=f'A27:SCALAR:{i}')
        scalar=tuple(ks.relation_profile(left,right,op) for op in ops)
        assert batched==scalar
        assert len(batched)==31


def test_public_batch_provider_preserves_operator_order_and_scalar_counts():
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel,get_relation_kernel
    from infinity_grid.v05_kernel_service_providers import _basis_records,_exact_relation_profile_service,_exact_relation_profile_batch_service
    configure_relation_kernel(scope_identity='A27:PROVIDER')
    basis=_basis_records(); left=basis['D4_PATH']; right=basis['D2_PATH']
    ops=tuple(get_relation_kernel().authority.operators)
    got=_exact_relation_profile_batch_service(left,right,ops)
    expected=tuple(_exact_relation_profile_service(left,right,op)['exact_outcome_count'] for op in ops)
    assert tuple(got['exact_outcome_counts'])==expected
    assert len(expected)==31


def test_s7d2_full_p124_batch_path_matches_scalar_reference_exactly():
    import infinity_grid.g6_s7_evaluators as ev
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel,get_relation_kernel
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import bind_kernel_view,current_kernel_view
    from infinity_grid.v05_kernel_service_providers import _basis_records,build_kernel_service_providers,_exact_relation_service,_exact_relation_profile_service
    configure_relation_kernel(scope_identity='A27:S7D2:P124',max_cache_entries=512,max_cache_bytes=128*1024*1024)
    ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator'
    spec=get_evaluator_spec(ref); bind_kernel_view(spec,build_kernel_service_providers(spec))
    ev._BASIS_RECORD_CACHE=None; ev._D2_CHILD_PREFIX_CACHE.clear()
    basis=_basis_records(); refs=tuple(sorted(basis)); ops=tuple(get_relation_kernel().authority.operators)
    # Find one real exact child deterministically.
    child=None
    for op in ops:
        rel=_exact_relation_service(basis['D2_PATH'],basis['D2_BROOM'],op)
        if rel['children']:
            child=rel['children'][0]; break
    assert child is not None
    metrics={'exact_relation_profile_call_count':0,'exact_relation_profile_batch_call_count':0}
    got=ev._d2_child_count_prefix(current_kernel_view(), child, basis=basis,basis_refs=refs,operators=ops,prefix_count=124,metrics=metrics,execution_context_normalization='FACTOR_SWAP_LEFT_V1')
    expected=[]
    for bref in refs:
        seed=basis[bref]
        for op in ops:
            expected.append(int(_exact_relation_profile_service(child,seed,op)['exact_outcome_count']))
    assert got==tuple(expected)
    assert metrics['exact_relation_profile_call_count']==124
    assert metrics['exact_relation_profile_batch_call_count']==0
    assert metrics['exact_relation_profile_family_call_count']==1
