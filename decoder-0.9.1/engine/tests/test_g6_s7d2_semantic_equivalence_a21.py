from __future__ import annotations

# A21 bounded semantic-equivalence validation for the accepted S7D2 optimizer.
# The two parent states below are frozen members of the certified A6 S1 depth-1
# collision class S1:D1:0000:fd33820b02cf0b4e95e10372daf7c3287622190fb19bb7b90ee39cae566d674e.

PARENT_A = {'H_classes': ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'], 'edge_operators': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [4, 6]], 'edges': [[0, 1], [1, 2], [2, 3], [2, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [9, 11], [0, 10]], 'n': 12}
PARENT_B = {'H_classes': ['C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C'], 'edge_operators': [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [4, 6]], 'edges': [[0, 1], [1, 2], [2, 3], [2, 4], [5, 6], [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [2, 6]], 'n': 12}


def _bind(ref):
    from infinity_grid.v05_stage_registry import get_evaluator_spec
    from infinity_grid.v05_kernel_services import bind_kernel_view
    from infinity_grid.v05_kernel_service_providers import build_kernel_service_providers
    spec=get_evaluator_spec(ref)
    bind_kernel_view(spec,build_kernel_service_providers(spec))


def _fixture():
    from infinity_grid.exact_tree_relation_kernel import configure_relation_kernel, get_relation_kernel
    from infinity_grid.v05_kernel_service_providers import _basis_records
    configure_relation_kernel(scope_identity='S7D2:A21:SEMANTIC:EQUIV',max_cache_entries=0,max_cache_bytes=1,max_relation_cache_entries=2048,max_relation_cache_bytes=128*1024*1024)
    basis=_basis_records(); refs=tuple(sorted(basis)); ops=tuple(get_relation_kernel().authority.operators)
    return basis,refs,ops


def _expand_norm_bag(norm_signature, ops):
    import infinity_grid.g6_s7_evaluators as ev
    assert norm_signature[0]=='S7_D2_OUTER_SUCCESSOR_SIG1_FACTOR_SWAP_REP_PREFIX_MULTISET'
    assert norm_signature[1]==124
    op_index={op:i for i,op in enumerate(ops)}
    values=[]
    for value,multiplicity in norm_signature[2]:
        tag,counts=value
        assert tag=='S7_CHILD_D1_BRANCH_COUNT_PREFIX' and len(counts)==124
        expanded=[]
        for ri in range(4):
            base=ri*31
            for oi,(a,b) in enumerate(ops):
                expanded.append(counts[base+oi])
                expanded.append(counts[base+op_index[(b,a)]])
        values.extend([('S7_CHILD_D1_BRANCH_COUNT_PREFIX',tuple(expanded))]*int(multiplicity))
    return ev._multiset_signature(values)


def test_a21_fixed_a6_collision_pair_literal_248_equals_normalized_124_joint_bags():
    import infinity_grid.g6_s7_evaluators as ev
    basis,refs,ops=_fixture()
    ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator'
    asym=next(op for op in ops if op[0]!=op[1])
    oi=ops.index(asym)
    # Use D2_PATH to keep the bounded fixture smaller while preserving multiple exact children.
    ri=refs.index('D2_PATH')
    literal_outer=ri*31*2 + oi*2  # LEFT coordinate in frozen 248 order
    normalized_outer=ri*31 + oi
    for parent in (PARENT_A,PARENT_B):
        _bind(ref)
        literal=ev.s7_depth2_outer_prefix_evaluator({'state_tree':parent,'basis_refs':refs,'operator_basis':ops,'outer_context_index':literal_outer,'inner_prefix_context_count':248})
        _bind(ref)
        normalized=ev.s7_depth2_outer_prefix_evaluator({'state_tree':parent,'basis_refs':refs,'operator_basis':ops,'outer_context_index':normalized_outer,'inner_prefix_context_count':124,'execution_context_normalization':'FACTOR_SWAP_LEFT_V1'})
        assert literal['outcome_count']>1
        assert literal['outcome_count']==normalized['outcome_count']
        assert literal['signature'][2]==_expand_norm_bag(normalized['signature'],ops)


def test_a21_outer_right_coordinate_maps_exactly_to_transposed_left_representative():
    import infinity_grid.g6_s7_evaluators as ev
    basis,refs,ops=_fixture()
    ref='infinity_grid.g6_s7_evaluators:s7_depth2_outer_prefix_evaluator'
    asym=next(op for op in ops if op[0]!=op[1])
    oi=ops.index(asym); ti=ops.index((asym[1],asym[0])); ri=refs.index('D2_PATH')
    literal_right=ri*31*2 + oi*2 + 1
    normalized_transposed=ri*31 + ti
    _bind(ref)
    literal=ev.s7_depth2_outer_prefix_evaluator({'state_tree':PARENT_A,'basis_refs':refs,'operator_basis':ops,'outer_context_index':literal_right,'inner_prefix_context_count':248})
    _bind(ref)
    normalized=ev.s7_depth2_outer_prefix_evaluator({'state_tree':PARENT_A,'basis_refs':refs,'operator_basis':ops,'outer_context_index':normalized_transposed,'inner_prefix_context_count':124,'execution_context_normalization':'FACTOR_SWAP_LEFT_V1'})
    assert literal['outcome_count']==normalized['outcome_count']
    assert literal['signature'][2]==_expand_norm_bag(normalized['signature'],ops)


def test_a21_empty_successor_relation_has_exact_empty_joint_bag():
    import infinity_grid.g6_s7_evaluators as ev
    got=ev.s7_depth2_parent_profile_multiset_evaluator({'inner_prefix_context_count':8,'child_profile_counts':[]})
    assert got['outcome_count']==0
    assert got['signature'][2]==tuple()


def test_a21_joint_vectors_preserve_correlation_that_equal_marginals_lose():
    import infinity_grid.g6_s7_evaluators as ev
    a=[('S7_CHILD_D1_BRANCH_COUNT_PREFIX',(1,4)),('S7_CHILD_D1_BRANCH_COUNT_PREFIX',(3,2))]
    b=[('S7_CHILD_D1_BRANCH_COUNT_PREFIX',(1,2)),('S7_CHILD_D1_BRANCH_COUNT_PREFIX',(3,4))]
    assert sorted(x[1][0] for x in a)==sorted(x[1][0] for x in b)
    assert sorted(x[1][1] for x in a)==sorted(x[1][1] for x in b)
    assert ev._multiset_signature(a)!=ev._multiset_signature(b)


def test_a21_singleton_skip_is_exact_monotone_partition_refinement():
    import infinity_grid.g6_s7_depth2_completion as d2
    rows=[{'state_token':x,'state':{'n':1}} for x in ('a','b','c')]
    groups=[{'depth1_class_id':'C','members':rows}]
    p1=8; o1=0
    m1={f'S7D2-PARENT-PREFIX-s1-P{p1:03d}-O{o1:03d}-a':'X',f'S7D2-PARENT-PREFIX-s1-P{p1:03d}-O{o1:03d}-b':'Y',f'S7D2-PARENT-PREFIX-s1-P{p1:03d}-O{o1:03d}-c':'Y'}
    g1=d2._refine_parent_prefix(groups,m1,panel='s1',prefix_context_count=p1,outer_context_index=o1)
    assert sorted(tuple(r['state_token'] for r in g['members']) for g in g1)==[('a',),('b','c')]
    # Later coordinates are evaluated only for the surviving non-singleton b/c block.
    p2=32; o2=1
    m2={f'S7D2-PARENT-PREFIX-s1-P{p2:03d}-O{o2:03d}-b':'B',f'S7D2-PARENT-PREFIX-s1-P{p2:03d}-O{o2:03d}-c':'C'}
    g2=d2._refine_parent_prefix(g1,m2,panel='s1',prefix_context_count=p2,outer_context_index=o2)
    assert sorted(tuple(r['state_token'] for r in g['members']) for g in g2)==[('a',),('b',),('c',)]
