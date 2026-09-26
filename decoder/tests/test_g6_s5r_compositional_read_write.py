from infinity_grid.g6_stage_executors import _basis
from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel
from infinity_grid.g6_s5r_crw_kernel import parent_candidates_from_observer_canons, parent_candidates_from_observer_canons_profiled, _legacy_parent_candidates_from_observer_canons, tree_from_rooted_canon, attachment_response_descriptor, exact_observer_state, decode_exact_observer_state
from infinity_grid.v05_stage_architecture import audit_callable
from infinity_grid.g6_s5r_crw_evaluators import observer_inversion_descriptor_evaluator


def test_crw_evaluator_passes_controller_stage_architecture_gate():
    assert audit_callable(observer_inversion_descriptor_evaluator)['status']=='PASS'


def test_crw_frozen_observer_inverts_all_four_seed_carriers():
    k=ExactTreeRelationKernel(max_cache_entries=0); basis=_basis(); probe=basis['D2_PATH']
    for name,parent in sorted(basis.items()):
        rel=k.relation(parent,probe,(0,0))
        cand=parent_candidates_from_observer_canons(rel.canons,probe,(0,0))
        actual=k.prepare(parent,cache=False).unrooted_canon
        assert cand==(actual,), name
        rebuilt=tree_from_rooted_canon(cand[0])
        assert k.prepare(rebuilt,cache=False).unrooted_canon==actual
        assert attachment_response_descriptor(rebuilt)==attachment_response_descriptor(parent)


def test_crw_evaluator_is_in_accepted_worker_registry():
    from infinity_grid.v05_stage_registry import CONTROLLER_ONLY_WORKER_EVALUATORS
    assert 'infinity_grid.g6_s5r_crw_evaluators:observer_inversion_descriptor_evaluator' in CONTROLLER_ONLY_WORKER_EVALUATORS


def test_crw_s1_identifier_uses_exact_canon_repr_when_historical_index_has_no_id_fields():
    from infinity_grid.g6_s5r_compositional_read_write import _s1_identifier
    a=_s1_identifier({'exact_canon_repr':'canon-a'})
    b=_s1_identifier({'exact_canon_repr':'canon-b'})
    assert len(a)==64 and len(b)==64 and a!=b


def _first_child(k,left,right):
    rel=k.relation(left,right,(0,0))
    assert rel.children
    return rel.children[0]


def test_crw_fast_inversion_is_structurally_exact_on_live_size_strata():
    k=ExactTreeRelationKernel(max_cache_entries=128); b=_basis(); probe=b['D2_PATH']
    # 15/17/19/21-vertex parent strata matching the certified higher-U support.
    p10=_first_child(k,b['D2_PATH'],b['D2_BROOM'])
    p12=_first_child(k,b['D2_PATH'],b['D4_PATH'])
    p14=_first_child(k,b['D4_PATH'],b['D4_BROOM'])
    parents=(
      _first_child(k,p10,b['D2_PATH']),
      _first_child(k,p10,b['D4_PATH']),
      _first_child(k,p12,b['D4_BROOM']),
      _first_child(k,p14,b['D4_PATH']),
    )
    assert tuple(p.n for p in parents)==(15,17,19,21)
    for parent in parents:
        rel=k.relation(parent,probe,(0,0))
        legacy=_legacy_parent_candidates_from_observer_canons(rel.canons,probe,(0,0))
        fast,stats=parent_candidates_from_observer_canons_profiled(rel.canons,probe,(0,0))
        actual=k.prepare(parent,cache=False).unrooted_canon
        assert fast==legacy==(actual,)
        assert stats['component_splits'] <= stats['size_compatible_edges']
        assert stats['size_compatible_edges'] < stats['operator_edges']
        assert stats['component_canon_calls'] < 2*stats['operator_edges']


def test_crw_verified_decode_fast_path_is_exact_on_live_size_strata():
    k=ExactTreeRelationKernel(max_cache_entries=128); b=_basis(); probe=b['D2_PATH']
    p10=_first_child(k,b['D2_PATH'],b['D2_BROOM'])
    p12=_first_child(k,b['D2_PATH'],b['D4_PATH'])
    p14=_first_child(k,b['D4_PATH'],b['D4_BROOM'])
    parents=(
      _first_child(k,p10,b['D2_PATH']),
      _first_child(k,p10,b['D4_PATH']),
      _first_child(k,p12,b['D4_BROOM']),
      _first_child(k,p14,b['D4_PATH']),
    )
    for parent in parents:
        q=exact_observer_state(parent,probe,(0,0),kernel=k)
        decoded,stats=decode_exact_observer_state(q,probe,(0,0),kernel=k)
        assert k.prepare(decoded,cache=False).unrooted_canon==k.prepare(parent,cache=False).unrooted_canon
        assert stats['fast_verified_decode']==1
        assert 1 <= stats['fast_verified_outcomes_examined'] <= len(q)
        assert stats['observer_outcomes_short_circuited']==len(q)-stats['fast_verified_outcomes_examined']


def test_crw_verified_decode_falls_back_without_changing_empty_state_failure():
    k=ExactTreeRelationKernel(max_cache_entries=0); probe=_basis()['D2_PATH']
    try:
        decode_exact_observer_state((),probe,(0,0),kernel=k)
    except Exception as exc:
        assert 'CRW1_OBSERVER_DECODE_NOT_UNIQUE:0' in str(exc)
    else:
        raise AssertionError('empty q must remain non-decodable')
