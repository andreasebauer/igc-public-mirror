from __future__ import annotations

from infinity_grid.adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter
from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel
from infinity_grid.g6_stage_executors import _basis


def _profile_vs_retained_canon(kernel, left, right, op):
    lp, rp = kernel.prepare(left), kernel.prepare(right)
    prof = kernel.relation_profile(lp, rp, op)
    canons, _pairs, attempted, legal, candidates, _reused, _rerooted = kernel._relation_canon_core(lp, rp, op)
    assert prof.exact_outcome_count == len(canons)
    assert prof.attempted_owner_pairs == attempted
    assert prof.legal_owner_pairs == legal
    assert prof.rooted_owner_pair_candidates == candidates
    return prof, lp, rp, len(canons)


def _all_c_tree(n, edges):
    edges=tuple(tuple(map(int,e)) for e in edges)
    return DecoratedG4Tree(
        int(n), edges, tuple("C" for _ in range(int(n))),
        tuple((0,0) for _ in edges),
    )


def test_a24r_equal_size_is_conservative_exact_fallback():
    b=_basis(); k=ExactTreeRelationKernel(adapter=G4AcceptedAdapter())
    prof, lp, rp, exact = _profile_vs_retained_canon(k,b['D2_PATH'],b['D2_BROOM'],(0,0))
    assert lp.tree.n == rp.tree.n == 5
    assert exact == prof.exact_outcome_count
    assert prof.child_canon_constructions == prof.rooted_owner_pair_candidates
    assert k.metrics()['relation_profile_bridge_fingerprint_fast_hits'] == 0
    assert k.metrics()['relation_profile_bridge_fingerprint_fallbacks'] == 1


def test_a24r_exact_rooted_cavity_ambiguity_fixture_falls_back_10_to_9():
    # Registered recovery counterexample: the larger tree contains an internal
    # 3-vertex rooted cavity indistinguishable from the external path-3 at the
    # relevant cut label.  Candidate product 10 therefore collapses to 9 exact
    # outputs and MUST never take the product fast path.
    small=_all_c_tree(3,((0,1),(1,2)))
    large=_all_c_tree(6,((0,1),(0,4),(1,2),(1,3),(4,5)))
    k=ExactTreeRelationKernel(adapter=G4AcceptedAdapter(),max_relation_cache_entries=0,max_relation_cache_bytes=0)
    prof, lp, rp, exact = _profile_vs_retained_canon(k,small,large,(0,0))
    assert prof.rooted_owner_pair_candidates == 10
    assert exact == 9
    assert k._new_bridge_fingerprint_is_unique(lp,rp,(0,0)) is False
    assert prof.child_canon_constructions == 10
    assert k.metrics()['relation_profile_bridge_fingerprint_fallbacks'] == 1


def test_a24r_positive_unequal_size_fastpath_is_differentially_exact():
    b=_basis(); ad=G4AcceptedAdapter(); k=ExactTreeRelationKernel(adapter=ad,max_relation_cache_entries=0,max_relation_cache_bytes=0)
    # Build a real larger retained G6-style carrier, then choose the first
    # operator for which the exact certificate succeeds.  The test requires at
    # least one positive case and compares it against the retained canonical path.
    rel=k.relation(b['D2_PATH'],b['D2_BROOM'],(0,0))
    assert rel.children
    large=rel.children[0]
    hit=None
    for seed_name in ('D2_PATH','D2_BROOM','D4_PATH','D4_BROOM'):
        seed=b[seed_name]
        if seed.n == large.n: continue
        for op in ad.operator_basis():
            lp,rp=k.prepare(large),k.prepare(seed)
            if not (lp.legal_owners(op[0]) and rp.legal_owners(op[1])): continue
            if k._new_bridge_fingerprint_is_unique(lp,rp,op):
                hit=(seed,op); break
        if hit is not None: break
    assert hit is not None
    prof, _lp, _rp, exact = _profile_vs_retained_canon(k,large,hit[0],hit[1])
    assert prof.exact_outcome_count == exact == prof.rooted_owner_pair_candidates
    assert prof.child_canon_constructions == 0
    assert k.metrics()['relation_profile_bridge_fingerprint_fast_hits'] >= 1
    assert k.metrics()['relation_profile_bridge_fingerprint_canon_avoided'] >= prof.rooted_owner_pair_candidates


def test_a24r_full_bounded_differential_panel_matches_retained_canonical_core():
    b=_basis(); ad=G4AcceptedAdapter(); ops=tuple(ad.operator_basis())
    k=ExactTreeRelationKernel(adapter=ad,max_relation_cache_entries=0,max_relation_cache_bytes=0)
    states=[b['D2_PATH'],b['D2_BROOM'],b['D4_PATH'],b['D4_BROOM']]
    for left,right,op in ((b['D2_PATH'],b['D4_BROOM'],(0,1)),(b['D4_PATH'],b['D2_BROOM'],(4,6))):
        rel=k.relation(left,right,op)
        if rel.children: states.append(rel.children[0])
    comparisons=0
    for state in states:
        for seed_name in ('D2_PATH','D4_BROOM'):
            seed=b[seed_name]
            for op in ops:
                _profile_vs_retained_canon(k,state,seed,op)
                comparisons += 1
    assert comparisons >= 4*2*31
    m=k.metrics()
    assert m['relation_profile_bridge_fingerprint_fast_hits'] > 0
    assert m['relation_profile_bridge_fingerprint_fallbacks'] > 0
    assert m['relation_profile_bridge_fingerprint_canon_avoided'] > 0


def test_a24r_provider_exposes_operational_fastpath_metrics_only():
    import inspect
    import infinity_grid.v05_kernel_service_providers as p
    import infinity_grid.g6_s7_evaluators as e
    assert "'relation_profile_'" in inspect.getsource(p._relation_metrics)
    src=inspect.getsource(e.s7_depth2_child_profile_generation_evaluator)
    assert 'relation_profile_bridge_fingerprint_fast_hits' in src
    assert 'relation_profile_bridge_fingerprint_fallbacks' in src
