from infinity_grid.g6_stage_executors import _basis
from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel
from infinity_grid.g6_fresh_marker_kernel import marker_observer_state, decode_marker_observer_state, write_marker_states
from infinity_grid.g6_marker_reference_oracle import legacy_marker_decode_oracle


def _kernel():
    return ExactTreeRelationKernel(
        scope_identity="TEST_G6_MARKER_SHARED_CACHE",
        max_cache_entries=128,max_cache_bytes=32*1024*1024,
        max_relation_cache_entries=128,max_relation_cache_bytes=32*1024*1024,
        max_observer_q_cache_entries=256,max_observer_q_cache_bytes=64*1024*1024,
        max_observer_decode_cache_entries=128,max_observer_decode_cache_bytes=32*1024*1024,
    )

def test_marker_q_cache_is_exact_and_semantics_preserving():
    b=_basis(); k=_kernel(); x=b["D2_PATH"]
    q1=marker_observer_state(x,kernel=k); m1=k.metrics()
    q2=marker_observer_state(x,kernel=k); m2=k.metrics()
    assert q2==q1
    assert m2["observer_q_cache_hits"]>m1["observer_q_cache_hits"]
    legacy=legacy_marker_decode_oracle(q1)
    assert legacy["candidate_count"]==1
    assert legacy["candidates"][0]==k.prepare(x).unrooted_canon

def test_marker_decode_cache_preserves_full_child_agreement_semantics():
    b=_basis(); k=_kernel(); x=b["D2_BROOM"]
    q=marker_observer_state(x,kernel=k)
    p1,s1=decode_marker_observer_state(q,kernel=k); m1=k.metrics()
    p2,s2=decode_marker_observer_state(q,kernel=k); m2=k.metrics()
    assert k.prepare(p1).unrooted_canon==k.prepare(x).unrooted_canon
    assert k.prepare(p2).unrooted_canon==k.prepare(x).unrooted_canon
    assert s1==s2
    assert m2["observer_decode_cache_hits"]>m1["observer_decode_cache_hits"]

def test_marker_write_fuses_raw_projection_without_second_relation_pass():
    b=_basis(); k=_kernel(); left,right=b["D2_PATH"],b["D2_BROOM"]; op=(0,0)
    ql=marker_observer_state(left,kernel=k); qr=marker_observer_state(right,kernel=k)
    diagnostics={}
    abstract=write_marker_states(ql,qr,op,kernel=k,diagnostics=diagnostics)
    assert diagnostics["fused_raw_projection"] is True
    assert diagnostics["projected_marker_states"]==abstract
    assert len(diagnostics["raw_child_rows"])==diagnostics["raw_child_count"]
    for record,q,expected_can in diagnostics["raw_child_rows"]:
        dec,_=decode_marker_observer_state(q,kernel=k)
        assert k.prepare(dec).unrooted_canon==expected_can

def test_marker_write_fused_projection_matches_independent_raw_reobservation():
    b=_basis(); k=_kernel(); left,right=b["D2_PATH"],b["D2_BROOM"]; op=(0,0)
    ql=marker_observer_state(left,kernel=k); qr=marker_observer_state(right,kernel=k)
    diagnostics={}
    abstract=write_marker_states(ql,qr,op,kernel=k,diagnostics=diagnostics)
    rel=k.relation(left,right,op)
    independent=tuple(sorted({marker_observer_state(c,kernel=k) for c in rel.children},key=repr))
    assert abstract==independent


def test_marker_q_seeds_exact_decode_cache_on_marker_free_domain():
    b=_basis(); k=_kernel(); x=b["D4_PATH"]
    q=marker_observer_state(x,kernel=k)
    before=k.metrics()
    dec,stats=decode_marker_observer_state(q,kernel=k)
    after=k.metrics()
    assert k.prepare(dec).unrooted_canon==k.prepare(x).unrooted_canon
    assert stats["cache_seeded_from_exact_marker_observation"] is True
    assert after["observer_decode_cache_hits"]>before["observer_decode_cache_hits"]
