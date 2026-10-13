from __future__ import annotations

import inspect

from infinity_grid.g6_stage_executors import _basis
from infinity_grid.g6_s5r_crw_kernel import exact_observer_state, write_exact_observer_states
from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel

def test_crw1_abstract_write_signature_has_no_raw_parent_parameter():
    names=list(inspect.signature(write_exact_observer_states).parameters)
    assert names[:3]==['left_state','right_state','operator']
    assert 'left' not in names and 'right' not in names and 'parent' not in names

def test_crw1_q_only_write_matches_raw_on_basis_panel():
    b=_basis(); k=ExactTreeRelationKernel(max_cache_entries=64)
    probe=b['D2_PATH']
    for left_name,right_name,op in [
        ('D2_PATH','D2_BROOM',(0,0)),
        ('D4_PATH','D2_PATH',(1,0)),
        ('D4_BROOM','D4_PATH',(0,1)),
    ]:
        l,r=b[left_name],b[right_name]
        ql=exact_observer_state(l,probe,kernel=k)
        qr=exact_observer_state(r,probe,kernel=k)
        abstract=write_exact_observer_states(ql,qr,op,probe=probe,kernel=k)
        raw=k.relation(k.prepare(l),k.prepare(r),op)
        expected=tuple(sorted({exact_observer_state(c,probe,kernel=k) for c in raw.children},key=repr))
        assert abstract==expected

def test_crw1_empty_relation_is_exact_empty_set():
    b=_basis(); k=ExactTreeRelationKernel(max_cache_entries=64); probe=b['D2_PATH']
    found=None
    for l in b.values():
        for r in b.values():
            for op in k.authority.operators:
                if not k.relation(k.prepare(l),k.prepare(r),op).canons:
                    found=(l,r,op); break
            if found: break
        if found: break
    if found:
        l,r,op=found
        ql=exact_observer_state(l,probe,kernel=k)
        qr=exact_observer_state(r,probe,kernel=k)
        assert write_exact_observer_states(ql,qr,op,probe=probe,kernel=k)==tuple()


def test_crw1_q_only_write_diagnostics_reuse_same_decoded_parents():
    from infinity_grid.g6_stage_executors import _basis
    from infinity_grid.g6_s5r_crw_kernel import exact_observer_state, write_exact_observer_states
    from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel
    b=_basis(); k=ExactTreeRelationKernel(max_cache_entries=64); probe=b['D2_PATH']
    l,r=b['D2_PATH'],b['D2_BROOM']
    ql=exact_observer_state(l,probe,kernel=k); qr=exact_observer_state(r,probe,kernel=k)
    d={}
    out=write_exact_observer_states(ql,qr,(0,0),probe=probe,kernel=k,diagnostics=d)
    assert out
    assert d['left_decoded_unrooted_canon']==k.prepare(l).unrooted_canon
    assert d['right_decoded_unrooted_canon']==k.prepare(r).unrooted_canon
    assert d['abstract_exact_child_count']>=d['abstract_projected_child_state_count']
