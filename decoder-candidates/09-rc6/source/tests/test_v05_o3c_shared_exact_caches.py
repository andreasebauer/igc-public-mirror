from __future__ import annotations

from infinity_grid.exact_tree_relation_kernel import ExactTreeRelationKernel
from infinity_grid.g5_capabilities import _g5_s1_carriers
from infinity_grid.g6_s5r_crw_kernel import exact_observer_state, decode_exact_observer_state


def _kernel(*, cached: bool) -> ExactTreeRelationKernel:
    if cached:
        return ExactTreeRelationKernel(
            scope_identity='O3C:CACHED', max_cache_entries=64, max_cache_bytes=16*1024*1024,
            max_relation_cache_entries=64, max_relation_cache_bytes=16*1024*1024,
            max_observer_q_cache_entries=128, max_observer_q_cache_bytes=16*1024*1024,
            max_observer_decode_cache_entries=128, max_observer_decode_cache_bytes=16*1024*1024,
        )
    return ExactTreeRelationKernel(
        scope_identity='O3C:UNCACHED', max_cache_entries=64, max_cache_bytes=16*1024*1024,
        max_relation_cache_entries=0, max_relation_cache_bytes=0,
        max_observer_q_cache_entries=0, max_observer_q_cache_bytes=0,
        max_observer_decode_cache_entries=0, max_observer_decode_cache_bytes=0,
    )


def test_o3c_relation_cache_is_exact_and_hits_only_after_exact_key_match():
    b=_g5_s1_carriers(); left,right=b['D4_PATH'],b['D4_BROOM']; op=(0,0)
    cold=_kernel(cached=False).relation(left,right,op)
    k=_kernel(cached=True); first=k.relation(left,right,op); before=k.metrics(); second=k.relation(left,right,op); after=k.metrics()
    assert first.canons==cold.canons==second.canons
    assert first.children==cold.children==second.children
    assert after['relation_cache_hits']==before['relation_cache_hits']+1
    # Reversing complete exact parents is a different key even if a future case happened to share a digest.
    k.relation(right,left,op); later=k.metrics()
    assert later['relation_cache_misses']>=after['relation_cache_misses']+1


def test_o3c_observer_q_and_decode_caches_preserve_exact_outputs():
    b=_g5_s1_carriers(); parent=b['D4_BROOM']; probe=b['D2_PATH']; op=(0,0)
    uncached=_kernel(cached=False); q0=exact_observer_state(parent,probe,op,kernel=uncached); d0,s0=decode_exact_observer_state(q0,probe,op,kernel=uncached)
    cached=_kernel(cached=True); q1=exact_observer_state(parent,probe,op,kernel=cached); d1,s1=decode_exact_observer_state(q1,probe,op,kernel=cached)
    m1=cached.metrics(); q2=exact_observer_state(parent,probe,op,kernel=cached); d2,s2=decode_exact_observer_state(q2,probe,op,kernel=cached); m2=cached.metrics()
    assert q0==q1==q2
    assert cached.prepare(d0).unrooted_canon==cached.prepare(d1).unrooted_canon==cached.prepare(d2).unrooted_canon
    assert s1==s2  # cache does not alter the historical diagnostic payload
    assert m2['observer_q_cache_hits']>=m1['observer_q_cache_hits']+1
    assert m2['observer_decode_cache_hits']>=m1['observer_decode_cache_hits']+1


def test_o3c_exact_caches_are_bounded_and_clearable():
    b=_g5_s1_carriers(); k=ExactTreeRelationKernel(
        scope_identity='O3C:BOUND',max_cache_entries=4,max_cache_bytes=1024*1024,
        max_relation_cache_entries=1,max_relation_cache_bytes=16*1024*1024,
        max_observer_q_cache_entries=1,max_observer_q_cache_bytes=16*1024*1024,
        max_observer_decode_cache_entries=1,max_observer_decode_cache_bytes=16*1024*1024)
    k.relation(b['D2_PATH'],b['D2_BROOM'],(0,0)); k.relation(b['D4_PATH'],b['D4_BROOM'],(0,0))
    assert k.metrics()['relation_cache_entries']<=1
    assert k.metrics()['relation_cache_evictions']>=1
    k.clear(); m=k.metrics()
    assert m['relation_cache_entries']==0 and m['observer_q_cache_entries']==0 and m['observer_decode_cache_entries']==0


def test_o3c_cache_state_is_partition_independent_simulated_1_vs_4_workers():
    b=_g5_s1_carriers(); names=('D2_PATH','D2_BROOM','D4_PATH','D4_BROOM'); ops=((0,0),(0,1),(1,0),(2,4))
    tasks=[(i,names[i%4],names[(i+1)%4],ops[i%4]) for i in range(12)]
    def run(parts:int):
        out=[]
        for part in range(parts):
            k=_kernel(cached=True)
            for i,l,r,op in tasks:
                if i%parts!=part: continue
                rel=k.relation(b[l],b[r],op)
                out.append((i,rel.canons))
        return tuple(sorted(out))
    assert run(1)==run(4)
