from collections import defaultdict
from itertools import combinations_with_replacement, permutations, product
from types import SimpleNamespace
import random

import pytest

from infinity_grid.o7_edge_canonicalization import (
    canonicalize_edges, canonicalize_with_witness, state_identity,
    verify_relabeling_witness, _edge,
)
from infinity_grid.o6_resource_observer import ResourceMerkle


def context(owners=2, capacity=2):
    site = ((capacity, 2, 0, 0, 0, 0, 0), (capacity, 2, 0, 0, 0, 0, 0))
    block = (site, site)
    bag = (((block,),),)
    parent = SimpleNamespace(h6=(bag, bag), exact_color='TEST_PARENT', pid='P')
    return SimpleNamespace(n=owners, parents=(parent,) * owners)


PATHS = tuple((a, 0, 0, 0, i) for a in range(2) for i in range(2))


def parent_maps():
    for bags in permutations(range(2)):
        for sites in product(tuple(permutations(range(2))), repeat=2):
            yield {p:(bags[p[0]],0,0,0,sites[p[0]][p[4]]) for p in PATHS}


def exhaustive_key(edges):
    best = None
    for owners in permutations(range(2)):
        for left, right in product(tuple(parent_maps()), repeat=2):
            maps = (left, right)
            key = tuple(sorted(_edge(owners[c],maps[c][p],t,owners[d],maps[d][q],u)
                               for c,p,t,d,q,u in ((e[0],tuple(e[1:6]),e[6],e[7],tuple(e[8:13]),e[13]) for e in edges)))
            if best is None or key < best: best = key
    return best


def test_exact_partition_against_independent_full_automorphism_enumeration():
    ctx=context()
    singles=[_edge(0,p,0,1,q,0) for p,q in product(PATHS,repeat=2)]
    forward, backward = {}, {}
    # Every unordered state of up to two edges, including all parallel edges.
    for n in range(3):
        for edges in combinations_with_replacement(singles,n):
            old=exhaustive_key(edges)
            result=canonicalize_with_witness(ctx,edges); new=result['edges']
            assert verify_relabeling_witness(ctx,edges,new,result['witness'])
            assert canonicalize_edges(ctx,new)==new
            assert forward.setdefault(old,new)==new
            assert backward.setdefault(new,old)==old
    assert len(forward)==len(backward)


def test_joint_ownership_not_individual_site_orbits():
    ctx=context()
    same=(_edge(0,PATHS[0],0,1,PATHS[0],0),_edge(0,PATHS[1],0,1,PATHS[0],0))
    split=(_edge(0,PATHS[0],0,1,PATHS[0],0),_edge(0,PATHS[2],0,1,PATHS[0],0))
    assert canonicalize_edges(ctx,same)!=canonicalize_edges(ctx,split)


def test_typed_parallel_edges_reversal_and_relabeling():
    ctx=context()
    edges=(_edge(0,PATHS[0],0,1,PATHS[2],1),)*2
    expected=canonicalize_edges(ctx,edges)
    for mapping in parent_maps():
        moved=tuple(_edge(1,mapping[tuple(e[1:6])],e[6],0,tuple(e[8:13]),e[13]) for e in edges)
        assert canonicalize_edges(ctx,moved)==expected
    changed=(_edge(0,PATHS[0],1,1,PATHS[2],1),)*2
    assert canonicalize_edges(ctx,changed)!=expected


def test_identity_is_context_bound_and_occurrence_independent():
    a=context(); b=context(owners=3)
    edges=(_edge(0,PATHS[0],0,1,PATHS[0],0),)
    assert state_identity(a,edges)['object_identity']!=state_identity(b,edges)['object_identity']
    a.name='lane A'; first=state_identity(a,edges)['object_identity']
    a.name='lane B'; a.parents[0].pid='alias'
    assert state_identity(a,edges)['object_identity']==first
    c=context(capacity=3)
    assert state_identity(a,edges)['object_identity']!=state_identity(c,edges)['object_identity']


def test_nonisomorphic_carriers_and_distinct_owner_colors_cannot_merge():
    a=context(); b=context()
    b.parents=tuple(SimpleNamespace(h6=p.h6,exact_color=str(i),pid=p.pid) for i,p in enumerate(b.parents))
    edges=(_edge(0,PATHS[0],0,1,PATHS[0],1),)
    reversed_types=(_edge(0,PATHS[0],1,1,PATHS[0],0),)
    assert canonicalize_edges(a,edges)==canonicalize_edges(a,reversed_types)
    assert canonicalize_edges(b,edges)!=canonicalize_edges(b,reversed_types)


def test_reject_invalid_inputs_and_forged_witness():
    ctx=context(capacity=1); edge=_edge(0,PATHS[0],0,1,PATHS[0],0)
    for edges in [(edge,edge), ((-1,*edge[1:]),), ((0,*edge[1:7],0,*edge[8:]),), ((True,*edge[1:]),)]:
        with pytest.raises(ValueError): canonicalize_edges(ctx,edges)
    result=canonicalize_with_witness(ctx,(edge,));witness=result['witness'];witness['node_map'][1][3]=[99]
    with pytest.raises(ValueError):verify_relabeling_witness(ctx,(edge,),result['edges'],witness)
