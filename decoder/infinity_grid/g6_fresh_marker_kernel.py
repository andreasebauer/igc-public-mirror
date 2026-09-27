from __future__ import annotations

"""Exact fresh-D-marker observer kernel for the registered G6 S5/S6 repair.

This is semantic kernel code, not an evaluator.  The public evaluator surface is
exposed only through restricted KernelView services.  Structural values are the
scientific authority; digests never decide equality.
"""
from typing import Any
from .adapters.g4_accepted import DecoratedG4Tree
from .exact_tree_relation_kernel import (
    ExactTreeRelationKernel, get_relation_kernel, reconstruct_tree_from_rooted_canon,
    tree_to_record,
)

MARKER_H='D'
MARKER_OPERATOR=(0,0)
_MARKER=DecoratedG4Tree(1,tuple(),(MARKER_H,),tuple())
_Q_CACHE_VERSION='G6_FRESH_D_MARKER_Q_CACHE_V1'
_DECODE_CACHE_VERSION='G6_FRESH_D_MARKER_DECODE_CACHE_V1'

class MarkerKernelError(RuntimeError): pass

def marker_leaf_record()->dict[str,Any]:
    return tree_to_record(_MARKER)

def marker_observer_state(tree:DecoratedG4Tree, *, kernel:ExactTreeRelationKernel|None=None, marker_operator=MARKER_OPERATOR):
    k=kernel or get_relation_kernel(); op=tuple(marker_operator)
    prepared=k.prepare(tree)
    key=(_Q_CACHE_VERSION,prepared.unrooted_canon,op)
    hit,cached=k.observer_q_cache_lookup(key)
    if hit:
        return cached
    rel=k.relation(prepared,_MARKER,op)
    q=tuple(rel.canons)
    k.observer_q_cache_store(key,q)
    # Exact q_D is globally injective on the registered marker-free G6 domain.
    # Seed the decode cache from the known source tree while q_D is already in
    # hand. The independent legacy oracle remains the cold semantic cross-check.
    if MARKER_H not in prepared.tree.H_classes:
        dkey=(_DECODE_CACHE_VERSION,q)
        dresult=(prepared.tree,{
            'marker_child_count':len(q),
            'recovered_parent_canon':prepared.unrooted_canon,
            'all_children_same_parent':True,
            'cache_seeded_from_exact_marker_observation':True,
        })
        k.observer_decode_cache_store(dkey,dresult)
    return q

def _delete_unique_D(child:DecoratedG4Tree)->DecoratedG4Tree:
    ds=[i for i,h in enumerate(child.H_classes) if h==MARKER_H]
    if len(ds)!=1: raise MarkerKernelError('MARKER_UNIQUE_D_REQUIRED')
    d=ds[0]; incident=[i for i,e in enumerate(child.edges) if d in e]
    if len(incident)!=1: raise MarkerKernelError('MARKER_D_MUST_BE_LEAF')
    keep=[i for i in range(child.n) if i!=d]; remap={old:i for i,old in enumerate(keep)}
    edges=[]; ops=[]
    for i,((u,v),op) in enumerate(zip(child.edges,child.edge_operators)):
        if i==incident[0]: continue
        if d in (u,v): raise MarkerKernelError('MARKER_D_EXTRA_EDGE')
        edges.append((remap[u],remap[v])); ops.append(tuple(op))
    return DecoratedG4Tree(len(keep),tuple(edges),tuple(child.H_classes[i] for i in keep),tuple(ops))

def decode_marker_observer_state(state, *, kernel:ExactTreeRelationKernel|None=None):
    k=kernel or get_relation_kernel(); q=tuple(state)
    if not q: raise MarkerKernelError('MARKER_Q_EMPTY')
    key=(_DECODE_CACHE_VERSION,q)
    hit,cached=k.observer_decode_cache_lookup(key)
    if hit:
        return cached
    recovered=[]
    for canon in q:
        child=reconstruct_tree_from_rooted_canon(tuple(canon))
        parent=_delete_unique_D(child)
        recovered.append((k.prepare(parent).unrooted_canon,parent))
    first=recovered[0][0]
    if any(can!=first for can,_ in recovered[1:]): raise MarkerKernelError('MARKER_Q_PARENT_DISAGREEMENT')
    parent=next(t for can,t in recovered if can==first)
    result=(parent, {'marker_child_count':len(q),'recovered_parent_canon':first,'all_children_same_parent':True})
    k.observer_decode_cache_store(key,result)
    return result

def write_marker_states(left_state,right_state,operator, *, kernel:ExactTreeRelationKernel|None=None, diagnostics=None):
    k=kernel or get_relation_kernel(); left,ls=decode_marker_observer_state(left_state,kernel=k); right,rs=decode_marker_observer_state(right_state,kernel=k)
    rel=k.relation(left,right,tuple(operator))
    child_rows=tuple((tree_to_record(c),marker_observer_state(c,kernel=k),k.prepare(c).unrooted_canon) for c in rel.children)
    states=tuple(sorted({q for _record,q,_can in child_rows},key=repr))
    if diagnostics is not None:
        diagnostics.update({
          'left_decoded_unrooted_canon':k.prepare(left).unrooted_canon,
          'right_decoded_unrooted_canon':k.prepare(right).unrooted_canon,
          'left_decode_stats':ls,'right_decode_stats':rs,
          'raw_child_count':len(rel.canons),
          'raw_child_canons':tuple(rel.canons),
          'raw_child_rows':child_rows,
          'projected_marker_states':states,
          'fused_raw_projection':True,
        })
    return states
