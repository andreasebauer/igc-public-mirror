from __future__ import annotations

"""Pure structural helpers for G6:S5R compositional read/write investigation.

No execution authority, subprocess, or durable I/O is present here.  The key
operation attempts to reconstruct a parent only from the frozen observer's
complete exact child-canon relation by deleting the known fixed probe component.
"""

from typing import Any, Iterable, Mapping

from .adapters.g4_accepted import DecoratedG4Tree
from .exact_tree_relation_kernel import (
    ExactTreeRelationKernel, ExactTreeRelationKernelError, get_relation_kernel,
    reconstruct_tree_from_rooted_canon, split_tree_components, edge_side_sizes,
)
from .g6_reference_oracles import legacy_parent_candidates_from_observer_canons as _legacy_parent_candidates_from_observer_canons


class G6CRWKernelError(RuntimeError):
    pass


def tree_from_rooted_canon(canon: tuple[Any,...]) -> DecoratedG4Tree:
    """Compatibility wrapper; implementation is kernel-owned."""
    try:
        return reconstruct_tree_from_rooted_canon(canon)
    except ExactTreeRelationKernelError as exc:
        raise G6CRWKernelError(str(exc)) from exc


def _split_components(tree: DecoratedG4Tree, edge_index: int) -> tuple[DecoratedG4Tree,DecoratedG4Tree]:
    try:
        return split_tree_components(tree,edge_index)
    except ExactTreeRelationKernelError as exc:
        raise G6CRWKernelError(str(exc)) from exc


def _edge_side_sizes(tree: DecoratedG4Tree) -> tuple[tuple[int,int], ...]:
    try:
        return edge_side_sizes(tree)
    except ExactTreeRelationKernelError as exc:
        raise G6CRWKernelError(str(exc)) from exc


def parent_candidates_from_observer_canons_profiled(observer_canons: Iterable[tuple[Any,...]], probe: DecoratedG4Tree,
                                                      operator: tuple[int,int]=(0,0), *, kernel=None) -> tuple[tuple[tuple[Any,...],...],dict[str,int]]:
    """Exact observer inversion with a size-pruned edge search.

    Deleting the true graft edge must expose the fixed probe as one connected
    component.  Therefore that side must have exactly ``probe.n`` vertices.
    Component size is an isomorphism invariant, so rejecting all other edges is
    exact and cannot remove a valid candidate.  Only size-compatible edges are
    materialized/canonicalized.  Structural canonical tuples remain the sole
    equality authority.
    """
    k=kernel or get_relation_kernel(); probe_can=k.prepare(probe).unrooted_canon; probe_n=int(probe.n); all_sets=[]
    stats={'observer_outcomes':0,'edges_scanned':0,'operator_edges':0,'size_compatible_edges':0,
           'component_splits':0,'component_canon_calls':0}
    for child_can in observer_canons:
        stats['observer_outcomes'] += 1
        child=tree_from_rooted_canon(tuple(child_can)); sizes=_edge_side_sizes(child); cands=set()
        for i,op in enumerate(child.edge_operators):
            stats['edges_scanned'] += 1
            if tuple(op)!=tuple(operator): continue
            stats['operator_edges'] += 1
            sx,sy=sizes[i]
            if sx!=probe_n and sy!=probe_n: continue
            stats['size_compatible_edges'] += 1
            x,y=_split_components(child,i); stats['component_splits'] += 1
            cx=cy=None
            if sx==probe_n:
                cx=k.prepare(x).unrooted_canon; stats['component_canon_calls'] += 1
                if cx==probe_can:
                    cy=k.prepare(y).unrooted_canon; stats['component_canon_calls'] += 1
                    cands.add(cy)
            if sy==probe_n:
                if cy is None:
                    cy=k.prepare(y).unrooted_canon; stats['component_canon_calls'] += 1
                if cy==probe_can:
                    if cx is None:
                        cx=k.prepare(x).unrooted_canon; stats['component_canon_calls'] += 1
                    cands.add(cx)
        if not cands: return tuple(),stats
        all_sets.append(cands)
    if not all_sets: return tuple(),stats
    common=set.intersection(*all_sets)
    return tuple(sorted(common,key=repr)),stats


def parent_candidates_from_observer_canons(observer_canons: Iterable[tuple[Any,...]], probe: DecoratedG4Tree,
                                             operator: tuple[int,int]=(0,0)) -> tuple[tuple[Any,...],...]:
    candidates,_stats=parent_candidates_from_observer_canons_profiled(observer_canons,probe,operator)
    return candidates


def _single_child_candidates_profiled(child_can: tuple[Any,...], probe: DecoratedG4Tree, operator: tuple[int,int],
                                       *, kernel: ExactTreeRelationKernel, probe_can: tuple[Any,...], probe_n: int
                                       ) -> tuple[set[tuple[Any,...]],dict[str,int]]:
    """Exact candidate parents exposed by one observer child.

    This is the single-outcome form of the accepted size-pruned inversion.  It is
    used only by the verified fast decoder below; structural canonical tuples
    remain the equality authority.
    """
    stats={'observer_outcomes':1,'edges_scanned':0,'operator_edges':0,'size_compatible_edges':0,
           'component_splits':0,'component_canon_calls':0}
    child=tree_from_rooted_canon(tuple(child_can)); sizes=_edge_side_sizes(child); cands:set[tuple[Any,...]]=set()
    for i,op in enumerate(child.edge_operators):
        stats['edges_scanned'] += 1
        if tuple(op)!=tuple(operator): continue
        stats['operator_edges'] += 1
        sx,sy=sizes[i]
        if sx!=probe_n and sy!=probe_n: continue
        stats['size_compatible_edges'] += 1
        x,y=_split_components(child,i); stats['component_splits'] += 1
        cx=cy=None
        if sx==probe_n:
            cx=kernel.prepare(x).unrooted_canon; stats['component_canon_calls'] += 1
            if cx==probe_can:
                cy=kernel.prepare(y).unrooted_canon; stats['component_canon_calls'] += 1
                cands.add(cy)
        if sy==probe_n:
            if cy is None:
                cy=kernel.prepare(y).unrooted_canon; stats['component_canon_calls'] += 1
            if cy==probe_can:
                if cx is None:
                    cx=kernel.prepare(x).unrooted_canon; stats['component_canon_calls'] += 1
                cands.add(cx)
    return cands,stats


def _add_inversion_stats(total: dict[str,int], part: Mapping[str,int]) -> None:
    for key in ('observer_outcomes','edges_scanned','operator_edges','size_compatible_edges','component_splits','component_canon_calls'):
        total[key]=int(total.get(key,0))+int(part.get(key,0))


def attachment_response_descriptor(tree: DecoratedG4Tree, *, kernel=None) -> tuple[Any,...]:
    """Exact anonymous typed attachment interface upper bound.

    Multiplicity of owners in one automorphism orbit is not stored: exact relation SET
    semantics already deduplicates equal rooted-owner realizations.  Each retained rooted
    message is a complete isomorphism class, not an owner identity.
    """
    k=kernel or get_relation_kernel()
    p=k.prepare(tree)
    rows=[]
    for typ in range(7):
        vals={p.rooted_canons[v] for v in p.rooted_owner_representatives(typ)}
        rows.append((typ,tuple(sorted(vals,key=repr))))
    return tuple(rows)


def exact_observer_state(tree: DecoratedG4Tree, probe: DecoratedG4Tree,
                         observer_operator: tuple[int,int]=(0,0), *, kernel=None) -> tuple[tuple[Any,...], ...]:
    """Public CRW1 state q(X): exact frozen one-probe outcome relation.

    The return value is the structurally sorted SET of exact child canonical forms.
    No parent topology or owner identity is added to the retained state.
    """
    if kernel is None:
        kernel=get_relation_kernel()
    op=tuple(observer_operator); key=(tree,probe,op)
    hit,cached=kernel.observer_q_cache_lookup(key)
    if hit:
        return cached
    out=tuple(kernel.relation(kernel.prepare(tree), kernel.prepare(probe), op).canons)
    kernel.observer_q_cache_store(key,out)
    return out


def decode_exact_observer_state(observer_state: Iterable[tuple[Any,...]], probe: DecoratedG4Tree,
                                observer_operator: tuple[int,int]=(0,0), *, kernel=None
                                ) -> tuple[DecoratedG4Tree, dict[str,int]]:
    """Decode a parent using only q(X) plus the fixed public probe/operator.

    Fast path: intersect exact parent candidates outcome-by-outcome.  As soon as
    exactly one candidate remains, reconstruct it and recompute q(candidate).  If
    that complete structural relation equals the supplied q-state, the candidate
    is already proved to satisfy every unvisited outcome and decoding can stop.
    This is an exact verification, not a heuristic or hash shortcut.

    Any non-verifying/adversarial input falls back to the historical complete
    intersection so the pre-optimization malformed-input semantics are retained.
    """
    state=tuple(observer_state)
    if kernel is None:
        kernel=get_relation_kernel()
    op=tuple(observer_operator); cache_key=(state,probe,op)
    hit,cached=kernel.observer_decode_cache_lookup(cache_key)
    if hit:
        cached_tree,cached_stats=cached
        return cached_tree,dict(cached_stats)
    probe_can=kernel.prepare(probe).unrooted_canon; probe_n=int(probe.n)
    common:set[tuple[Any,...]]|None=None
    fast_stats={'observer_outcomes':0,'edges_scanned':0,'operator_edges':0,'size_compatible_edges':0,
                'component_splits':0,'component_canon_calls':0}
    for idx,child_can in enumerate(state):
        cands,part=_single_child_candidates_profiled(tuple(child_can),probe,tuple(observer_operator),
                                                      kernel=kernel,probe_can=probe_can,probe_n=probe_n)
        _add_inversion_stats(fast_stats,part)
        common=set(cands) if common is None else common.intersection(cands)
        if not common:
            break
        if len(common)==1:
            candidate_can=next(iter(common))
            candidate=tree_from_rooted_canon(candidate_can)
            if exact_observer_state(candidate,probe,op,kernel=kernel)==state:
                fast_stats.update({
                    'fast_verified_decode':1,
                    'fast_verified_outcomes_examined':idx+1,
                    'observer_outcomes_total':len(state),
                    'observer_outcomes_short_circuited':len(state)-(idx+1),
                    'decode_candidate_count':1,
                })
                kernel.observer_decode_cache_store(cache_key,(candidate,dict(fast_stats)))
                return candidate,fast_stats
            break
    # Preserve exact historical behavior for empty, ambiguous, or malformed q.
    candidates,stats=parent_candidates_from_observer_canons_profiled(state,probe,observer_operator,kernel=kernel)
    stats.update({
        'fast_verified_decode':0,
        'fast_verified_outcomes_examined':int(fast_stats.get('observer_outcomes',0)),
        'observer_outcomes_total':len(state),
        'observer_outcomes_short_circuited':0,
        'decode_candidate_count':len(candidates),
    })
    if len(candidates)!=1:
        raise G6CRWKernelError(f"CRW1_OBSERVER_DECODE_NOT_UNIQUE:{len(candidates)}")
    decoded=tree_from_rooted_canon(candidates[0])
    kernel.observer_decode_cache_store(cache_key,(decoded,dict(stats)))
    return decoded,stats


def write_exact_observer_states(left_state: Iterable[tuple[Any,...]], right_state: Iterable[tuple[Any,...]],
                                operator: tuple[int,int], *, probe: DecoratedG4Tree,
                                observer_operator: tuple[int,int]=(0,0), kernel=None,
                                diagnostics: dict[str,Any] | None=None) -> tuple[tuple[tuple[Any,...], ...], ...]:
    """CRW1 abstract relation-valued write law using q-values only.

    Inputs are retained observer states, not raw parent carriers. The raw exact
    carrier is reconstructed transiently from q by the CRW0 inversion, then the
    frozen G5 exact graft algebra is applied. Each exact child is projected back
    to q before return. No hidden parent data is supplied to this function.

    ``diagnostics`` is an execution-only output channel.  When supplied, it records
    exact canons/stats of the q-derived transient parents so an evaluator can verify
    those same decodes without running the inversion a second time.  It does not
    accept or alter scientific input state.
    """
    if kernel is None:
        kernel=get_relation_kernel()
    left,lstats=decode_exact_observer_state(tuple(left_state),probe,observer_operator,kernel=kernel)
    right,rstats=decode_exact_observer_state(tuple(right_state),probe,observer_operator,kernel=kernel)
    lp=kernel.prepare(left); rp=kernel.prepare(right)
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update({
            'left_decoded_unrooted_canon':lp.unrooted_canon,
            'right_decoded_unrooted_canon':rp.unrooted_canon,
            'left_decode_stats':dict(lstats),
            'right_decode_stats':dict(rstats),
        })
    rel=kernel.relation(lp,rp,tuple(operator))
    out={exact_observer_state(child,probe,observer_operator,kernel=kernel) for child in rel.children}
    if diagnostics is not None:
        diagnostics['abstract_exact_child_count']=len(rel.canons)
        diagnostics['abstract_projected_child_state_count']=len(out)
    return tuple(sorted(out,key=repr))
