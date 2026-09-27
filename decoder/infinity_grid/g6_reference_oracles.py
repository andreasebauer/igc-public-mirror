from __future__ import annotations

"""Explicit independent G6 reference-oracle implementations.

These algorithms intentionally remain separate from PRIMARY kernel services so
independent verification retains meaning.  Production evaluators may reach them
only through a REFERENCE_ONLY KernelView service.
"""
from collections import deque
from typing import Any, Iterable
from .adapters.g4_accepted import DecoratedG4Tree, G4AcceptedAdapter

class G6ReferenceOracleError(RuntimeError): pass

def _thaw(value: Any) -> Any:
    if not isinstance(value, tuple) or not value: raise G6ReferenceOracleError('malformed frozen decoration')
    tag=value[0]
    if tag=='str' and len(value)==2: return str(value[1])
    if tag=='int' and len(value)==2: return int(value[1])
    if tag=='seq' and len(value)==2: return tuple(_thaw(x) for x in value[1])
    raise G6ReferenceOracleError(f'unsupported frozen decoration tag {tag!r}')

def legacy_tree_from_rooted_canon(canon: tuple[Any,...]) -> DecoratedG4Tree:
    ad=G4AcceptedAdapter(); table=ad.relation_class_table(); reverse={key:klass for klass,key,_caps in table}
    if len(reverse)!=len(table): raise G6ReferenceOracleError('non-unique G4 vertex key table')
    edges=[]; ops=[]; classes=[]
    def walk(node):
        if not isinstance(node,tuple) or len(node)!=3 or node[0]!='V': raise G6ReferenceOracleError('malformed rooted tree canon')
        key=_thaw(node[1])
        if key not in reverse: raise G6ReferenceOracleError('unknown canonical vertex key')
        idx=len(classes); classes.append(reverse[key]); children=node[2]
        if not isinstance(children,tuple): raise G6ReferenceOracleError('malformed rooted children')
        for entry in children:
            if not isinstance(entry,tuple) or len(entry)!=2: raise G6ReferenceOracleError('malformed rooted branch')
            op=_thaw(entry[0])
            if not isinstance(op,tuple) or len(op)!=2: raise G6ReferenceOracleError('malformed endpoint operator')
            child=walk(entry[1]); edges.append((idx,child)); ops.append((int(op[0]),int(op[1])))
        return idx
    walk(canon)
    return DecoratedG4Tree(len(classes),tuple(edges),tuple(classes),tuple(ops))

def _component(tree: DecoratedG4Tree, keep: set[int]) -> DecoratedG4Tree:
    order=sorted(keep); remap={old:i for i,old in enumerate(order)}; edges=[]; ops=[]
    for (u,v),op in zip(tree.edges,tree.edge_operators):
        if u in keep and v in keep: edges.append((remap[u],remap[v])); ops.append(tuple(op))
    return DecoratedG4Tree(len(order),tuple(edges),tuple(tree.H_classes[i] for i in order),tuple(ops))

def _split(tree: DecoratedG4Tree, edge_index: int):
    a,b=tree.edges[edge_index]; adj=[[] for _ in range(tree.n)]
    for i,(u,v) in enumerate(tree.edges):
        if i==edge_index: continue
        adj[u].append(v); adj[v].append(u)
    seen={a}; q=deque([a])
    while q:
        v=q.popleft()
        for u in adj[v]:
            if u not in seen: seen.add(u); q.append(u)
    other=set(range(tree.n))-seen
    if not other or b not in other: raise G6ReferenceOracleError('split failed')
    return _component(tree,seen),_component(tree,other)

def legacy_parent_candidates_from_observer_canons(observer_canons: Iterable[tuple[Any,...]], probe: DecoratedG4Tree,
                                                     operator: tuple[int,int]=(0,0)) -> tuple[tuple[Any,...],...]:
    """Historical complete inversion retained only as an independent oracle."""
    ad=G4AcceptedAdapter(); probe_can=ad.unrooted_canon(probe); all_sets=[]
    for child_can in observer_canons:
        child=legacy_tree_from_rooted_canon(tuple(child_can)); cands=set()
        for i,op in enumerate(child.edge_operators):
            if tuple(op)!=tuple(operator): continue
            x,y=_split(child,i); cx,cy=ad.unrooted_canon(x),ad.unrooted_canon(y)
            if cx==probe_can: cands.add(cy)
            if cy==probe_can: cands.add(cx)
        if not cands: return tuple()
        all_sets.append(cands)
    if not all_sets: return tuple()
    return tuple(sorted(set.intersection(*all_sets),key=repr))
