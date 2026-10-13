from __future__ import annotations

"""Independent fresh-marker decode path used only behind a REFERENCE_ONLY service.

It deliberately does not import the production fresh-marker helper.
"""
from typing import Any
from .adapters.g4_accepted import DecoratedG4Tree
from .exact_tree_relation_kernel import get_relation_kernel, reconstruct_tree_from_rooted_canon, tree_to_record

class MarkerReferenceOracleError(RuntimeError): pass

def legacy_marker_decode_oracle(observer_state):
    q=tuple(observer_state)
    if not q: return {'candidate_count':0,'candidates':tuple(),'tree':None}
    k=get_relation_kernel(); out=[]
    for canon in q:
        child=reconstruct_tree_from_rooted_canon(tuple(canon))
        ds=[i for i,h in enumerate(child.H_classes) if h=='D']
        if len(ds)!=1: return {'candidate_count':0,'candidates':tuple(),'tree':None}
        d=ds[0]; incident=[i for i,e in enumerate(child.edges) if d in e]
        if len(incident)!=1: return {'candidate_count':0,'candidates':tuple(),'tree':None}
        keep=[i for i in range(child.n) if i!=d]; remap={old:i for i,old in enumerate(keep)}; edges=[]; ops=[]
        for i,((u,v),op) in enumerate(zip(child.edges,child.edge_operators)):
            if i==incident[0]: continue
            if d in (u,v): return {'candidate_count':0,'candidates':tuple(),'tree':None}
            edges.append((remap[u],remap[v])); ops.append(tuple(op))
        p=DecoratedG4Tree(len(keep),tuple(edges),tuple(child.H_classes[i] for i in keep),tuple(ops))
        out.append((k.prepare(p).unrooted_canon,p))
    uniq={can for can,_ in out}
    if len(uniq)!=1: return {'candidate_count':len(uniq),'candidates':tuple(sorted(uniq,key=repr)),'tree':None}
    can=next(iter(uniq)); p=next(p for c,p in out if c==can)
    return {'candidate_count':1,'candidates':(can,),'tree':tree_to_record(p)}
