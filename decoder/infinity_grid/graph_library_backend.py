"""Qualified typed-multigraph owner automorphisms using igraph/BLISS.

No canonical state naming is changed. Edge occurrence gadget permutations are
projected to owners BEFORE closure, so duplicate occurrences do not inflate the
returned owner group. Unsupported auto inputs use the retained implementation;
backend selection is inspectable. Runtime/library errors never trigger fallback.
"""
from collections import Counter, deque
_COUNTS = Counter()

def supported(n, colors, edges):
    return (type(n) is int and 1 <= n <= 6 and len(colors) == n
            and all(isinstance(c, (tuple,list)) and len(c)==7 and all(type(x) is int and x>=0 for x in c) for c in colors)
            and isinstance(edges, (tuple,list)) and len(edges)<=6
            and all(type(e) is tuple and len(e)==4 and all(type(x) is int for x in e)
                    and 0<=e[0]<e[1]<n and 0<=e[2]<7 and 0<=e[3]<7 for e in edges))

def select_backend(n, colors, edges, backend=None):
    choice='auto' if backend is None else backend
    if choice not in ('auto','legacy','igraph_bliss'):
        raise ValueError('Unknown automorphism backend: '+str(choice))
    fits=supported(n,colors,edges)
    if choice=='igraph_bliss' and not fits:
        raise ValueError('Outside qualified igraph automorphism domain')
    return 'igraph_bliss' if choice=='igraph_bliss' or (choice=='auto' and fits) else 'legacy'

def usage():
    return dict(sorted(_COUNTS.items()))

def _edge(u,v,a,b):
    return (u,v,a,b) if u<v else (v,u,b,a)

def automorphisms(n, colors, edges, legacy, backend=None):
    choice=select_backend(n,colors,edges,backend);_COUNTS[choice]+=1
    if choice=='legacy':return legacy(n,colors,edges)
    import igraph
    labels=[('owner',tuple(c)) for c in colors];gadget_edges=[]
    for u,v,a,b in edges:
        left=len(labels);right=left+1;labels.extend([('endpoint',a),('endpoint',b)])
        gadget_edges.extend([(u,left),(left,right),(right,v)])
    vocabulary={label:i for i,label in enumerate(sorted(set(labels)))}
    color_ids=[vocabulary[label] for label in labels]
    graph=igraph.Graph(n=len(labels),edges=gadget_edges,directed=False)
    generators=graph.automorphism_group(sh='fl',color=color_ids)
    projected=[]
    for generator in generators:
        if sorted(generator)!=list(range(len(labels))) or any(labels[i]!=labels[generator[i]] for i in range(len(labels))):
            raise RuntimeError('Invalid BLISS colored gadget generator')
        transformed=Counter(tuple(sorted((generator[u],generator[v]))) for u,v in gadget_edges)
        if transformed!=Counter(tuple(sorted(e)) for e in gadget_edges):raise RuntimeError('BLISS generator edge replay failed')
        p=tuple(generator[:n])
        if sorted(p)!=list(range(n)):raise RuntimeError('BLISS owner projection failed')
        projected.append(p)
    seen=_owner_group_closure_sympy(n, projected)
    original=Counter(edges)
    for p in seen:
        if any(colors[i]!=colors[p[i]] for i in range(n)) or Counter(_edge(p[u],p[v],a,b) for u,v,a,b in edges)!=original:
            raise RuntimeError('Projected owner map replay failed')
    return sorted(seen)


def _owner_group_closure_legacy(n, projected):
    identity=tuple(range(n));seen={identity};pending=deque([identity])
    while pending:
        current=pending.popleft()
        for generator in projected:
            composed=tuple(generator[current[i]] for i in range(n))
            if composed not in seen:
                if len(seen)>=720:raise RuntimeError('Owner group exceeds qualified bound')
                seen.add(composed);pending.append(composed)
    return sorted(seen)


def _owner_group_closure_sympy(n, projected):
    """Sorted owner maps, retaining explicit fixed points and the 720-map bound."""
    if type(n) is not int or not 1 <= n <= 6:
        raise ValueError('Outside qualified owner group domain')
    identity = tuple(range(n))
    if any(len(p) != n or any(type(i) is not int for i in p)
           or sorted(p) != list(identity) for p in projected):
        raise ValueError('Invalid owner generator')
    from sympy.combinatorics import Permutation, PermutationGroup
    group = PermutationGroup([Permutation(list(identity), size=n)] +
                             [Permutation(list(p), size=n) for p in projected])
    order = int(group.order())
    if not 1 <= order <= 720:
        raise RuntimeError('Owner group exceeds qualified bound')
    maps = [tuple(p) for p in group.generate_schreier_sims(af=True)]
    if (len(maps) != order or len(set(maps)) != order or identity not in maps
            or any(len(p) != n or sorted(p) != list(identity) for p in maps)):
        raise RuntimeError('Invalid SymPy owner group output')
    return sorted(maps)
