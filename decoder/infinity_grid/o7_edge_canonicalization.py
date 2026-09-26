"""Bounded C1 canonical edges under frozen site and owner symmetries."""
from __future__ import annotations

import hashlib
import itertools
from collections import Counter, defaultdict


def _parts(edge):
    return edge[0], tuple(edge[1:6]), edge[6], edge[7], tuple(edge[8:13]), edge[13]


def _edge(c, path, typ, d, other, other_type):
    if c < d:
        return (c, *path, typ, d, *other, other_type)
    return (d, *other, other_type, c, *path, typ)


def _maps(ctx, active, *, compact):
    classes = [tuple(cls) for cls in ctx.color_classes]
    sources = [tuple(o for o in cls if o in active) for cls in classes]
    choices = [itertools.permutations(cls[:len(src)] if compact else cls,
                                     len(src)) for cls, src in zip(classes, sources)]
    for combo in itertools.product(*choices):
        mapping = list(range(ctx.n))
        for src, dst in zip(sources, combo):
            for old, new in zip(src, dst):
                mapping[old] = new
        yield mapping


def _full(ctx, edges, active, incident, used):
    colors = {}
    for owner, path in active:
        port, free = ctx.parents[owner].h6[path[0]][path[1]][path[2]][path[3]][path[4]]
        remaining = tuple(f - used[(owner, path, typ)] for typ, f in enumerate(free))
        if min(remaining) < 0:
            raise ValueError('E7 oversubscribed')
        colors[(owner, path)] = (ctx.parents[owner].base_p2k[path], port, remaining)
    best = None
    for mapping in _maps(ctx, {o for o, _ in active}, compact=False):
        signatures = {node: hashlib.sha256(repr(colors[node]).encode()).hexdigest()
                      for node in active}
        for _ in range(12):
            updated = {}
            for node in active:
                neighbors = sorted((ta, mapping[nb[0]], colors[nb], tb,
                                    signatures[nb]) for ta, nb, tb in incident[node])
                updated[node] = hashlib.sha256(repr((colors[node], tuple(neighbors))).encode()).hexdigest()
            if all(updated[node] == signatures[node] for node in active):
                break
            before = sorted(sorted(x for x in active if signatures[x] == sig)
                            for sig in set(signatures.values()))
            after = sorted(sorted(x for x in active if updated[x] == sig)
                           for sig in set(updated.values()))
            signatures = updated
            if len(before) == len(after) and [len(g) for g in before] == [len(g) for g in after]:
                break
        remap = {}
        for owner in range(ctx.n):
            groups = defaultdict(list)
            for node in active:
                if node[0] == owner:
                    groups[ctx.parents[owner].base_p2k[node[1]]].append(node)
            for orbit, nodes in groups.items():
                def order(node):
                    neighbors = tuple(sorted((ta, mapping[nb[0]], colors[nb], tb,
                                              signatures[nb]) for ta, nb, tb in incident[node]))
                    return signatures[node], colors[node][1:], neighbors, node[1]
                targets = ctx.parents[owner].base_k2p[orbit]
                for node, target in zip(sorted(nodes, key=order), targets):
                    remap[node] = target
        candidate = tuple(sorted(_edge(mapping[c], remap[(c, p)], t,
                                       mapping[d], remap[(d, q)], u)
                                 for c, p, t, d, q, u in map(_parts, edges)))
        if best is None or candidate < best:
            best = candidate
    return best


def canonicalize_edges(ctx, edges):
    edges = tuple(tuple(edge) for edge in edges)
    if not edges:
        return ()
    active, incident, used = set(), defaultdict(list), Counter()
    for c, p, t, d, q, u in map(_parts, edges):
        left, right = (c, p), (d, q)
        active.update((left, right))
        incident[left].append((t, right, u))
        incident[right].append((u, left, t))
        used[(c, p, t)] += 1
        used[(d, q, u)] += 1

    orbit_keys = {(o, ctx.parents[o].base_p2k[p]) for o, p in active}
    if len(orbit_keys) != len(active):
        return _full(ctx, edges, active, incident, used)

    remap = {}
    for owner, path in active:
        parent = ctx.parents[owner]
        port, free = parent.h6[path[0]][path[1]][path[2]][path[3]][path[4]]
        if any(f - used[(owner, path, typ)] < 0 for typ, f in enumerate(free)):
            raise ValueError('E7 oversubscribed')
        orbit = parent.base_p2k[path]
        remap[(owner, path)] = parent.base_k2p[orbit][0]

    best = None
    for mapping in _maps(ctx, {o for o, _ in active}, compact=True):
        candidate = tuple(sorted(_edge(mapping[c], remap[(c, p)], t,
                                       mapping[d], remap[(d, q)], u)
                                 for c, p, t, d, q, u in map(_parts, edges)))
        if best is None or candidate < best:
            best = candidate
    return best
