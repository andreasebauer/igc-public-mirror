"""Independent finite oracle: permutations, Floyd distances and endpoint copies.

No production helper is imported here. This audits diagnostics, not a certified
carrier population or complete labelled successor Counter.
"""
from collections import Counter
from functools import lru_cache
from itertools import permutations, combinations
import hashlib
import json


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False).encode()).hexdigest()


def normalized_edge(u, v, a, b):
    return (u, v, a, b) if u < v else (v, u, b, a)


@lru_cache(None)
def owner_maps(caps, edges):
    support = tuple(tuple(x > 0 for x in row) for row in caps)
    original = Counter(edges)
    return tuple(p for p in permutations(range(len(caps)))
                 if all(support[i] == support[p[i]] for i in range(len(caps)))
                 and Counter(normalized_edge(p[u], p[v], a, b)
                             for u, v, a, b in edges) == original)


def components(n, edges, removed_vertex=None, removed_edge=None):
    remaining = set(range(n)) - {removed_vertex}
    count = 0
    while remaining:
        count += 1
        reached = {min(remaining)}
        while True:
            expanded = set(reached)
            for i, (u, v) in enumerate(edges):
                if i == removed_edge or removed_vertex in (u, v):
                    continue
                if u in reached:
                    expanded.add(v)
                if v in reached:
                    expanded.add(u)
            if expanded == reached:
                break
            reached = expanded
        remaining -= reached
    return count


@lru_cache(None)
def graph_metrics(n, edges):
    inf = n + 1
    distances = [[0 if u == v else inf for v in range(n)] for u in range(n)]
    for u, v in edges:
        distances[u][v] = distances[v][u] = 1
    for k in range(n):
        for u in range(n):
            for v in range(n):
                distances[u][v] = min(distances[u][v], distances[u][k] + distances[k][v])
    connected = components(n, edges) == 1
    return {
        "connected": connected,
        "dist": distances,
        "diameter": max(map(max, distances)) if connected else None,
        "distance_sum": sum(distances[u][v] for u, v in combinations(range(n), 2)) if connected else None,
        "bridges": sum(components(n, edges, removed_edge=i) > 1
                       for i in range(len(edges))) if connected else 0,
        "articulations": sum(components(n, edges, removed_vertex=i) > 1
                             for i in range(n)) if connected and n > 2 else 0,
    }


@lru_cache(None)
def service_effect(n, edges, pair):
    before = graph_metrics(n, edges)
    if not before["connected"]:
        return None
    after = graph_metrics(n, tuple(sorted(edges + (pair,))))
    u, v = pair
    return (before["dist"][u][v], before["distance_sum"] - after["distance_sum"],
            before["diameter"] - after["diameter"], before["bridges"] - after["bridges"],
            before["articulations"] - after["articulations"], int(pair in edges))


def summary(caps, edges, bridge_pairs):
    n = len(caps)
    top = tuple(sorted((u, v) for u, v, a, b in edges))
    # Expand distinct endpoints rather than use the production multiplication.
    labelled_copies = Counter()
    for u, v in combinations(range(n), 2):
        for a, b in bridge_pairs:
            for left_endpoint in range(caps[u][a]):
                for right_endpoint in range(caps[v][b]):
                    labelled_copies[(u, v, a, b)] += 1
    labels = sorted(labelled_copies)
    group = owner_maps(caps, edges)
    action_orbits = {frozenset(normalized_edge(p[u], p[v], a, b) for p in group)
                    for u, v, a, b in labels}
    structural = Counter()
    weighted = Counter()
    for u, v, a, b in labels:
        effect = service_effect(n, top, (u, v))
        if effect is not None:
            structural[(a, b, effect)] += 1
            weighted[(a, b, effect, tuple(sorted((caps[u][a], caps[v][b]))))] += 1
    parent = list(range(n))

    def root(x):
        while parent[x] != x:
            x = parent[x]
        return x

    for p in group:
        for i, j in enumerate(p):
            parent[root(i)] = root(j)
    return {
        "legal_action_labels": len(labels),
        "total_action_copies": sum(labelled_copies.values()),
        "type_pair_support": len({(a, b) for u, v, a, b in labels}),
        "owner_pair_support": len({(u, v) for u, v, a, b in labels}),
        "action_orbits": len(action_orbits),
        "service_classes": len(structural),
        "service_signature_sha256": digest([[repr(k), value] for k, value in
                                             sorted(structural.items(), key=lambda item: repr(item[0]))]),
        "resource_weighted_service_classes": len(weighted),
        "automorphism_size": len(group),
        "owner_orbits": len({root(i) for i in range(n)}),
    }
