"""Bounded igraph metrics with the scanner's exact legacy output contract."""
from collections import Counter
from math import isinf


def supported(n, pairs):
    return (type(n) is int and 1 <= n <= 8 and len(pairs) <= 10
            and all(type(e) in (tuple, list) and len(e) == 2
                    and all(type(v) is int and 0 <= v < n for v in e)
                    for e in pairs))


def select_backend(n, pairs, backend=None):
    choice = 'auto' if backend is None else backend
    if choice not in ('auto', 'legacy', 'igraph'):
        raise ValueError('Unknown graph metrics backend: ' + str(choice))
    fits = supported(n, pairs)
    if choice == 'igraph' and not fits:
        raise ValueError('Outside qualified igraph graph metrics domain')
    return 'igraph' if fits and choice != 'legacy' else 'legacy'


def graph_basic(n, pairs, legacy, *, backend=None):
    if select_backend(n, pairs, backend) == 'legacy':
        return legacy(n, pairs)
    import igraph
    edges = [tuple(sorted(e)) for e in pairs]
    graph = igraph.Graph(n=n, edges=edges, directed=False)
    dist = [[None if isinf(d) else int(d) for d in row]
            for row in graph.distances()]
    connected = all(d is not None for row in dist for d in row)
    if connected:
        eccentricities = [max(row) for row in dist]
        diameter = max(eccentricities)
        radius = min(eccentricities)
        distance_sum = sum(dist[i][j] for i in range(n) for j in range(i + 1, n))
        shells = sorted(tuple(Counter(row).get(k, 0) for k in range(max(row) + 1))
                        for row in dist)
    else:
        diameter = radius = distance_sum = shells = None
    # Triangles are features of simple support, not edge occurrences or loops.
    support = graph.copy()
    support.simplify(multiple=True, loops=True)
    return {
        'degree': sorted(graph.degree(loops=True), reverse=True),
        'beta': len(edges) - n + 1 if connected else None,
        'diameter': diameter, 'radius': radius,
        'triangles': len(support.list_triangles()),
        'articulations': len(graph.articulation_points()) if connected and n > 2 else 0,
        'bridges': len(graph.bridges()) if connected else 0,
        'distance_sum': distance_sum, 'shells': shells,
        'parallel_multiplicities': sorted(Counter(edges).values(), reverse=True),
        'connected': connected, 'dist': dist,
    }
