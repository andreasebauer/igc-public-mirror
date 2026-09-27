from __future__ import annotations

"""Small independent semantic oracles used at adapter boundaries.

These helpers intentionally avoid Decoder's recursive message implementation.
They are bounded reference constructions for verification, not production
canonicalizers and not sources of scientific promotion authority.
"""

import itertools
from typing import Any, Callable, Sequence


def brute_force_typed_tree_canon(
    *, n: int, edges: Sequence[Sequence[int]], vertex_labels: Sequence[Any],
    endpoint_labels: Sequence[Sequence[int]], vertex_key: Callable[[Any], Any] = lambda x: x,
) -> tuple[Any, ...]:
    """Exact typed-tree isomorphism oracle by enumeration of all vertex permutations.

    ``endpoint_labels[i] == (a,b)`` is bound to ``edges[i] == (u,v)``.  If a
    permutation reverses the canonical stored endpoint order, the pair is swapped.
    This representation is algorithmically independent of recursive tree messages.
    Intended only for small bounded verification cases.
    """
    n = int(n)
    if len(vertex_labels) != n or len(edges) != len(endpoint_labels):
        raise ValueError("typed oracle arity mismatch")
    labels = [vertex_key(x) for x in vertex_labels]
    reps = []
    for perm_tuple in itertools.permutations(range(n)):
        new_labels = [None] * n
        for old, new in enumerate(perm_tuple):
            new_labels[new] = labels[old]
        out_edges = []
        for e, op in zip(edges, endpoint_labels):
            if len(e) != 2 or len(op) != 2:
                raise ValueError("typed oracle edge shape mismatch")
            u, v = int(e[0]), int(e[1])
            a, b = int(op[0]), int(op[1])
            pu, pv = perm_tuple[u], perm_tuple[v]
            if pu < pv:
                out_edges.append((pu, pv, a, b))
            else:
                out_edges.append((pv, pu, b, a))
        reps.append((tuple(new_labels), tuple(sorted(out_edges))))
    return min(reps)
