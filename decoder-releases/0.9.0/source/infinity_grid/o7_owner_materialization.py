"""Bounded O7 edge-incidence materialization over immutable O6 resources."""
from __future__ import annotations

from collections import Counter


def materialize_owner(base_h6, edges, owner):
    """Consume one free port for each attached edge endpoint."""
    incidence = Counter()
    for edge in edges:
        if len(edge) != 14:
            raise ValueError('O7 edge must have 14 fields')
        for current, path, typ in ((edge[0], tuple(edge[1:6]), edge[6]),
                                   (edge[7], tuple(edge[8:13]), edge[13])):
            if current == owner:
                incidence[(path, typ)] += 1

    result = []
    for a, o5 in enumerate(base_h6):
        blocks5 = []
        for j, o4 in enumerate(o5):
            blocks4 = []
            for k, o3 in enumerate(o4):
                blocks3 = []
                for v, block in enumerate(o3):
                    sites = []
                    for i, (port, free) in enumerate(block):
                        path = (a, j, k, v, i)
                        remaining = tuple(value - incidence[(path, typ)]
                                          for typ, value in enumerate(free))
                        if min(remaining) < 0:
                            raise ValueError('E7 oversubscribed')
                        sites.append((port, remaining))
                    blocks3.append(tuple(sites))
                blocks4.append(tuple(blocks3))
            blocks5.append(tuple(blocks4))
        result.append(tuple(blocks5))
    return tuple(result)
