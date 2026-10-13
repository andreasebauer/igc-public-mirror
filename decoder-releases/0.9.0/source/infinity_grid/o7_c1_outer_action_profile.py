"""Independent O7 outer-action enumeration over the pinned O6 observer.

The inherited O6 actions and Merkle primitive retain their historical producer.
This module replaces only the outer orbit enumeration and multiplicity logic.
"""
from __future__ import annotations

from collections import Counter, defaultdict


def hybrid_profile(ctx, edges, templates, pairs, *, resource_observer=None):
    from sys import modules
    engine = modules['_ig_c1_original_pinned_engine']
    o6 = engine.O6
    data = engine.current_owner_data(ctx, edges)
    base = [node[1].base_sig for node in data]
    counts = Counter()

    if resource_observer is None:
        resource_observer = o6.resource_profile
    for owner, (resource, *_rest) in enumerate(data):
        for (label, child), multiplicity in resource_observer(resource, templates, pairs).items():
            children = base.copy()
            children[owner] = child
            counts[(label, o6._digest_container('R7', children))] += multiplicity

    # Build eligible pointed-resource classes once for each owner and type.
    eligible = []
    for _resource, root, _entries, _groups, by_path in data:
        by_type = [defaultdict(list) for _ in range(7)]
        for path, (p, free, leaf, pointed) in by_path.items():
            for typ, remaining in enumerate(free):
                if remaining:
                    by_type[typ][pointed].append((path, p, free, leaf))
        eligible.append((root, by_type))

    for left in range(ctx.n):
        for right in range(left + 1, ctx.n):
            root_l, types_l = eligible[left]
            root_r, types_r = eligible[right]
            for t_l, t_r in pairs:
                for group_l in types_l[t_l].values():
                    path_l, p_l, f_l, leaf_l = min(group_l, key=lambda x: x[0])
                    next_l = list(f_l)
                    next_l[t_l] -= 1
                    child_l = o6._successor_merkle_sig(root_l, {leaf_l: (p_l, tuple(next_l))})
                    for group_r in types_r[t_r].values():
                        path_r, p_r, f_r, leaf_r = min(group_r, key=lambda x: x[0])
                        next_r = list(f_r)
                        next_r[t_r] -= 1
                        child_r = o6._successor_merkle_sig(root_r, {leaf_r: (p_r, tuple(next_r))})
                        children = base.copy()
                        children[left], children[right] = child_l, child_r
                        label = ('R7', min(t_l, t_r), max(t_l, t_r))
                        counts[(label, o6._digest_container('R7', children))] += len(group_l) * len(group_r)
    return counts


def independent_o6_hybrid_profile(ctx, edges, templates, pairs):
    from .o6_resource_observer import resource_profile
    return hybrid_profile(ctx, edges, templates, pairs, resource_observer=resource_profile)


def independent_merkle_profile(ctx, edges, templates, pairs, *, materializer=None):
    """New O6 actions and O7 grouping with new outer/pointed Merkle paths."""
    from sys import modules
    from .o6_resource_observer import ResourceMerkle, _bag_hash, resource_profile

    engine = modules['_ig_c1_original_pinned_engine']
    if materializer is None:
        materializer = engine.materialize_owner
    owners = [materializer(p.h6, edges, i)
              for i, p in enumerate(ctx.parents)]
    trees = [ResourceMerkle(owner) for owner in owners]
    base = [tree.digest for tree in trees]
    counts = Counter()
    for owner, resource in enumerate(owners):
        for (label, child), multiplicity in resource_profile(resource, templates, pairs).items():
            children = base.copy()
            children[owner] = child
            counts[(label, _bag_hash('R7', children))] += multiplicity

    eligible = []
    for tree in trees:
        by_type = [defaultdict(list) for _ in range(7)]
        for path, (port, free, leaf) in tree.sites.items():
            pointed = tree.pointed(leaf)
            for typ, remaining in enumerate(free):
                if remaining:
                    by_type[typ][pointed].append((path, port, free, leaf))
        eligible.append(by_type)

    for left in range(ctx.n):
        for right in range(left + 1, ctx.n):
            for t_l, t_r in pairs:
                for group_l in eligible[left][t_l].values():
                    _path, port_l, free_l, leaf_l = min(group_l, key=lambda x: x[0])
                    next_l = list(free_l)
                    next_l[t_l] -= 1
                    child_l = trees[left].successor({leaf_l: (port_l, tuple(next_l))})
                    for group_r in eligible[right][t_r].values():
                        _path, port_r, free_r, leaf_r = min(group_r, key=lambda x: x[0])
                        next_r = list(free_r)
                        next_r[t_r] -= 1
                        child_r = trees[right].successor({leaf_r: (port_r, tuple(next_r))})
                        children = base.copy()
                        children[left], children[right] = child_l, child_r
                        label = ('R7', min(t_l, t_r), max(t_l, t_r))
                        counts[(label, _bag_hash('R7', children))] += len(group_l) * len(group_r)
    return counts


def independent_materialization_profile(ctx, edges, templates, pairs):
    from .o7_owner_materialization import materialize_owner
    return independent_merkle_profile(ctx, edges, templates, pairs,
                                      materializer=materialize_owner)
