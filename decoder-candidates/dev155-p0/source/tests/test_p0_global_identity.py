"""Independent incidence-graph and immutable-fixture identity regressions."""
from itertools import combinations_with_replacement, product
from types import SimpleNamespace

import networkx as nx
import pytest

from infinity_grid.o7_edge_canonicalization import (
    IDENTITY_SCHEMA, LEGACY_IDENTITY_SCHEMA, _edge, _tree, state_identity,
    verify_relabeling_witness,
)


def parent(color='A', capacities=(3, 4), reverse=False):
    bags = []
    for capacity in capacities:
        site = ((capacity, 2, 0, 0, 0, 0, 0), (capacity, 2, 0, 0, 0, 0, 0))
        bags.append(((((site,),),),))
    return SimpleNamespace(exact_color=color, h6=tuple(reversed(bags)) if reverse else tuple(bags))


def context(reverse=False, capacities=(3, 4)):
    return SimpleNamespace(n=2, parents=(parent(capacities=capacities, reverse=reverse), parent('B')))


def independent_graph(ctx, edges):
    # Encodes ownership, typed resources, endpoint correlation and multiplicity
    # without importing production graph construction or canonical labeling.
    graph = nx.Graph()

    def tree(owner, path, value, depth):
        key = ('tree', owner, path)
        color = ('owner', ctx.parents[owner].exact_color) if depth == 0 else ('bag', depth)
        if depth == 5:
            color = ('site', tuple(value[0]), tuple(value[1]))
        graph.add_node(key, color=color)
        if depth < 5:
            for i, child in enumerate(value):
                graph.add_edge(key, tree(owner, path + (i,), child, depth + 1))
        return key

    for owner, p in enumerate(ctx.parents):
        tree(owner, (), p.h6, 0)
    for i, edge in enumerate(edges):
        center = ('edge', i)
        graph.add_node(center, color=('edge',))
        for side, offset in enumerate((0, 7)):
            endpoint = ('endpoint', i, side)
            graph.add_node(endpoint, color=('endpoint', edge[offset + 6]))
            graph.add_edge(center, endpoint)
            graph.add_edge(endpoint, ('tree', edge[offset], tuple(edge[offset + 1:offset + 6])))
    return graph


def isomorphic(a, ea, b, eb):
    return nx.is_isomorphic(independent_graph(a, ea), independent_graph(b, eb),
                            node_match=nx.algorithms.isomorphism.categorical_node_match('color', None))


P0 = (0, 0, 0, 0, 0)
P1 = (1, 0, 0, 0, 0)


def test_reproduced_false_merge_is_separated_and_legacy_alias_retained():
    left, right = context(), context(reverse=True)
    edge = (_edge(0, P0, 0, 1, P0, 0),)
    a, b = state_identity(left, edge), state_identity(right, edge)
    assert _tree(left.parents[0].h6) == _tree(right.parents[0].h6)
    assert not isomorphic(left, edge, right, edge)
    assert a['legacy_context_identity'] == b['legacy_context_identity']
    assert a['object_identity'] != b['object_identity']
    assert a['identity_schema'] == IDENTITY_SCHEMA
    assert a['legacy_context_identity_schema'] == LEGACY_IDENTITY_SCHEMA
    assert 'edges' not in a['identity_payload']  # Original coordinates are outside the global key.


@pytest.mark.parametrize('left_type,right_type', product(range(2), repeat=2))
def test_nonidentical_bag_relabeling_transports_edges_to_the_same_identity(left_type, right_type):
    left, right = context(), context(reverse=True)
    edges = (_edge(0, P0, left_type, 1, P1, right_type),) * 2
    moved = (_edge(0, P1, left_type, 1, P1, right_type),) * 2
    a, b = state_identity(left, edges), state_identity(right, moved)
    assert isomorphic(left, edges, right, moved)
    assert a['identity_payload'] == b['identity_payload']
    assert a['object_identity'] == b['object_identity']
    assert verify_relabeling_witness(left, edges, a['edges'], a['witness'])
    assert verify_relabeling_witness(right, moved, b['edges'], b['witness'])


def test_all_bounded_states_match_independent_isomorphism_partition():
    ctx = context()
    singles = [_edge(0, p, t, 1, q, u)
               for p, q in product((P0, P1), repeat=2)
               for t, u in ((0, 0), (0, 1))]
    states = [()] + [(e,) for e in singles] + list(combinations_with_replacement(singles, 2))
    keys = [state_identity(ctx, edges)['object_identity'] for edges in states]
    graphs = [independent_graph(ctx, edges) for edges in states]
    for i, a in enumerate(graphs):
        for j in range(i + 1):
            equal = nx.is_isomorphic(a, graphs[j], node_match=nx.algorithms.isomorphism.categorical_node_match('color', None))
            assert (keys[i] == keys[j]) == equal


def test_actual_semantic_colors_and_unused_resources_cannot_be_replaced_by_palette_numbers():
    a, b = context(capacities=(3, 4)), context(capacities=(5, 6))
    edges = (_edge(0, P0, 0, 1, P0, 0),)
    assert not isomorphic(a, edges, b, edges)
    assert state_identity(a, edges)['object_identity'] != state_identity(b, edges)['object_identity']
    assert state_identity(a, ())['object_identity'] != state_identity(b, ())['object_identity']


def test_owner_slot_contract_and_occurrence_aliases_remain_explicit():
    a = context()
    b = SimpleNamespace(n=2, parents=tuple(reversed(a.parents)))
    edge = (_edge(0, P0, 0, 1, P1, 0),)
    moved = (_edge(1, P0, 0, 0, P1, 0),)
    assert isomorphic(a, edge, b, moved)
    assert state_identity(a, edge)['object_identity'] != state_identity(b, moved)['object_identity']
    first = state_identity(a, edge)
    a.name = 'different occurrence'; a.parents[0].pid = 'different prototype alias'
    assert state_identity(a, edge)['identity_payload'] == first['identity_payload']
    identical = SimpleNamespace(n=2, parents=(parent(), parent()))
    assert state_identity(identical, edge)['object_identity'] == state_identity(identical, moved)['object_identity']


def test_identity_strict_resource_and_capacity_validation():
    ctx = context(capacities=(1, 4))
    edge = _edge(0, P0, 0, 1, P0, 0)
    with pytest.raises(ValueError, match='oversubscribed'):
        state_identity(ctx, (edge, edge))
    with pytest.raises(ValueError):
        state_identity(ctx, ((True, *edge[1:]),))


def test_native_o6_resource_collision_witness_is_separated():
    from dataclasses import replace
    from pathlib import Path
    from infinity_grid.frontier import _load_engine
    from infinity_grid.o7_owner_materialization import materialize_owner
    root = Path('/tmp/ig_o_materialized_discovery_runtime_b_fqhbxc/o7/Infinity_Grid_O7_COMPACT_REPLAY_ROOT_v1_2026-08-29')
    if not root.is_dir():
        pytest.skip('Exact retained native fixture unavailable; synthetic independent controls remain required')
    engine = _load_engine(root)
    records = engine.load_parent_records()
    parent_id = 'O6P:66a20d2196504084a036c4f7399cf1655359b13af5337287f2081a409bc3bf8d:3,3:d665ae55aa52fb6c'
    other_id = 'O6P:e3759f2aafd495565e5a69cfb59a97370e357d82275da802ce9f9ab2beb86c60:3,2:22bfb0feba1417ad'
    parents = {p.pid: p for p in records.values()}
    original, other = parents[parent_id], parents[other_id]
    bags = list(original.h6); bags[0], bags[1] = bags[1], bags[0]
    changed = replace(original, h6=tuple(bags))
    a = engine.LaneContext('P0_REGRESSION', [original, other], [original.pid, other.pid])
    b = engine.LaneContext('P0_REGRESSION', [changed, other], [original.pid, other.pid])
    edges = ((0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0),)
    left, right = state_identity(a, edges), state_identity(b, edges)
    assert _tree(original.h6) == _tree(changed.h6)
    assert _tree(materialize_owner(original.h6, edges, 0)) != _tree(materialize_owner(changed.h6, edges, 0))
    assert left['legacy_context_identity'] == right['legacy_context_identity']
    assert left['object_identity'] != right['object_identity']
