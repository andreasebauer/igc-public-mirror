"""Preregistered exact comparisons for the bounded owner-group replacement."""
import itertools
import math
import sys
import pytest
from infinity_grid import graph_library_backend as lib, regime_scanner as scanner

@pytest.mark.parametrize('n', range(1, 7))
def test_every_single_generator(n):
    count = 0
    for p in itertools.permutations(range(n)):
        assert lib._owner_group_closure_sympy(n, [p]) == lib._owner_group_closure_legacy(n, [p])
        count += 1
    assert count == math.factorial(n)

@pytest.mark.parametrize('n', range(1, 6))
def test_every_ordered_generator_pair(n):
    count = 0
    permutations = list(itertools.permutations(range(n)))
    for p in permutations:
        for q in permutations:
            assert lib._owner_group_closure_sympy(n, [p, q]) == lib._owner_group_closure_legacy(n, [p, q])
            count += 1
    assert count == math.factorial(n) ** 2

def test_six_owner_selection_and_full_symmetric_group():
    permutations = list(itertools.permutations(range(6)))
    for i in range(96):
        p, q = permutations[(i * 37) % 720], permutations[(i * 113 + 17) % 720]
        expected = lib._owner_group_closure_legacy(6, [p, q])
        assert lib._owner_group_closure_sympy(6, [p, q]) == expected
        assert lib._owner_group_closure_sympy(6, [q, p, tuple(range(6)), p, q]) == expected
    adjacent = []
    for i in range(5):
        p = list(range(6)); p[i], p[i+1] = p[i+1], p[i]; adjacent.append(tuple(p))
    actual = lib._owner_group_closure_sympy(6, adjacent)
    assert actual == lib._owner_group_closure_legacy(6, adjacent)
    assert len(actual) == 720

def test_identity_fixed_points_and_refusals():
    for n in range(1, 7):
        identity = tuple(range(n))
        for gens in [[], [identity]]:
            assert lib._owner_group_closure_sympy(n, gens) == lib._owner_group_closure_legacy(n, gens) == [identity]
    for n, gens in [(0, []), (7, []), (2, [(0, 0)]), (2, [(0,)]), (2, [(False, True)])]:
        with pytest.raises(ValueError): lib._owner_group_closure_sympy(n, gens)

def test_missing_library_does_not_fall_back(monkeypatch):
    monkeypatch.setitem(sys.modules, 'sympy.combinatorics', None)
    with pytest.raises(ModuleNotFoundError): scanner._automorphisms(2, [(1,)*7]*2, [])
    assert scanner._automorphisms(2, [(1,)*7]*2, [], backend='legacy') == [(0, 1), (1, 0)]

def test_invalid_library_output_does_not_fall_back(monkeypatch):
    import sympy.combinatorics as groups
    class BadGroup:
        def __init__(self, *args, **kwargs): pass
        def order(self): return 2
        def generate_schreier_sims(self, **kwargs): return iter([[0, 1], [0, 1]])
    monkeypatch.setattr(groups, 'PermutationGroup', BadGroup)
    with pytest.raises(RuntimeError, match='Invalid SymPy owner group output'):
        scanner._automorphisms(2, [(1,)*7]*2, [])

@pytest.mark.parametrize('n', range(1, 5))
def test_typed_graph_integration(n):
    slots = [(u,v,a,b) for u in range(n) for v in range(u+1,n) for a in range(2) for b in range(2)]
    count = 0
    for colors in [[(1,)*7]*n, [((i%2),)*7 for i in range(n)]]:
        for m in range(4):
            for edges in itertools.combinations_with_replacement(slots, m):
                assert scanner._automorphisms(n, colors, edges, backend='igraph_bliss') == scanner._automorphisms_legacy(n, colors, edges)
                count += 1
    assert count == 2 * math.comb(len(slots)+3, 3)

def test_downstream_action_summary():
    class State:
        owner_caps = [(2,)*7]*4
        typed_edges = [(0,1,0,1),(1,2,1,0),(2,3,0,1),(0,3,1,0)]
        top_pairs = [(u,v) for u,v,a,b in typed_edges]
    pairs = list(itertools.product(range(7), repeat=2))
    assert scanner._action_aggregate(State(), pairs, automorphism_backend='igraph_bliss') == scanner._action_aggregate(State(), pairs, automorphism_backend='legacy')
