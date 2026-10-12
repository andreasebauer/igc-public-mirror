"""Execute only as registered Decoder validation; frozen exhaustive domain."""
from itertools import combinations_with_replacement, combinations
from math import comb
import sys
import pytest
from infinity_grid import regime_scanner as scanner
from infinity_grid import graph_metrics_library as library

@pytest.mark.parametrize('n', range(1, 7))
def test_exhaustive_multigraph_metrics(n):
    possible = list(combinations_with_replacement(range(n), 2))
    count = 0
    for m in range(6):
        for edges in combinations_with_replacement(possible, m):
            before = tuple(edges)
            expected = scanner._graph_basic_legacy(n, edges)
            assert scanner._graph_basic(n, edges, backend='igraph') == expected, (n, edges)
            assert edges == before
            count += 1
    assert count == comb(len(possible) + 5, 5)
    print('EXHAUSTIVE_METRICS', n, count, flush=True)


def test_orientation_generator_and_no_input_mutation():
    e = [[2, 1], [1, 0], [2, 1], [1, 1]]
    before = [x[:] for x in e]
    assert scanner._graph_basic(3, (x for x in e)) == scanner._graph_basic_legacy(3, e)
    assert e == before


def test_dispatch_empty_and_unsupported_inputs():
    for n, e in [(0, []), (9, [(0, 1)]), (2, [(0, 1)]*11), (2, [('0', '1')]), (2, [(False, True)])]:
        assert library.select_backend(n, e) == 'legacy'
        assert scanner._graph_basic(n, e) == scanner._graph_basic_legacy(n, e)
        with pytest.raises(ValueError): scanner._graph_basic(n, e, backend='igraph')
    with pytest.raises(ValueError): scanner._graph_basic(2, [], backend='typo')
    with pytest.raises(IndexError): scanner._graph_basic(2, [(0, 2)])


def test_no_silent_library_fallback(monkeypatch):
    monkeypatch.setitem(sys.modules, 'igraph', None)
    with pytest.raises(ModuleNotFoundError): scanner._graph_basic(2, [(0, 1)])
    assert scanner._graph_basic(2, [(0, 1)], backend='legacy') == scanner._graph_basic_legacy(2, [(0, 1)])


def test_library_runtime_error_propagates(monkeypatch):
    import igraph
    def broken(*args, **kwargs): raise RuntimeError('declared backend failure')
    monkeypatch.setattr(igraph, 'Graph', broken)
    with pytest.raises(RuntimeError, match='declared backend failure'): scanner._graph_basic(2, [(0, 1)])


def test_downstream_service_effects(monkeypatch):
    for n in range(2, 5):
        possible = list(combinations(range(n), 2))
        for m in range(1, 5):
            for edges in combinations_with_replacement(possible, m):
                edges = list(edges)
                if not scanner._graph_basic_legacy(n, edges)['connected']: continue
                for pair in possible:
                    actual = scanner._service_effect(n, edges, pair)
                    with monkeypatch.context() as context:
                        context.setattr(scanner, '_graph_basic', scanner._graph_basic_legacy)
                        expected = scanner._service_effect(n, edges, pair)
                    assert actual == expected, (n, edges, pair)
