from infinity_grid.o7_c1_graph_comparison import ARCHIVE, compare_c1_outer_graphs
from infinity_grid.regime_scanner import _graph_basic, _graph_basic_legacy
from itertools import combinations_with_replacement
import random


def test_historical_c1_qualified_outer_graph_subset():
    result = compare_c1_outer_graphs()
    assert result['status'] == 'PASS'
    assert result['counts'] == {'MATCH': 192}
    assert len(result['cases']) == 192


def test_historical_c1_archive_identity(tmp_path):
    altered = tmp_path / ARCHIVE.name
    altered.write_bytes(ARCHIVE.read_bytes() + b'altered')
    try:
        compare_c1_outer_graphs(altered)
    except ValueError as error:
        assert 'checksum mismatch' in str(error)
    else:
        raise AssertionError('Archive drift accepted')


def test_extended_graph_domain_against_retained_independent_calculation():
    # Deterministic diversity across connected/disconnected, looped, and
    # parallel multigraphs, including the full new n=8, m=10 boundary.
    rng = random.Random(0xC1A72026)
    compared = 0
    for n in range(1, 9):
        pairs = list(combinations_with_replacement(range(n), 2))
        for m in range(11):
            for _ in range(128):
                edges = [rng.choice(pairs) for _ in range(m)]
                assert _graph_basic(n, edges, backend='igraph') == _graph_basic_legacy(n, edges), (n, edges)
                compared += 1
    assert compared == 11264
