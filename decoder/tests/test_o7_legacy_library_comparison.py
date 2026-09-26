"""Registered Decoder VALIDATION for the bounded O7 library comparison."""
from pathlib import Path
from itertools import combinations_with_replacement

import pytest

from infinity_grid.o7_legacy_library_comparison import ARCHIVE, compare_o7_survivor_topology
from infinity_grid.regime_scanner import _graph_basic, _graph_basic_legacy


def test_immutable_o7_survivors_against_igraph(tmp_path):
    result = compare_o7_survivor_topology()
    assert result['status'] == 'PASS'
    assert result['counts'] == {'MATCH': 205}
    assert len(result['records']) == 205
    damaged = tmp_path / 'altered_o7_root.zip'
    damaged.write_bytes(ARCHIVE.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        compare_o7_survivor_topology(damaged)


def test_six_edge_extension_exhaustive_small_graphs():
    # Exhaust every six-occurrence multigraph, including loops and parallels,
    # for n=1..4. The 205-row historical test covers actual n=5,6 O7 cases.
    checked = 0
    for n in range(1, 5):
        possible = [(u, v) for u in range(n) for v in range(u, n)]
        for edges in combinations_with_replacement(possible, 6):
            assert _graph_basic(n, edges, backend='igraph') == _graph_basic_legacy(n, edges), (n, edges)
            checked += 1
    assert checked == 5496
