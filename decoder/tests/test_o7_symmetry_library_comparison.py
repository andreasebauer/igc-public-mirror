"""Registered validation of BLISS/SymPy owner permutations on O7 inputs."""
from itertools import combinations_with_replacement

import pytest

from infinity_grid.o7_legacy_library_comparison import ARCHIVE
from infinity_grid.o7_symmetry_library_comparison import compare_o7_typed_owner_symmetries
from infinity_grid.regime_scanner import _automorphisms, _automorphisms_legacy


def test_immutable_o7_typed_owner_symmetries(tmp_path):
    result = compare_o7_typed_owner_symmetries()
    assert result['status'] == 'PASS' and result['counts'] == {'MATCH': 205}
    assert len(result['records']) == 205
    tampered = tmp_path / 'altered.zip'
    tampered.write_bytes(ARCHIVE.read_bytes() + b'altered')
    with pytest.raises(ValueError, match='checksum mismatch'):
        compare_o7_typed_owner_symmetries(tampered)


def test_six_edge_typed_multigraph_extension_small_domain():
    checked = 0
    for n in range(2, 5):
        color = (0, 0, 0, 0, 0, 0, 0)
        pairs = [(u, v) for u in range(n) for v in range(u + 1, n)]
        for chosen in combinations_with_replacement(pairs, 6):
            edges = [(u, v, 1 if i % 3 else 2, 2 if i % 3 else 1)
                     for i, (u, v) in enumerate(chosen)]
            colors = [color] * n
            assert _automorphisms(n, colors, edges, backend='igraph_bliss') == _automorphisms_legacy(n, colors, edges)
            checked += 1
    assert checked == 491
