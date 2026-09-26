"""Registered Decoder validation against original post-O7 structural CSV."""
import pytest

from infinity_grid.o7_historical_structural_comparison import ORIGINAL, compare_structural_audit


def test_original_post_o7_csv_matches_igraph(tmp_path):
    result = compare_structural_audit()
    assert result['status'] == 'PASS'
    assert result['row_counts'] == {'MATCH': 205}
    assert all(result['aggregate_checks'].values())
    assert result['observed_local_shell_profiles'] == 28
    damaged = tmp_path / 'changed.zip'
    damaged.write_bytes(ORIGINAL.read_bytes() + b'changed')
    with pytest.raises(ValueError, match='checksum mismatch'):
        compare_structural_audit(damaged)
