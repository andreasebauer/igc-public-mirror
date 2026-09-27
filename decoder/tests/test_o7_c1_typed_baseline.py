"""Pinned original-producer reproducibility, without new typed-science claim."""
import pytest

from infinity_grid.o7_c1_typed_baseline import reconstruct_original_case


def test_original_c1_76_case_zero(tmp_path):
    result = reconstruct_original_case(tmp_path, kind='O7xO6', index=0)
    assert result['status'] == 'PASS'
    assert all(result['checks'].values())
    assert result['observed_profile'] == {
        'action_successor_pairs': 606120,
        'total_action_copies': 826771,
        'digest': 'd687fb9ae0148fece145eebf0d6495e5cfed2601def900d3b23da30da35ecacc',
    }


def test_original_c1_77_case_zero(tmp_path):
    result = reconstruct_original_case(tmp_path, kind='O7xO7', index=0)
    assert result['status'] == 'PASS'
    assert all(result['checks'].values())
    assert result['observed_profile'] == {
        'action_successor_pairs': 1973514,
        'total_action_copies': 1973514,
        'digest': 'eab59d0cd747eada744449466cd379581bb763f6b22837e71ceef5a58e52e793',
    }


def test_source_drift_refused(tmp_path, monkeypatch):
    from infinity_grid import o7_c1_typed_baseline as baseline
    changed = tmp_path / 'altered.zip'
    changed.write_bytes(baseline.ORIGINAL.read_bytes() + b'altered')
    monkeypatch.setattr(baseline, 'ORIGINAL', changed)
    with pytest.raises(ValueError, match='checksum mismatch'):
        baseline.reconstruct_original_case(tmp_path / 'restore')
