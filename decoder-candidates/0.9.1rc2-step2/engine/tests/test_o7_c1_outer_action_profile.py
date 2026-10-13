"""Registered hybrid C1 profile comparison; inherited O6 observer is pinned."""
from infinity_grid.o7_c1_outer_action_profile import hybrid_profile
from infinity_grid.o7_c1_typed_baseline import reconstruct_original_case


def test_outer_action_76_profile_exact(tmp_path):
    result = reconstruct_original_case(tmp_path, kind='O7xO6', index=0,
                                       profile_calculator=hybrid_profile)
    assert result['status'] == 'PASS', result
    assert result['observed_profile'] == {
        'digest': 'd687fb9ae0148fece145eebf0d6495e5cfed2601def900d3b23da30da35ecacc',
        'action_successor_pairs': 606120, 'total_action_copies': 826771}


def test_outer_action_77_profile_exact(tmp_path):
    result = reconstruct_original_case(tmp_path, kind='O7xO7', index=0,
                                       profile_calculator=hybrid_profile)
    assert result['status'] == 'PASS', result
    assert result['observed_profile'] == {
        'digest': 'eab59d0cd747eada744449466cd379581bb763f6b22837e71ceef5a58e52e793',
        'action_successor_pairs': 1973514, 'total_action_copies': 1973514}
