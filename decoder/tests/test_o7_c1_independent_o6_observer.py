"""Bounded comparison of new nested O6 action interpreter to pinned C1 evidence."""
from infinity_grid.o7_c1_outer_action_profile import independent_o6_hybrid_profile
from infinity_grid.o7_c1_typed_baseline import reconstruct_original_case


def test_independent_o6_observer_c1_76_case_zero(tmp_path):
    observed = reconstruct_original_case(
        tmp_path, kind='O7xO6', index=0,
        profile_calculator=independent_o6_hybrid_profile)
    assert observed['status'] == 'PASS', observed
    assert observed['observed_profile'] == {
        'digest': 'd687fb9ae0148fece145eebf0d6495e5cfed2601def900d3b23da30da35ecacc',
        'action_successor_pairs': 606120,
        'total_action_copies': 826771,
    }
