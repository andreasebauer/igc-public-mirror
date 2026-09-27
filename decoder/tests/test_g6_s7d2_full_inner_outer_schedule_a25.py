from __future__ import annotations

import inspect


def test_a25_active_schedule_pays_full_inner_vector_once_per_outer_component():
    import infinity_grid.g6_s7_depth2_completion as d2
    assert d2.EXECUTION_CONTEXT_COUNT == 124
    assert d2.INNER_PREFIX_SCHEDULE == (124,)
    assert sum(d2.INNER_PREFIX_SCHEDULE) == 124
    assert sum((1, 8, 32, 64, 124)) == 229
    assert d2.EXECUTION_STRATEGY_ID == "FULL_INNER_OUTER_MONOTONE_FACTOR_SWAP_NORMALIZED_V2"


def test_a25_handler_keeps_exact_monotone_singleton_pruning_and_factor_swap_science_binding():
    import infinity_grid.g6_s7_depth2_completion as d2
    src = inspect.getsource(d2.run_g6_s7_depth2_completion)
    assert "for prefix_count in INNER_PREFIX_SCHEDULE" in src
    assert "for outer_idx in range(EXECUTION_CONTEXT_COUNT)" in src
    assert 'if len(g["members"]) > 1' in src
    assert "FACTOR_SWAP_NORMALIZATION_ID" in src
    assert '"ordinary_context_count": 248' in src
    assert '"execution_context_count": EXECUTION_CONTEXT_COUNT' in src
    assert '"inner_prefix_schedule_execution_only": list(INNER_PREFIX_SCHEDULE)' in src
    assert '"execution_strategy": EXECUTION_STRATEGY_ID' in src
    assert "run_content_indexed_generation" not in src


def test_a25_is_execution_only_and_does_not_relax_completion_requirement():
    import infinity_grid.g6_s7_depth2_completion as d2
    src = inspect.getsource(d2.run_g6_s7_depth2_completion)
    assert "full_for_survivors" in src
    assert 'raise G6S7Depth2Error("S7D2_INCOMPLETE_FULL_OBSERVER")' in src
    assert '"g6_graduated": False' in src
    assert '"global_minimality_earned": False' in src
