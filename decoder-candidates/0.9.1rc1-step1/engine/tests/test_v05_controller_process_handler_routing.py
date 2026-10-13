from __future__ import annotations

from infinity_grid.v05_controller_event_loop import _CONTROLLER_PROCESS_HANDLERS


def test_controller_process_handler_allowlist_is_exact_and_engine_owned():
    assert _CONTROLLER_PROCESS_HANDLERS == frozenset({
        "infinity_grid.change_validation:validate_revision",
        "infinity_grid.representative_qualification:handler",
        "infinity_grid.v05_engineering_stage:engineering_job_handler",
        "infinity_grid.replay_root_job:replay_root_job_handler",
    })


def test_general_science_handlers_are_not_process_exempt():
    assert "infinity_grid.controller_only_fixture:stage_handler" not in _CONTROLLER_PROCESS_HANDLERS
    assert "infinity_grid.g6_s8_intrinsic_descriptor:run_g6_s8" not in _CONTROLLER_PROCESS_HANDLERS
