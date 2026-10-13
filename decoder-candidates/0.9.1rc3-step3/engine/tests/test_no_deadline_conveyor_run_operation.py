import os
import tempfile
import time
from pathlib import Path

from infinity_grid.conveyor import operational_triggers
from infinity_grid.run_operations import RunOperations, RunOperationsError


POLICY = "NO_AUTOMATIC_RUNTIME_DEADLINE_V1"


def test_conveyor_and_run_operation_ignore_time_only_limits_but_keep_non_time_limits():
    assert os.environ.get("IG_DECODER_EXECUTION_POLICY") == POLICY

    chain_plan = {
        "finite_limits": {
            "max_level_attempts": 10,
            "max_unique_candidates": 10,
            "max_level_wall_seconds": 1,
            "max_total_wall_seconds": 1,
            "max_total_storage_bytes": 10_000_000,
            "minimum_free_bytes": 1,
        }
    }
    triggers = operational_triggers(
        chain_plan=chain_plan,
        attempts=1,
        unique_candidates=1,
        level_wall_seconds=120.0,
        chain_wall_seconds=240.0,
        total_storage_bytes=0,
        free_bytes=10_000_000,
    )
    assert "O_LEVEL_TIME_BUDGET" not in triggers
    assert "O_CHAIN_TIME_BUDGET" not in triggers

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        ops = object.__new__(RunOperations)
        ops.resource_budget = {
            "max_wall_seconds_total": 0.001,
            "max_wall_seconds_per_depth": 0.001,
            "max_run_dir_bytes": 10_000_000,
            "max_mirror_dir_bytes": 10_000_000,
            "max_rss_bytes": 10**15,
        }
        ops.run_dir = root / "run"
        ops.mirror_root = root / "mirror"
        ops.run_dir.mkdir()
        ops.mirror_root.mkdir()
        ops._run_started_perf = time.perf_counter() - 5.0

        result = ops.check_resource_budget(depth=7, depth_elapsed_seconds=9.0)
        assert result["status"] == "PASS"
        assert result["observed"]["wall_seconds_total"] > 0.001

        ops.resource_budget["max_run_dir_bytes"] = -1
        try:
            ops.check_resource_budget(depth=7, depth_elapsed_seconds=9.0)
        except RunOperationsError as exc:
            assert "max_run_dir_bytes" in str(exc)
        else:
            raise AssertionError("non-time run-directory limit was not enforced")
