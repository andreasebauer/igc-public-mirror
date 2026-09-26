from __future__ import annotations

"""Controller-owned entrypoint for the registered replay root job.

The handler is intentionally thin.  Durable orchestration belongs to the
Decoder runtime, while scientific node execution remains unbound until P4.
"""

from typing import Any, Mapping

from .v05_chain import ChainExecutionResult
from .v05_origin_guard import require_controller_execution_origin


HANDLER_REF = "infinity_grid.replay_root_job:replay_root_job_handler"


def replay_root_job_handler(stage: Mapping[str, Any], runtime) -> ChainExecutionResult:
    require_controller_execution_origin("replay-root-job")
    if stage.get("stage_id") != "REPLAY:L0_TO_G8:ROOT":
        raise RuntimeError("REPLAY_ROOT_STAGE_REQUIRED")
    result = runtime.run_registered_replay_root(
        dict(stage["execution"].get("parameters") or {}),
        dict(stage.get("input_artifacts") or {}),
    )
    return ChainExecutionResult(result=result)
