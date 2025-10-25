"""
Ledger contracts (signatures only; no implementation).

Notes:
- No secrets in code. Use environment (PGHOST, PGUSER, PGPASSWORD, PGDATABASE) or DSN.
- All path building is delegated to DB PathRegistry (function or view).
"""

from typing import Dict, Iterable, List, Literal, Optional, Sequence, Tuple

Status = Literal["queued", "running", "written", "failed"]

def create_run_id(*, mode: str, sim_id: Optional[int], run_set_id: Optional[int]) -> int:
    """Create and return a run_id row to group seeded jobs for this invocation."""
    raise NotImplementedError

def create_jobs_for_sim(
    *,
    run_id: int,
    sim_id: int,
    frame_start: int,
    frame_end: int,
    frame_stride: int,
) -> int:
    """
    Insert one simulation job per frame (phase='SIM', step=1, status='queued').
    Returns inserted count.
    """
    raise NotImplementedError

def create_metric_jobs_for_frames(
    *,
    run_id: int,
    run_set_id: int,
    frame_start: int,
    frame_end: int,
    frame_stride: int,
    selected_metric_step_ids: Sequence[int],
) -> int:
    """
    Insert one metric job per (frame x selected metric step), status='queued'.
    Returns inserted count.
    """
    raise NotImplementedError

def finalize_job_identity_and_path(*, run_id: int) -> int:
    """
    For each seeded job in run_id:
      - Fill typed fields (group_id, step_id, phase, frame, job_type, params, etc.).
      - Resolve output path via DB PathRegistry and persist it to the job row.
    Returns finalized count.
    """
    raise NotImplementedError

def get_jobs_for_run(
    *,
    run_id: int,
    frame: Optional[int] = None,
    stage_pp: Optional[int] = None,
    status: Optional[Status] = None,
) -> List[Dict]:
    """
    Return job dicts from the unified job view for this run.
    Used by OE to iterate frames and stages in order.
    """
    raise NotImplementedError

def get_frame_order_for_run(*, run_id: int) -> List[int]:
    """Return the sorted list of frame indices present for this run."""
    raise NotImplementedError

def get_stage_order_for_run(*, run_id: int) -> List[int]:
    """
    Return the sorted list of pp stages present (e.g., [0, 1, 3, ...]).
    OE uses this to walk pp0 -> pp1 -> pp3 -> ...
    """
    raise NotImplementedError

def update_job_status(
    *,
    job_id: int,
    to_status: Status,
    set_start: bool = False,
    set_finish: bool = False,
    error_message: Optional[str] = None,
) -> None:
    """
    Transition: queued->running->written|failed.
    Timestamps: set_start -> started_at; set_finish -> finished_at.
    Persist error_message when to_status='failed'.
    """
    raise NotImplementedError

def log_execution(
    *,
    job_id: int,
    runtime_ms: int,
    queue_wait_ms: int = 0,
    was_aliased: bool = False,           # always False in v1
    reused_step_id: Optional[int] = None, # always None in v1
    reuse_metric_id: Optional[int] = None,# always None in v1
    learning_note: Optional[str] = None,
) -> None:
    """Append a row to JobExecutionLog."""
    raise NotImplementedError

def log_error(*, job_id: int, message: str) -> None:
    """Append a row to JobErrors."""
    raise NotImplementedError

def mark_run_failed(*, run_id: int, message: str) -> None:
    """Mark run as failed with message."""
    raise NotImplementedError

def mark_run_finished(*, run_id: int) -> None:
    """Mark run as fully finished."""
    raise NotImplementedError