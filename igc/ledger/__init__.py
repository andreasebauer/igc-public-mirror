"""
Ledger — DB access (contracts only).

DB is the source of truth:
- Job rows (SimMetricJobs or equivalent)
- Unified job view (full_job_ledger_extended_view or equivalent)
- PathRegistry (DB-driven path resolution)
- Execution and error logs
"""

from .core import (
    create_jobs_for_sim,
    create_metric_jobs_for_frames,
    finalize_job_identity_and_path,
    get_jobs_for_run,
    update_job_status,
    log_execution,
    log_error,
    create_run_id,
    get_frame_order_for_run,
    get_stage_order_for_run,
    mark_run_failed,
    mark_run_finished,
)