"""
OE public surface (signatures only; no implementation).

Staging rule:
- Metrics are grouped by ppN stage labels (integers). Execution order is strictly ascending:
  [0, 1, 3, 4, 5, ...] — i.e., pp2 is simply skipped unless present in the selected set.

All path construction uses DB PathRegistry (no string templates in code).
"""

from typing import Dict, Iterable, Iterator, List, Literal, Optional, Tuple, Union

Mode = Literal["SIMS", "METRICS"]

def seed_jobs(
    *,
    mode: Mode,
    sim_id: Optional[int],
    run_set_id: Optional[int],
    frame_start: int,
    frame_end: int,
    frame_stride: int,
    selected_metric_step_ids: Optional[List[int]] = None,
) -> int:
    """
    Seeding Pass 1:
    - If mode == 'SIMS': insert one simulation job per frame (phase='SIM', step=1).
    - If mode == 'METRICS': insert one metric job per (frame x selected metric step).
    Returns a run_id grouping all seeded jobs for this invocation.
    """
    raise NotImplementedError

def finalize_seeded_jobs(*, run_id: int) -> int:
    """
    Seeding Pass 2:
    - For each newly-seeded job in this run_id, fill missing typed fields
      and assign a unique output path+filename via the DB PathRegistry.
    - Intermediate steps (1/2) are .npy; step 3 finals use the first output type listed by DB.
    Returns the number of finalized jobs.
    """
    raise NotImplementedError

def run(*, run_id: int) -> Iterator[Dict]:
    """
    Execute jobs for the run_id, strictly sequential per frame.
    Order per frame:
      for stage in sorted(pp stages): for job in stage: execute
    Yields structured event dicts for the Run Monitor (Page F).
    Stop-on-error: first failed job marks the run failed and halts.
    """
    raise NotImplementedError

def pause(*, run_id: int) -> None:
    """Request a cooperative pause at the next safe checkpoint."""
    raise NotImplementedError

def resume(*, run_id: int) -> None:
    """Resume a paused run."""
    raise NotImplementedError

def stop(*, run_id: int) -> None:
    """Gracefully stop after the current job completes."""
    raise NotImplementedError

def status(*, run_id: int) -> Dict:
    """
    Return a summary:
      - counts by status (queued/running/written/failed)
      - current frame and stage (if running)
      - last error (if any)
    """
    raise NotImplementedError