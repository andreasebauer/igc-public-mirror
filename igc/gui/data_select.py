"""Page C — Data Selection (frames to analyze).

Public interface:
- list_runs(sim_id: int | None = None) -> list[dict]      # known DB runs
- index_path(path: str) -> int                            # registers a path; returns run_id
- describe_run(run_id: int) -> dict                       # fields, frames, summary
- select_frame_range(run_id: int, at_start: int, at_end: int, stride: int) -> dict
"""

from typing import Dict, List, Optional

def list_runs(sim_id: Optional[int] = None) -> List[Dict]:
    return []

def index_path(path: str) -> int:
    return 0

def describe_run(run_id: int) -> Dict:
    return {}

def select_frame_range(run_id: int, at_start: int, at_end: int, stride: int) -> Dict:
    return {"run_id": run_id, "at_start": at_start, "at_end": at_end, "stride": stride}