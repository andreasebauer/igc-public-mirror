"""Page F — Run Monitor.

Public interface:
- start_run(plan_id: int) -> None
- get_status(plan_id: int) -> dict
- pause_run(plan_id: int) -> None
- resume_run(plan_id: int) -> None
- stop_current_member(plan_id: int) -> None
- stop_all(plan_id: int) -> None
"""

from typing import Dict

def start_run(plan_id: int) -> None:
    return

def get_status(plan_id: int) -> Dict:
    return {"state": "idle", "progress": 0.0}

def pause_run(plan_id: int) -> None:
    return

def resume_run(plan_id: int) -> None:
    return

def stop_current_member(plan_id: int) -> None:
    return

def stop_all(plan_id: int) -> None:
    return