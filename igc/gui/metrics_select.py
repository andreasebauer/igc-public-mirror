"""Page D — Metrics Selection (table-driven).

Public interface:
- list_available_metrics(run_id: int) -> list[dict]
- resolve_dependencies(selected_metric_ids: list[int]) -> dict
- validate_closed_plan(resolution: dict) -> list[str]       # empty list = ok
- save_plan(run_id: int, resolved_metric_ids: list[int]) -> int  # returns plan_id
"""

from typing import Dict, List

def list_available_metrics(run_id: int) -> List[Dict]:
    return []

def resolve_dependencies(selected_metric_ids: List[int]) -> Dict:
    # returns {"final": [ids], "auto_added_parents": [ids]}
    return {"final": selected_metric_ids, "auto_added_parents": []}

def validate_closed_plan(resolution: Dict) -> List[str]:
    return []

def save_plan(run_id: int, resolved_metric_ids: List[int]) -> int:
    return 0