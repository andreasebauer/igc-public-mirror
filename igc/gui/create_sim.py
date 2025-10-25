"""Page B — Create Simulation (spec only).

Public interface:
- load_defaults(sim_id: int | None) -> dict
- validate_spec(spec: dict) -> list[str]     # empty list = ok
- save_simulation(spec: dict) -> int         # returns sim_id
"""

from typing import Dict, List, Optional

def load_defaults(sim_id: Optional[int] = None) -> Dict:
    return {}

def validate_spec(spec: Dict) -> List[str]:
    return []

def save_simulation(spec: Dict) -> int:
    return 0