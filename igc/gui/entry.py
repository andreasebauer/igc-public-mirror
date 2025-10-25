"""Page A — Entry.

Public interface (called by CLI or server routes):
- list_actions() -> list[str]
- choose_create_sim() -> None            # navigates to Page B
- choose_run_metrics() -> None           # navigates to Page C
"""

from typing import List

def list_actions() -> List[str]:
    return ["Create Simulation", "Run Metrics"]

def choose_create_sim() -> None:
    # navigation stub to Page B
    return

def choose_run_metrics() -> None:
    # navigation stub to Page C
    return