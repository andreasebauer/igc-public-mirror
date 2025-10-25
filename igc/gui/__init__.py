"""
GUI workflow (decoupled V1)

Pages:
- A: Entry (choose Create Simulation or Run Metrics)
- B: Create Simulation (spec only, no run)
- C: Data Selection (pick frames / range)
- D: Metrics Selection (choose exact steps; deps auto-added)
- F: Run Monitor (observe OE; pause/resume/stop)
"""
__all__ = [
    "entry",
    "create_sim",
    "data_select",
    "metrics_select",
    "run_monitor",
]