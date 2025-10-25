"""
IGC — Infinity Grid (Python Orchestrator & Analysis)

Modules
-------
- oe: Orchestrator (includes sweep expansion; sequential CPU f64)
- ledger: Postgres access (DB is the contract; no file configs)
- runner: Executes one step in memory (pure compute)
- sims: Modular simulator (params from `Simulations` table)
- metrics: Metric kernels (selection from DB; parentage enforced)
- writer: The only file I/O + artifact recording
- gui: Operator workflow (A, B, C, D, F)
- utilities: Logging, hashing, NPY I/O, retention helpers
"""

__all__ = [
    "oe",
    "ledger",
    "runner",
    "sims",
    "metrics",
    "writer",
    "gui",
    "utilities",
]

__version__ = "0.0.1"