"""
OE (Orchestrator) v1 — sequential CPU f64

Responsibilities:
- Seeding Pass 1: insert Sim/Metric jobs for the selected run (no reuse).
- Seeding Pass 2: finalize each job (fill missing fields; assign unique output path via PathRegistry in DB).
- Execution: sequential per frame; per-frame staging order is pp0 -> pp1 -> pp3 -> ...; stop on first failure.
- Handoff: Runner computes; Writer persists; Ledger records status/artifacts/logs.
"""

from .core import (
    seed_compute_jobs,
    finalize_seeded_jobs,
    run,
    pause,
    resume,
    stop,
    status,
)