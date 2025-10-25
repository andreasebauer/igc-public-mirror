# Infinity Grid — Python Orchestrator & Analysis (IGC)

This repository contains the Python rewrite of the Infinity Grid stack:

- **oe** – Orchestrator (includes sweep expansion; sequential CPU f64).
- **ledger** – DB access layer (Postgres; DB is the contract).
- **runner** – Executes one step in memory (pure compute).
- **sims** – Modular simulator (same math, different regimes/scales via `Simulations`).
- **metrics** – Metric kernels (table-driven selection; staging from DB parentage).
- **writer** – The only component that writes files + records artifacts.
- **gui** – Operator workflow (A: Entry, B: Create Simulation, C: Data Select, D: Metrics Select, F: Run Monitor).
- **utilities** – Logging, hashing, NPY I/O, retention, small helpers.

## Development principles

- Determinism first.  
- DB is the contract (no file/JSON configs).  
- Pure compute functions; I/O isolated in Writer.  
- CPU-only, float64.  
- Sequential execution, resumable and idempotent.  

## Repo layout