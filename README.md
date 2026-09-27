# Infinity Grid source archive

| Area | Purpose |
| --- | --- |
| [`decoder/`](decoder/) | Exact dev84 revision 5 candidate; engineering-qualified, not activated |
| [`decoder-import/`](decoder-import/README.txt) | Source hashes, verifier, provenance and qualification receipt |
| [`microscope/`](microscope/) | Independently maintained public viewer |
| [`shared-data/`](shared-data/README.txt) | Storage prototype; no complete recovery migration |
| `igc/` and root `pyproject.toml` | Legacy IGC orchestrator, separate from Decoder |

For Decoder, first run `python decoder-import/verify_source.py`, then follow
[the import guide](decoder-import/README.txt). Installing the repository root
installs the legacy package. Pin an exact commit for recovery.

CI checks source integrity only. Scientific runs and qualification use the
Decoder native controller. Preserve independent backups; this checkout does
not contain the full scientific recovery closure.

---

Decoder source import: see [decoder-import/README.txt](decoder-import/README.txt).
Shared data storage: see [shared-data/README.txt](shared-data/README.txt).

# Infinity Grid — Python Orchestrator & Analysis (IGC)

This repository contains the Python rewrite of the Infinity Grid stack:

- **oe** – Orchestrator (includes sweep expansion; sequential CPU f64).
- **ledger** – DB access layer (Postgres 17; DB is the contract).
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