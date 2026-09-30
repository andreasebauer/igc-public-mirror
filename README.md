# Infinity Grid source repository

**Decoder starts at [DECODER_READ_FIRST.txt](DECODER_READ_FIRST.txt).**

The consolidation branch contains exact dev137 source, runtime reconstruction,
mode checks, one dependency catalog and the current RC ledger. **Unqualified;
full RC OPEN.** No release or runtime pointer is promoted.

Run `python decoder-admin/decoder.py status` and
`python decoder-admin/decoder.py verify-source` from this directory.

| Area | Purpose |
| --- | --- |
| `decoder/` | Exact dev137 source and tests |
| `decoder-admin/` | Single recovery entry point, dependency catalog, status |
| `decoder-import/` | Source integrity and historical import provenance |
| `microscope/`, `decoder-preview/` | Existing independently maintained interfaces |
| `shared-data/` | Historical storage prototype; Drive remains the data store |
| Root `pyproject.toml`, `igc/` | Legacy package; not the Decoder installation root |

Large runtime archives and private/scientific evidence remain on Drive and are
located by immutable hashes. Five candidate fixtures and complete historical
science recovery closure remain open. CI verifies repository integrity only.

---

## Legacy IGC documentation (historical)

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