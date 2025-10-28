from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import os, json, time
from igc.db.pg import cx, fetchone_dict, execute
from igc.common import hashutil
from igc.common import paths as igc_paths

STORE = Path(os.environ.get("IGC_STORE", "/data/igc"))

# --- Hash helpers ------------------------------------------------------------

def stable_hash(obj: Any) -> str:
    return hashutil.hash_json(obj)  # canonical JSON-based hash from your repo

def compute_effective_hash(base_spec: Dict[str, Any], overrides: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
    eff: Dict[str, Any] = dict(base_spec)
    eff.update(overrides or {})
    base_spec_hash = stable_hash(base_spec)
    overrides_hash = stable_hash(overrides or {})
    effective_spec_hash = stable_hash(eff)
    return base_spec_hash, overrides_hash, effective_spec_hash, eff

# --- Path planning for Frame 0 (run-overrides.json) --------------------------

def plan_frame0_dir(sim_row: Dict[str, Any]) -> Path:
    """
    Use your existing path templates to compute a sim root and Frame_0000 path.
    Fallback to STORE/sim-{id}/Frame_0000_s00 if template helpers are minimal.
    """
    simname = sim_row.get("name") or f"sim-{sim_row['id']}"
    # If igc_paths exposes builders, use them; otherwise fallback:
    sim_root = STORE / simname
    frame0 = sim_root / "Frame_0000_s00"
    frame0.mkdir(parents=True, exist_ok=True)
    return frame0

def write_run_overrides_json(frame0: Path,
                             sim_id: int,
                             base_hash: str,
                             overrides_hash: str,
                             effective_hash: str,
                             overrides: Dict[str, Any]) -> Path:
    payload = {
        "sim_id": sim_id,
        "base_spec_hash": base_hash,
        "overrides_hash": overrides_hash,
        "effective_spec_hash": effective_hash,
        "mode": "run",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overrides": overrides,
        "notes": "Run-mode overrides; base spec unchanged in DB."
    }
    p = frame0 / "run_overrides.json"
    p.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return p

# --- Enqueue job (simmetricjobs + initial jobexecutionlog) -------------------

def enqueue_sim_job(sim_id: int,
                    effective_spec_hash: str,
                    bundle_level: Optional[str] = None,
                    is_visualization: Optional[bool] = None,
                    precision: Optional[str] = None) -> int:
    """
    Creates a queued job in simmetricjobs and writes initial jobexecutionlog row.
    Returns job id.
    """
    with cx() as conn:
        row = fetchone_dict(conn,
            "INSERT INTO public.simmetricjobs (simid, status, bundle_level, is_visualization, precision, spec_hash) "
            "VALUES (%s, 'queued', %s, %s, %s, %s) RETURNING jobid",
            (sim_id, bundle_level, is_visualization, precision, effective_spec_hash))
        job_id = int(row["jobid"])
        execute(conn,
            "INSERT INTO public.jobexecutionlog (jobid, simid, status, queued_at, spec_hash) "
            "VALUES (%s, %s, 'queued', NOW(), %s)",
            (job_id, sim_id, effective_spec_hash))
        return job_id

# --- Terminalization & flushing policy (optional call from runner) -----------

def finalize_and_flush(job_id: int, sim_id: int, status: str = "finished") -> None:
    """
    Marks job terminal in jobexecutionlog, then flushes simmetricjobs row.
    Intended to be called by OE when run ends; safe if idempotent.
    """
    with cx() as conn:
        execute(conn,
            "INSERT INTO public.jobexecutionlog (jobid, simid, status, finished_at) VALUES (%s, %s, %s, NOW())",
            (job_id, sim_id, status))
        execute(conn, "DELETE FROM public.simmetricjobs WHERE jobid=%s", (job_id,))
