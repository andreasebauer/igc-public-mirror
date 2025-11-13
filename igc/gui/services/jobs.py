from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
from pathlib import Path
import os, json, time

from igc.db.pg import cx, fetchone_dict, execute
from igc.common import hashutil
from igc.common import paths as igc_paths
from igc.oe import core as oe_core  # use OE’s sim_label + run token

# Unified default root; OE core also targets /data/simulations
STORE = Path(os.environ.get("IGC_STORE", "/data/simulations"))

# --- Hash helpers ------------------------------------------------------------

def stable_hash(obj: Any) -> str:
    return hashutil.hash_json(obj)  # canonical JSON-based hash from your repo

def compute_effective_hash(base_spec: Dict[str, Any],
                           overrides: Dict[str, Any]) -> Tuple[str, str, str, Dict[str, Any]]:
    eff: Dict[str, Any] = dict(base_spec)
    eff.update(overrides or {})
    base_spec_hash = stable_hash(base_spec)
    overrides_hash = stable_hash(overrides or {})
    effective_spec_hash = stable_hash(eff)
    return base_spec_hash, overrides_hash, effective_spec_hash, eff

# --- Path planning for Frame 0 (run-overrides.json) --------------------------

def plan_frame0_dir(sim_row: Dict[str, Any]) -> Path:
    """
    Stage under the *same* run root OE uses:
      /data/simulations/{sim_label}/{run_token}/Frame_0000
    We reuse OE’s run token so GUI and OE write to the identical timestamp folder.
    """
    sim_id = int(sim_row["id"])
    sim_label = (sim_row.get("label") or str(sim_id))

    # Use the same stable run token as OE (create if missing so OE will reuse it)
    tt = oe_core._RUN_TOKEN.setdefault(sim_id, oe_core._time_token_utc())

    frame0 = STORE / sim_label / tt / "Frame_0000"
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

# --- Enqueue job (simmetricjobs only; OE will log terminal statuses) ---------

def enqueue_sim_job(sim_id: int,
                    effective_spec_hash: str,
                    bundle_level: Optional[str] = None,
                    is_visualization: Optional[bool] = None,
                    precision: Optional[str] = None) -> int:
    """
    Return an existing seeded job id for this sim (if any).
    No non-terminal logging is written here; OE core owns terminal logs.
    """
    with cx() as conn:
        row = fetchone_dict(conn,
            "SELECT jobid FROM public.simmetjobs WHERE simid=%s ORDER BY createdate ASC LIMIT 1",
            (sim_id,))
        job_id = int(row["jobid"]) if row and row.get("jobid") is not None else None
        return job_id or 0

# --- Terminalization & flushing (optional helper) ----------------------------

def finalize_and_flush(job_id: int, sim_id: int, status: str = "finished") -> None:
    """
    Optional helper: remove a simmetjobs row when a job is fully consumed.
    Does NOT write jobexecutionlog; OE core writes 'written'/'failed'.
    """
    with cx() as conn:
        execute(conn, "DELETE FROM public.simmetjobs WHERE jobid=%s", (job_id,))