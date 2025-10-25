"""
OE (Orchestrator) — Pass 1 & Pass 2 (Swift parity, simplified)

- seed_jobs(sim_id): uses Ledger to create SimMetricJobs from job_seed_view
  with jobtype="auto", jobsubtype="", priority=0.

- finalize_seeded_jobs(sim_id): resolves output paths and freezes identity:
  - find unresolved jobs (output_path NULL/empty) from full_job_ledger_extended_view
  - decide extension: steps 1/2 -> .npy ; step 3 -> first of metric_output_types (default .csv)
  - render canonical path from DB PathRegistry (id=1) templates:
      time/sim/frame/group/metric/step/field/( metricName.type | intermediate.npy )
  - write final path + identity to SimMetricJobs via ledger.update_seeded_job(...)

This file keeps OE stateless and CPU-only; path logic is DB-driven via PathRegistry.
"""

from typing import Dict, List, Optional
import os
import psycopg

from igc import ledger


# ---- DB helper (env-driven) ----
def _connect():
    dsn = os.getenv("PGDSN")
    if dsn:
        return psycopg.connect(dsn)
    return psycopg.connect()  # reads PG* vars (PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGPORT)


# -------------------
# Public OE functions
# -------------------
def seed_jobs(*, sim_id: int) -> None:
    """
    Pass 1 (Swift parity):
      createJobsForSimulation(simID: simID, jobtype: "auto", jobsubtype: "", priority: 0)
    """
    ledger.create_jobs_for_sim(
        sim_id=sim_id,
        jobtype="auto",
        jobsubtype="",
        priority=0,
    )


def finalize_seeded_jobs(*, sim_id: int) -> int:
    """
    Pass 2 (Swift parity):
      - load all jobs for sim_id
      - filter unresolved (output_path empty)
      - compute ext and job_type
      - render canonical path via PathRegistry templates
      - update job row via ledger.update_seeded_job(...)
    Returns number of finalized jobs.
    """
    # Pull everything for this sim from the unified view
    jobs: List[Dict] = ledger.fetch_job_ledger_record(sim_id=sim_id)
    unresolved = [j for j in jobs if not j.get("output_path")]

    # Load templates once
    T = _get_path_templates()
    finalized = 0

    for j in unresolved:
        job_id   = int(j["job_id"])
        group_id = int(j["group_id"])
        step_id  = int(j["step_id"])          # DB step id (1/2/3)
        phase    = int(j.get("job_phase") or 0)            # pp stage numeric
        frame    = int(j.get("job_frame") or 0)
        step     = int(j.get("step", step_id))  # if 'step' absent, reuse step_id
        metric_output_types = j.get("metric_outputtypes", "")
        ext   = _ext_for(step, metric_output_types)

        # Render canonical path from templates
        final_path = _render_path_from_registry(j, ext, T)

        job_type = f"step_{step}"

        ledger.update_seeded_job(
            job_id=job_id,
            group_id=group_id,
            step_id=step_id,
            job_type=job_type,
            job_phase=phase,
            job_frame=frame,
            output_path=final_path,
        )
        finalized += 1

    return finalized


# -------------------------
# Helpers (kept local here)
# -------------------------
def _ext_for(step_num: int, metric_output_types: Optional[str]) -> str:
    # steps 1/2 => .npy (intermediates)
    if step_num in (1, 2):
        return ".npy"
    # step 3 => first of metric_output_types (csv|png|json|npy|txt), default .csv
    if not metric_output_types:
        return ".csv"
    tokens = (
        metric_output_types.strip("{}()[]").replace(" ", "").split(",")
        if isinstance(metric_output_types, str)
        else []
    )
    for t in tokens:
        t = t.lower()
        if t in ("csv", "png", "json", "npy", "txt"):
            return f".{t}"
    return ".csv"


def _time_token_utc() -> str:
    # same as Swift: "yyyyMMdd_HHmm" in UTC
    import datetime as _dt
    return _dt.datetime.utcnow().strftime("%Y%m%d_%H%M")


def _get_path_templates() -> Dict[str, str]:
    """
    Reads PathRegistry (id=1) and returns a dict of templates.
    Expected columns:
      timestamptemplate, frametemplate, simnametemplate, groupnametemplate,
      metricidtemplate, steptemplate, filenametemplate, intermediate, fieldtemplate
    """
    sql = """
        SELECT timestamptemplate, frametemplate, simnametemplate, groupnametemplate,
               metricidtemplate, steptemplate, filenametemplate, intermediate, fieldtemplate
        FROM "PathRegistry"
        WHERE id = 1
    """
    with _connect() as conn, conn.cursor() as cur:
        cur.execute(sql)
        row = cur.fetchone()
        if not row:
            raise RuntimeError("PathRegistry row id=1 not found")
        cols = [d[0] for d in cur.description]
        return {k: v for k, v in zip(cols, row)}


def _safe(s: Optional[str], fallback: str) -> str:
    return s if (isinstance(s, str) and len(s) > 0) else fallback


def _render_path_from_registry(j: Dict, ext: str, templates: Dict[str, str]) -> str:
    """
    Renders the canonical path using PathRegistry templates.
    - For steps 1/2: use 'intermediate' template (e.g., '/intermediate.npy')
    - For step 3: use 'filenametemplate' with {metricName}.{type}
    Directory order (as you specified):
      Time/{timestamp}/Sim/{simLabel}/Frame/{frame}/Group/{groupName}/
      Metric/{metricName}/Step/{step}/Field/{stepID}/<file>
    """
    # Resolve tokens from job view columns with robust fallbacks
    sim_id   = int(j["sim_id"])
    group_id = int(j["group_id"])
    frame    = int(j.get("job_frame") or 0)
    step_id  = int(j["step_id"])
    step     = int(j.get("step", step_id))  # keep Swift parity
    sim_label    = _safe(j.get("sim_label")    or j.get("simlabel"),    str(sim_id))
    group_name   = _safe(j.get("group_name")   or j.get("groupname"),   str(group_id))
    metric_name  = _safe(j.get("metric_name")  or j.get("metricname"),  str(j.get("metric_id")))
    type_token   = ext[1:] if ext.startswith(".") else ext  # 'csv', 'npy', ...

    tt   = _time_token_utc()
    T    = templates

    # base root for store (env or default)
    root = os.environ.get("IGC_STORE", "/data/igc")

    # Compose directories from the registry templates
    parts = [
        root,
        T["timestamptemplate"].format(timestamp=tt).lstrip("/"),
        T["simnametemplate"].format(simLabel=sim_label).lstrip("/"),
        T["frametemplate"].format(frame=frame).lstrip("/"),
        T["groupnametemplate"].format(groupName=group_name).lstrip("/"),
        T["metricidtemplate"].format(metricName=metric_name).lstrip("/"),
        T["steptemplate"].format(step=step).lstrip("/"),
        T["fieldtemplate"].format(stepID=step_id).lstrip("/"),
    ]

    # filename component depends on step
    if step in (1, 2):
        tail = T["intermediate"].lstrip("/")  # e.g., "intermediate.npy"
    else:
        tail = T["filenametemplate"].format(metricName=metric_name, type=type_token).lstrip("/")

    # Join with "/" carefully (avoid doubles)
    path = "/".join(p.strip("/") for p in parts if p)
    return f"{path}/{tail}"


# ------------------------------
# Stubs to keep imports satisfied
# ------------------------------
def run(*, run_id: int):
    """Execution loop will be implemented next (pp0 -> pp1 -> pp3, sequential, stop-on-error)."""
    raise NotImplementedError


def pause(*, run_id: int) -> None:
    """Cooperative pause at next safe checkpoint (stub)."""
    return None


def resume(*, run_id: int) -> None:
    """Resume a paused run (stub)."""
    return None


def stop(*, run_id: int) -> None:
    """Gracefully stop after current job completes (stub)."""
    return None


def status(*, run_id: int) -> Dict:
    """Return a summary for the run (stub)."""
    return {"run_id": run_id, "state": "unknown"}