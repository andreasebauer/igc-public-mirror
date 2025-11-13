"""
OE (Orchestrator) — Pass 1 & Pass 2 (Swift parity, simplified)

- seed_jobs(sim_id): uses Ledger to create SimMetricJobs from job_seed_view
  with jobtype="auto", jobsubtype="", priority=0.

- finalize_seeded_jobs(sim_id): resolves output paths and freezes identity:
  - find unresolved jobs (output_path NULL/empty) from full_job_ledger_extended_view
  - decide extension: steps 1/2 -> .npy ; step 3 -> first of metric_output_types (default .csv)
  - render canonical path via internal flat builder:
	      /data/simulations/{sim_label}/{timestamp}/Frame_{NNNN}/
  - write final path + identity to SimMetricJobs via ledger.update_seeded_job(...)
This file keeps OE stateless and CPU-only; path logic is DB-driven via PathRegistry.
"""

from typing import Dict, List, Optional
import os
import psycopg

from igc import ledger

# ---- DB helper (env-driven) ----
from igc.db.pg import cx, execute
def _connect():
    dsn = os.getenv("PGDSN")
    if dsn:
        return psycopg.connect(dsn)
    return psycopg.connect()  # reads PG* vars (PGHOST, PGUSER, PGPASSWORD, PGDATABASE, PGPORT)

# -------------------
# Public OE functions
# -------------------

def seed_compute_jobs(*, sim_id: int) -> int:
    """
    SIM compute seeding: create frame=0 compute job (output_type='state').
    """
    return ledger.create_compute_template_for_sim(sim_id=sim_id)

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

    # --- estimate per-field memory/size from simulation grid (double precision) ---
    try:
        with _connect() as _c, _c.cursor() as _cur:
            _cur.execute("SELECT COALESCE(gridx,0), COALESCE(gridy,0), COALESCE(gridz,0) FROM public.simulations WHERE id=%s", (sim_id,))
            _row = _cur.fetchone() or (0,0,0)
            _gx, _gy, _gz = (int(_row[0]), int(_row[1]), int(_row[2]))
    except Exception as e:
        # hard-fail: we cannot continue without valid grid dims
        request_abort("grid dimension fetch failed")
        raise
    _elems = max(0, _gx * _gy * _gz)
    _bytes_per_field = _elems * 8
    _mem_grid_mb = _bytes_per_field / (1024.0 * 1024.0)
    _mem_pipeline_mb = 0.0
    _mem_total_mb = _mem_grid_mb + _mem_pipeline_mb    
    finalized = 0

    # set a stable run token once per finalize pass for this sim
    if sim_id not in _RUN_TOKEN:
        _RUN_TOKEN[sim_id] = _time_token_utc()    

    for j in unresolved:
        job_id = int(j["job_id"])
        # no steps as jobs: group comes from metric’s group (if present), step=0
        group_id = int(j.get("group_id") or j.get("met_group_id") or 0)
        step_id  = 0
        phase    = int(j.get("job_phase") or 0)
        frame    = int(j.get("job_frame") or 0)
        step     = 0
        metric_output_types = j.get("metric_outputtypes", "")  # from big_view (met_outputtypes), may be ''
        ext = _ext_for(step, metric_output_types)

        # Render canonical path from templates
        final_path = _render_path_from_registry(j, ext)
        # Ensure output hints & estimates are present on the job row
        try:
            _out_type = j.get("output_type") or None
            with _connect() as _c, _c.cursor() as _cur:
                _cur.execute("""
                    UPDATE public.simmetjobs
                    SET output_extension = %s,
                        output_type      = COALESCE(output_type, %s),
                        mime_type        = %s,
                        output_size_bytes= COALESCE(output_size_bytes, %s),
                        mem_grid_mb      = COALESCE(mem_grid_mb, %s),
                        mem_pipeline_mb  = COALESCE(mem_pipeline_mb, %s),
                        mem_total_mb     = COALESCE(mem_total_mb, %s)
                    WHERE jobid = %s
                """, (
                    ext,
                    _out_type,
                    "application/octet-stream",
                    _bytes_per_field,
                    _mem_grid_mb,
                    _mem_pipeline_mb,
                    _mem_total_mb,
                    job_id,
                ))
                _c.commit()
        except Exception:
            pass        

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

def _safe(s: Optional[str], fallback: str) -> str:
    return s if (isinstance(s, str) and len(s) > 0) else fallback
# --- OE Viewer in-memory event log (per-process, per-sim) ----------------------
from collections import defaultdict, deque

import signal, threading

# ---- Process-wide abort machinery -------------------------------------------
ABORT = threading.Event()

def request_abort(reason: str = "") -> None:
    """Set abort flag; callers should notice and stop ASAP."""
    ABORT.set()
    if reason:
        print(f"[OE] ⚠ abort requested: {reason}")

def _install_signal_handlers():
    def _h(sig, _frame):
        request_abort(f"signal {sig}")
    try:
        signal.signal(signal.SIGINT, _h)
        signal.signal(signal.SIGTERM, _h)
    except Exception:
        # Not fatal (e.g., environment disallows handlers)
        pass

# { sim_id: deque([{"seq":int,"ts":iso,"status":str,"frame":int,"jobid":int,"path":str,"filename":str,"ms":int}, ...]) }
_VIEW = defaultdict(lambda: deque(maxlen=20000))
_SEQ  = defaultdict(int)

def _vlog(sim_id: int, *, status: str, frame: int | None = None, jobid: int | None = None,
          path: str | None = None, filename: str | None = None, ms: int | None = None) -> None:
    _SEQ[sim_id] += 1
    _VIEW[sim_id].append({
        "seq": _SEQ[sim_id],
        "ts":  _time_token_utc(),  # simple readable tick
        "status": status,
        "frame": frame,
        "jobid": jobid,
        "path": path or "",
        "filename": filename or "",
        "ms": ms or 0,
    })

def get_viewer_events(sim_id: int, after_seq: int) -> list[dict]:
    """Return list of viewer events with seq > after_seq (ascending)."""
    if sim_id not in _VIEW: return []
    return [e for e in _VIEW[sim_id] if int(e.get("seq", 0)) > int(after_seq or 0)]

# Stable per-sim run token (timestamp), set once per finalize pass
_RUN_TOKEN: Dict[int, str] = {}

def _render_path_from_registry(j: Dict, ext: str) -> str:
    """
    Simplified flat layout for simulation output paths.
    
    New structure (ignores PathRegistry templates):
        /data/simulations/{sim_label}/{timestamp}/sim_meta.json          ← sim metadata (written elsewhere)
        /data/simulations/{sim_label}/{timestamp}/Frame_{NNNN}/          ← compute bundles
            psi.npy, pi.npy, eta.npy, phi_field.npy, frame_info.json
        /data/simulations/{sim_label}/{timestamp}/Frame_{NNNN}/{metric}.{ext}
            ← for metrics or exports (flat, no Group/Metric/Step/Field)
    
    - Compute jobs (output_type == "state") return the frame directory itself.
    - Metric/export jobs return a full file path within the frame directory.
    """

    import os

    # --- Fixed storage root on 16TB disk ---
    root = "/data/simulations"
    if not os.path.isabs(root):
        root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)

    # --- Resolve tokens from job record ---
    sim_id = int(j["sim_id"])
    sim_label = _safe(j.get("sim_label") or j.get("simlabel"), "").strip()
    if not sim_label:
        # DB fallback: use simulations.label if present
        try:
            with _connect() as _c, _c.cursor() as _cur:
                _cur.execute("SELECT label FROM public.simulations WHERE id=%s", (sim_id,))
                row = _cur.fetchone()
                if row and row[0]:
                    sim_label = str(row[0]).strip()
        except Exception as e:
            request_abort("failed to fetch simulation label")
            raise
    sim_label = _safe(sim_label, f"Sim_{sim_id}")
    frame = int(j.get("job_frame") or 0)
    metric_name = _safe(
        j.get("metric_name") or j.get("metricname") or j.get("output_type"),
        str(j.get("metric_id")),
    )
    type_token = ext[1:] if ext.startswith(".") else ext  # e.g. 'npy', 'csv', etc.

    # --- Timestamped simulation root ---
    tt = _RUN_TOKEN.setdefault(sim_id, _time_token_utc())
    sim_root = os.path.join(root, sim_label, tt)

    # --- Frame directory ---
    frame_dir = os.path.join(sim_root, f"Frame_{frame:04d}")
    os.makedirs(frame_dir, exist_ok=True)

    # --- Decide what to return ---
    out_type = (j.get("output_type") or "").lower()
    if out_type == "state":
        # Compute jobs: return the directory; saver writes the npy/json bundle here.
        return frame_dir

    # Metric or export jobs: single file in the frame directory
    filename = f"{metric_name}.{type_token}".lstrip("/")
    return os.path.join(frame_dir, filename)

# ------------------------------
# Stubs to keep imports satisfied
# ------------------------------

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

def maintain_frame_window(*, sim_id: int, window: int, max_frame: int, template_frame: int = 0) -> int:
    """
    Keep frames [maxCompleted+1 .. min(maxCompleted+window, max_frame)] present in SimMetricJobs.
    Uses ledger.fetch_frame_stats + ledger.insert_jobs_for_frames_like_frame.
    Finalizes newly inserted rows with finalize_seeded_jobs(sim_id) when needed.
    Returns number of inserted rows.
    """
    assert window > 0, "window must be > 0"
    stats = ledger.fetch_frame_stats(sim_id=sim_id)
    mc = max(0, stats.get("maxCompleted", 0))
    ms = max(-1, stats.get("maxSeeded", -1))
    target_max = min(mc + window, max_frame)

    if ms < target_max:
        start = max(ms + 1, 0)
        inserted = ledger.insert_jobs_for_frames_like_frame(
            sim_id=sim_id, start_frame=start, end_frame=target_max, template_frame=template_frame
        )
        if inserted > 0:
            finalize_seeded_jobs(sim_id=sim_id)
        return inserted
    return 0


def seed_frames_all_at_once(*, sim_id: int, end_frame: int, template_frame: int = 0) -> int:
    """
    Convenience: clone template_frame rows into [0..end_frame], then finalize paths.
    Returns number of inserted rows.
    """
    inserted = ledger.insert_jobs_for_frames_like_frame(
        sim_id=sim_id, start_frame=0, end_frame=end_frame, template_frame=template_frame
    )
    if inserted > 0:
        finalize_seeded_jobs(sim_id=sim_id)
    return inserted

def run(*, sim_id: int) -> None:
    """
    Sequential job executor (v1, no real compute yet).

    Walks all jobs for sim_id ordered by frame, pp stage, step_id.
    For each queued job:
      - mark running
      - simulate runner + writer (placeholder)
      - mark written
      - log execution time
    Stops immediately if any job fails.
    """

    from time import perf_counter
    print(f"[OE] ▶ run(sim_id={sim_id}) starting")
    # ensure one stable run token per sim run
    _RUN_TOKEN.setdefault(sim_id, _time_token_utc())
    print(f"[OE] ▶ run token tt={_RUN_TOKEN[sim_id]}")
    # fresh run: clear abort and install signal handlers
    ABORT.clear()
    _install_signal_handlers()
    # Get all jobs for this simulation
    jobs = ledger.fetch_job_ledger_record(sim_id=sim_id)
    # sort by frame, then job_phase (pp stage), then step_id
    jobs.sort(key=lambda j: (int(j.get("job_frame", 0)), int(j.get("job_phase", 0)), int(j.get("step_id", 0))))

    total = len(jobs)
    processed = 0
    start_all = perf_counter()

    for j in jobs:
        # stop immediately if abort was requested (signal or prior error)
        if ABORT.is_set():
            print("[OE] ⛔ abort flag set before next job; stopping.")
            break        
        job_id = int(j["job_id"])
        status = (j.get("job_status") or "").lower()

        if status not in ("queued", "created"):
            continue  # skip already processed or non-runnable

        try:
            ledger.update_job_status_single(job_id=job_id, to_status="running", set_start=True)
            # viewer log: running           
            frame = int(j.get("job_frame", 0))
            step  = int(j.get("step_id", 0))
            print(f"[OE] ▶ running job {job_id} (frame={frame}, step={step})")

            t0 = perf_counter()
            out_type = (j.get("output_type") or "").lower()
            if out_type == "state":
                from pathlib import Path
                import json, numpy as np
                from igc.ledger.sim import get_simulation_full
                from igc.sim.grid_constructor import GridConfig, build_humming_grid
                from igc.sim.coupler import CouplerConfig, Coupler
                from igc.sim.injector import Injector
                from igc.sim.integrator import IntegratorConfig, Integrator

                sim = get_simulation_full(int(j["sim_id"])) or {}
                Nx = int(sim.get("gridx") or 0); Ny = int(sim.get("gridy") or 0); Nz = int(sim.get("gridz") or 0)
                at_start = int(j.get("job_frame") or 0)
                # build initial state
                cfg = GridConfig(shape=(Nx,Ny,Nz), periodic=True)
                state = build_humming_grid(cfg, substeps_per_at=int(sim.get("substeps_per_at") or 48),
                                           initial_at=at_start)

                integ = Integrator(
                    Coupler(CouplerConfig()),
                    Injector([]),
                    IntegratorConfig(
                        substeps_per_at=int(sim.get("substeps_per_at") or 48),
                        dt_per_at=float(sim.get("dt_per_at") or 1.0),
                        stride_frames_at=1,
                    ),
                )

                fdir = Path(str(j.get("output_path") or "")).resolve()
                fdir.mkdir(parents=True, exist_ok=True)

                # run a single At and save arrays (mirrors saver)
                substeps = integ.cfg.substeps_per_at
                dt = integ.cfg.dt_per_at / substeps
                psi, pi, eta, phi_field = state.psi, state.pi, state.eta, state.phi_field
                for _sub in range(substeps):
                    K = {'lambda_eta': integ.coupler.cfg.lambda_eta,
                         'C_pi_to_eta': integ.coupler.cfg.C_pi_to_eta,
                         'lambda_phi': integ.coupler.cfg.lambda_phi,
                         'C_eta_to_phi': integ.coupler.cfg.C_eta_to_phi}
                    pi  += dt * (-psi)
                    psi += dt *  pi
                    eta += dt * (-K["lambda_eta"] * eta + K["C_pi_to_eta"] * np.abs(pi))
                    phi_field += dt * (-K["lambda_phi"] * phi_field + K["C_eta_to_phi"] * np.abs(eta))
                    np.maximum(phi_field, 0.0, out=phi_field)

                def _save(name, arr):
                    p = fdir / f"{name}.npy"; np.save(str(p), arr); return str(p)
                files = {"psi": _save("psi", psi), "pi": _save("pi", pi),
                         "eta": _save("eta", eta), "phi_field": _save("phi_field", phi_field)}
                info = {"sim_id": int(j["sim_id"]), "frame": at_start,
                        "substeps_per_at": substeps, "files": {k: {"path": v} for k,v in files.items()}}
                (fdir / "frame_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
                # --- viewer events (in-memory only; not jobexecutionlog) ---
                _vlog(int(j["sim_id"]), status="frame_start", frame=at_start, jobid=job_id, path=str(fdir))
                for _name, _p in files.items():
                    _vlog(int(j["sim_id"]), status="file_written", frame=at_start, jobid=job_id,
                          path=str(fdir), filename=f"{_name}.npy")
                _vlog(int(j["sim_id"]), status="file_written", frame=at_start, jobid=job_id,
                      path=str(fdir), filename="frame_info.json")                
                
            else:
                import time; time.sleep(0.01)

            duration_ms = int((perf_counter() - t0) * 1000)

            # write + log
            ledger.log_execution(job=j, runtime_ms=duration_ms, queue_wait_ms=0)
            ledger.update_job_status_single(job_id=job_id, to_status="written", set_finish=True)
            _vlog(int(j["sim_id"]), status="frame_done",
                  frame=int(j.get("job_frame", 0)), jobid=job_id, ms=duration_ms)            
            # viewer log: written
            try:
                with _connect() as conn, conn.cursor() as cur:
                    cur.execute(
                        "INSERT INTO public.jobexecutionlog (jobid, simid, status) VALUES (%s, %s, %s)",
                        (job_id, int(j["sim_id"]), "written"),
                    )
                    conn.commit()
            except Exception:
                pass            
            processed += 1
        except Exception as e:
            # mark failed, flip abort, and raise to surface failure to caller
            try:
                ledger.log_error(job=j, message=str(e))
            except Exception:
                pass
            try:
                ledger.update_job_status_single(job_id=job_id, to_status="failed", set_finish=True)
            except Exception:
                pass
            request_abort(f"job {job_id} failed")
            print(f"[OE] ✖ job {job_id} failed: {e}")
            raise

    total_ms = int((perf_counter() - start_all) * 1000)
    if ABORT.is_set():
        print(f"[OE] ❌ run aborted after {processed}/{total} jobs in {total_ms} ms")
        # raise if we reached here due to external abort (no exception thrown in loop)
        raise RuntimeError("OE aborted")
    print(f"[OE] ✅ run complete: {processed}/{total} jobs processed in {total_ms} ms")

    # emit a run_complete viewer event so the GUI can show a final message
    try:
        sim_for_vlog = int(jobs[-1].get("sim_id")) if jobs else None
        if sim_for_vlog is not None:
            _vlog(sim_for_vlog, status="run_complete", frame=None, jobid=None, ms=total_ms)
    except Exception:
        pass

# --- GUI-facing seeding helper (idempotent) ---
def seed_metric_jobs(sim_id: int, metric_ids: list[int], frames: list[int], phases: list[int]) -> int:
    if not metric_ids or not frames or not phases:
        return 0
    tuples = [(sim_id, mid, fr, ph) for mid in metric_ids for fr in frames for ph in phases]
    sql = """
    INSERT INTO public.simmetjobs(simid, metricid, frame, phase, status, createdate)
    SELECT %s, %s, %s, %s, 'queued', now()
    WHERE NOT EXISTS (
      SELECT 1 FROM public.simmetjobs j
      WHERE j.simid=%s AND j.metricid=%s AND j.frame=%s AND j.phase=%s
    )
    """
    inserted = 0
    with _connect() as conn, conn.cursor() as cur:
        for (sid, mid, fr, ph) in tuples:
            cur.execute(sql, (sid, mid, fr, ph, sid, mid, fr, ph))
            if cur.rowcount == 1:
                inserted += 1
        conn.commit()
    return inserted
