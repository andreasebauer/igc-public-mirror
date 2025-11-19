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
from igc.sim.validator import validate_sim_config

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

# Stable per-sim *metric* run token (timestamp), set once per metrics seeding
_METRIC_RUN_TOKEN: Dict[int, str] = {}

# Guard so we only run the sim suite once per sim_id inside this process.
_SIM_RUN_DONE: Dict[int, bool] = {}

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

    # Metric or export jobs: place outputs in a per-metrics-run subfolder under each frame.
    # This allows multiple metric runs on the same sim without overwriting prior outputs.
    metrics_run_token = _METRIC_RUN_TOKEN.get(sim_id) or _time_token_utc()
    _METRIC_RUN_TOKEN[sim_id] = metrics_run_token
    metrics_root = os.path.join(frame_dir, metrics_run_token)
    os.makedirs(metrics_root, exist_ok=True)

    filename = f"{metric_name}.{type_token}".lstrip("/")
    return os.path.join(metrics_root, filename)

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


def _derive_run_kind(kind: str | None, jobs: list[dict]) -> str:
    """
    Decide which kind of run this is: 'sim', 'metrics', or 'both'.

    If an explicit kind is given and recognized, it wins.
    Otherwise, infer from the presence of state and/or metric jobs.
    """
    if kind:
        k = kind.lower()
        if k in ("sim", "metrics", "both"):
            return k

    has_state = any((j.get("output_type") or "").lower() == "state" for j in jobs)
    has_metric = any(j.get("metric_id") is not None for j in jobs)

    if has_state and has_metric:
        return "both"
    if has_metric:
        return "metrics"
    # default: pure sim or no explicit type
    return "sim"


def _job_is_in_scope(j: dict, run_kind: str) -> bool:
    """
    Filter jobs by run_kind:

      - 'sim'     → only non-metric jobs (state / legacy)
      - 'metrics' → only metric jobs
      - 'both'    → all jobs
    """
    is_metric = j.get("metric_id") is not None

    if run_kind == "sim":
        # skip metric jobs; keep state/unknown
        return not is_metric
    if run_kind == "metrics":
        # only metric jobs
        return is_metric
    # 'both' or anything else: process everything
    return True


def _run_sim_suite_for_sim(sim_id: int) -> None:
    """
    Run the full simulation for sim_id using the fused integrator/saver.

    This mirrors igc.gui.create_sim.run_simulation, but:
      - uses the canonical /data/simulations root (or IGC_STORE),
      - writes under {store}/{label}/{tt}/Frame_XXXX,
        where {tt} is OE's _RUN_TOKEN[sim_id].
    """

    import os
    from pathlib import Path

    from igc.ledger.sim import get_simulation_full
    from igc.sim.grid_constructor import GridConfig, build_humming_grid
    from igc.sim.coupler import CouplerConfig, Coupler
    from igc.sim.injector import Injector, InjectionEvent
    from igc.sim.integrator import IntegratorConfig, Integrator

    # 1) Load full Simulations row
    sim = get_simulation_full(sim_id)
    if not sim:
        raise RuntimeError(f"simulation not found for id={sim_id}")

    Nx = int(sim.get("gridx") or 0)
    Ny = int(sim.get("gridy") or 0)
    Nz = int(sim.get("gridz") or 0)
    if Nx <= 0 or Ny <= 0 or Nz <= 0:
        raise RuntimeError(f"invalid grid dims for sim {sim_id}: {Nx}x{Ny}x{Nz}")

    # 2) Build humming grid (true IG vacuum at At=0)
    cfg = GridConfig(shape=(Nx, Ny, Nz))
    substeps_per_at = int(sim.get("substeps_per_at") or 48)
    state = build_humming_grid(cfg, substeps_per_at=substeps_per_at, initial_at=0)

    # Center index for seeding
    cx0 = state.psi.shape[0] // 2
    cy0 = state.psi.shape[1] // 2
    cz0 = state.psi.shape[2] // 2

    # 3) Coupler configuration (D/C/λ + gate)
    coupler_cfg = CouplerConfig(
        D_psi=float(sim.get("d_psi") or 0.0),
        D_eta=float(sim.get("d_eta") or 0.0),
        D_phi=float(sim.get("d_phi") or 0.0),
        C_pi_to_eta=float(sim.get("c_pi_to_eta") or 1.0),
        C_eta_to_phi=float(sim.get("c_eta_to_phi") or 1.0),
        lambda_eta=float(sim.get("lambda_eta") or 1.0),
        lambda_phi=float(sim.get("lambda_phi") or 1.0),
        gate=str(sim.get("gate_name") or "linear"),
    )
    coupler = Coupler(coupler_cfg)

    # 4) Seeding events via Injector
    events: list[InjectionEvent] = []

    seed_type = (str(sim.get("seed_type") or "none")).lower()
    seed_strength = float(sim.get("seed_strength") or 0.0)

    if seed_type != "none" and seed_strength != 0.0:
        seed_field = (str(sim.get("seed_field") or "psi")).lower()
        seed_sigma = float(sim.get("seed_sigma") or 0.0)

        # center: grid center or explicit "x,y,z"
        if (sim.get("seed_center") or "center") == "center":
            center = (cx0, cy0, cz0)
        else:
            center = tuple(map(int, str(sim.get("seed_center")).split(",")))

        window = (
            float(sim.get("seed_phase_a") or 0.25),
            float(sim.get("seed_phase_b") or 0.30),
        )

        first_at = int(sim.get("seed_at") or 0)
        period_at = int(sim.get("seed_repeat_at") or 0)
        if period_at > 0:
            repeat = {"first_at": first_at, "period_at": period_at}
        else:
            repeat = {"first_at": first_at}

        events.append(
            InjectionEvent(
                kind=seed_type,
                field=seed_field,
                amplitude=seed_strength,
                sigma=seed_sigma,
                center=center,
                window=window,
                repeat=repeat,
            )
        )

    injector = Injector(events)

    # 5) Integrator configuration (dt, dx) and At range from seeded frames
    # Derive t_max from the seeded frame window so PDE length always matches
    # what OE/SimMetricJobs planned (0..maxSeeded).
    stats = ledger.fetch_frame_stats(sim_id=sim_id)
    max_seeded = int(stats.get("maxSeeded", 0))
    t_max = max_seeded + 1
    if t_max <= 0:
        raise RuntimeError(f"no seeded frames for sim {sim_id} (maxSeeded={max_seeded})")

    integ_cfg = IntegratorConfig(
        substeps_per_at=substeps_per_at,
        dt_per_at=float(sim.get("dt_per_at") or 1.0),
        dx=float(sim.get("dx") or 1.0),
        stride_frames_at=1,
    )

    # Sanity / stability check
    validate_sim_config(cfg, integ_cfg, coupler_cfg)

    # 6) Resolve store + label + OE run token → /data/simulations/{label}/{tt}/Frame_XXXX
    root = os.environ.get("IGC_STORE", "/data/simulations")
    store = Path(root)
    store.mkdir(parents=True, exist_ok=True)

    base_label = str(sim.get("label") or f"Sim_{sim_id}").strip()
    tt = _RUN_TOKEN.setdefault(sim_id, _time_token_utc())
    sim_label_with_tt = f"{base_label}/{tt}"

    # 7) Run the integrator – writes Frame_0000..Frame_{t_max-1}
    integ = Integrator(coupler, injector, integ_cfg)

    def _on_frame_saved(frame_idx: int, at_val: int) -> None:
        """Emit viewer event for each saved PDE frame."""
        try:
            frame_dir = store / sim_label_with_tt / f"Frame_{frame_idx:04d}"
            _vlog(
                sim_id,
                status="frame_done",
                frame=frame_idx,
                jobid=None,
                path=str(frame_dir),
                ms=0,
            )
        except Exception:
            # logging must never break the integrator
            pass

    integ.run(
        store=store,
        sim_label=sim_label_with_tt,
        psi=state.psi,
        pi=state.pi,
        eta=state.eta,
        phi_field=state.phi_field,
        phi_cone=state.phi_cone,
        at_start=0,
        at_end=t_max,
        save_first_frame=True,
        header_stats=True,
        on_frame_saved=_on_frame_saved,
    )

    # Mark in-process guard so we don't run twice for the same sim_id.
    _SIM_RUN_DONE[sim_id] = True


def run(*, sim_id: int, kind: str | None = None) -> None:
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

    # Clear per-sim viewer events so each OE.run starts with a fresh log.
    # This prevents stale sim/metric runs from previous executions from
    # showing up in the OE viewer for the same sim_id.
    try:
        _VIEW.pop(sim_id, None)
        _SEQ[sim_id] = 0
    except Exception:
        # viewer reset must never break the runner
        pass

    _install_signal_handlers()
    # Get all jobs for this simulation
    jobs = ledger.fetch_job_ledger_record(sim_id=sim_id)
    # sort by frame, then job_phase (pp stage), then step_id
    jobs.sort(key=lambda j: (int(j.get("job_frame", 0)), int(j.get("job_phase", 0)), int(j.get("step_id", 0))))

    # Decide what kind of run this is (sim / metrics / both)
    run_kind = _derive_run_kind(kind, jobs)
    print(f"[OE] ▶ inferred run_kind={run_kind}")

    # viewer event: run starting (sim / metrics / both)
    try:
        start_status = {
            "sim": "run_sim_start",
            "metrics": "run_metrics_start",
            "both": "run_both_start",
        }.get(run_kind, "run_sim_start")
        _vlog(sim_id, status=start_status, frame=None, jobid=None)
    except Exception:
        # logging must not interfere with the runner
        pass

    # Preload metric pipelines for this run (only metrics present in the jobs)
    from igc.metrics import runner as metrics_runner  # local import to avoid OE import churn
    metric_ids = sorted(
        {
            int(j.get("metric_id"))
            for j in jobs
            if j.get("metric_id") is not None
        }
    )
    metric_pipelines = metrics_runner.load_metric_pipelines(metric_ids)

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

        # Skip jobs that are not part of this run kind (sim / metrics / both)
        if not _job_is_in_scope(j, run_kind):
            continue

        try:
            ledger.update_job_status_single(job_id=job_id, to_status="running", set_start=True)
            # viewer log: running           
            frame = int(j.get("job_frame", 0))
            step  = int(j.get("step_id", 0))
            print(f"[OE] ▶ running job {job_id} (frame={frame}, step={step})")

            t0 = perf_counter()
            out_type = (j.get("output_type") or "").lower()
            metric_id = j.get("metric_id")

            if out_type == "state":
                # ------------------------------------------------------------
                # STATE JOBS (SIMULATION FRAMES)
                # ------------------------------------------------------------
                # We never run PDE per job anymore. Instead:
                #   - The first frame-0 state job triggers the sim suite once
                #     for this sim_id (0..t_max), writing frames into
                #     /data/simulations/{label}/{tt}/Frame_XXXX.
                #   - All state jobs then become pure bookkeeping: status and
                #     logging are handled by the generic block below.
                # ------------------------------------------------------------
                simid = int(j["sim_id"])
                frame0 = int(j.get("job_frame") or 0)

                if (
                    frame0 == 0
                    and not _SIM_RUN_DONE.get(simid, False)
                    and run_kind in ("sim", "both")
                ):
                    print(f"[OE] ▶ starting sim suite for sim_id={simid} (job {job_id}, frame=0)")
                    _run_sim_suite_for_sim(simid)
                    print(f"[OE] ▶ sim suite completed for sim_id={simid}")
                # No per-job PDE here; generic logging below will mark this job written.

            else:
                # Metric job or unknown job type
                if metric_id:
                    # Metric job: delegate to metrics runner (pipelines preloaded above)
                    metric_name = (
                        j.get("metric_name")
                        or j.get("metricname")
                        or j.get("output_type")
                        or f"metric_{metric_id}"
                    )
                    _vlog(
                        int(j["sim_id"]),
                        status="metric_start",
                        frame=frame,
                        jobid=job_id,
                        filename=str(metric_name),
                    )
                    metrics_runner.execute_metric_job(j, metric_pipelines)
                    _vlog(
                        int(j["sim_id"]),
                        status="metric_done",
                        frame=frame,
                        jobid=job_id,
                        filename=str(metric_name),
                    )
                else:
                    # Legacy / unknown job type: do nothing but avoid crashing
                    import time
                    time.sleep(0.01)

            duration_ms = int((perf_counter() - t0) * 1000)

            # write + log
            ledger.log_execution(job=j, runtime_ms=duration_ms, queue_wait_ms=0)
            ledger.update_job_status_single(job_id=job_id, to_status="written", set_finish=True)

            # Only emit frame_done for metric jobs.
            # State-job frame events come from the integrator callback (on_frame_saved).
            if (j.get("output_type") or "").lower() != "state":
                _vlog(
                    int(j["sim_id"]),
                    status="frame_done",
                    frame=int(j.get("job_frame", 0)),
                    jobid=job_id,
                    ms=duration_ms,
                )
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
            try:
                # viewer log: error
                _vlog(
                    int(j.get("sim_id", 0) or 0),
                    status="error",
                    frame=int(j.get("job_frame", 0) or 0),
                    jobid=job_id,
                    path=str(e),
                )
            except Exception:
                # logging must not crash the runner
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
            try:
                complete_status = {
                    "sim": "run_sim_complete",
                    "metrics": "run_metrics_complete",
                    "both": "run_both_complete",
                }.get(run_kind, "run_complete")
                _vlog(
                    sim_for_vlog,
                    status=complete_status,
                    frame=None,
                    jobid=None,
                    ms=total_ms,
                )
            except Exception:
                # logging must not crash the runner
                pass
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
