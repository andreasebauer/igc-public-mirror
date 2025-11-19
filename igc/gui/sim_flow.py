from __future__ import annotations
from igc.ledger.sim import list_simulations, update_simulation, create_simulation, get_simulation_full, get_simulation_columns, get_simulation_full, get_simulation_columns, get_simulation_full, get_simulation_columns

from .vars import VARS
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from fastapi import APIRouter, Request, Form, Query, HTTPException
from fastapi.responses import RedirectResponse
from fastapi import BackgroundTasks
from igc.oe import core as oe_core
from igc.ledger.core import _connect
from fastapi.templating import Jinja2Templates
# minimal row→dict helpers (local; we only import _connect above)
def _rowdict(cur, row):
    if row is None:
        return None
    cols = [d[0] for d in cur.description]
    return {k: v for k, v in zip(cols, row)}

def fetchone_dict(conn, sql: str, params: dict | None = None):
    with conn.cursor() as cur:
        cur.execute(sql, params or {})
        return _rowdict(cur, cur.fetchone())

def fetchall_dict(conn, sql: str, params: dict | None = None):
    with conn.cursor() as cur:
        cur.execute(sql, params or {})
        cols = [d[0] for d in cur.description]
        rows = cur.fetchall() or []
        return [{k: v for k, v in zip(cols, r)} for r in rows]
from igc.ledger import sim as sims_svc
from igc.gui.services import jobs as jobs_svc
from igc.gui.services import forms as forms_svc

router = APIRouter(prefix="/sims", tags=["sims"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

# --- OE viewer: in-memory sim log buffer (notes) -------------------------------
# We keep short, non-durable, sim-level breadcrumbs here so the viewer can show
# phase/milestone messages even when no jobs exist yet (jobid is NOT NULL in DB).
def _simlog_append(app, sim_id: int, text: str) -> None:
    try:
        st = getattr(app, "state")
        if not hasattr(st, "_oe_log"):
            st._oe_log = {}
        buf = st._oe_log.setdefault(int(sim_id), [])
        buf.append({
            "logid": None,            # synthetic; DB tail uses ints
            "recorded_at": None,      # UI can render as "now" if None
            "status": "note",         # distinguish from job statuses
            "jobid": 0,               # synthetic (DB requires NOT NULL jobid)
            "simid": int(sim_id),
            "metricid": None, "stepid": None, "frame": None,
            "message": str(text),
            "output_path": ""
        })
        if len(buf) > 500:
            del buf[: len(buf) - 500]
    except Exception:
        pass

# ---------- 1) sim_start -----------------------------------------------------

@router.get("/start")
def sim_start(request: Request):
    return templates.TemplateResponse("sim_start.html", {"request": request})

# ---------- 2) sim_select (mode: run|edit|sweep) -----------------------------

@router.get("/select")
def sim_select(request: Request,
               mode: str = Query(..., regex="^(run|edit|sweep)$"),
               q: Optional[str]=Query(None),
               page: int = 1):
    # FastAPI injects "mode" from the query (?mode=run|edit|sweep); default via signature if present
    sims = list_simulations(limit=500)

    return templates.TemplateResponse(
        "sim_select.html",
        {
            "request": request,
            "mode": mode,
            "sims": sims,
        },
    )



def sim_select_post(mode: str = Form(...), sim_id: int = Form(...)):
    if mode == "run":
        url = f"/sims/edit?mode=run&sim_id={sim_id}"
    elif mode == "edit":
        url = f"/sims/edit?mode=edit&sim_id={sim_id}"
    else:  # sweep
        url = f"/sims/sweep?sim_id={sim_id}"
    return RedirectResponse(url=url, status_code=303)

# ---------- 3) sim_edit (mode: create|edit|run) ------------------------------

@router.get("/edit")
def sim_edit_get(request: Request,
                 mode: str = Query(..., regex="^(create|edit|run)$"),
                 sim_id: Optional[int] = Query(None)):
    # fetch full row if sim_id is present
    sim = get_simulation_full(sim_id) if sim_id is not None else None

    # debug: prove whether defaults arrive from DB
    log = logging.getLogger("uvicorn.error")
    if sim_id and sim:
        log.info(
            "sim_edit_get: id=%s default_gridx=%r default_stride=%r default_t_max=%r keys=%d",
            sim_id,
            sim.get("default_gridx"),
            sim.get("default_stride"),
            sim.get("default_t_max"),
            len(sim.keys()),
        )
    else:
        log.info("sim_edit_get: id=%s sim=%r", sim_id, sim)

    # overlay transient edits from /edit/apply
    try:
        overrides = getattr(request.app.state, "_pending_overrides", {}).get(sim_id, {})
    except Exception:
        overrides = {}
    if sim and overrides:
        sim = {**sim, **overrides}

    # try to fetch column metadata; if missing, fallback to keys from sim
    fields = get_simulation_columns()
    if (not fields) and sim:
        fields = [{"name": k, "data_type": "text", "description": ""} for k in sim.keys()]

    return templates.TemplateResponse(
        "sim_edit.html",
        {
            "header_right": (
                ("Run " if mode == "run" else ("Edit " if mode == "edit" else "")) +
                ((sim.get("label","") + " " + sim.get("name","")) if sim else "")
            ).strip(),
            "request": request,
            "mode": mode,
            "sim": sim,
            "fields": fields,
        },
    )

def sim_sweep_get(request: Request, sim_id: int = Query(...)):
    row = sims_svc.get_simulation(sim_id)
    if not row:
        raise HTTPException(404, "simulation not found")
    return templates.TemplateResponse("sim_sweep.html", {
        "request": request, "sim_id": sim_id, "base": row
    })

@router.post("/sweep")
async def sim_sweep_post(request: Request, sim_id: int = Form(...)):
    # Parse ranges; store transiently for confirm
    data = dict(await request.form())
    request.app.state._sweep_plan = {sim_id: data}
    return RedirectResponse(url=f"/sims/confirm?mode=sweep&sim_id={sim_id}", status_code=303)

# ---------- 5) sim_confirmation ---------------------------------------------

from fastapi import Query
from fastapi.responses import RedirectResponse

@router.get("/confirm")
def sim_confirm_get(
    request: Request,
    mode: str | None = Query(default=None, regex="^(create|edit|run|sweep)$"),
    sim_id: int | None = Query(default=None),
):
    # Gracefully handle bare /sims/confirm without required query params
    if mode is None or sim_id is None:
        return RedirectResponse(url="/sims/start", status_code=303)

    sim = get_simulation_full(sim_id) if mode != "create" else None
    fields = get_simulation_columns()

    # overlay transient edits captured by /sims/edit/apply
    try:
        overrides = getattr(request.app.state, "_pending_overrides", {}).get(sim_id, {})
    except Exception:
        overrides = {}
    if sim and overrides:
        sim = {**sim, **overrides}
    overrides = {}
    sweep = None
    if mode == "run":
        overrides = getattr(request.app.state, "_pending_overrides", {}).get(sim_id, {})
    elif mode == "sweep":
        sweep = getattr(request.app.state, "_sweep_plan", {}).get(sim_id, {})

    return templates.TemplateResponse(
        "sim_confirm.html",
        {
            "request": request,
            "mode": mode,
            "sim_id": sim_id,
            "header_right": "Confirmation",
            "overrides": overrides,
            "sweep": sweep,
            "fields": fields,
            "sim": sim,
        },
    )

@router.post("/confirm")
def sim_confirm_post(request: Request,
                     mode: str = Form(...),
                     sim_id: int = Form(...),
                     background_tasks: BackgroundTasks = None,
                     bundle_level: Optional[str] = Form(None),
                     is_visualization: Optional[bool] = Form(False),
                     precision: Optional[str] = Form(None)):
    # viewer note: pipeline has been initiated from sim_confirm
    _simlog_append(
        request.app,
        sim_id,
        "We are now preparing run data. Please wait for your simulation to start..."
    )                     
    if mode in {"create","edit"}:
        st = getattr(request.app, "state")
        pending = getattr(st, "_pending_overrides", {}).pop(sim_id, {})
        # persist to DB
        if mode == "edit":
            from igc.ledger.sim import update_simulation
            rc = update_simulation(sim_id, pending)
            print(f"[save/edit] sim_id={sim_id} keys={len(pending)} rowcount={rc}")
            # also clear any run_overrides for this sim_id
            try: getattr(st, "_run_overrides", {}).pop(sim_id, None)
            except Exception: pass
            return RedirectResponse(url="/sims/start?msg=Simulation+saved", status_code=303)
        else:
            from igc.ledger.sim import create_simulation
            new_id = create_simulation(pending)
            print(f"[save/create] new_id={new_id} keys={len(pending)}")
            return RedirectResponse(url="/sims/start?msg=Simulation+saved", status_code=303)

    if mode == "run":
        base = sims_svc.get_simulation_full(sim_id)
        if not base:
            raise HTTPException(404, "simulation not found")
        # Fresh run: assign new run token and clear any old jobs for this sim
        try:
            oe_core._RUN_TOKEN[sim_id] = oe_core._time_token_utc()
        except Exception:
            pass

        overrides = getattr(request.app.state, "_pending_overrides", {}).get(sim_id, {})
        # Persist only run-mode tunables into simulations so OE reads current values from DB
        # These are the fields that are editable in sim_edit (run mode) and should
        # affect this run: grid, cadence, gate & diffusion parameters.
        RUN_TUNABLE_KEYS = {
            "gridx", "gridy", "gridz",
            "phi_threshold",
            "t_max", "stride",
            "d_psi", "d_eta", "d_phi",
            "lambda_phi",
            "gate_name",
        }
        run_overrides = {k: v for k, v in (overrides or {}).items() if k in RUN_TUNABLE_KEYS}
        if run_overrides:
            try:
                from igc.ledger.sim import update_simulation
                rc = update_simulation(sim_id, run_overrides)
                _simlog_append(request.app, sim_id, f"overrides_persisted: {rc} fields")
                # refresh base row after persistence
                base = sims_svc.get_simulation_full(sim_id)
                # once persisted, clear pending overrides for this sim
                try:
                    getattr(request.app.state, "_pending_overrides", {}).pop(sim_id, None)
                except Exception:
                    pass
                # overrides are now in DB; treat as empty for hashing
                overrides = {}
            except Exception as e:
                _simlog_append(request.app, sim_id, f"overrides_persist_failed: {type(e).__name__}")

        base_hash, over_hash, eff_hash, effective = jobs_svc.compute_effective_hash(base, {})
        frame0 = jobs_svc.plan_frame0_dir(base)

        # --- always create sim_meta.json before any seeding / running ---
        try:
            from pathlib import Path as _Path
            import json as _json
            sim_label = (base.get("label") or f"Sim_{sim_id}").strip()
            tt = oe_core._RUN_TOKEN.setdefault(sim_id, oe_core._time_token_utc())
            run_root = _Path("/data/simulations") / sim_label / tt
            run_root.mkdir(parents=True, exist_ok=True)
            meta_path = run_root / "sim_meta.json"
            meta = dict(base)
            # Ensure sim_id is always present for simpicker and downstream tools
            if "sim_id" not in meta:
                meta["sim_id"] = sim_id
            meta_path.write_text(_json.dumps(meta, default=str, indent=2), encoding="utf-8")
            _simlog_append(request.app, sim_id, f"meta_written: {meta_path}")
        except Exception as e:
            _simlog_append(
                request.app,
                sim_id,
                f"meta_write_failed: {type(e).__name__}: {e}",
            )

        if background_tasks is not None:
            end_frame = int((base or {}).get("t_max") or 0)

            # seed frame-0 compute job (bundle of ψ/π/η/φ)
            oe_core.seed_compute_jobs(sim_id=sim_id)
            # (start-of-run message is already logged at sim_confirm_post entry)

            background_tasks.add_task(
                oe_core.seed_frames_all_at_once,
                sim_id=sim_id,
                end_frame=end_frame,
                template_frame=0,
            )
            _simlog_append(
                request.app,
                sim_id,
                f"Preparation done. Seeding frames 0..{end_frame} from template 0..."
            )

            background_tasks.add_task(
                oe_core.finalize_seeded_jobs,
                sim_id=sim_id,
            )
            _simlog_append(
                request.app,
                sim_id,
                "Finalize scheduled: resolving output paths for seeded jobs."
            )

            # start the simple sequential runner
            background_tasks.add_task(oe_core.run, sim_id=sim_id, kind="sim")
            _simlog_append(
                request.app,
                sim_id,
                "Runner scheduled: simulation data creation is now being started..."
            )
                                                      
        # Keep your existing enqueue to actually run the computation
        job_id = jobs_svc.enqueue_sim_job(sim_id, eff_hash, bundle_level, is_visualization, precision)

        # Redirect user to OE Viewer (live progress/gauges/logs by sim_id)
        _simlog_append(request.app, sim_id, "pipeline_redirect: opening OE viewer")        
        return RedirectResponse(url=f"/sims/oe/viewer?sim_id={sim_id}", status_code=303)
    if mode == "sweep":
        # Minimal: enqueue a single representative job; extend to full expansion as needed.
        base = sims_svc.get_simulation_full(sim_id)
        if not base:
            raise HTTPException(404, "simulation not found")
        plan = getattr(request.app.state, "_sweep_plan", {}).get(sim_id, {})
        # TODO: expand ranges -> multiple jobs. For now enqueue one with base.
        _, _, eff_hash, effective = jobs_svc.compute_effective_hash(base, {})
        job_id = jobs_svc.enqueue_sim_job(sim_id, eff_hash, bundle_level, is_visualization, precision)
        return RedirectResponse(url=f"/jobs?sim_id={sim_id}", status_code=303)

    raise HTTPException(400, "invalid mode")

# ---------- 6) sim_log (minimal placeholder; reuse your existing jobs pages) -

@router.get("/log")
def sim_log(request: Request, sim_id: Optional[int]=None):
    # You already have jobs monitor in run_monitor/web; link or embed here.
    return templates.TemplateResponse("sim_log.html", {"request": request, "sim_id": sim_id})


@router.post("/edit/apply", name="sim_edit_apply")
async def sim_edit_apply(request: Request):
    form = await request.form()
    mode = form.get("mode", "run")
    sim_id = int(form.get("sim_id")) if form.get("sim_id") else None
    # store overrides in app state (create dict once)
    st = getattr(request.app, "state")
    if not hasattr(st, "_pending_overrides"): st._pending_overrides = {}
    # take all posted fields except control keys
    override = {k: v for k, v in form.items() if k not in {"mode","sim_id"}}
    print(f"[apply] sim_id={sim_id} override_keys={list(override.keys())[:8]} count={len(override)}")
    if sim_id is not None:
        st._pending_overrides[sim_id] = override
        return RedirectResponse(url=f"/sims/confirm?mode={mode}&sim_id={sim_id}", status_code=303)
    return RedirectResponse(url=f"/sims/start", status_code=303)

# --- OE Viewer (page) ------------------------------------------------------
from fastapi import Query
from fastapi.responses import HTMLResponse

@router.get("/oe/viewer", response_class=HTMLResponse)
def oe_viewer_get(
    request: Request,
    sim_id: int = Query(...),
    kind: str = Query(default="sim"),
):
    """
    Opens the OE Viewer for a given sim_id.
    The template is static for now; data comes from JSON endpoints below.
    """
    kind_lower = (kind or "sim").lower()
    kind_suffix = ""
    if kind_lower == "metrics":
        kind_suffix = " · Metrics Run"
    elif kind_lower == "both":
        kind_suffix = " · Sim + Metrics Run"

    return templates.TemplateResponse(
        "oe_viewer.html",
        {
            "request": request,
            "sim_id": sim_id,
            "header_right": f"OE Viewer · Sim {sim_id}{kind_suffix}",
        },
    )

# --- OE Viewer JSON stubs (read-only) -------------------------------------

from fastapi.responses import JSONResponse

@router.get("/oe/run/{sim_id}/progress")
def oe_progress(sim_id: int):
    """
    Read-only progress snapshot for a simulation:
      - totals from simmetjobs
      - bytes from big_view (smj_output_size_bytes; written = sum over written jobs)
      - simple ETA and throughput estimates
    """
    from datetime import datetime, timezone

    with _connect() as conn:
        # Totals by status (simmetjobs)
        cnt_sql = """
        SELECT
          COUNT(*)                                             AS total_jobs,
          SUM(CASE WHEN status='written' THEN 1 ELSE 0 END)    AS done_jobs,
          SUM(CASE WHEN status='running' THEN 1 ELSE 0 END)    AS running_jobs,
          SUM(CASE WHEN status='failed'  THEN 1 ELSE 0 END)    AS failed_jobs
        FROM simmetjobs
        WHERE simid = %(sim_id)s;
        """
        totals = fetchone_dict(conn, cnt_sql, {"sim_id": sim_id}) or {}

        # Bytes + first start time from big_view
        bytes_sql = """
        SELECT
          COALESCE(SUM(smj_output_size_bytes), 0)                                                  AS bytes_expected,
          COALESCE(SUM(CASE WHEN smj_status = 'written' THEN smj_output_size_bytes ELSE 0 END), 0) AS bytes_written,
          MIN(smj_startdate)                                                                      AS first_start
        FROM big_view
        WHERE smj_simid = %(sim_id)s;
        """
        by = fetchone_dict(conn, bytes_sql, {"sim_id": sim_id}) or {}

    # --- totals ---
    total_jobs   = int(totals.get("total_jobs") or 0)
    done_jobs    = int(totals.get("done_jobs") or 0)
    running_jobs = int(totals.get("running_jobs") or 0)
    failed_jobs  = int(totals.get("failed_jobs") or 0)

    # --- percent ---
    percent = 0.0
    if total_jobs > 0:
        percent = (done_jobs / total_jobs) * 100.0

    # --- bytes ---
    bytes_expected = int(by.get("bytes_expected") or 0)
    bytes_written  = int(by.get("bytes_written")  or 0)

    # --- ETA & throughput estimates ---
    eta_seconds = None
    throughput_bps = None

    first_start = by.get("first_start")
    now = datetime.now(timezone.utc)

    if first_start is not None and done_jobs > 0:
        try:
            elapsed = (now - first_start).total_seconds()
            if elapsed > 1.0:
                # jobs per second
                jobs_per_sec = done_jobs / elapsed
                remaining_jobs = max(total_jobs - done_jobs, 0)
                if jobs_per_sec > 0 and remaining_jobs > 0:
                    eta_seconds = int(remaining_jobs / jobs_per_sec)

                # throughput: bytes_written / elapsed
                if bytes_written > 0:
                    throughput_bps = int(bytes_written / elapsed)
        except Exception:
            pass

    return JSONResponse({
        "sim_id": sim_id,
        "totals": {
            "total_jobs": total_jobs,
            "done_jobs": done_jobs,
            "running_jobs": running_jobs,
            "failed_jobs": failed_jobs,
        },
        "bytes": {
            "expected": bytes_expected,
            "written":  bytes_written,
        },
        "percent": round(percent, 2),
        "eta_seconds": eta_seconds,
        "throughput_bps": throughput_bps,
        "current": None,
        "server_time": now.isoformat(),
    })
@router.get("/oe/run/{sim_id}/gauges")
def oe_gauges(sim_id: int):
    """
    Live gauges for the viewer: CPU (proc/sys), RAM (used + predicted), disk free, bytes written.
    - CPU/RAM via psutil if available (falls back to 0).
    - Disk free via shutil.disk_usage on the simulations store.
    - bytes_written + mem_predicted_mb from SQL (existing schema).
    """
    import os, shutil
    from pathlib import Path
    # Optional psutil (don’t hard-require it)
    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None  # type: ignore

    # --- CPU / RAM (process + system) ---
    cpu_pct_proc = 0.0
    cpu_pct_sys  = 0.0
    ram_used_mb  = 0
    if psutil:
        try:
            p = psutil.Process(os.getpid())
            # interval=0.0 → non-blocking (first call returns 0, will warm up)
            cpu_pct_proc = float(p.cpu_percent(interval=0.1))
            cpu_pct_sys  = float(psutil.cpu_percent(interval=0.1))
            rss = getattr(p.memory_info(), "rss", 0)
            ram_used_mb = int(rss // (1024 * 1024))
        except Exception:
            pass

    # --- Disk free ---
    # Use the shared store root; match igc/gui/services/jobs.py
    STORE = Path(os.environ.get("IGC_STORE", "/data/simulations"))
    disk_free_mb = 0
    disk_total_mb = 0
    try:
        du = shutil.disk_usage(str(STORE))
        disk_free_mb  = int(du.free  // (1024 * 1024))
        disk_total_mb = int(du.total // (1024 * 1024))
    except Exception:
        pass

    # --- bytes_written + mem_predicted_mb from DB (reuse your existing queries) ---
    bytes_written = 0
    mem_predicted_mb = 0
    with _connect() as conn, conn.cursor() as cur:
        # 1) predicted memory from big_view
        cur.execute("""
            SELECT COALESCE(SUM(smj_mem_total_mb), 0) AS mem_predicted_mb
            FROM public.big_view
            WHERE smj_simid = %s
              AND smj_status IN ('running','queued','created','written')
        """, (sim_id,))
        row = cur.fetchone()
        if row and row[0] is not None:
            mem_predicted_mb = int(float(row[0]) or 0.0)

        # 2) sim_label from simulations
        cur.execute("SELECT label FROM public.simulations WHERE id = %s", (sim_id,))
        label_row = cur.fetchone()
        sim_label = (label_row[0].strip() if label_row and label_row[0] else f"Sim_{sim_id}")

    # 3) resolve run_root from sim_label + run token
    from igc.oe import core as oe_core  # local import to avoid cycles at module import time
    bytes_written = 0
    try:
        tt = oe_core._RUN_TOKEN.get(sim_id)
        if tt:
            run_root = Path("/data/simulations") / sim_label / tt
            if run_root.is_dir():
                # sum sizes of all files under the run root
                for dirpath, _, filenames in os.walk(run_root):
                    for name in filenames:
                        try:
                            p = Path(dirpath) / name
                            bytes_written += p.stat().st_size
                        except OSError:
                            pass
    except Exception:
        # leave bytes_written=0 if anything goes wrong
        pass
    return {
        "cpu_pct_proc":  cpu_pct_proc,
        "cpu_pct_sys":   cpu_pct_sys,
        "ram_used_mb":   ram_used_mb,
        "mem_predicted_mb": mem_predicted_mb,
        "disk_free_mb":  disk_free_mb,
        "disk_total_mb": disk_total_mb,
        "bytes_written": bytes_written,
    }
# --- Abort current OE run -----------------------------------------------------
@router.post("/oe/run/{sim_id}/abort")
def oe_abort(sim_id: int):
    """Signal the in-process OE runner to abort."""
    from igc.oe import core as oe_core
    oe_core.request_abort(f"user abort sim_id={sim_id}")
    return {"ok": True, "aborted": True, "sim_id": sim_id}

@router.get("/oe/run/{sim_id}/logs")
def oe_logs(request: Request, sim_id: int, after_id: int = 0, viewer_after: int = 0, limit: int = 200):
    """
    Runtime logs for a sim based on in-memory OE events and transient notes.
    """
    # pull sim-level notes from app state; never fail
    try:
        store = getattr(request.app.state, "_oe_log", {})
        # return notes once and remove them from the buffer
        notes = store.pop(sim_id, None) or store.pop(str(sim_id), []) or []
    except Exception:
        notes = []

    limit = max(1, min(int(limit or 200), 1000))
    # Runtime viewer logs no longer read from jobexecutionlog; they are driven
    # entirely by in-memory OE events plus transient notes.
    items = []
    next_after_id = int(after_id or 0)

    from datetime import datetime, timezone
    # in-memory viewer events (frame_start/file_written/frame_done)
    try:
        from igc.oe import core as oe_core
        viewer_events = oe_core.get_viewer_events(int(sim_id), int(viewer_after or 0)) or []
        viewer_next = viewer_events[-1]["seq"] if viewer_events else int(viewer_after or 0)
    except Exception:
        viewer_events = []
        viewer_next = int(viewer_after or 0)

    return JSONResponse({
        "notes": notes,
        "sim_id": sim_id,
        "items": items,
        "server_time": datetime.now(timezone.utc).isoformat(),
        "viewer": viewer_events,
        "viewer_next": int(viewer_next),
        "next_after_id": next_after_id,
    })