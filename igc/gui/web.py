from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import os

from fastapi import FastAPI, Request, Form, HTTPException, Query, BackgroundTasks
from igc.gui.services.metrics_data import list_metric_groups, list_metrics_by_group, list_assigned_metrics
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from igc.gui.entry import list_actions, db_health, recent_sims
from igc.gui.create_sim import load_defaults, save_simulation, run_simulation
from igc.gui.data_select import describe_run, select_frame_range
from igc.gui.run_monitor import list_active_jobs, job_detail, requeue_job, cancel_job
from igc.gui.sim_flow import router as sim_router
from igc.gui.sim_flow import _simlog_append

app = FastAPI(title="IGC GUI")
# Apple touch icons (served from igc/gui/static/)

_STATIC_DIR = Path(__file__).resolve().parent / "static"

@app.get("/apple-touch-icon.png", include_in_schema=False)

def apple_touch_icon():

    return FileResponse(_STATIC_DIR / "apple-touch-icon.png", media_type="image/png", headers={"Cache-Control": "public, max-age=604800"})

@app.get("/apple-touch-icon-precomposed.png", include_in_schema=False)

def apple_touch_icon_precomposed():

    return FileResponse(_STATIC_DIR / "apple-touch-icon-precomposed.png", media_type="image/png", headers={"Cache-Control": "public, max-age=604800"})
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
# expose registry in templates
from igc.gui.vars import VARS
try:
    templates.env.globals.update(VARS=VARS, keys=VARS.keys, routes=VARS.routes, ui=VARS.ui)
except Exception:
    pass

app.include_router(sim_router)
app.include_router(sim_router)
static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
def index(request: Request):
    health = db_health()
    sims = recent_sims()
    return templates.TemplateResponse("index.html", {"request": request, "health": health, "sims": sims, "actions": list_actions()})

@app.get("/sims/new")
def sims_new(request: Request, from_id: Optional[int] = None):
    defaults = load_defaults(from_id)
    return templates.TemplateResponse("sim_new.html", {"request": request, "d": defaults})

@app.post("/sims")
def sims_create(
    request: Request,
    label: str = Form(...),
    name: str = Form(...),
    gridx: int = Form(...),
    gridy: int = Form(...),
    gridz: int = Form(...),
    psi0_elsewhere: float = Form(...),
    psi0_center: float = Form(...),
    phi0: float = Form(1.0),
    eta0: float = Form(1e-12),
    substeps_per_at: int = Form(48),
    dt_per_at: float = Form(1.0),
    dx: float = Form(1.0)
):
    spec = dict(
        label=label, name=name,
        gridx=gridx, gridy=gridy, gridz=gridz,
        psi0_elsewhere=psi0_elsewhere, psi0_center=psi0_center,
        phi0=phi0, eta0=eta0,
        substeps_per_at=substeps_per_at, dt_per_at=dt_per_at, dx=dx
    )
    try:
        sim_id = save_simulation(spec)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url=f"/sims/id/{sim_id}", status_code=303)

@app.get("/sims/id/{sim_id}")
def sims_detail(request: Request, sim_id: int):
    run = describe_run(sim_id)
    if "error" in run:
        raise HTTPException(status_code=404, detail=run["error"])
    sim = run["sim"]
    frames = run["frames"]
    return templates.TemplateResponse("sim_detail.html", {"request": request, "sim": sim, "frames": frames})

@app.post("/sims/id/{sim_id}/run")
def sims_run(sim_id: int, ats: int = Form(1), save_first: Optional[bool]=Form(False), stats: Optional[bool]=Form(False)):
    try:
        run_simulation(sim_id, ats=int(ats), save_first=bool(save_first), header_stats=bool(stats))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return RedirectResponse(url=f"/sims/id/{sim_id}", status_code=303)


@app.get("/jobs")
def jobs_page(request: Request, sim_id: Optional[int]=None, limit: int = 200):
    rows = list_active_jobs(sim_id=sim_id, limit=limit)
    return templates.TemplateResponse("jobs.html", {"request": request, "rows": rows, "sim_id": sim_id, "limit": limit, "header_right": "Running Jobs"})

@app.get("/jobs/{job_id}")
def job_page(request: Request, job_id: int):
    row = job_detail(job_id)
    if not row:
        raise HTTPException(status_code=404, detail="job not found")
    return templates.TemplateResponse("job_detail.html", {"request": request, "row": row})

@app.post("/jobs/{job_id}/requeue")
def job_requeue(job_id: int):
    requeue_job(job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)

@app.post("/jobs/{job_id}/cancel")
def job_cancel(job_id: int):
    cancel_job(job_id)
    return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)

# --- IGC VARS/URLQ REGISTRATION ---
try:
    from .vars import VARS
    from .utils import urlq
    # Expose registry + helper to all templates
    templates.env.globals.update(
        VARS=VARS,             # full registry
        routes=VARS.routes,    # shorthand: routes.*
        ui=VARS.ui,            # shorthand: ui.*
        keys=VARS.keys,        # shorthand: keys.*
        urlq=urlq,             # helper for query-string URLs
    )
except Exception as _e:
    # If templates is not defined here, ignore (some apps init elsewhere)
    pass

# ========== Metrics: SimPicker & Select (new) =================================
from igc.gui.services.simpicker_service import read_sim_meta
from igc.ledger.core import (
    upsert_pathregistry_simroot, sim_exists, load_metric_catalog_grouped,
    load_selected_metric_ids, overwrite_simmetricmatcher
)
from igc.oe.core import seed_metric_jobs
from fastapi import Form
from fastapi.responses import RedirectResponse

@app.get("/metrics/simpicker")
def metrics_simpicker(request: Request, base: str = "/data/simulations"):
    """Render the server-side folder picker at the given base path."""
    import os
    base_abs = os.path.abspath(base) if base else "/"
    if not os.path.isdir(base_abs):
        base_abs = "/"
    dirs = []
    try:
        for e in os.scandir(base_abs):
            if e.is_dir():
                dirs.append({"name": e.name, "path": os.path.join(base_abs, e.name)})
    except Exception:
        pass
    dirs.sort(key=lambda d: d["name"].lower())
    parent = os.path.dirname(base_abs.rstrip(os.sep)) or "/"
    browse = {"base": base_abs, "parent": parent, "dirs": dirs}
    return templates.TemplateResponse(
        "create_simpicker.html",
        {
            "request": request,
            "preview": {},
            "availability": {"frames": [], "phases": [0]},
            "error": None,
            "browse": browse,
            "back_url": "/",
            "primary_kind": "link",
            "primary_label": "Next",
            "primary_url": "#",
            "primary_disabled": True,
        },
    )

# ------------------ File-browse allowlist + API for jsTree -------------------
import os
from urllib.parse import unquote

# Only allow browsing under these absolute prefixes (no trailing slash required)
ALLOWLIST_PREFIXES = ("/data/simulations",)

def _canon_allowed(path: str) -> str | None:
    """Return canonical absolute path if it's inside one of ALLOWLIST_PREFIXES, else None."""
    if not path:
        return None
    p = os.path.abspath(os.path.expanduser(path))
    # Normalize and ensure p is within allowed prefixes
    for prefix in ALLOWLIST_PREFIXES:
        pref = os.path.abspath(os.path.expanduser(prefix))
        if p == pref or p.startswith(pref + os.sep):
            return p
    return None


@app.get("/api/fs_tree")
def api_fs_tree(
    base: str | None = Query(default=None),
    parent: str | None = Query(default=None),
):
    """
    JSON for jsTree: returns immediate subdirectories of the requested root.
    Accepts either ?base= or ?parent=. Falls back to the first allowed root.
    Example return: [{"id": "/data/in/A1", "text": "A1", "children": true}, ...]
    """
    from urllib.parse import unquote
    import os

    # Resolve raw -> absolute -> allowlisted
    raw = base or parent or "/data/simulations"
    raw = unquote(raw or "")
    # ALLOWLIST_PREFIXES and _canon_allowed must already exist in this module
    try:
        allowed = _canon_allowed(raw) or os.path.abspath(ALLOWLIST_PREFIXES[0])
    except NameError:
        # Fallback: basic allow under /data/in
        roots = ("/data/in",)
        p = os.path.abspath(os.path.expanduser(raw))
        allowed = p if any(p == r or p.startswith(r + os.path.sep) for r in roots) else roots[0]

    nodes = []
    try:
        for entry in os.scandir(allowed):
            if entry.is_dir(follow_symlinks=False):
                child = os.path.join(allowed, entry.name)
                nodes.append({"id": child, "text": entry.name, "children": True})
    except Exception:
        pass

    nodes.sort(key=lambda n: n["text"].lower())
    return nodes

# ----------  Preview partial (metrics/simpicker/preview) -----------------------
@app.get("/metrics/simpicker/preview")
def metrics_simpicker_preview(request: Request, path: str):
    """
    Serve the right-pane preview for the selected simulation folder.
    Accepts ?path= (absolute folder path). Reads sim_meta.json inside that folder.
    """
    from igc.gui.services.simpicker_service import read_sim_meta

    allowed = _canon_allowed(path)
    if allowed is None:
        return templates.TemplateResponse(
            "partials/preview_not_allowed.html",
            {"request": request, "path": path},
            status_code=403,
        )
    try:
        preview, availability = read_sim_meta(allowed)
        ctx = {
            "request": request,
            "preview": preview,
            "availability": availability,
            "browse": {"base": allowed},
        }
        return templates.TemplateResponse("partials/preview.html", ctx)
    except Exception as e:
        msg = str(e).lower()
        # suppress UI when sim_meta.json is simply missing in a selected folder
        if "sim_meta.json not found" in msg:
            # render empty (no text) instead of an error box
            return templates.TemplateResponse("partials/preview.html", {"request": request, "preview": {}, "availability": {"frames": [], "phases":[0]}, "browse":{"base": allowed}})
        ctx = {"request": request, "error": str(e), "path": allowed}
        return templates.TemplateResponse("partials/preview_error.html", ctx)
@app.get("/metrics/select/{sim_id}")
def metrics_select_page_new(
    request: Request,
    sim_id: int,
    run_root: str | None = Query(default=None),
):
    # New metrics selection page (dual-list UI)
    run = describe_run(sim_id)
    if "error" in run:
        raise HTTPException(status_code=404, detail=run["error"])
    sim = run["sim"]
    groups = list_metric_groups()
    # Determine whether this metrics run is for a sweep root, based on run_root or sim metadata.
    is_sweep = False
    sweep_count = 0
    sweep_d_psi_start = None
    sweep_d_psi_end = None
    sweep_d_psi_step = None

    if run_root and "/Sweep/" in run_root:
        is_sweep = True
    if sim.get("_is_sweep"):
        is_sweep = True
        try:
            sweep_count = int(sim.get("_sweep_count") or 0)
        except Exception:
            sweep_count = 0
        sweep_d_psi_start = sim.get("_sweep_d_psi_start")
        sweep_d_psi_end = sim.get("_sweep_d_psi_end")
        sweep_d_psi_step = sim.get("_sweep_d_psi_step")
    # If this is a sweep (run_root under /Sweep/), but sim does not carry sweep metadata,
    # re-use the SimPicker logic to derive sweep info from the filesystem.
    if run_root and "/Sweep/" in run_root and not sim.get("_is_sweep"):
        try:
            from igc.gui.services.simpicker_service import read_sim_meta
            preview, _avail = read_sim_meta(run_root)
            is_sweep = True
            try:
                sweep_count = int(preview.get("_sweep_count") or sweep_count)
            except Exception:
                pass
            if preview.get("_sweep_d_psi_start") is not None:
                sweep_d_psi_start = preview.get("_sweep_d_psi_start")
            if preview.get("_sweep_d_psi_end") is not None:
                sweep_d_psi_end = preview.get("_sweep_d_psi_end")
            if preview.get("_sweep_d_psi_step") is not None:
                sweep_d_psi_step = preview.get("_sweep_d_psi_step")
        except Exception:
            # If anything goes wrong, keep existing defaults;
            # metrics_select still works, only sweep summary may be less detailed.
            pass            
    metrics_by_group = list_metrics_by_group()

    assigned_metric_ids = list_assigned_metrics(sim_id)
    return templates.TemplateResponse(
        "metrics_select.html",
        {
            "request": request,
            "sim": sim,
            "groups": groups,
            "metrics_by_group": metrics_by_group,
            "assigned_metric_ids": assigned_metric_ids,
            "back_url": "/metrics/simpicker",
            "primary_kind": "submit",
            "primary_label": "Next",
            "primary_url": None,
            "primary_disabled": (len(assigned_metric_ids) == 0),
            "primary_form_id": "metricsForm",
            "header_right": f"{sim['label']} · {sim['name']}",
            "is_sweep": is_sweep,
            "sweep_count": sweep_count,
            "sweep_d_psi_start": sweep_d_psi_start,
            "sweep_d_psi_end": sweep_d_psi_end,
            "sweep_d_psi_step": sweep_d_psi_step,            
            "run_root": run_root or "",
        },
    )

@app.post("/metrics/{sim_id}/confirm")
def metrics_confirm(
    request: Request,
    sim_id: int,
    metric_ids: List[int] = Form([]),
    run_root: str | None = Form(default=None),
):
    if not metric_ids:
        raise HTTPException(status_code=400, detail="No metrics selected.")

    run = describe_run(sim_id)
    if "error" in run:
        raise HTTPException(status_code=404, detail=run["error"])
    sim = run["sim"]

    # Build grouped summary using data services
    selected = {int(x) for x in metric_ids}
    groups = list_metric_groups()                 # [{'id':1,'name':'observables','count':22}, ...]
    all_by_group = list_metrics_by_group()        # { group_id: [{id,name,desc,out}, ...], ... }

    confirm_groups = []
    for g in groups:
        gid = g["id"]
        gmets = [m for m in all_by_group.get(gid, []) if m["id"] in selected]
        confirm_groups.append({"id": gid, "name": g["name"], "metrics": gmets})
    is_sweep = bool(run_root) and "/Sweep/" in run_root        

    return templates.TemplateResponse(
        "metrics_confirm.html",
        {
            "request": request,
            "sim": sim,
            "groups": confirm_groups,
            "total_selected": len(selected),
            "metric_ids": list(selected),
            "header_right": "Confirmation",
            "is_sweep": is_sweep,            
            "run_root": run_root or "",
        },
    )

@app.post("/metrics/{sim_id}/save")
def metrics_save(
    request: Request,
    background_tasks: BackgroundTasks,
    sim_id: int,
    metric_ids: List[int] = Form([]),
    run_root: str = Form(""),
):
    # Clear any leftover OE viewer notes for this sim before a fresh metrics run
    try:
        st = getattr(request.app, "state", None)
        if st is not None and hasattr(st, "_oe_log"):
            st._oe_log.pop(int(sim_id), None)
    except Exception:
        pass

    # Metrics run preparation note for OE viewer
    _simlog_append(
        request.app,
        sim_id,
        "We are now preparing metrics run data. Please wait for your metrics to start..."
    )

    # Persist: enable selected; disable any previously enabled but not selected
    selected = {int(x) for x in metric_ids}

    from igc.db.pg import cx, fetchall_dict, execute
    from igc.ledger import core as ledger
    from igc.oe import core as oe_core
    from igc.gui.metrics_select import discover_frames_in_root
    from time import perf_counter
    
    # 1) Persist selection in simmetricmatcher
    with cx() as conn:
        # current enabled set
        rows = fetchall_dict(
            conn,
            "SELECT metric_id FROM simmetricmatcher WHERE sim_id=%s AND enabled=true",
            (sim_id,),
        )
        current = {r["metric_id"] for r in rows}

        # disable removed
        to_disable = current - selected
        for mid in to_disable:
            execute(
                conn,
                "UPDATE simmetricmatcher SET enabled=false, updated_at=now() "
                "WHERE sim_id=%s AND metric_id=%s AND enabled=true",
                (sim_id, mid),
            )

        # upsert selected -> enabled=true
        for mid in selected:
            execute(
                conn,
                """
                INSERT INTO simmetricmatcher (sim_id, metric_id)
                VALUES (%s, %s)
                ON CONFLICT (sim_id, metric_id)
                DO UPDATE SET enabled=true, updated_at=now()
                """,
                (sim_id, mid),
            )

    # 2) Flush job queue and seed metric jobs based on frames on disk
        # Force OE to use the actual disk run_root for this metrics session
        import os
        tt = os.path.basename(run_root.rstrip("/"))
        try:
            oe_core._RUN_TOKEN[sim_id] = tt
        except Exception:
            pass

    # Metrics sweep: if run_root is a Sweep folder, run metrics once per member run_root.
    is_sweep = bool(run_root) and "/Sweep/" in run_root

    if selected and is_sweep:
        from igc.gui.metrics_select import discover_sweep_members
        import os
        # Ensure a fresh sweep abort state for metrics sweeps.
        # We do NOT call request_abort_sweep here, because that flag is also
        # consulted by the new sweep loop itself via should_abort_sweep(sim_id).
        # If we set it before starting, the first iteration would immediately bail out.
        try:
            if hasattr(oe_core, "clear_sweep_abort"):
                oe_core.clear_sweep_abort(sim_id)
        except Exception:
            pass

        def _metrics_sweep_loop() -> None:
            members = list(discover_sweep_members(run_root))
            total_members = len(members)

            # Announce sweep start with total member count
            try:
                _simlog_append(
                    request.app,
                    sim_id,
                    f"[sweep] metrics sweep starting ({total_members} sims)"
                )
            except Exception:
                pass

            for idx, member_root in enumerate(members, start=1):
                # If any sim or metrics sweep for this sim_id requested abort, stop.
                try:
                    if hasattr(oe_core, "should_abort_sweep") and oe_core.should_abort_sweep(sim_id):
                        # Optional: small note so you see this in the viewer
                        try:
                            _simlog_append(request.app, sim_id, "[sweep] metrics sweep abort requested; stopping.")
                        except Exception:
                            pass
                        break
                except Exception:
                    pass

                # Announce this sweep member as running
                try:
                    _simlog_append(
                        request.app,
                        sim_id,
                        f"[sweep] metrics sweep {idx}/{total_members} sims running"
                    )
                except Exception:
                    pass

                try:
                    # Set RUN_TOKEN correctly for sweep members:
                    # it must preserve "Sweep/<stamp>/<member>", not just the leaf folder name.
                    try:
                        # Example:
                        # run_root     = /data/simulations/A1/Sweep/20251120_1046
                        # member_root  = /data/simulations/A1/Sweep/20251120_1046/A1_s_psi_0_2
                        # label_root   = /data/simulations/A1
                        label_root = os.path.dirname(os.path.dirname(run_root.rstrip("/")))
                        # Relative token: Sweep/20251120_1046/A1_s_psi_0_2
                        tt_member = os.path.relpath(member_root, start=label_root)
                    except Exception:
                        # Fallback — still better than cutting off the sweep path
                        tt_member = os.path.basename(member_root.rstrip("/"))

                    try:
                        oe_core._RUN_TOKEN[sim_id] = tt_member
                    except Exception:
                        pass

                    # Flush all jobs before each member run
                    with cx() as conn2:
                        execute(conn2, "TRUNCATE TABLE public.simmetjobs")

                    # Discover frames for this member run_root
                    frames_member = discover_frames_in_root(member_root)
                    if not frames_member:
                        continue

                    phases = [0]
                    # Seed metric jobs and finalize paths for this member
                    oe_core.seed_metric_jobs(sim_id, sorted(selected), frames_member, phases)
                    try:
                        oe_core.finalize_seeded_jobs(sim_id=sim_id)
                    except Exception:
                        # finalization failures should not stop the sweep; jobs can still exist
                        pass

                    # Run metrics for this member, appending logs in the OE viewer
                    t0 = perf_counter()
                    oe_core.run(sim_id=sim_id, kind="metrics", append_view=True)
                    run_secs = perf_counter() - t0

                    # Emit a sweep note so you see the member boundary and runtime.
                    try:
                        _simlog_append(
                            request.app,
                            sim_id,
                            f"[sweep] metrics sweep {idx}/{total_members} sims done (runtime {run_secs:.2f}s)"
                        )
                    except Exception:
                        pass

                except Exception:
                    # Do not abort the entire sweep on a single member failure
                    continue

            # After finishing all members, announce completion and clear abort flag
            try:
                _simlog_append(
                    request.app,
                    sim_id,
                    f"[sweep] metrics sweep completed ({total_members}/{total_members} sims)"
                )
            except Exception:
                pass

            try:
                if hasattr(oe_core, "clear_sweep_abort"):
                    oe_core.clear_sweep_abort(sim_id)
            except Exception:
                pass

        if background_tasks is not None:
            background_tasks.add_task(_metrics_sweep_loop)
        else:
            _metrics_sweep_loop()

        return RedirectResponse(
            url=f"/sims/oe/viewer?sim_id={sim_id}&kind=metrics", status_code=303
        )

    # Non-sweep metrics run: seed and run metrics in a background task so the viewer
    # can attach immediately.
    if selected and not is_sweep:
        def _metrics_single_run() -> None:
            # Flush ALL jobs before a fresh metrics run (simmetjobs is a pure work queue)
            with cx() as conn:
                execute(conn, "TRUNCATE TABLE public.simmetjobs")

            # Discover frames for this metrics run from the explicit run_root on disk
            frames = discover_frames_in_root(run_root)
            if not frames:
                return

            phases = [0]
            # Seed metric jobs and finalize paths
            oe_core.seed_metric_jobs(sim_id, sorted(selected), frames, phases)
            try:
                oe_core.finalize_seeded_jobs(sim_id=sim_id)
            except Exception:
                # finalization failures should not stop the run; jobs can still exist
                pass

            # Run metrics for this sim_id; append logs to the existing viewer buffer
            oe_core.run(sim_id=sim_id, kind="metrics", append_view=True)

        if background_tasks is not None:
            background_tasks.add_task(_metrics_single_run)
        else:
            _metrics_single_run()

    # 4) Redirect to OE viewer for this sim
    return RedirectResponse(url=f"/sims/oe/viewer?sim_id={sim_id}&kind=metrics", status_code=303)

@app.get("/sims/{sim_id}/frames/{frame_idx}/preview/{field}")
def sim_frame_preview(sim_id: int, frame_idx: int, field: str):
    """
    Serve preview_psi.png / preview_phi.png / preview_eta.png for a given frame.
    """
    from fastapi import HTTPException
    from fastapi.responses import FileResponse
    import os
    from pathlib import Path

    from igc.db.pg import cx, fetchall_dict
    from igc.oe import core as oe_core

    # Only allow known fields
    if field not in ("psi", "phi", "eta", "collapse_mask", "p_k", "coh_length"):
        raise HTTPException(status_code=404, detail="unknown field")

    # Resolve current run token for this sim_id
    tt = oe_core._RUN_TOKEN.get(sim_id)
    if not tt:
        raise HTTPException(status_code=404, detail="no active run token")

    # Resolve sim label from DB
    with cx() as conn:
        rows = fetchall_dict(conn, "SELECT label FROM public.simulations WHERE id=%s", (sim_id,))
        if not rows:
            raise HTTPException(status_code=404, detail="simulation not found")
        sim_label = str(rows[0].get("label") or f"Sim_{sim_id}").strip()

    root = os.environ.get("IGC_STORE", "/data/simulations")
    frame_dir = Path(root) / sim_label / tt / f"Frame_{int(frame_idx):04d}"

    # Map field -> preview filename
    name = {
        "psi": "preview_psi.png",
        "phi": "preview_phi.png",
        "eta": "preview_eta.png",
        "collapse_mask": "preview_collapse_mask.png",
        "p_k": "preview_p_k.png",
        "coh_length": "preview_coh_length.png",
    }.get(field)

    path = frame_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="preview not found")

    return FileResponse(str(path), media_type="image/png")