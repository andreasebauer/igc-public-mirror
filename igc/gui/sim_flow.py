from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Optional
from fastapi import APIRouter, Request, Form, Query, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from igc.gui.services import simulations as sims_svc
from igc.gui.services import jobs as jobs_svc
from igc.gui.services import forms as forms_svc

router = APIRouter(prefix="/sims", tags=["sims"])
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

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
    items = sims_svc.list_simulations(q=q, limit=100, offset=(page-1)*100)
    return templates.TemplateResponse("sim_select.html", {
        "request": request, "mode": mode, "items": items, "q": q or "", "page": page
    })

@router.post("/select")
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
    base_spec: Dict[str, Any] = {}
    if mode in {"edit","run"}:
        if not sim_id:
            raise HTTPException(400, "sim_id required")
        row = sims_svc.get_simulation(sim_id)
        if not row:
            raise HTTPException(404, "simulation not found")
        base_spec = dict(row)
    else:
        # create mode: show defaults
        base_spec = forms_svc.EDITABLE_DEFAULTS.copy()
    return templates.TemplateResponse("sim_edit.html", {
        "request": request, "mode": mode, "sim_id": sim_id, "spec": base_spec
    })

@router.post("/edit")
async def sim_edit_post(request: Request,
                        mode: str = Form(...),
                        sim_id: Optional[int] = Form(None)):
    form = dict(await request.form())
    clean, errs = forms_svc.validate(mode, form)
    if errs:
        return templates.TemplateResponse("sim_edit.html", {
            "request": request, "mode": mode, "sim_id": sim_id, "spec": form, "errors": errs
        })
    # create/edit: persist; run: keep ephemeral (no DB updates)
    if mode == "create":
        new_id = sims_svc.insert_simulation(clean)
        return RedirectResponse(url=f"/sims/confirm?mode=create&sim_id={new_id}", status_code=303)
    elif mode == "edit":
        if not sim_id:
            raise HTTPException(400, "sim_id required for edit")
        sims_svc.update_simulation(sim_id, clean)
        return RedirectResponse(url=f"/sims/confirm?mode=edit&sim_id={sim_id}", status_code=303)
    else:  # run (ephemeral overrides)
        if not sim_id:
            raise HTTPException(400, "sim_id required for run")
        # store overrides temporarily in session-like cookie or pass via confirm URL as needed.
        request.session if hasattr(request, "session") else None  # optional if you enable SessionMiddleware
        request.app.state._run_overrides = {sim_id: clean}  # minimal in-process cache
        return RedirectResponse(url=f"/sims/confirm?mode=run&sim_id={sim_id}", status_code=303)

# ---------- 4) sim_sweep -----------------------------------------------------

@router.get("/sweep")
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

@router.get("/confirm")
def sim_confirm_get(request: Request,
                    mode: str = Query(..., regex="^(create|edit|run|sweep)$"),
                    sim_id: int = Query(...)):
    base = sims_svc.get_simulation(sim_id) if mode != "create" else None
    overrides = {}
    sweep = None
    if mode == "run":
        overrides = getattr(request.app.state, "_run_overrides", {}).get(sim_id, {})
    elif mode == "sweep":
        sweep = getattr(request.app.state, "_sweep_plan", {}).get(sim_id, {})
    return templates.TemplateResponse("sim_confirmation.html", {
        "request": request, "mode": mode, "sim_id": sim_id,
        "base": base, "overrides": overrides, "sweep": sweep
    })

@router.post("/confirm")
def sim_confirm_post(request: Request,
                     mode: str = Form(...),
                     sim_id: int = Form(...),
                     bundle_level: Optional[str] = Form(None),
                     is_visualization: Optional[bool] = Form(False),
                     precision: Optional[str] = Form(None)):
    if mode in {"create","edit"}:
        # Already persisted in /edit; just go home with a toast
        return RedirectResponse(url="/sims/start?msg=Simulation+saved", status_code=303)

    if mode == "run":
        base = sims_svc.get_simulation(sim_id)
        if not base:
            raise HTTPException(404, "simulation not found")
        overrides = getattr(request.app.state, "_run_overrides", {}).get(sim_id, {})
        base_hash, over_hash, eff_hash, effective = jobs_svc.compute_effective_hash(base, overrides)
        frame0 = jobs_svc.plan_frame0_dir(base)
        run_overrides_path = jobs_svc.write_run_overrides_json(frame0, sim_id, base_hash, over_hash, eff_hash, overrides)
        job_id = jobs_svc.enqueue_sim_job(sim_id, eff_hash, bundle_level, is_visualization, precision)
        return RedirectResponse(url=f"/jobs/{job_id}", status_code=303)

    if mode == "sweep":
        # Minimal: enqueue a single representative job; extend to full expansion as needed.
        base = sims_svc.get_simulation(sim_id)
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
