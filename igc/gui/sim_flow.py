from __future__ import annotations
from igc.ledger.sim import list_simulations, update_simulation, create_simulation, get_simulation_full, get_simulation_columns, get_simulation_full, get_simulation_columns, get_simulation_full, get_simulation_columns

from .vars import VARS
from pathlib import Path
from typing import Any, Dict, Optional
import logging
from fastapi import APIRouter, Request, Form, Query, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from igc.ledger import sim as sims_svc
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

    return templates.TemplateResponse("sim_edit.html", {"header_right": (("Run " if mode == "run" else ("Edit " if mode == "edit" else "")) + ((sim.get("label","") + " " + sim.get("name","")) if sim else "")).strip(), 
        "request": request,
        "mode": mode,
        "sim": sim,
        "fields": fields,
    })

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
    return templates.TemplateResponse("sim_confirm.html", {"request": request, "mode": mode, "sim_id": sim_id, "header_right": "Confirmation",
        "overrides": overrides, "sweep": sweep
        , "fields": fields
    , "sim": sim})

@router.post("/confirm")
def sim_confirm_post(request: Request,
                     mode: str = Form(...),
                     sim_id: int = Form(...),
                     bundle_level: Optional[str] = Form(None),
                     is_visualization: Optional[bool] = Form(False),
                     precision: Optional[str] = Form(None)):
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
        base = sims_svc.get_simulation(sim_id)
        if not base:
            raise HTTPException(404, "simulation not found")
        overrides = getattr(request.app.state, "_pending_overrides", {}).get(sim_id, {})
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
