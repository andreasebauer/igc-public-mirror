from __future__ import annotations
from pathlib import Path
from typing import Optional, List
import os

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from igc.gui.entry import list_actions, db_health, recent_sims
from igc.gui.create_sim import load_defaults, save_simulation, run_simulation
from igc.gui.data_select import describe_run, select_frame_range
from igc.gui.metrics_select import list_metrics_for_sim, validate_selection, seed_jobs_for_frames
from igc.gui.run_monitor import list_active_jobs, job_detail, requeue_job, cancel_job
from igc.gui.sim_flow import router as sim_router

app = FastAPI(title="IGC GUI")
BASE_DIR = Path(__file__).parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))
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

@app.get("/metrics/{sim_id}")
def metrics_select_page(request: Request, sim_id: int):
    run = describe_run(sim_id)
    if "error" in run:
        raise HTTPException(status_code=404, detail=run["error"])
    sim = run["sim"]
    frames = run["frames"]
    metrics = list_metrics_for_sim(sim_id)
    return templates.TemplateResponse("metrics_select.html", {"request": request, "sim": sim, "frames": frames, "metrics": metrics})

@app.post("/metrics/{sim_id}/seed")
def metrics_seed(sim_id: int, frame_start: int = Form(...), frame_end: int = Form(...), stride: int = Form(1), metric_ids: List[int] = Form([])):
    err = validate_selection(metric_ids)
    if err:
        raise HTTPException(status_code=400, detail="; ".join(err))
    run = describe_run(sim_id)
    if "error" in run:
        raise HTTPException(status_code=404, detail=run["error"])
    frames = select_frame_range(run["frames"], frame_start, frame_end, stride)
    if not frames:
        raise HTTPException(status_code=400, detail="no frames selected")
    seed_jobs_for_frames(sim_id, metric_ids, frames)
    return RedirectResponse(url=f"/jobs?sim_id={sim_id}", status_code=303)

@app.get("/jobs")
def jobs_page(request: Request, sim_id: Optional[int]=None, limit: int = 200):
    rows = list_active_jobs(sim_id=sim_id, limit=limit)
    return templates.TemplateResponse("jobs.html", {"request": request, "rows": rows, "sim_id": sim_id, "limit": limit})

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
