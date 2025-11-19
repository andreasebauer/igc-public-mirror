from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional
import os

from igc.db.pg import cx, fetchone_dict, fetchall_dict, execute
from igc.sim.grid_constructor import GridConfig, build_humming_grid
from igc.sim.coupler import CouplerConfig, Coupler
from igc.sim.injector import Injector, InjectionEvent
from igc.sim.integrator import IntegratorConfig, Integrator
from igc.sim.saver import HeaderOptions
from igc.sim.validator import validate_sim_config

STORE = Path(os.environ.get("IGC_STORE", "/data/igc"))

@dataclass
class SimSpec:
    label: str
    name: str
    gridx: int
    gridy: int
    gridz: int
    psi0_elsewhere: float
    psi0_center: float
    phi0: float
    eta0: float
    substeps_per_at: int = 48
    dt_per_at: float = 1.0
    dx: float = 1.0
    d_psi: float = 0.0
    d_eta: float = 0.0
    d_phi: float = 0.0
    c_pi_to_eta: float = 1.0
    c_eta_to_phi: float = 1.0
    lambda_eta: float = 1.0
    lambda_phi: float = 1.0
    gate_name: str = "linear"
    seed_type: str = "none"
    seed_field: str = "psi"
    seed_strength: float = 0.0
    seed_sigma: float = 0.0
    seed_center: str = "center"
    seed_phase_a: float = 13/48
    seed_phase_b: float = 14/48
    seed_repeat_at: Optional[int] = None
    phi_threshold: float = 0.5
    precision: str = "float64"

def load_defaults(sim_id: Optional[int]=None) -> dict:
    if sim_id is None:
        return asdict(SimSpec(
            label="DEV_gui",
            name="GUI-created sim",
            gridx=128, gridy=128, gridz=128,
            psi0_elsewhere=1.0, psi0_center=1.00002,
            phi0=1.0, eta0=1e-12
        ))
    with cx() as conn:
        row = fetchone_dict(conn, "select * from simulations where id=%s", (sim_id,))
        if not row: raise ValueError("simulation not found")
        return dict(row)

def validate_spec(spec: dict) -> list[str]:
    errs:list[str] = []
    for k in ["label","name","gridx","gridy","gridz","psi0_elsewhere","psi0_center","phi0","eta0"]:
        if k not in spec or spec[k] in (None,""):
            errs.append(f"missing {k}")
    if spec.get("gridx",0) <=0 or spec.get("gridy",0)<=0 or spec.get("gridz",0)<=0:
        errs.append("grid dims must be > 0")
    if spec.get("seed_phase_a",0) >= spec.get("seed_phase_b",1):
        errs.append("seed_phase_a must be < seed_phase_b")
    return errs

def save_simulation(spec: dict) -> int:
    errs = validate_spec(spec)
    if errs: raise ValueError("; ".join(errs))
    with cx() as conn:
        simid_row = fetchone_dict(conn, "select coalesce(max(simid),0)+1 as next_sid from simulations")
        simid = simid_row["next_sid"]
        execute(conn, """
            insert into simulations (
              simid, name, label, description, gridx, gridy, gridz,
              psi0_center, psi0_elsewhere, phi0, eta0, phi_threshold,
              substeps_per_at, dt_per_at, dx,
              d_psi, d_eta, d_phi, c_pi_to_eta, c_eta_to_phi, lambda_eta, lambda_phi, gate_name,
              seed_type, seed_field, seed_strength, seed_sigma, seed_center, seed_phase_a, seed_phase_b, seed_repeat_at,
              precision, status, createdate
            )
            values (
              %(simid)s, %(name)s, %(label)s, %(description)s, %(gridx)s, %(gridy)s, %(gridz)s,
              %(psi0_center)s, %(psi0_elsewhere)s, %(phi0)s, %(eta0)s, %(phi_threshold)s,
              %(substeps_per_at)s, %(dt_per_at)s, %(dx)s,
              %(d_psi)s, %(d_eta)s, %(d_phi)s, %(c_pi_to_eta)s, %(c_eta_to_phi)s, %(lambda_eta)s, %(lambda_phi)s, %(gate_name)s,
              %(seed_type)s, %(seed_field)s, %(seed_strength)s, %(seed_sigma)s, %(seed_center)s, %(seed_phase_a)s, %(seed_phase_b)s, %(seed_repeat_at)s,
              %(precision)s, 'new', %(createdate)s
            )
        """, {
            **spec,
            "simid": simid,
            "description": spec.get("description","created via gui"),
            "createdate": datetime.utcnow()
        })
        row = fetchone_dict(conn, "select id from simulations where simid=%s order by id desc limit 1", (simid,))
        return row["id"]

def run_simulation(sim_id: int, ats: int, save_first: bool=True, header_stats: bool=True) -> int:
    with cx() as conn:
        sim = fetchone_dict(conn, "select * from simulations where id=%s", (sim_id,))
        if not sim: raise ValueError("simulation not found")

    cfg = GridConfig(shape=(sim["gridx"], sim["gridy"], sim["gridz"]))
    state = build_humming_grid(cfg, substeps_per_at=sim["substeps_per_at"], initial_at=0)

    # Legacy initial-condition overrides (ψ0/φ0/η0) — disabled for now.
    # We keep the true humming vacuum at At=0 and apply all seeds via Injector at At=3.
    #
    # state.psi[...] = sim["psi0_elsewhere"]
    # delta = (sim["psi0_center"] - sim["psi0_elsewhere"]) if sim["psi0_center"] is not None else 0.0
    # state.psi[cx0,cy0,cz0] += delta
    # state.phi_field[...] = sim["phi0"]
    # state.eta[...] = sim["eta0"]

    # We still compute the center index here, for use as the default InjectionEvent center.
    cx0 = state.psi.shape[0]//2
    cy0 = state.psi.shape[1]//2
    cz0 = state.psi.shape[2]//2

    coupler = Coupler(CouplerConfig(
        D_psi=sim["d_psi"], D_eta=sim["d_eta"], D_phi=sim["d_phi"],
        C_pi_to_eta=sim["c_pi_to_eta"], C_eta_to_phi=sim["c_eta_to_phi"],
        lambda_eta=sim["lambda_eta"], lambda_phi=sim["lambda_phi"],
        gate=sim["gate_name"]
    ))
    events=[]
    if (sim["seed_type"] or "none").lower()!="none" and float(sim["seed_strength"] or 0.0)!=0.0:
        events.append(InjectionEvent(
            kind=(sim["seed_type"] or "once").lower(),
            field=(sim["seed_field"] or "psi").lower(),
            amplitude=float(sim["seed_strength"]),
            sigma=float(sim["seed_sigma"] or 0.0),
            center=(cx0,cy0,cz0) if (sim["seed_center"] or "center")=="center" else tuple(map(int,(sim["seed_center"].split(",")))),
            window=(float(sim["seed_phase_a"] or 0.25), float(sim["seed_phase_b"] or 0.30)),
            repeat=(
                {
                    "first_at": int(sim.get("seed_at") or 0),
                    "period_at": int(sim["seed_repeat_at"])
                }
                if sim.get("seed_repeat_at") else
                {
                    "first_at": int(sim.get("seed_at") or 0)
                }
            )
        ))
    injector = Injector(events)
    integ_cfg = IntegratorConfig(
        substeps_per_at=sim["substeps_per_at"],
        dt_per_at=sim["dt_per_at"],
        dx=float(sim.get("dx") or 1.0),
        stride_frames_at=1,
    )

    # Sanity / stability check before running a potentially large sim
    validate_sim_config(cfg, integ_cfg, coupler.cfg)

    integ = Integrator(coupler, injector, integ_cfg)
    integ.run(
        store=STORE,
        sim_label=sim["label"],
        psi=state.psi,
        pi=state.pi,
        eta=state.eta,
        phi_field=state.phi_field,
        phi_cone=state.phi_cone,
        at_start=0,
        at_end=int(ats),
        save_first_frame=bool(save_first),
        header_stats=bool(header_stats),
    )
    return ats
