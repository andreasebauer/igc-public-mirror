from __future__ import annotations
from typing import Any, Dict, List, Tuple

# Minimal, mode-aware validator; extend ranges as you like.
# Returns (clean_payload, errors). clean_payload uses keys matching DB columns.

EDITABLE_DEFAULTS = {
    # identity
    "simid": None, "name": None, "label": None, "description": None,
    # grid & timing
    "gridx": 128, "gridy": 128, "gridz": 128, "t_max": 100, "substeps_per_at": 1,
    # cadence
    "stride": 1, "quarter_substeps": False, "final_only": False,
    # initial fields
    "psi0_center": 1.00000, "psi0_elsewhere": 1.0, "phi0": 1.0, "eta0": 1e-12,
    # gates
    "phi_threshold": 0.5, "alpha": None,
    # diffusion / couplings
    "d_psi": 1.0, "d_phi": 1.0, "d_eta": 1.0,
    "c_psiphi": 1.0, "c_psieta": 0.25, "c_phipsi": 1.0, "c_phieta": 0.0,
    "lambda_phi": 1.0, "lambda_eta": 1.0,
    # seeding/injector
    "seed_type": "none", "seed_field": "psi", "seed_strength": 0.0, "seed_sigma": 1.0,
    "seed_center": None, "seed_phase_a": None, "seed_phase_b": None, "seed_repeat_at": None,
    "injector_json": None,
    # advanced configs
    "coupler_json": None, "integrator_json": None,
}

RUN_EDITABLE = {
    "gridx","gridy","gridz","t_max","substeps_per_at",
    "stride","quarter_substeps","final_only",
    "phi_threshold","alpha",
    "d_psi","d_phi","d_eta",
    "c_psiphi","c_psieta","c_phipsi","c_phieta",
    "lambda_phi","lambda_eta",
    "seed_type","seed_field","seed_strength","seed_sigma","seed_center",
    "seed_phase_a","seed_phase_b","seed_repeat_at",
    "injector_json","coupler_json","integrator_json",
}

def coerce_types(data: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k,v in data.items():
        if isinstance(v, str) and v.strip() == "":
            out[k] = None
            continue
        if k in {"gridx","gridy","gridz","t_max","substeps_per_at","stride"}:
            out[k] = int(v) if v is not None else None
        elif k in {
            "psi0_center","psi0_elsewhere","phi0","eta0","phi_threshold","alpha",
            "d_psi","d_phi","d_eta","c_psiphi","c_psieta","c_phipsi","c_phieta",
            "lambda_phi","lambda_eta","seed_strength","seed_sigma"
        }:
            out[k] = float(v) if v is not None else None
        elif k in {"quarter_substeps","final_only"}:
            out[k] = bool(v) if isinstance(v, bool) else (str(v).lower() in {"1","true","on","yes"})
        else:
            out[k] = v
    return out

def validate(mode: str, payload: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    mode: create|edit|run
    """
    clean = EDITABLE_DEFAULTS.copy()
    clean.update(coerce_types(payload))
    errs: Dict[str,str] = {}

    # Required fields
    req = ["name","label","gridx","gridy","gridz","t_max","stride"]
    for k in req:
        if clean.get(k) in (None, "") and mode in {"create","edit"}:
            errs[k] = "required"

    # Basic ranges
    for k in ["gridx","gridy","gridz"]:
        v = clean.get(k)
        if v is not None and int(v) <= 0:
            errs[k] = "must be > 0"
    if clean.get("t_max") is not None and int(clean["t_max"]) < 0:
        errs["t_max"] = "must be >= 0"
    if clean.get("stride") is not None and int(clean["stride"]) < 1:
        errs["stride"] = "must be >= 1"
    pt = clean.get("phi_threshold")
    if pt is not None and not (0.0 <= float(pt) <= 1.5):
        errs["phi_threshold"] = "expected 0.0–1.5 (adjust per model)"

    # Lock fields in run mode (drop any changes not allowed)
    if mode == "run":
        clean = {k:v for k,v in clean.items() if k in RUN_EDITABLE}

    return clean, errs
