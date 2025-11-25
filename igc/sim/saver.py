from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional
import json
import numpy as np

from igc.common.paths import ensure_dirs
from igc.common.hashutil import sha256_file

@dataclass
class HeaderOptions:
    write_stats: bool = False  # toggle tiny means/min/max as quick health check

def _stats(a: np.ndarray) -> Dict[str, float]:
    return {
        "mean": float(np.nanmean(a)),
        "min":  float(np.nanmin(a)),
        "max":  float(np.nanmax(a)),
        "nonfinite": int(np.sum(~np.isfinite(a)))
    }

def save_frame(store: Path,
               sim_label: str,
               frame: int,
               *,
               psi: np.ndarray,
               pi: np.ndarray,
               eta: np.ndarray,
               phi_field: np.ndarray,
               phi_cone: Optional[np.ndarray] = None,
               at: int,
               substeps_per_at: int,
               tact_phase: float,
               coupler_id: Optional[str] = None,
               injector_events: Optional[list] = None,
               header_opts: HeaderOptions = HeaderOptions()) -> Dict[str, str]:
    """
    Writes psi.npy, pi.npy, eta.npy, phi_field.npy and a small frame_info.json
    Returns dict of {name:path_str} for the written files.
    """
    _, fdir = ensure_dirs(store, sim_label, frame)
    files = {}

    # Save arrays (f64)
    files["psi"]       = str(fdir / "psi.npy")
    files["pi"]        = str(fdir / "pi.npy")
    files["eta"]       = str(fdir / "eta.npy")
    files["phi_field"] = str(fdir / "phi_field.npy")
    if phi_cone is not None:
        files["phi_cone"] = str(fdir / "phi_cone.npy")

    np.save(files["psi"], psi)
    np.save(files["pi"],  pi)
    np.save(files["eta"], eta)
    np.save(files["phi_field"], phi_field)
    if phi_cone is not None:
        np.save(files["phi_cone"], phi_cone)

    # Build header (provenance + optional quick stats)
    info = {
        "sim_label": sim_label,
        "frame": frame,
        "at": at,
        "substeps_per_at": substeps_per_at,
        "tact_phase": tact_phase,
        "coupler_id": coupler_id,
        "injector_events": injector_events or [],
        "files": {
            "psi":       {"path": files["psi"]},
            "pi":        {"path": files["pi"]},
            "eta":       {"path": files["eta"]},
            "phi_field": {"path": files["phi_field"]},
        }
    }
    if phi_cone is not None:
        info["files"]["phi_cone"] = {"path": files["phi_cone"]}

    # add sizes & hashes
    for key, meta in info["files"].items():
        p = Path(meta["path"])
        meta["shape"] = list(eval(key).shape) if False else None  # kept None to avoid large headers
        meta["dtype"] = "float64"
        meta["bytes"] = p.stat().st_size
        meta["sha256"] = sha256_file(p)

    if header_opts.write_stats:
        info["quick_stats"] = {
            "psi": _stats(psi),
            "pi": _stats(pi),
            "eta": _stats(eta),
            "phi_field": _stats(phi_field),
        }
        if phi_cone is not None:
            info["quick_stats"]["phi_cone"] = _stats(phi_cone)

    header_path = fdir / "frame_info.json"
    with header_path.open("w") as f:
        json.dump(info, f, indent=2)
    files["frame_info"] = str(header_path)

    return files

def save_substep(store: Path,
                 sim_label: str,
                 frame: int,
                 substep: int,
                 *,
                 psi: np.ndarray,
                 pi: np.ndarray,
                 eta: np.ndarray,
                 phi_field: np.ndarray,
                 phi_cone: Optional[np.ndarray] = None):
    """
    Save raw triad arrays for a single substep under:
      Frame_XXXX/Sub_YY/
    """
    _, frame_dir = ensure_dirs(store, sim_label, frame)
    subdir = frame_dir / f"Sub_{substep:02d}"
    subdir.mkdir(parents=True, exist_ok=True)

    np.save(subdir / "psi.npy",        psi)
    np.save(subdir / "pi.npy",         pi)
    np.save(subdir / "eta.npy",        eta)
    np.save(subdir / "phi_field.npy",  phi_field)

    if phi_cone is not None:
        np.save(subdir / "phi_cone.npy", phi_cone)
