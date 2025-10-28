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

    np.save(files["psi"], psi)
    np.save(files["pi"],  pi)
    np.save(files["eta"], eta)
    np.save(files["phi_field"], phi_field)

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

    header_path = fdir / "frame_info.json"
    with header_path.open("w") as f:
        json.dump(info, f, indent=2)
    files["frame_info"] = str(header_path)

    return files
