import json, os
from typing import Dict, Any, Tuple

ALLOWLIST_PREFIXES = ("/data/in", "/data/sims", "/mnt", "/media")

def _is_allowed_path(p: str) -> bool:
    try:
        ap = os.path.abspath(p)
        return any(ap.startswith(pref) for pref in ALLOWLIST_PREFIXES)
    except Exception:
        return False

def read_sim_meta(abs_path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not _is_allowed_path(abs_path):
        raise ValueError("Path not allowed. Use /data/in (recommended) or /data/sims, or a mounted folder under /mnt or /media.")
    meta_path = os.path.join(abs_path, "sim_meta.json")
    if not os.path.isfile(meta_path):
        raise ValueError("sim_meta.json not found in selected folder.")
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
    except Exception:
        raise ValueError("Could not parse sim_meta.json.")
    if "sim_id" not in meta:
        raise ValueError("sim_meta.json must contain sim_id.")

    preview = {
        "sim_id": meta.get("sim_id"),
        "name": meta.get("name"),
        "label": meta.get("label"),
        "gridx": meta.get("gridx"),
        "gridy": meta.get("gridy"),
        "gridz": meta.get("gridz"),
        "t_max": meta.get("t_max"),
        "stride": meta.get("stride"),
        "substeps_per_at": meta.get("substeps_per_at"),
        "dx": meta.get("dx"),
        "dt_per_at": meta.get("dt_per_at"),
    }

    frames = set()
    phases = set()
    try:
        for entry in os.scandir(abs_path):
            if entry.is_dir() and entry.name.isdigit():
                frames.add(int(entry.name))
                sub = os.path.join(abs_path, entry.name)
                for ph in os.listdir(sub):
                    if str(ph).isdigit():
                        phases.add(int(ph))
    except Exception:
        pass
    if not phases:
        phases = {0}

    availability = {"frames": sorted(frames), "phases": sorted(phases)}
    return preview, availability