import json, os
from typing import Dict, Any, Tuple

ALLOWLIST_PREFIXES = ("/data/simulations",)

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

    # normalize keys for display (do not mutate metadata on disk)
    _id  = meta.get("id") or meta.get("sim.id") or meta.get("sim_id")
    _lbl = meta.get("label") or meta.get("sim.label")
    _desc= meta.get("description") or meta.get("sim.description")

    # normalize label/description/id for display but include full meta
    _id  = meta.get("id") or meta.get("sim.id") or meta.get("sim_id")
    _lbl = meta.get("label") or meta.get("sim.label")
    _desc= meta.get("description") or meta.get("sim.description")

    if _id is not None: meta["id"] = _id
    if _lbl is not None: meta["label"] = _lbl
    if _desc is not None: meta["description"] = _desc

    # expose full meta to preview
    preview = meta

    frames = set()
    phases = set()
    try:
        import re

        def _frame_index(name: str):
            # Current convention: directories like "Frame_0000", "frame0001", etc.
            # Fallback: numeric suffix anywhere in the name.
            m = re.search(r"(\d+)$", name)
            return int(m.group(1)) if m else None

        for entry in os.scandir(abs_path):
            if not entry.is_dir():
                continue
            name = entry.name
            # Accept both "Frame_0000" style and plain numeric dir names, just in case.
            idx = None
            if name.lower().startswith("frame"):
                idx = _frame_index(name)
            elif name.isdigit():
                idx = int(name)
            if idx is not None:
                frames.add(idx)
                # For now, phases are just {0}; if you later introduce per-frame phases,
                # you can refine this to inspect subdirectories.
        if not phases:
            phases = {0}
    except Exception:
        pass

    availability = {"frames": sorted(frames), "phases": sorted(phases)}
    return preview, availability
