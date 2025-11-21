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
    sweep_info = None

    if not os.path.isfile(meta_path):
        # If this looks like a Sweep root, try to derive sim + sweep info from member runs.
        if "/Sweep/" in abs_path:
            member_meta_paths = []
            try:
                for entry in os.scandir(abs_path):
                    if not entry.is_dir():
                        continue
                    cand = os.path.join(entry.path, "sim_meta.json")
                    if os.path.isfile(cand):
                        member_meta_paths.append(cand)
            except Exception:
                member_meta_paths = []

            if not member_meta_paths:
                raise ValueError("sim_meta.json not found in selected folder.")

            # Use the first member as base sim
            meta_path = member_meta_paths[0]

            # Compute sweep info (for d_psi) across all members, if available
            d_psi_vals = []
            for mp in member_meta_paths:
                try:
                    with open(mp, "r") as fh:
                        data = json.load(fh)
                    v = data.get("d_psi")
                    if isinstance(v, (int, float, str)):
                        try:
                            d_psi_vals.append(float(v))
                        except Exception:
                            pass
                except Exception:
                    continue

            d_psi_vals = sorted(d_psi_vals)
            sweep_count = len(d_psi_vals)
            d_psi_start = d_psi_vals[0] if sweep_count >= 1 else None
            d_psi_end = d_psi_vals[-1] if sweep_count >= 1 else None
            if sweep_count > 1 and d_psi_start is not None and d_psi_end is not None:
                d_psi_step = (d_psi_end - d_psi_start) / (sweep_count - 1)
            else:
                d_psi_step = None

            sweep_info = {
                "_is_sweep": True,
                "_sweep_root": abs_path,
                "_sweep_count": sweep_count,
                "_sweep_d_psi_start": d_psi_start,
                "_sweep_d_psi_end": d_psi_end,
                "_sweep_d_psi_step": d_psi_step,
            }
        else:
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

    if sweep_info:
        try:
            preview.update(sweep_info)
        except Exception:
            # Do not let sweep metadata break preview rendering
            pass    

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
