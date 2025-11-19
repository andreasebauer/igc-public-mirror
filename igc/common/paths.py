from pathlib import Path
from typing import Tuple

def sim_root(store: Path, sim_label: str) -> Path:
    """
    Canonical simulation root. OE already provides sim_label in the form
    "<label>/<tt>", so we must not prepend 'Sim_' anymore.
    """
    # sim_label may already include subdirectories (e.g. "A1/20250120_0815")
    return store / sim_label

def frame_dir(store: Path, sim_label: str, frame: int) -> Path:
    return sim_root(store, sim_label) / f"Frame_{frame:04d}"

def ensure_dirs(store: Path, sim_label: str, frame: int) -> Tuple[Path, Path]:
    sroot = sim_root(store, sim_label)
    fdir  = frame_dir(store, sim_label, frame)
    sroot.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    return sroot, fdir
