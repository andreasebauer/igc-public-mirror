from pathlib import Path
from typing import Tuple

def sim_root(store: Path, sim_label: str) -> Path:
    return store / f"Sim_{sim_label}"

def frame_dir(store: Path, sim_label: str, frame: int) -> Path:
    return sim_root(store, sim_label) / f"Frame_{frame:04d}"

def ensure_dirs(store: Path, sim_label: str, frame: int) -> Tuple[Path, Path]:
    sroot = sim_root(store, sim_label)
    fdir  = frame_dir(store, sim_label, frame)
    sroot.mkdir(parents=True, exist_ok=True)
    fdir.mkdir(parents=True, exist_ok=True)
    return sroot, fdir
