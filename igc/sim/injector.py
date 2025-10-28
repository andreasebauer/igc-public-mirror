from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np

@dataclass(frozen=True)
class InjectionEvent:
    kind: str           # "once" | "train"
    field: str          # typically "psi"
    amplitude: float    # epsilon > 0
    sigma: float        # Gaussian radius in voxels (if > 0), 0 = voxel
    center: Tuple[int,int,int]
    window: Tuple[float,float]  # tact phase window [a,b)
    repeat: Optional[Dict] = None  # e.g., {"period_at": 4}

class Injector:
    """Applies deterministic nudges aligned to tact-phase windows."""

    def __init__(self, events: Optional[List[InjectionEvent]] = None):
        self.events = events or []
        # track "once" executions keyed by (at, sub) to avoid duplicates
        self._fired_once = set()

    def maybe_apply(self, at: int, sub: int, tact_phase: float,
                    psi: np.ndarray, pi: np.ndarray, eta: np.ndarray, phi_field: np.ndarray
                    ) -> List[Dict]:
        if not self.events:
            return []  # no injections in pp0

        H, W, Z = psi.shape
        applied = []
        for ev in self.events:
            a, b = ev.window
            in_window = (tact_phase >= a) and (tact_phase < b)
            if not in_window:
                continue

            key = (ev.kind, at, sub, ev.center, ev.amplitude, ev.sigma)
            if ev.kind == "once" and key in self._fired_once:
                continue

            cx, cy, cz = ev.center
            cx = max(0, min(H-1, cx)); cy = max(0, min(W-1, cy)); cz = max(0, min(Z-1, cz))

            if ev.sigma <= 0.0:
                psi[cx, cy, cz] += ev.amplitude
            else:
                # Gaussian kernel write
                r = int(max(1, round(3*ev.sigma)))
                xs = slice(max(0, cx-r), min(H, cx+r+1))
                ys = slice(max(0, cy-r), min(W, cy+r+1))
                zs = slice(max(0, cz-r), min(Z, cz+r+1))
                X, Y, Zz = np.meshgrid(
                    np.arange(xs.start, xs.stop),
                    np.arange(ys.start, ys.stop),
                    np.arange(zs.start, zs.stop),
                    indexing='ij'
                )
                K = np.exp(-((X-cx)**2 + (Y-cy)**2 + (Zz-cz)**2) / (2*ev.sigma**2))
                psi[xs, ys, zs] += ev.amplitude * K

            applied.append({"kind": ev.kind, "field": ev.field,
                            "amp": ev.amplitude, "sigma": ev.sigma,
                            "center": ev.center, "at": at, "sub": sub, "tact_phase": tact_phase})
            if ev.kind == "once":
                self._fired_once.add(key)

        return applied
