"""
GridConstructor: builds the humming, meta-stable vacuum state
- f64 fields: psi, pi, eta, phi_field
- lawful phase diversity via a Kronecker/Fibonacci phase pattern
- phi_field = 1.0 (open), eta ~ 1e-12 (tiny), psi ~ 0, pi tiny lawful
"""
from dataclasses import dataclass
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class GridConfig:
    shape: Tuple[int, int, int]  # (Nx, Ny, Nz) or (Nx, Ny, 1) for 2-D use
    periodic: bool = True

@dataclass
class GridState:
    psi: np.ndarray
    pi: np.ndarray
    eta: np.ndarray
    phi_field: np.ndarray
    # directional cones: shape (6, Nx, Ny, Nz)
    # direction index convention: 0=+x, 1=-x, 2=+y, 3=-y, 4=+z, 5=-z
    phi_cone: np.ndarray
    at: int
    substeps_per_at: int
    tact_phase: float  # in [0,1)

def _kronecker_phases(shape: Tuple[int,int,int]) -> np.ndarray:
    """Create a deterministic, lawful phase diversity map in [0,1).
    Uses golden ratio-based incommensurate slopes."""
    Nx, Ny, Nz = shape
    # Golden ratio and a second incommensurate constant
    phi = (1 + 5**0.5) / 2
    kx, ky, kz = 1/phi, 1/phi**2, (1/phi + 1/phi**2) % 1
    xs = np.arange(Nx)[:, None, None]
    ys = np.arange(Ny)[None, :, None]
    zs = np.arange(Nz)[None, None, :]
    phase = (kx*xs + ky*ys + kz*zs) % 1.0
    return phase

    return phase

def build_humming_grid(cfg: GridConfig,
                       substeps_per_at: int = 48,
                       initial_at: int = 0) -> GridState:
    Nx, Ny, Nz = cfg.shape
    psi = np.zeros((Nx, Ny, Nz), dtype=np.float64)
    pi  = np.zeros_like(psi)
    eta = np.full_like(psi, 1e-12)        # tiny, nonzero memory
    # scalar gate baseline: nearly closed in the true IG vacuum
    phi_field = np.full_like(psi, 1e-12)

    # directional cones: start nearly closed in all 6 directions
    phi_cone = np.full((6, Nx, Ny, Nz), 1e-12, dtype=np.float64)

    # lawful tiny momentum to start the hum, phase-distributed
    phase = _kronecker_phases(cfg.shape)  # [0,1)
    # center ε for momentum; small enough to stay within tiny hum regime
    eps = 1e-12
    pi += eps * np.cos(2*np.pi*phase)

    state = GridState(
        psi=psi,
        pi=pi,
        eta=eta,
        phi_field=phi_field,
        phi_cone=phi_cone,
        at=initial_at,
        substeps_per_at=substeps_per_at,
        tact_phase=0.0,
    )
    return state