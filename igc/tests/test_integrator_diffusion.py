# igc/tests/test_integrator_diffusion.py
"""
Test that the Integrator's PDE layer actually uses the spatial diffusion term
D_eta * ∇²η to spread memory from a localized peak.

We set:
- η = 1 at the center of a 7x7x7 grid, 0 elsewhere
- ψ = 0, π = 0, φ = 1 everywhere
- D_eta > 0, D_psi = D_phi = 0
- lambda_eta = lambda_phi = 0, C_* = 0

After one At (with a single substep), the 6 direct neighbors of the center
should have η > 0 if the diffusion term is wired correctly.
"""

import numpy as np
from pathlib import Path

from igc.sim.grid_constructor import GridConfig, build_humming_grid
from igc.sim.coupler import CouplerConfig, Coupler
from igc.sim.injector import Injector
from igc.sim.integrator import IntegratorConfig, Integrator


def run_one_at_diffusion(
    D_eta: float = 0.1,
    dt_per_at: float = 0.1,
    grid_n: int = 7,
) -> np.ndarray:
    """
    Helper: run the integrator for a single At on a small grid with a delta-like
    η at the center and only D_eta > 0. Returns the η field after one At.
    """
    Nx = Ny = Nz = grid_n
    cx = cy = cz = grid_n // 2

    cfg = GridConfig(shape=(Nx, Ny, Nz), periodic=True)
    state = build_humming_grid(cfg, substeps_per_at=1, initial_at=0)

    # Overwrite humming initial conditions:
    # eta: delta at center; psi = 0; pi = 0; phi = 1
    state.psi[...] = 0.0
    state.pi[...] = 0.0
    state.eta[...] = 0.0
    state.eta[cx, cy, cz] = 1.0
    state.phi_field[...] = 1.0

    # Coupler: only D_eta > 0; all others zero
    coupler_cfg = CouplerConfig(
        D_psi=0.0,
        D_eta=D_eta,
        D_phi=0.0,
        C_pi_to_eta=0.0,
        C_eta_to_phi=0.0,
        lambda_eta=0.0,
        lambda_phi=0.0,
        gate="linear",
    )
    coupler = Coupler(coupler_cfg)

    injector = Injector([])

    integ_cfg = IntegratorConfig(
        substeps_per_at=1,
        dt_per_at=dt_per_at,
        dx=1.0,
        stride_frames_at=1,
    )
    integ = Integrator(coupler, injector, integ_cfg)

    # Use a temporary store path (we ignore the saved files)
    store_path = Path("/tmp/igc_diffusion_test_store")
    store_path.mkdir(parents=True, exist_ok=True)

    # Single At: at_start=0, at_end=1
    integ.run(
        store=store_path,
        sim_label="TEST_DIFFUSION",
        psi=state.psi,
        pi=state.pi,
        eta=state.eta,
        phi_field=state.phi_field,
        phi_cone=state.phi_cone,
        at_start=0,
        at_end=1,
        save_first_frame=False,
        header_stats=False,
    )

    return state.eta.copy()


def test_integrator_diffusion_neighbors_gain_mass():
    """
    With D_eta > 0 and φ = 1, starting from a delta in η at the center, the 6 direct
    neighbors should acquire positive η after one At.
    """
    grid_n = 7
    eta_after = run_one_at_diffusion(D_eta=0.1, dt_per_at=0.1, grid_n=grid_n)
    cx = cy = cz = grid_n // 2

    center_val = eta_after[cx, cy, cz]
    neighbors = [
        eta_after[cx + 1, cy, cz],
        eta_after[cx - 1, cy, cz],
        eta_after[cx, cy + 1, cz],
        eta_after[cx, cy - 1, cz],
        eta_after[cx, cy, cz + 1],
        eta_after[cx, cy, cz - 1],
    ]

    # Center should still be positive (not all mass instantly diffused away)
    assert center_val > 0.0, f"center η became non-positive: {center_val}"

    # All 6 direct neighbors should have gained some η
    for i, v in enumerate(neighbors):
        assert v > 0.0, f"neighbor {i} did not gain η (value={v})"

    # At least one neighbor should have a reasonably noticeable value
    assert max(neighbors) > 1e-3, f"neighbors too small: {neighbors}"


if __name__ == "__main__":
    test_integrator_diffusion_neighbors_gain_mass()
    print("test_integrator_diffusion_neighbors_gain_mass passed")