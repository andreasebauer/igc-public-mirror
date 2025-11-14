# igc/tests/test_integrator_sho.py
"""
Test the reversible ψ–π oscillator core of the Integrator with spatial
terms and dissipative couplings turned off.

We solve:
    ψ' = π
    π' = -ψ

with D_* = 0, lambda_* = 0, C_* = 0, so that η and φ stay constant and
only the local oscillator runs. For small dt_per_at, the numerical
solution should match cos(t), -sin(t) up to a modest tolerance.
"""

import math
from pathlib import Path
import numpy as np

from igc.sim.grid_constructor import GridConfig, build_humming_grid
from igc.sim.coupler import CouplerConfig, Coupler
from igc.sim.injector import Injector
from igc.sim.integrator import IntegratorConfig, Integrator


def run_one_at_sho(dt_per_at: float = 0.25, substeps_per_at: int = 64):
    """
    Helper: run the integrator for a single At on a 1x1x1 grid with
    pure SHO dynamics (no diffusion, no dissipative feedback).
    """
    # 3x3x3 grid; still uniform, but large enough for np.gradient(edge_order=2)
    cfg = GridConfig(shape=(3, 3, 3), periodic=True)
    state = build_humming_grid(cfg, substeps_per_at=substeps_per_at, initial_at=0)

    # Overwrite the humming initial conditions for a clean SHO test:
    # ψ(0) = 1, π(0) = 0, η = 0, φ = 1
    state.psi[...] = 1.0
    state.pi[...] = 0.0
    state.eta[...] = 0.0
    state.phi_field[...] = 1.0

    # Coupler with all spatial and dissipative terms turned off
    coupler_cfg = CouplerConfig(
        D_psi=0.0,
        D_eta=0.0,
        D_phi=0.0,
        C_pi_to_eta=0.0,
        C_eta_to_phi=0.0,
        lambda_eta=0.0,
        lambda_phi=0.0,
        gate="linear",
    )
    coupler = Coupler(coupler_cfg)

    # No injections
    injector = Injector([])

    integ_cfg = IntegratorConfig(
        substeps_per_at=substeps_per_at,
        dt_per_at=dt_per_at,
        dx=1.0,
        stride_frames_at=1,
    )
    integ = Integrator(coupler, injector, integ_cfg)

    # Use a temporary store directory for saver; we won't inspect the files
    store_path = Path("/tmp/igc_sho_test_store")
    store_path.mkdir(parents=True, exist_ok=True)

    # Single At: at_start=0, at_end=1
    integ.run(
        store=store_path,
        sim_label="TEST_SHO",
        psi=state.psi,
        pi=state.pi,
        eta=state.eta,
        phi_field=state.phi_field,
        at_start=0,
        at_end=1,
        save_first_frame=False,
        header_stats=False,
    )

    # After one At, physical time advanced is dt_per_at
    psi_final = state.psi[0, 0, 0]
    pi_final = state.pi[0, 0, 0]
    return psi_final, pi_final


def test_integrator_sho_matches_cos_sin():
    """
    With D_* = 0 and all dissipative couplings zero, the ψ–π integrator
    should approximate:

        ψ(t) ≈ cos(t),  π(t) ≈ -sin(t)

    for t = dt_per_at and sufficiently small dt_per_at.
    """
    dt_per_at = 0.25  # small step for better accuracy
    substeps = 64

    psi_num, pi_num = run_one_at_sho(dt_per_at=dt_per_at, substeps_per_at=substeps)

    # Exact solution at t = dt_per_at for ψ(0)=1, π(0)=0
    t = dt_per_at
    psi_exact = math.cos(t)
    pi_exact = -math.sin(t)

    # Tolerances: loose enough for explicit scheme, tight enough to catch regressions
    assert abs(psi_num - psi_exact) < 1e-2, f"psi_num={psi_num}, psi_exact={psi_exact}"
    assert abs(pi_num - pi_exact) < 1e-2, f"pi_num={pi_num}, pi_exact={pi_exact}"


if __name__ == "__main__":
    # Allow running directly via `python igc/tests/test_integrator_sho.py`
    test_integrator_sho_matches_cos_sin()
    print("test_integrator_sho_matches_cos_sin passed")