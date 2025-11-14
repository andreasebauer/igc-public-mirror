# igc/tests/test_cones_behavior.py
"""
Cone behavior sanity tests.

These are not unit tests in the strict numerical sense; they are "physics sanity"
checks that run a tiny cone-mode simulation and compute some simple metrics so
we can reason about cone dynamics.

We check two things:
  1) With an initial ψ gradient in +x only, φ_cone[+x] grows more than other directions.
  2) phi_field equals the envelope (max) over φ_cone directions.
"""

import numpy as np
from pathlib import Path

from igc.sim.grid_constructor import GridConfig, build_humming_grid
from igc.sim.coupler import CouplerConfig, Coupler
from igc.sim.injector import Injector
from igc.sim.integrator import IntegratorConfig, Integrator


def _run_tiny_cone_sim(grid_n: int = 9,
                       substeps: int = 8,
                       ats: int = 1):
    """
    Run a tiny cone-mode sim on a grid_n^3 box with a simple ψ profile that has
    a gradient only in +x.

    Returns (psi, phi_field, phi_cone, eta) after the run.
    """
    Nx = Ny = Nz = grid_n
    cfg = GridConfig(shape=(Nx, Ny, Nz), periodic=True)
    state = build_humming_grid(cfg, substeps_per_at=substeps, initial_at=0)

    # Overwrite humming initial conditions:
    # ψ has a step in +x at the center slab; π = 0; η = 0; cones all open initially.
    psi = state.psi
    pi = state.pi
    eta = state.eta
    phi_field = state.phi_field
    phi_cone = state.phi_cone

    psi[...] = 0.0
    cx = grid_n // 2
    psi[cx + 1, :, :] = 1.0  # gradient in +x direction only
    pi[...] = 0.0
    eta[...] = 0.0
    phi_field[...] = 1.0
    phi_cone[...] = 1.0

    # Coupler: enable cone and η dynamics, modest strengths
    coupler_cfg = CouplerConfig(
        D_psi=0.05,
        D_eta=0.05,
        D_phi=0.05,
        C_pi_to_eta=0.0,   # keep η quiet here to focus on skin-driven cones
        C_eta_to_phi=0.0,  # no η->φ coupling in this test
        lambda_eta=0.0,
        lambda_phi=0.1,
        gate="cones",
    )
    coupler = Coupler(coupler_cfg)
    injector = Injector([])

    integ_cfg = IntegratorConfig(
        substeps_per_at=substeps,
        dt_per_at=1.0,
        dx=1.0,
        stride_frames_at=999999,  # no intermediate frame saving
    )
    integ = Integrator(coupler, injector, integ_cfg)

    store_path = Path("/tmp/igc_cone_behavior_test_store")
    store_path.mkdir(parents=True, exist_ok=True)

    integ.run(
        store=store_path,
        sim_label="TEST_CONES",
        psi=psi,
        pi=pi,
        eta=eta,
        phi_field=phi_field,
        phi_cone=phi_cone,
        at_start=0,
        at_end=ats,
        save_first_frame=False,
        header_stats=False,
    )

    return psi.copy(), phi_field.copy(), phi_cone.copy(), eta.copy()


def test_cone_directionality_from_skin():
    """
    With an initial ψ gradient only in +x, φ_cone[+x] should increase more than
    other directions on average.
    """
    psi, phi_field, phi_cone, eta = _run_tiny_cone_sim()

    # Direction indices: 0=+x,1=-x,2=+y,3=-y,4=+z,5=-z
    phi_xp = phi_cone[0]
    phi_xm = phi_cone[1]
    phi_yp = phi_cone[2]
    phi_ym = phi_cone[3]
    phi_zp = phi_cone[4]
    phi_zm = phi_cone[5]

    mean_xp = float(np.mean(phi_xp))
    mean_xm = float(np.mean(phi_xm))
    mean_yp = float(np.mean(phi_yp))
    mean_ym = float(np.mean(phi_ym))
    mean_zp = float(np.mean(phi_zp))
    mean_zm = float(np.mean(phi_zm))

    print("\n[cone metrics] mean φ_cone per direction:",
          f"x+={mean_xp:.4f}, x-={mean_xm:.4f}, "
          f"y+={mean_yp:.4f}, y-={mean_ym:.4f}, "
          f"z+={mean_zp:.4f}, z-={mean_zm:.4f}")

    # Basic sanity: x+ should be ≥ all others (since skin is only in +x)
    assert mean_xp >= mean_xm - 1e-6
    assert mean_xp >= mean_yp - 1e-6
    assert mean_xp >= mean_ym - 1e-6
    assert mean_xp >= mean_zp - 1e-6
    assert mean_xp >= mean_zm - 1e-6


def test_phi_field_matches_envelope_of_cones():
    """
    phi_field is supposed to be the scalar envelope (max) over φ_cone directions.
    Check that this holds numerically after the run.
    """
    psi, phi_field, phi_cone, eta = _run_tiny_cone_sim()

    envelope = np.max(phi_cone, axis=0)
    diff = np.abs(phi_field - envelope)
    max_diff = float(np.max(diff))

    print(f"\n[phi_field vs envelope] max |phi_field - max(phi_cone)| = {max_diff:.3e}")
    assert max_diff < 1e-12