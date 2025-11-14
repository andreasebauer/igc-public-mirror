# igc/tests/test_sim_hum_and_dna.py
"""
Simulation-level sanity tests for the cone-based IG integrator:

1) Humming grid (true IG vacuum):
   - build_humming_grid should produce:
       psi ≈ 0 (code baseline),
       pi tiny lawful hum,
       eta ≈ 1e-12,
       phi_field ≈ 1e-12,
       phi_cone ≈ 1e-12.

   Metrics: RMS of pi, |∇ψ|, mean eta, mean phi_field, mean / anisotropy of phi_cone.

2) DNA injection test:
   - Two runs (control vs DNA=3.14e-7 at At=3, substep ~13/48).
   - Evolve for 20 Ats (fast enough for tests).
   - Metrics: ψ variance, ψ max, η total, phi_cone anisotropy for both runs.
   - Assert that the DNA run deviates non-trivially from control;
     print metrics so they can be inspected.
"""

import numpy as np
from pathlib import Path

from igc.sim.grid_constructor import GridConfig, build_humming_grid
from igc.sim.coupler import CouplerConfig, Coupler
from igc.sim.injector import Injector, InjectionEvent
from igc.sim.integrator import IntegratorConfig, Integrator
from igc.sim.operators import laplace, compute_directional_skin


def _grad_sq(psi: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """Helper: compute |∇ψ|^2 with periodic BCs."""
    psi_xp = np.roll(psi, -1, axis=0)
    psi_xm = np.roll(psi,  1, axis=0)
    psi_yp = np.roll(psi, -1, axis=1)
    psi_ym = np.roll(psi,  1, axis=1)
    psi_zp = np.roll(psi, -1, axis=2)
    psi_zm = np.roll(psi,  1, axis=2)

    gx = (psi_xp - psi_xm) / (2.0 * dx)
    gy = (psi_yp - psi_ym) / (2.0 * dx)
    gz = (psi_zp - psi_zm) / (2.0 * dx)
    return gx*gx + gy*gy + gz*gz


def test_humming_grid_true_ig_vacuum():
    """
    Confirm that build_humming_grid creates the true IG vacuum and print basic metrics.
    """
    Nx = Ny = Nz = 8
    cfg = GridConfig(shape=(Nx, Ny, Nz), periodic=True)
    substeps = 48
    state = build_humming_grid(cfg, substeps_per_at=substeps, initial_at=0)

    psi = state.psi
    pi = state.pi
    eta = state.eta
    phi_field = state.phi_field
    phi_cone = state.phi_cone

    # Basic value checks
    assert np.allclose(psi, 0.0)

    max_pi = float(np.max(np.abs(pi)))
    std_pi = float(np.std(pi))
    assert max_pi > 0.0
    assert max_pi < 1e-9
    assert std_pi > 0.0

    assert np.allclose(eta, 1e-12)
    assert np.allclose(phi_field, 1e-12)
    assert phi_cone.shape[0] == 6
    for d in range(6):
        assert np.allclose(phi_cone[d], 1e-12)

    # Metrics
    pi_rms = float(np.sqrt(np.mean(pi**2)))
    gradpsi_sq = _grad_sq(psi, dx=1.0)
    gradpsi_rms = float(np.sqrt(np.mean(gradpsi_sq)))
    eta_mean = float(np.mean(eta))
    phi_mean = float(np.mean(phi_field))
    phi_dir_means = [float(np.mean(phi_cone[d])) for d in range(6)]
    phi_aniso = float(np.std(phi_dir_means))

    print("\n[Vacuum metrics]")
    print(f"  pi_rms       = {pi_rms:.3e}")
    print(f"  gradpsi_rms  = {gradpsi_rms:.3e}")
    print(f"  eta_mean     = {eta_mean:.3e}")
    print(f"  phi_mean     = {phi_mean:.3e}")
    print(f"  phi_dir_means= {[f'{m:.3e}' for m in phi_dir_means]}")
    print(f"  phi_aniso    = {phi_aniso:.3e}")

    # Loose sanity: gradψ is basically zero, η/φ ~ 1e-12, anisotropy ~0
    assert gradpsi_rms < 1e-15
    assert abs(eta_mean - 1e-12) < 1e-16
    assert abs(phi_mean - 1e-12) < 1e-16
    assert phi_aniso < 1e-15


def _run_with_or_without_dna(inject_dna: bool,
                             grid_n: int = 24,
                             substeps_per_at: int = 48,
                             ats: int = 20):
    """
    Helper: run a sim either with or without a DNA injection at At=3, substep ≈13/48.
    Returns (psi, phi_cone, eta) at final At.
    """
    Nx = Ny = Nz = grid_n
    cfg = GridConfig(shape=(Nx, Ny, Nz), periodic=True)
    state = build_humming_grid(cfg, substeps_per_at=substeps_per_at, initial_at=0)

    psi = state.psi
    pi = state.pi
    eta = state.eta
    phi_field = state.phi_field
    phi_cone = state.phi_cone

    # Coupler: moderate values in cone mode
    coupler_cfg = CouplerConfig(
        D_psi=0.05,
        D_eta=0.02,
        D_phi=0.02,
        C_pi_to_eta=0.5,
        C_eta_to_phi=0.5,
        lambda_eta=0.01,
        lambda_phi=0.01,
        gate="cones",
    )
    coupler = Coupler(coupler_cfg)

    events = []
    if inject_dna:
        cx = Nx // 2
        cy = Ny // 2
        cz = Nz // 2
        dna_amp = 3.14e-7  # 0.000000314

        phase_center = 13.0 / 48.0
        window_half = 1e-3

        events.append(
            InjectionEvent(
                kind="once",
                field="psi",
                amplitude=dna_amp,
                sigma=0.0,
                center=(cx, cy, cz),
                window=(phase_center - window_half, phase_center + window_half),
                repeat=None,
            )
        )
    injector = Injector(events)

    integ_cfg = IntegratorConfig(
        substeps_per_at=substeps_per_at,
        dt_per_at=1.0,
        dx=1.0,
        stride_frames_at=999999,
    )
    integ = Integrator(coupler, injector, integ_cfg)

    store_path = Path("/tmp/igc_sim_hum_dna_test_store")
    store_path.mkdir(parents=True, exist_ok=True)

    integ.run(
        store=store_path,
        sim_label="TEST_DNA" if inject_dna else "TEST_CONTROL",
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

    return psi.copy(), phi_cone.copy(), eta.copy()


def test_dna_injection_metrics():
    """
    Compare control vs DNA runs and print metric differences.

    We check:
      - ψ variance and max amplitude difference
      - total η difference
      - φ_cone directional anisotropy difference
    """
    psi_ctrl, phi_cone_ctrl, eta_ctrl = _run_with_or_without_dna(inject_dna=False, ats=20)
    psi_dna,  phi_cone_dna,  eta_dna  = _run_with_or_without_dna(inject_dna=True,  ats=20)

    # Basic ψ metrics
    var_ctrl = float(np.var(psi_ctrl))
    var_dna  = float(np.var(psi_dna))
    max_ctrl = float(np.max(np.abs(psi_ctrl)))
    max_dna  = float(np.max(np.abs(psi_dna)))

    # η halo metric
    eta_total_ctrl = float(np.sum(eta_ctrl))
    eta_total_dna  = float(np.sum(eta_dna))

    # φ_cone anisotropy
    dir_means_ctrl = [float(np.mean(phi_cone_ctrl[d])) for d in range(6)]
    dir_means_dna  = [float(np.mean(phi_cone_dna[d])) for d in range(6)]
    phi_aniso_ctrl = float(np.std(dir_means_ctrl))
    phi_aniso_dna  = float(np.std(dir_means_dna))

    # Differences
    d_var = var_dna - var_ctrl
    d_max = max_dna - max_ctrl
    d_eta_total = eta_total_dna - eta_total_ctrl
    d_phi_aniso = phi_aniso_dna - phi_aniso_ctrl

    print("\n[DNA vs control metrics]")
    print(f"  ψ_var_ctrl   = {var_ctrl:.3e}")
    print(f"  ψ_var_dna    = {var_dna:.3e}")
    print(f"  Δψ_var       = {d_var:.3e}")
    print(f"  ψ_max_ctrl   = {max_ctrl:.3e}")
    print(f"  ψ_max_dna    = {max_dna:.3e}")
    print(f"  Δψ_max       = {d_max:.3e}")
    print(f"  η_total_ctrl = {eta_total_ctrl:.3e}")
    print(f"  η_total_dna  = {eta_total_dna:.3e}")
    print(f"  Δη_total     = {d_eta_total:.3e}")
    print(f"  φ_aniso_ctrl = {phi_aniso_ctrl:.3e}")
    print(f"  φ_aniso_dna  = {phi_aniso_dna:.3e}")
    print(f"  Δφ_aniso     = {d_phi_aniso:.3e}")

    # ------------------------------
    # Topological metrics: Betti numbers & Euler characteristic
    # ------------------------------
    from gudhi import CubicalComplex

    # Use η field for topology (cleaner than ψ)
    eta_ctrl_f = eta_ctrl.astype(float)
    eta_dna_f  = eta_dna.astype(float)

    cc_ctrl = CubicalComplex(top_dimensional_cells=eta_ctrl_f)
    cc_dna  = CubicalComplex(top_dimensional_cells=eta_dna_f)

    cc_ctrl.compute_persistence()
    cc_dna.compute_persistence()

    betti_ctrl = cc_ctrl.betti_numbers()
    betti_dna  = cc_dna.betti_numbers()

    # Ensure 3 entries (B0, B1, B2)
    while len(betti_ctrl) < 3:
        betti_ctrl.append(0)
    while len(betti_dna) < 3:
        betti_dna.append(0)

    B0_ctrl, B1_ctrl, B2_ctrl = betti_ctrl[:3]
    B0_dna,  B1_dna,  B2_dna  = betti_dna[:3]

    euler_ctrl = B0_ctrl - B1_ctrl + B2_ctrl
    euler_dna  = B0_dna  - B1_dna  + B2_dna

    print("\n[Topological metrics]")
    print(f"  Betti0_ctrl = {B0_ctrl}, Betti0_dna = {B0_dna}")
    print(f"  Betti1_ctrl = {B1_ctrl}, Betti1_dna = {B1_dna}")
    print(f"  Betti2_ctrl = {B2_ctrl}, Betti2_dna = {B2_dna}")
    print(f"  Euler_ctrl  = {euler_ctrl}")
    print(f"  Euler_dna   = {euler_dna}")

    # Topology here is informative (we are still in a single connected spacetime patch);
    # We keep it for inspection but do not assert on it in this early-time test.

    # Very loose sanity checks: something must change in amplitudes / halo / anisotropy.
    assert d_var > 1e-12 or d_max > 1e-9 or d_eta_total > 1e-15 or d_phi_aniso > 1e-15

    