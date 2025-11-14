from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

from igc.sim.operators import laplace, div_phi_grad_psi
from igc.sim.coupler import Coupler
from igc.sim.injector import Injector
from igc.sim.saver import save_frame, HeaderOptions
from igc.common.paths import ensure_dirs

@dataclass(frozen=True)
class IntegratorConfig:
    substeps_per_at: int = 48
    dt_per_at: float = 1.0        # normalized; dt = dt_per_at / substeps_per_at
    dx: float = 1.0               # spatial step (for Laplacians / transport; default 1.0)
    stride_frames_at: int = 1     # save every N At (we’ll use 1 here)

class Integrator:
    """Minimal fused integrator: reversible ψ–π + local η/φ feedback."""

    def __init__(self, coupler: Coupler, injector: Injector, cfg: IntegratorConfig):
        self.coupler = coupler
        self.injector = injector
        self.cfg = cfg

    def run(self, *,
            store,
            sim_label: str,
            psi: np.ndarray,
            pi: np.ndarray,
            eta: np.ndarray,
            phi_field: np.ndarray,
            at_start: int,
            at_end: int,
            save_first_frame: bool = True,
            header_stats: bool = False) -> None:

        substeps = self.cfg.substeps_per_at
        dt = self.cfg.dt_per_at / substeps
        dx = self.cfg.dx
        frame = 0

        # Optionally write first frame
        if save_first_frame:
            save_frame(store, sim_label, frame,
                       psi=psi, pi=pi, eta=eta, phi_field=phi_field,
                       at=at_start, substeps_per_at=substeps, tact_phase=0.0,
                       header_opts=HeaderOptions(write_stats=header_stats))
            frame += 1

        at = at_start
        while at < at_end:
            for sub in range(substeps):
                tact_phase = sub / substeps

                # 1) coefficients for this substep
                K = self.coupler.get(at, sub, tact_phase)
                D_psi = K["D_psi"]
                D_eta = K["D_eta"]
                D_phi = K["D_phi"]
                lambda_eta = K["lambda_eta"]
                lambda_phi = K["lambda_phi"]
                C_pi_to_eta = K["C_pi_to_eta"]
                C_eta_to_phi = K["C_eta_to_phi"]
                # gradient → phi coupling (for now fixed; later can be exposed via CouplerConfig)
                C_gradpsi_to_phi = 1.0

                # 1a) spatial operators (periodic BCs)
                div_flux_psi = div_phi_grad_psi(phi_field, psi, dx)  # ∇·(φ ∇ψ)
                lap_eta = laplace(eta, dx)
                lap_phi = laplace(phi_field, dx)

                # gradient of psi and its squared norm for |∇ψ|² term
                grad_psi = np.gradient(psi, dx, edge_order=2)
                grad_psi_norm_sq = grad_psi[0]**2 + grad_psi[1]**2 + grad_psi[2]**2

                # 2) reversible ψ–π core with gated transport
                # ψ' = π ; π' = -ψ + D_psi * ∇·(φ ∇ψ)
                pi += dt * (-psi + D_psi * div_flux_psi)
                psi += dt * pi

                # 3) dissipative feedback with diffusion
                # eta: η' = D_eta ∇²η - λ_eta η + C_pi_to_eta |π|
                eta += dt * (D_eta * lap_eta - lambda_eta * eta + C_pi_to_eta * np.abs(pi))

                # phi_field: φ' = D_phi ∇²φ - λ_phi φ + C_eta_to_phi |η| + C_gradpsi_to_phi |∇ψ|²
                phi_field += dt * (
                    D_phi * lap_phi
                    - lambda_phi * phi_field
                    + C_eta_to_phi * np.abs(eta)
                    + C_gradpsi_to_phi * grad_psi_norm_sq
                )
                # clip to non-negative (permission can't be negative)
                np.maximum(phi_field, 0.0, out=phi_field)

                # 4) (pp0) injection is off; still call maybe_apply (it will no-op)
                _events = self.injector.maybe_apply(at, sub, tact_phase, psi, pi, eta, phi_field)

            # end of At
            at += 1

            # Save at end of each At (simple stride 1 here)
            save_frame(store, sim_label, frame,
                       psi=psi, pi=pi, eta=eta, phi_field=phi_field,
                       at=at, substeps_per_at=substeps, tact_phase=0.0,
                       header_opts=HeaderOptions(write_stats=header_stats))
            frame += 1
