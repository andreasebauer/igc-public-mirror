from dataclasses import dataclass
from typing import Optional, List, Dict
import numpy as np

from igc.sim.coupler import Coupler
from igc.sim.injector import Injector
from igc.sim.saver import save_frame, HeaderOptions
from igc.common.paths import ensure_dirs

@dataclass(frozen=True)
class IntegratorConfig:
    substeps_per_at: int = 48
    dt_per_at: float = 1.0        # normalized; dt = dt_per_at / substeps_per_at
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
                # reversible ψ–π (local SHO with gentle damping on pi)
                # ψ' = π ; π' = -ψ - γ π  (γ = small damping -> here coupled into step below)
                pi += dt * (-psi)
                psi += dt * pi

                # 2) dissipative local feedback (simple, bounded)
                # eta: -λ_eta * eta + C * |pi|
                eta += dt * (-K["lambda_eta"] * eta + K["C_pi_to_eta"] * np.abs(pi))

                # phi_field: -λ_phi * phi + C * |eta|   (bounded to [0, +inf), but we’ll also clip later)
                phi_field += dt * (-K["lambda_phi"] * phi_field + K["C_eta_to_phi"] * np.abs(eta))
                # clip to non-negative (permission can't be negative)
                np.maximum(phi_field, 0.0, out=phi_field)

                # 3) (pp0) injection is off; still call maybe_apply (it will no-op)
                _events = self.injector.maybe_apply(at, sub, tact_phase, psi, pi, eta, phi_field)

            # end of At
            at += 1

            # Save at end of each At (simple stride 1 here)
            save_frame(store, sim_label, frame,
                       psi=psi, pi=pi, eta=eta, phi_field=phi_field,
                       at=at, substeps_per_at=substeps, tact_phase=0.0,
                       header_opts=HeaderOptions(write_stats=header_stats))
            frame += 1
