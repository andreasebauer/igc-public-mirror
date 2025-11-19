from dataclasses import dataclass
from typing import Optional, List, Dict, Callable
import numpy as np

from igc.sim.operators import (
    laplace_jit,
    div_phi_cone_psi_jit,
    compute_directional_skin_jit,
)
from igc.sim.kernels import pde_substep_jit
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
            phi_cone: np.ndarray,
            at_start: int,
            at_end: int,
            save_first_frame: bool = True,
            header_stats: bool = False,
            on_frame_saved: Optional[Callable[[int, int], None]] = None
        ) -> None:

        substeps = self.cfg.substeps_per_at
        dt = self.cfg.dt_per_at / substeps
        dx = self.cfg.dx
        frame = 0

        # Scratch buffers for fused PDE kernel
        psi_next = np.empty_like(psi)
        pi_next = np.empty_like(pi)
        eta_next = np.empty_like(eta)
        phi_cone_next = np.empty_like(phi_field, shape=(6, *psi.shape)) if phi_cone.ndim == 4 else np.empty_like(phi_cone)
        phi_field_next = np.empty_like(phi_field)

        # Optionally write first frame
        if save_first_frame:
            save_frame(store, sim_label, frame,
                       psi=psi, pi=pi, eta=eta, phi_field=phi_field, phi_cone=phi_cone,
                       at=at_start, substeps_per_at=substeps, tact_phase=0.0,
                       header_opts=HeaderOptions(write_stats=header_stats))
            if on_frame_saved is not None:
                on_frame_saved(frame, at_start)
            frame += 1

        at = at_start
        while at < at_end:
            # Run one At worth of substeps
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

                # 2) fused PDE substep (Numba kernel)
                pde_substep_jit(
                    psi, pi, eta, phi_cone, phi_field,
                    psi_next, pi_next, eta_next, phi_cone_next, phi_field_next,
                    D_psi, D_eta, D_phi,
                    lambda_eta, lambda_phi,
                    C_pi_to_eta, C_eta_to_phi, C_gradpsi_to_phi,
                    dx, dt,
                )

                # 3) swap buffers so updated fields become current for next substep
                psi, psi_next = psi_next, psi
                pi, pi_next = pi_next, pi
                eta, eta_next = eta_next, eta
                phi_cone, phi_cone_next = phi_cone_next, phi_cone
                phi_field, phi_field_next = phi_field_next, phi_field

                # 4) injection (still in Python; may modify fields in place)
                _events = self.injector.maybe_apply(at, sub, tact_phase, psi, pi, eta, phi_field)

            # End of this At
            at += 1

            # Save at end of each At (simple stride 1 here)
            save_frame(
                store,
                sim_label,
                frame,
                psi=psi,
                pi=pi,
                eta=eta,
                phi_field=phi_field,
                phi_cone=phi_cone,
                at=at,
                substeps_per_at=substeps,
                tact_phase=0.0,
                header_opts=HeaderOptions(write_stats=header_stats),
            )
            if on_frame_saved is not None:
                on_frame_saved(frame, at)
            frame += 1
