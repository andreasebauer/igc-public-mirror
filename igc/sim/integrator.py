from dataclasses import dataclass
from queue import Queue
from threading import Thread
from typing import Optional, List, Dict, Callable
import numpy as np
# from time import perf_counter
from igc.sim.operators import (
    laplace_jit,
    div_phi_cone_psi_jit,
    compute_directional_skin_jit,
)
from igc.sim.kernels import pde_substep_jit
from igc.sim.coupler import Coupler
from igc.sim.injector import Injector
from igc.sim.saver import save_frame, save_substep, HeaderOptions
from igc.common.paths import ensure_dirs

@dataclass(frozen=True)
class IntegratorConfig:
    substeps_per_at: int = 48
    dt_per_at: float = 1.0        # normalized; dt = dt_per_at / substeps_per_at
    dx: float = 1.0               # spatial step (for Laplacians / transport; default 1.0)
    stride_frames_at: int = 1     # save every N At (we’ll use 1 here)
    save_substeps: bool = False        # if True, allow substep snapshots/diagnostics
    substep_save_stride: int = 0       # save every N-th substep (0 = none, 1 = all)

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
        
        # Timing accumulators per frame
        # pde_times: Dict[int, float] = {}
        # writer_times: Dict[int, float] = {}

        # Helper to push timing info to OE viewer
        # def _timelog(msg: str):
        #     try:
        #         # two-level import to avoid circular imports
        #         from igc.gui.sim_flow import _simlog_append
        #         # viewer logs expect (app, sim_id, message); Integrator does not know sim_id,
        #         # so timelog only prints to stdout; OE runner will attach times via on_frame_saved.
        #         print(f"[TIMING] {msg}")
        #     except Exception:
        #         pass        

        # Asynchronous writer: queue frames for background saving
        write_queue: "Queue[tuple]" = Queue(maxsize=2)

        def _writer():
            """
            Dedicated writer thread: pulls frame data from write_queue and writes
            them to disk using save_frame(). Calls on_frame_saved() when finished.
            """
            while True:
                item = write_queue.get()
                if item is None:
                    break
                (frame_idx, at_val, psi_w, pi_w, eta_w, phi_field_w, phi_cone_w) = item
        #         wt0 = perf_counter()
                save_frame(
                    store,
                    sim_label,
                    frame_idx,
                    psi=psi_w,
                    pi=pi_w,
                    eta=eta_w,
                    phi_field=phi_field_w,
                    phi_cone=phi_cone_w,
                    at=at_val,
                    substeps_per_at=substeps,
                    tact_phase=0.0,
                    header_opts=HeaderOptions(write_stats=header_stats),
                )

                if on_frame_saved is not None:
                    on_frame_saved(frame_idx, at_val)
        #         writer_times[frame_idx] = perf_counter() - wt0
        #         _timelog(f"writer frame {frame_idx} runtime {writer_times[frame_idx]:.3f}s")
                write_queue.task_done()

        writer_thread = Thread(target=_writer, daemon=True)
        writer_thread.start()

        # Scratch buffers for fused PDE kernel
        psi_next = np.empty_like(psi)
        pi_next = np.empty_like(pi)
        eta_next = np.empty_like(eta)
        phi_cone_next = np.empty_like(phi_field, shape=(6, *psi.shape)) if phi_cone.ndim == 4 else np.empty_like(phi_cone)
        phi_field_next = np.empty_like(phi_field)

        # Optionally write first frame
        if save_first_frame:
            write_queue.put((
                frame,
                at_start,
                psi.copy(),
                pi.copy(),
                eta.copy(),
                phi_field.copy(),
                phi_cone.copy(),
            ))
            frame += 1
        #     _timelog(f"PDE frame {frame-1} runtime {pde_times.get(frame-1, 0.0):.3f}s")
        #     _timelog(f"Writer frame {frame-1} runtime {writer_times.get(frame-1, 0.0):.3f}s")
        at = at_start
        while at < at_end:
            # Start PDE timer for this frame
        #     pde_t0 = perf_counter()            
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
                k_psi_restore = float(K.get("k_psi_restore", 1.0))                
                # gradient → phi coupling (for now fixed; later can be exposed via CouplerConfig)
                C_gradpsi_to_phi = 1.0

                # 2) fused PDE substep (Numba kernel)
                pde_substep_jit(
                    psi, pi, eta, phi_cone, phi_field,
                    psi_next, pi_next, eta_next, phi_cone_next, phi_field_next,
                    D_psi, D_eta, D_phi,
                    lambda_eta, lambda_phi,
                    C_pi_to_eta, C_eta_to_phi, C_gradpsi_to_phi,
                    k_psi_restore,
                    dx, dt,
                )

                # 2b) optionally save this substep for diagnostics (before buffer swap)
                if self.cfg.save_substeps and self.cfg.substep_save_stride > 0:
                    if (sub % self.cfg.substep_save_stride) == 0:
                        save_substep(
                            store,
                            sim_label,
                            frame,
                            sub,
                            psi=psi_next,
                            pi=pi_next,
                            eta=eta_next,
                            phi_field=phi_field_next,
                            phi_cone=phi_cone_next,
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

            # Record PDE time for this frame
            # pde_times[frame] = perf_counter() - pde_t0

            # Save at end of each At (simple stride 1 here)
            write_queue.put((
                frame,
                at,
                psi.copy(),
                pi.copy(),
                eta.copy(),
                phi_field.copy(),
                phi_cone.copy(),
            ))
            frame += 1
        #     _timelog(f"PDE frame {frame-1} runtime {pde_times.get(frame-1, 0.0):.3f}s")
        #     _timelog(f"Writer frame {frame-1} runtime {writer_times.get(frame-1, 0.0):.3f}s")
        # Finish writer thread
        write_queue.put(None)
        writer_thread.join()

