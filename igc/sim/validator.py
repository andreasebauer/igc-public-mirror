# igc/sim/validator.py
"""
Basic stability / sanity checks for simulation configs.

This does not prove full PDE stability, but catches obviously unsafe
combinations of (dt, dx, D_*, lambda_*).
"""

from __future__ import annotations

from typing import List
from igc.sim.grid_constructor import GridConfig
from igc.sim.coupler import CouplerConfig
from igc.sim.integrator import IntegratorConfig


def validate_sim_config(grid_cfg: GridConfig,
                        integ_cfg: IntegratorConfig,
                        coup_cfg: CouplerConfig) -> None:
    """
    Raise ValueError if the configuration is obviously unstable or nonsensical.

    Checks:
    - grid dims > 0
    - substeps_per_at > 0, dt > 0
    - decay stability: lambda * dt <= 2  (explicit Euler for x' = -lambda x)
    - diffusion CFL: dt <= dx^2 / (2 * d * D_max) with d=3, if any D_* > 0
    """
    errs: List[str] = []

    # Grid sanity
    Nx, Ny, Nz = grid_cfg.shape
    if Nx <= 0 or Ny <= 0 or Nz <= 0:
        errs.append(f"invalid grid shape {grid_cfg.shape}; all dims must be > 0")

    # Time step sanity
    substeps = integ_cfg.substeps_per_at
    if substeps <= 0:
        errs.append(f"substeps_per_at must be > 0 (got {substeps})")
    dt_per_at = float(integ_cfg.dt_per_at)
    if dt_per_at <= 0.0:
        errs.append(f"dt_per_at must be > 0 (got {dt_per_at})")
    else:
        dt = dt_per_at / substeps if substeps > 0 else 0.0
        if dt <= 0.0:
            errs.append(f"effective dt must be > 0 (got {dt})")
    dx = float(integ_cfg.dx)

    # Only proceed with numeric checks if we have a positive dt
    if not errs:
        dt = dt_per_at / substeps

        # Decay stability: lambda * dt <= 2 for simple x' = -lambda x
        if coup_cfg.lambda_eta > 0.0 and coup_cfg.lambda_eta * dt > 2.0:
            errs.append(
                f"lambda_eta * dt = {coup_cfg.lambda_eta * dt:.3g} > 2; "
                f"explicit Euler may be unstable (lambda_eta={coup_cfg.lambda_eta}, dt={dt:.3g})"
            )
        if coup_cfg.lambda_phi > 0.0 and coup_cfg.lambda_phi * dt > 2.0:
            errs.append(
                f"lambda_phi * dt = {coup_cfg.lambda_phi * dt:.3g} > 2; "
                f"explicit Euler may be unstable (lambda_phi={coup_cfg.lambda_phi}, dt={dt:.3g})"
            )

        # Diffusion CFL: dt <= dx^2 / (2 * d * D_max), d=3
        D_max = max(coup_cfg.D_psi, coup_cfg.D_eta, coup_cfg.D_phi)
        if D_max > 0.0 and dx > 0.0:
            d = 3.0
            dt_max = (dx * dx) / (2.0 * d * D_max)
            if dt > dt_max:
                errs.append(
                    f"dt={dt:.3g} exceeds diffusion CFL limit dt_max={dt_max:.3g} "
                    f"for D_max={D_max:.3g}, dx={dx:.3g} (explicit scheme may blow up)"
                )

    if errs:
        # Join all errors into a single message so callers can surface it to UI / logs.
        raise ValueError("; ".join(errs))