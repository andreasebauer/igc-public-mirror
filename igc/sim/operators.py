# igc/sim/operators.py
"""
Basic discrete operators for the IG grid (periodic BCs, 3D).

- laplace(a, dx):
    3D 6-neighbour Laplacian with periodic boundaries.
    Δa ≈ (a_{+x} + a_{-x} + a_{+y} + a_{-y} + a_{+z} + a_{-z} - 6 a) / dx^2

- div_phi_grad_psi(phi, psi, dx):
    ∇·(phi ∇psi) with central differences and periodic BCs.
    This is the cone-gated transport term: where φ is high, ψ flows; where φ is low, ψ is trapped.
"""

from __future__ import annotations
import numpy as np


def laplace(a: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    3D 6-neighbour Laplacian with periodic boundaries.

    Δa ≈ (a_{+x} + a_{-x} + a_{+y} + a_{-y} + a_{+z} + a_{-z} - 6 a) / dx^2
    """
    axp = np.roll(a, -1, axis=0)
    axm = np.roll(a,  1, axis=0)
    ayp = np.roll(a, -1, axis=1)
    aym = np.roll(a,  1, axis=1)
    azp = np.roll(a, -1, axis=2)
    azm = np.roll(a,  1, axis=2)
    return (axp + axm + ayp + aym + azp + azm - 6.0 * a) / (dx * dx)


def div_phi_grad_psi(phi: np.ndarray, psi: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Compute ∇·(phi ∇psi) with central differences and periodic BCs.

    grad_psi ≈ ((ψ_{+x} - ψ_{-x})/(2dx), ...)
    flux = phi * grad_psi
    div flux ≈ (flux_{+x} - flux_{-x})/(2dx) + ...
    """
    # gradients of psi (central differences, periodic)
    psi_xp = np.roll(psi, -1, axis=0)
    psi_xm = np.roll(psi,  1, axis=0)
    psi_yp = np.roll(psi, -1, axis=1)
    psi_ym = np.roll(psi,  1, axis=1)
    psi_zp = np.roll(psi, -1, axis=2)
    psi_zm = np.roll(psi,  1, axis=2)

    grad_x = (psi_xp - psi_xm) / (2.0 * dx)
    grad_y = (psi_yp - psi_ym) / (2.0 * dx)
    grad_z = (psi_zp - psi_zm) / (2.0 * dx)

    # gated fluxes
    flux_x = phi * grad_x
    flux_y = phi * grad_y
    flux_z = phi * grad_z

    # divergence of flux (central differences on flux, periodic)
    fx_xp = np.roll(flux_x, -1, axis=0)
    fx_xm = np.roll(flux_x,  1, axis=0)
    fy_yp = np.roll(flux_y, -1, axis=1)
    fy_ym = np.roll(flux_y,  1, axis=1)
    fz_zp = np.roll(flux_z, -1, axis=2)
    fz_zm = np.roll(flux_z,  1, axis=2)

    div_x = (fx_xp - fx_xm) / (2.0 * dx)
    div_y = (fy_yp - fy_ym) / (2.0 * dx)
    div_z = (fz_zp - fz_zm) / (2.0 * dx)

    return div_x + div_y + div_z