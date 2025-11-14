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
from numba import njit, prange

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

def div_phi_cone_psi(phi_cone: np.ndarray, psi: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Directional cone transport operator (prototype).

    phi_cone has shape (6, Nx, Ny, Nz) with direction indices:
      0 = +x, 1 = -x, 2 = +y, 3 = -y, 4 = +z, 5 = -z

    For now we implement a simple directional flux:
      F_x+ = φ_x+(x) * (ψ(x+e_x) - ψ(x))
      F_x- = φ_x-(x) * (ψ(x-e_x) - ψ(x))
    similarly for y,z, and we sum these as a net divergence-like term.

    This is not yet a fully conservative anisotropic discretization, but it
    respects the basic invariants:
      - if φ_cone is zero everywhere, result is zero
      - if ψ is constant, result is zero for any φ_cone
    and provides a clear hook for directional gating.
    """
    # unpack directional cones
    phi_xp = phi_cone[0]
    phi_xm = phi_cone[1]
    phi_yp = phi_cone[2]
    phi_ym = phi_cone[3]
    phi_zp = phi_cone[4]
    phi_zm = phi_cone[5]

    # neighbor ψ values (periodic BCs)
    psi_xp = np.roll(psi, -1, axis=0)
    psi_xm = np.roll(psi,  1, axis=0)
    psi_yp = np.roll(psi, -1, axis=1)
    psi_ym = np.roll(psi,  1, axis=1)
    psi_zp = np.roll(psi, -1, axis=2)
    psi_zm = np.roll(psi,  1, axis=2)

    # directional flux contributions (local net effect)
    flux = (
        phi_xp * (psi_xp - psi)
        + phi_xm * (psi_xm - psi)
        + phi_yp * (psi_yp - psi)
        + phi_ym * (psi_ym - psi)
        + phi_zp * (psi_zp - psi)
        + phi_zm * (psi_zm - psi)
    )

    # scale like a Laplacian term
    return flux / (dx * dx)

def compute_directional_skin(psi: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Compute directional ψ-skin for each triad.

    skin_dir = max(ψ(neighbor_dir) - ψ(here), 0)^2

    Returns an array shape (6, Nx,Ny,Nz):

      index 0 = +x
      index 1 = -x
      index 2 = +y
      index 3 = -y
      index 4 = +z
      index 5 = -z

    Properties:
      - if ψ is constant, skin = 0  for all directions
      - if ψ has a jump only in +x, only skin_xp is nonzero
      - directional skin allows φ_cone to respond asymmetrically
    """
    # neighbor ψ values (periodic BCs)
    psi_xp = np.roll(psi, -1, axis=0)
    psi_xm = np.roll(psi,  1, axis=0)
    psi_yp = np.roll(psi, -1, axis=1)
    psi_ym = np.roll(psi,  1, axis=1)
    psi_zp = np.roll(psi, -1, axis=2)
    psi_zm = np.roll(psi,  1, axis=2)

    # directional differences
    dx_xp = psi_xp - psi
    dx_xm = psi_xm - psi
    dx_yp = psi_yp - psi
    dx_ym = psi_ym - psi
    dx_zp = psi_zp - psi
    dx_zm = psi_zm - psi

    # positive directional skin: max(Δψ,0)^2
    skin_xp = np.maximum(dx_xp, 0.0)**2
    skin_xm = np.maximum(dx_xm, 0.0)**2
    skin_yp = np.maximum(dx_yp, 0.0)**2
    skin_ym = np.maximum(dx_ym, 0.0)**2
    skin_zp = np.maximum(dx_zp, 0.0)**2
    skin_zm = np.maximum(dx_zm, 0.0)**2

    return np.stack(
        [skin_xp, skin_xm, skin_yp, skin_ym, skin_zp, skin_zm],
        axis=0
    )
@njit(parallel=True)
def laplace_jit(a: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Numba-accelerated 3D 6-neighbour Laplacian with periodic BCs.
    Mirrors laplace(a, dx) but uses explicit loops for speed.
    """
    Nx, Ny, Nz = a.shape
    out = np.empty_like(a)
    inv_dx2 = 1.0 / (dx * dx)
    for i in prange(Nx):
        ip = 0 if i + 1 == Nx else i + 1
        im = Nx - 1 if i == 0 else i - 1
        for j in range(Ny):
            jp = 0 if j + 1 == Ny else j + 1
            jm = Ny - 1 if j == 0 else j - 1
            for k in range(Nz):
                kp = 0 if k + 1 == Nz else k + 1
                km = Nz - 1 if k == 0 else k - 1
                out[i, j, k] = (
                    a[ip, j, k] + a[im, j, k] +
                    a[i, jp, k] + a[i, jm, k] +
                    a[i, j, kp] + a[i, j, km] -
                    6.0 * a[i, j, k]
                ) * inv_dx2
    return out


@njit(parallel=True)
def div_phi_cone_psi_jit(phi_cone: np.ndarray, psi: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Numba-accelerated directional cone transport operator.

    phi_cone has shape (6, Nx, Ny, Nz):
      0 = +x, 1 = -x, 2 = +y, 3 = -y, 4 = +z, 5 = -z

    This mirrors div_phi_cone_psi: local net flux with periodic BCs.
    """
    Nx, Ny, Nz = psi.shape
    out = np.empty_like(psi)
    inv_dx2 = 1.0 / (dx * dx)

    phi_xp = phi_cone[0]
    phi_xm = phi_cone[1]
    phi_yp = phi_cone[2]
    phi_ym = phi_cone[3]
    phi_zp = phi_cone[4]
    phi_zm = phi_cone[5]

    for i in prange(Nx):
        ip = 0 if i + 1 == Nx else i + 1
        im = Nx - 1 if i == 0 else i - 1
        for j in range(Ny):
            jp = 0 if j + 1 == Ny else j + 1
            jm = Ny - 1 if j == 0 else j - 1
            for k in range(Nz):
                kp = 0 if k + 1 == Nz else k + 1
                km = Nz - 1 if k == 0 else k - 1

                psi_here = psi[i, j, k]
                flux = 0.0

                flux += phi_xp[i, j, k] * (psi[ip, j, k] - psi_here)
                flux += phi_xm[i, j, k] * (psi[im, j, k] - psi_here)
                flux += phi_yp[i, j, k] * (psi[i, jp, k] - psi_here)
                flux += phi_ym[i, j, k] * (psi[i, jm, k] - psi_here)
                flux += phi_zp[i, j, k] * (psi[i, j, kp] - psi_here)
                flux += phi_zm[i, j, k] * (psi[i, j, km] - psi_here)

                out[i, j, k] = flux * inv_dx2

    return out


@njit(parallel=True)
def compute_directional_skin_jit(psi: np.ndarray, dx: float = 1.0) -> np.ndarray:
    """
    Numba-accelerated directional ψ-skin:

      skin_dir = max(ψ(neighbor_dir) - ψ(here), 0)^2

    Returns shape (6, Nx,Ny,Nz) with same direction indexing as phi_cone:
      0 = +x, 1 = -x, 2 = +y, 3 = -y, 4 = +z, 5 = -z
    """
    Nx, Ny, Nz = psi.shape
    skin = np.zeros((6, Nx, Ny, Nz), dtype=psi.dtype)

    for i in prange(Nx):
        ip = 0 if i + 1 == Nx else i + 1
        im = Nx - 1 if i == 0 else i - 1
        for j in range(Ny):
            jp = 0 if j + 1 == Ny else j + 1
            jm = Ny - 1 if j == 0 else j - 1
            for k in range(Nz):
                kp = 0 if k + 1 == Nz else k + 1
                km = Nz - 1 if k == 0 else k - 1

                here = psi[i, j, k]

                dx_xp = psi[ip, j, k] - here
                dx_xm = psi[im, j, k] - here
                dx_yp = psi[i, jp, k] - here
                dx_ym = psi[i, jm, k] - here
                dx_zp = psi[i, j, kp] - here
                dx_zm = psi[i, j, km] - here

                v = dx_xp
                skin[0, i, j, k] = v * v if v > 0.0 else 0.0
                v = dx_xm
                skin[1, i, j, k] = v * v if v > 0.0 else 0.0
                v = dx_yp
                skin[2, i, j, k] = v * v if v > 0.0 else 0.0
                v = dx_ym
                skin[3, i, j, k] = v * v if v > 0.0 else 0.0
                v = dx_zp
                skin[4, i, j, k] = v * v if v > 0.0 else 0.0
                v = dx_zm
                skin[5, i, j, k] = v * v if v > 0.0 else 0.0

    return skin