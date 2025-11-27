# igc/sim/kernels.py
"""
Numba-accelerated fused PDE kernel for one IG substep.

This performs, in one pass over the grid:
  - cone-gated ψ transport (using φ_cone)
  - ψ–π reversible dynamics
  - η halo diffusion + decay + |π|-source
  - directional cone update for φ_cone[d]
  - scalar φ_field envelope (max over cones)

All updates are written into *_next arrays; caller is responsible
for swapping buffers between substeps.
"""

from __future__ import annotations
import numpy as np
from numba import njit, prange


@njit(parallel=True)
def pde_substep_jit(
    psi, pi, eta, phi_cone, phi_field,
    psi_next, pi_next, eta_next, phi_cone_next, phi_field_next,
    D_psi, D_eta, D_phi,
    lambda_eta, lambda_phi, lambda_psi,
    C_pi_to_eta, C_eta_to_phi, C_gradpsi_to_phi,
    C_psi_to_eta, C_eta_to_psi, C_phi_to_psi,
    gamma_pi,    
    k_psi_restore,
    dx, dt,
):
    Nx, Ny, Nz = psi.shape
    inv_dx2 = 1.0 / (dx * dx)
    half_dx_inv = 0.5 / dx

    for i in prange(Nx):
        ip = 0 if i + 1 == Nx else i + 1
        im = Nx - 1 if i == 0 else i - 1
        for j in range(Ny):
            jp = 0 if j + 1 == Ny else j + 1
            jm = Ny - 1 if j == 0 else j - 1
            for k in range(Nz):
                kp = 0 if k + 1 == Nz else k + 1
                km = Nz - 1 if k == 0 else k - 1

                # --- Local fields ---
                psi_here = psi[i, j, k]
                pi_here  = pi[i, j, k]
                eta_here = eta[i, j, k]

                # neighbors of psi
                psi_ip = psi[ip, j, k]; psi_im = psi[im, j, k]
                psi_jp = psi[i, jp, k]; psi_jm = psi[i, jm, k]
                psi_kp = psi[i, j, kp]; psi_km = psi[i, j, km]

                # cone weights
                phi_xp = phi_cone[0, i, j, k]
                phi_xm = phi_cone[1, i, j, k]
                phi_yp = phi_cone[2, i, j, k]
                phi_ym = phi_cone[3, i, j, k]
                phi_zp = phi_cone[4, i, j, k]
                phi_zm = phi_cone[5, i, j, k]

                # --- cone-gated ψ transport (div_phi_cone_psi) ---
                flux = 0.0
                flux += phi_xp * (psi_ip       - psi_here)
                flux += phi_xm * (psi_im       - psi_here)
                flux += phi_yp * (psi_jp       - psi_here)
                flux += phi_ym * (psi_jm       - psi_here)
                flux += phi_zp * (psi_kp       - psi_here)
                flux += phi_zm * (psi_km       - psi_here)
                div_flux_psi = C_phi_to_psi * flux * inv_dx2

                # --- η Laplacian ---
                lap_eta = (
                    eta[ip, j, k] + eta[im, j, k] +
                    eta[i, jp, k] + eta[i, jm, k] +
                    eta[i, j, kp] + eta[i, j, km] -
                    6.0 * eta_here
                ) * inv_dx2

                # --- ∇ψ and |∇ψ|^2 ---
                gx = (psi_ip - psi_im) * half_dx_inv
                gy = (psi_jp - psi_jm) * half_dx_inv
                gz = (psi_kp - psi_km) * half_dx_inv
                grad_sq = gx*gx + gy*gy + gz*gz

                # --- directional ψ-skin (positive part squared) ---
                dx_xp = psi_ip - psi_here
                dx_xm = psi_im - psi_here
                dx_yp = psi_jp - psi_here
                dx_ym = psi_jm - psi_here
                dx_zp = psi_kp - psi_here
                dx_zm = psi_km - psi_here

                skin_xp = dx_xp*dx_xp if dx_xp > 0.0 else 0.0
                skin_xm = dx_xm*dx_xm if dx_xm > 0.0 else 0.0
                skin_yp = dx_yp*dx_yp if dx_yp > 0.0 else 0.0
                skin_ym = dx_ym*dx_ym if dx_ym > 0.0 else 0.0
                skin_zp = dx_zp*dx_zp if dx_zp > 0.0 else 0.0
                skin_zm = dx_zm*dx_zm if dx_zm > 0.0 else 0.0

                # --- π, ψ updates (reversible SHO + gated transport + η→ψ coupling) ---
                pi_new  = pi_here + dt * (
                    -k_psi_restore * psi_here
                    - lambda_psi * pi_here         # damping on π
                    + gamma_pi                     # constant drive term                    
                    + D_psi * div_flux_psi
                    + C_eta_to_psi * eta_here
                )
                psi_new = psi_here + dt * pi_new

                # --- η update (halo) ---
                eta_new = eta_here + dt * (
                    D_eta * lap_eta
                    - lambda_eta * eta_here
                    + C_pi_to_eta * abs(pi_new)
                    + C_psi_to_eta * abs(psi_here)
                )

                # --- directional cone updates ---
                phi_new0 = phi_cone[0, i, j, k]
                phi_new1 = phi_cone[1, i, j, k]
                phi_new2 = phi_cone[2, i, j, k]
                phi_new3 = phi_cone[3, i, j, k]
                phi_new4 = phi_cone[4, i, j, k]
                phi_new5 = phi_cone[5, i, j, k]

                # helper: update one cone direction
                # x+
                lap_phi0 = (
                    phi_cone[0, ip, j, k] + phi_cone[0, im, j, k] +
                    phi_cone[0, i, jp, k] + phi_cone[0, i, jm, k] +
                    phi_cone[0, i, j, kp] + phi_cone[0, i, j, km] -
                    6.0 * phi_new0
                ) * inv_dx2
                val = phi_new0 + dt * (
                    D_phi * lap_phi0
                    - lambda_phi * phi_new0
                    + C_eta_to_phi * abs(eta_new)
                    + C_gradpsi_to_phi * skin_xp
                )
                phi_new0 = 0.0 if val < 0.0 else val

                # x-
                lap_phi1 = (
                    phi_cone[1, ip, j, k] + phi_cone[1, im, j, k] +
                    phi_cone[1, i, jp, k] + phi_cone[1, i, jm, k] +
                    phi_cone[1, i, j, kp] + phi_cone[1, i, j, km] -
                    6.0 * phi_new1
                ) * inv_dx2
                val = phi_new1 + dt * (
                    D_phi * lap_phi1
                    - lambda_phi * phi_new1
                    + C_eta_to_phi * abs(eta_new)
                    + C_gradpsi_to_phi * skin_xm
                )
                phi_new1 = 0.0 if val < 0.0 else val

                # y+
                lap_phi2 = (
                    phi_cone[2, ip, j, k] + phi_cone[2, im, j, k] +
                    phi_cone[2, i, jp, k] + phi_cone[2, i, jm, k] +
                    phi_cone[2, i, j, kp] + phi_cone[2, i, j, km] -
                    6.0 * phi_new2
                ) * inv_dx2
                val = phi_new2 + dt * (
                    D_phi * lap_phi2
                    - lambda_phi * phi_new2
                    + C_eta_to_phi * abs(eta_new)
                    + C_gradpsi_to_phi * skin_yp
                )
                phi_new2 = 0.0 if val < 0.0 else val

                # y-
                lap_phi3 = (
                    phi_cone[3, ip, j, k] + phi_cone[3, im, j, k] +
                    phi_cone[3, i, jp, k] + phi_cone[3, i, jm, k] +
                    phi_cone[3, i, j, kp] + phi_cone[3, i, j, km] -
                    6.0 * phi_new3
                ) * inv_dx2
                val = phi_new3 + dt * (
                    D_phi * lap_phi3
                    - lambda_phi * phi_new3
                    + C_eta_to_phi * abs(eta_new)
                    + C_gradpsi_to_phi * skin_ym
                )
                phi_new3 = 0.0 if val < 0.0 else val

                # z+
                lap_phi4 = (
                    phi_cone[4, ip, j, k] + phi_cone[4, im, j, k] +
                    phi_cone[4, i, jp, k] + phi_cone[4, i, jm, k] +
                    phi_cone[4, i, j, kp] + phi_cone[4, i, j, km] -
                    6.0 * phi_new4
                ) * inv_dx2
                val = phi_new4 + dt * (
                    D_phi * lap_phi4
                    - lambda_phi * phi_new4
                    + C_eta_to_phi * abs(eta_new)
                    + C_gradpsi_to_phi * skin_zp
                )
                phi_new4 = 0.0 if val < 0.0 else val

                # z-
                lap_phi5 = (
                    phi_cone[5, ip, j, k] + phi_cone[5, im, j, k] +
                    phi_cone[5, i, jp, k] + phi_cone[5, i, jm, k] +
                    phi_cone[5, i, j, kp] + phi_cone[5, i, j, km] -
                    6.0 * phi_new5
                ) * inv_dx2
                val = phi_new5 + dt * (
                    D_phi * lap_phi5
                    - lambda_phi * phi_new5
                    + C_eta_to_phi * abs(eta_new)
                    + C_gradpsi_to_phi * skin_zm
                )
                phi_new5 = 0.0 if val < 0.0 else val

                # write cones
                phi_cone_next[0, i, j, k] = phi_new0
                phi_cone_next[1, i, j, k] = phi_new1
                phi_cone_next[2, i, j, k] = phi_new2
                phi_cone_next[3, i, j, k] = phi_new3
                phi_cone_next[4, i, j, k] = phi_new4
                phi_cone_next[5, i, j, k] = phi_new5

                # scalar envelope φ_field
                phi_f = phi_new0
                if phi_new1 > phi_f: phi_f = phi_new1
                if phi_new2 > phi_f: phi_f = phi_new2
                if phi_new3 > phi_f: phi_f = phi_new3
                if phi_new4 > phi_f: phi_f = phi_new4
                if phi_new5 > phi_f: phi_f = phi_new5
                phi_field_next[i, j, k] = phi_f

                # store updated ψ, π, η
                psi_next[i, j, k] = psi_new
                pi_next[i, j, k]  = pi_new
                eta_next[i, j, k] = eta_new
                