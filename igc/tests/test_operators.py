# tests/test_operators.py
"""
Basic tests for igc.sim.operators (laplace, div_phi_grad_psi).

These are small, fast sanity checks to catch obvious sign / indexing bugs
in the discrete operators we use in the PDE layer.
"""

import numpy as np

from igc.sim.operators import laplace, div_phi_grad_psi


def test_laplace_delta_center():
    """
    Laplacian of a delta at the center (with periodic BCs) should:
    - be negative at the center
    - be positive and equal at the 6 direct neighbors
    - sum to ~0 over the whole grid
    """
    Nx = Ny = Nz = 5
    a = np.zeros((Nx, Ny, Nz), dtype=float)
    cx = cy = cz = 2  # center index
    a[cx, cy, cz] = 1.0

    dx = 1.0
    L = laplace(a, dx)

    # Center value: (0+0+0+0+0+0 - 6*1) / dx^2 = -6
    assert np.isclose(L[cx, cy, cz], -6.0)

    # 6 direct neighbors should all have the same positive value (= +1)
    neighbors = [
        L[cx + 1, cy, cz],
        L[cx - 1, cy, cz],
        L[cx, cy + 1, cz],
        L[cx, cy - 1, cz],
        L[cx, cy, cz + 1],
        L[cx, cy, cz - 1],
    ]
    for v in neighbors:
        assert np.isclose(v, 1.0)

    # The discrete Laplacian of a compact pattern should sum to ~0
    assert np.isclose(L.sum(), 0.0)


def test_div_phi_grad_psi_zero_when_phi_zero():
    """
    If phi is zero everywhere, ∇·(phi ∇psi) must be zero for any psi.
    """
    Nx = Ny = Nz = 4
    psi = np.random.rand(Nx, Ny, Nz)
    phi = np.zeros_like(psi)
    dx = 1.0

    div_val = div_phi_grad_psi(phi, psi, dx)
    assert np.allclose(div_val, 0.0)


def test_div_phi_grad_psi_zero_for_constant_psi():
    """
    If psi is constant, grad psi = 0, so ∇·(phi ∇psi) must be zero
    for any phi.
    """
    Nx = Ny = Nz = 4
    psi = np.ones((Nx, Ny, Nz), dtype=float) * 3.14  # constant field
    phi = np.random.rand(Nx, Ny, Nz)
    dx = 1.0

    div_val = div_phi_grad_psi(phi, psi, dx)
    assert np.allclose(div_val, 0.0)