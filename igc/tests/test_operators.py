# tests/test_operators.py
"""
Basic tests for igc.sim.operators (laplace, div_phi_grad_psi).

These are small, fast sanity checks to catch obvious sign / indexing bugs
in the discrete operators we use in the PDE layer.
"""

import numpy as np

from igc.sim.operators import laplace, div_phi_grad_psi, div_phi_cone_psi

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

    div_val = div_phi_grad_psi(phi, psi, dx)
    assert np.allclose(div_val, 0.0)

def test_compute_directional_skin_zero_for_constant_psi():
    """
    If psi is constant, all directional skin components must be zero.
    """
    Nx = Ny = Nz = 5
    psi = np.ones((Nx,Ny,Nz))
    from igc.sim.operators import compute_directional_skin
    skin = compute_directional_skin(psi, dx=1.0)
    assert np.allclose(skin, 0.0)


def test_compute_directional_skin_positive_only_in_one_direction():
    """
    Create a psi field with a gradient only in +x direction.
    Only skin_xp should be nonzero.
    """
    Nx = Ny = Nz = 5
    psi = np.zeros((Nx,Ny,Nz))
    # create a slope in +x only
    psi[1,:,:] = 1.0  # neighbor in +x direction

    from igc.sim.operators import compute_directional_skin
    skin = compute_directional_skin(psi, dx=1.0)

    skin_xp, skin_xm, skin_yp, skin_ym, skin_zp, skin_zm = skin

    assert np.any(skin_xp > 0.0)
    assert np.allclose(skin_xm, 0.0)
    assert np.allclose(skin_yp, 0.0)
    assert np.allclose(skin_ym, 0.0)
    assert np.allclose(skin_zp, 0.0)
    assert np.allclose(skin_zm, 0.0)

def test_div_phi_cone_psi_zero_when_phi_cone_zero():
    """
    If phi_cone is zero everywhere, directional cone transport must be zero
    for any psi.
    """
    Nx = Ny = Nz = 4
    psi = np.random.rand(Nx, Ny, Nz)
    phi_cone = np.zeros((6, Nx, Ny, Nz), dtype=float)
    dx = 1.0

    div_val = div_phi_cone_psi(phi_cone, psi, dx)
    assert np.allclose(div_val, 0.0)


def test_div_phi_cone_psi_zero_for_constant_psi():
    """
    If psi is constant, directional cone transport must be zero for any
    directional phi_cone (no skin, no flux).
    """
    Nx = Ny = Nz = 4
    psi = np.ones((Nx, Ny, Nz), dtype=float) * 2.718
    # Random directional cones, but should not matter for constant psi
    phi_cone = np.random.rand(6, Nx, Ny, Nz)
    dx = 1.0

    div_val = div_phi_cone_psi(phi_cone, psi, dx)
    assert np.allclose(div_val, 0.0)    