# igc/tests/test_topology_static.py
"""
Static topology sanity test.

We construct a simple 3D field with two disconnected "blobs" and verify that
6-connected-component labeling finds exactly 2 components. For a field that
is just a union of two disjoint solid clusters, this corresponds to:

  Betti0 = 2, Betti1 = 0, Betti2 = 0
  Euler characteristic χ = 2

This is a predictable, non-dynamic regime: "real structure" by construction.
"""

import numpy as np
import cc3d


def test_two_disconnected_blobs():
    Nx = Ny = Nz = 24
    field = np.zeros((Nx, Ny, Nz), dtype=np.uint8)

    # First blob: small cube around (6,6,6)
    field[4:8, 4:8, 4:8] = 1

    # Second blob: small cube around (17,17,17)
    field[16:20, 16:20, 16:20] = 1

    # 6-connectivity for standard "face-adjacent" neighbors
    labels, n_comp = cc3d.connected_components(field, connectivity=6, return_N=True)

    print(f"\n[Static topology] number of connected components = {n_comp}")

    # We expect exactly 2 components
    assert n_comp == 2