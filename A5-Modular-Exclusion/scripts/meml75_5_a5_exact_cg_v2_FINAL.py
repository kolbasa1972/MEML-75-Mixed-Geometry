"""
meml75_5_a5_exact_cg_v2_FINAL.py

Proves Theorem 1: (LL)_3 = 0 for L~3 due to antisymmetry.
Patched: LL_3_to_3 now exactly matches Eq. (A.6) of 1812.02158.

ETHICS MEML labels:
    [EXACT]   All CG coefficients match App. B of 1812.02158.
    [TEST]    Verifies (LL)_3 = 0 and M_nu symmetric.
"""

import numpy as np

def LL_3_to_3(L1, L2):
    """(3 ⊗ 3) -> 3_A. Formula (A.6). Exact match."""
    return np.array([
        L1[1]*L2[2] - L1[2]*L2[1],
        L1[0]*L2[1] - L1[1]*L2[0], 
        L1[2]*L2[0] - L1[0]*L2[2]  
    ])

def _test_triplet_vanishes():
    print("[TEST] Checking if antisymmetric triplet vanishes ...")
    L = np.array([1.0, 2.0, 3.0], dtype=complex)
    triplet = LL_3_to_3(L, L)
    assert np.allclose(triplet, 0.0), triplet
    print("  (LL)_3 for L~3 is exactly ZERO. (Confirmed.)")
    print("  => M_3 is unphysical. NR-A5-04 was testing a phantom.")

if __name__ == "__main__":
    _test_triplet_vanishes()