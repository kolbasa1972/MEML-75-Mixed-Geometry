"""
meml75_5_a5_exact_fit_v11_L3_w4_FINAL.py

Case II: Physically correct L~3 at pure weight 4.
Fixes: Correct dof = +2 (5 obs, 3 shape params).

ETHICS MEML labels:
    [EXACT]   Exact CG for 3x3 -> 1,5 and 5x5 -> 1,5.
    [HONEST]  Enforces |tau| >= 1.
    [HONEST]  dof = N_obs(5) - N_shape_params(3) = +2.
"""

import numpy as np
from scipy.optimize import differential_evolution
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# 1. Weight 2 Forms
# ============================================================
def get_Y2(tau):
    q = np.exp(2j * np.pi * tau)
    q15 = np.exp(2j * np.pi * tau / 5.0)
    t1 = 1.0 + 6.0*q + 6.0*q**2 + 6.0*q**3
    t2 = np.sqrt(5.0) * q15 * (1.0 + 3.0*q + 3.0*q**2)
    t3 = np.sqrt(5.0) * q15**2 * (1.0 + 3.0*q + 3.0*q**2)
    t4 = np.sqrt(5.0) * q15**3 * (1.0 + 3.0*q)
    t5 = np.sqrt(5.0) * q15**4 * (1.0 + 3.0*q)
    return np.array([t1, t2, t3, t4, t5], dtype=complex)

# ============================================================
# 2. Weight 4 Forms (5x5 -> 1, 5)
# ============================================================
def Y5x5_to_1(A, B):
    """5x5 -> 1 (A.37). T-covariant."""
    return A[0]*B[0] + A[1]*B[4] + A[2]*B[3] + A[3]*B[2] + A[4]*B[1]

def Y5x5_to_5_S1(A, B):
    """5x5 -> 5 (A.42). T-covariant."""
    return np.array([
        2*A[0]*B[0] + A[1]*B[4] + A[2]*B[3] + A[3]*B[2] + A[4]*B[1],
        -np.sqrt(3)*A[0]*B[1] - np.sqrt(3)*A[1]*B[0] + np.sqrt(2)*A[2]*B[4] + np.sqrt(2)*A[4]*B[2],
        -np.sqrt(2)*A[0]*B[2] - 2*A[1]*B[1] - np.sqrt(2)*A[3]*B[0] + np.sqrt(2)*A[4]*B[3],
        np.sqrt(2)*A[0]*B[3] - np.sqrt(2)*A[1]*B[2] + np.sqrt(2)*A[2]*B[1] - np.sqrt(2)*A[3]*B[0],
        np.sqrt(2)*A[0]*B[4] + np.sqrt(2)*A[1]*B[3] + np.sqrt(2)*A[3]*B[1] + np.sqrt(2)*A[4]*B[0]
    ], dtype=complex)

def get_Y4(tau):
    """Returns Y1^(4) and Y5^(4)."""
    Y2 = get_Y2(tau)
    return Y5x5_to_1(Y2, Y2), Y5x5_to_5_S1(Y2, Y2)

# ============================================================
# 3. L~3 Mass Matrix at Weight 4
# ============================================================
def LL_3_to_1(L1, L2):
    """3x3 -> 1 (A.5). T-covariant."""
    return L1[0]*L2[0] + L1[1]*L2[2] + L1[2]*L2[1]

def LL_3_to_5(L1, L2):
    """3x3 -> 5 (A.7). T-covariant."""
    return np.array([
        2*L1[0]*L2[0] - L1[1]*L2[2] - L1[2]*L2[1],
        -np.sqrt(3)*(L1[0]*L2[1] + L1[1]*L2[0]),
        np.sqrt(6)*L1[1]*L2[1],
        np.sqrt(6)*L1[2]*L2[2],
        -np.sqrt(3)*(L1[0]*L2[2] + L1[2]*L2[0])
    ], dtype=complex)

def get_M_L3_w4(tau, c1, c5):
    """M_nu for L~3, pure weight 4."""
    Y1, Y5 = get_Y4(tau)
    
    Q = lambda x: (c1 * LL_3_to_1(x, x) * Y1 + 
                   c5 * Y5x5_to_1(LL_3_to_5(x, x), Y5))
    
    M = np.zeros((3,3), dtype=complex)
    for i in range(3):
        for j in range(3):
            ei = np.zeros(3, complex); ei[i] = 1
            ej = np.zeros(3, complex); ej[j] = 1
            M[i,j] = 0.5 * (Q(ei+ej) - Q(ei) - Q(ej))
    return M

# ============================================================
# 4. Statistics
# ============================================================
EXP = {
    "s12sq":     {"val": 0.304,  "sigma": 0.012},
    "s13sq":     {"val": 0.0222, "sigma": 0.0007},
    "s23sq":     {"val": 0.573,  "sigma": 0.020},
    "dm21_dm31": {"val": 0.0296, "sigma": 0.0003},
    "sin_delta": {"val": -0.259, "sigma": 0.25},
}

def pmns_from_mnu(M):
    M = np.asarray(M, dtype=complex)
    H = M.conj().T @ M
    evals, evecs = np.linalg.eigh(H)
    evals = np.maximum(evals, 0.0)
    absm = np.sqrt(evals)
    U = evecs
    s13 = abs(U[0, 2])
    c13 = np.sqrt(max(0.0, 1.0 - s13 * s13))
    if c13 < 1e-12: return {k: np.nan for k in EXP}
    s12 = abs(U[0, 1]) / c13
    s23 = abs(U[1, 2]) / c13
    c12 = np.sqrt(max(0.0, 1.0 - s12**2))
    c23 = np.sqrt(max(0.0, 1.0 - s23**2))
    J = np.imag(U[0, 0] * U[1, 1] * np.conj(U[0, 1]) * np.conj(U[1, 0]))
    denom = s12 * c12 * s23 * c23 * s13 * c13 * c13
    sin_delta = (J / denom) if denom > 1e-14 else 0.0
    m2 = absm ** 2
    dm21 = m2[1] - m2[0]
    dm31 = m2[2] - m2[0]
    dm21_dm31 = dm21 / dm31 if abs(dm31) > 1e-30 else np.inf
    return {
        "s12sq": float(s12 ** 2), "s13sq": float(s13 ** 2),
        "s23sq": float(s23 ** 2), "sin_delta": float(sin_delta),
        "dm21_dm31": float(dm21_dm31)
    }

def chi2_from_params(p):
    if any(np.isnan(v) or np.isinf(v) for v in p.values()): return 1e12
    c2 = 0.0
    for k in EXP: c2 += ((p[k] - EXP[k]["val"]) / EXP[k]["sigma"]) ** 2
    return float(c2)

# ============================================================
# 5. Scan
# ============================================================
def scan_L3_w4(seed=42):
    print("[SCAN] L~3 pure weight 4 (physically correct, 3 generations)")
    print("[DOF]    +2 (5 obs, 3 shape params: tau(2) + c1/c5(1))")
    # Enforce |tau| >= 1 by Im(tau) >= 1
    bounds = [(-0.5, 0.5), (1.0, 3.0), (-5.0, 5.0)]
    
    def objective(x):
        tau = x[0] + 1j * x[1]
        if abs(tau) < 1.0: return 1e12
        try:
            # c5 = 1.0 fixed as scale, c1 = x[2]
            M = get_M_L3_w4(tau, x[2], 1.0)
            return chi2_from_params(pmns_from_mnu(M))
        except Exception: return 1e12
    
    res = differential_evolution(objective, bounds, seed=seed, maxiter=300, popsize=30, polish=True)
    
    print(f"[SCAN] Best chi2 = {res.fun:.4f}")
    print(f"[SCAN] tau = {res.x[0]:.4f} + {res.x[1]:.4f}i  (|tau| = {abs(res.x[0]+1j*res.x[1]):.4f})")
    print(f"[SCAN] c1/c5 = {res.x[2]:.4f}")
    
    M = get_M_L3_w4(res.x[0] + 1j * res.x[1], res.x[2], 1.0)
    p = pmns_from_mnu(M)
    print("\n[BREAKDOWN]")
    for k in EXP:
        val, exp, err = p[k], EXP[k]["val"], EXP[k]["sigma"]
        print(f"  {k:10s}: model = {val:+.6f}, exp = {exp:+.6f}, pull = {(val-exp)/err:+.2f} sigma")

if __name__ == "__main__":
    scan_L3_w4()