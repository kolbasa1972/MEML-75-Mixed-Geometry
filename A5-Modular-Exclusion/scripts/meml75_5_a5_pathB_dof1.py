"""
meml75_5_a5_pathB_dof1.py

Path B: dof=+1 configurations for minimal A_5 modular model.

B1: one weight-6 quintet (c6b=0), real coefficients.
    Params: Re tau, Im tau, c4, c6a  -> 4 params on 5 observables -> dof=+1.
B2: both weight-6 quintets with c6a=c6b=c6, real coefficients.
    Params: Re tau, Im tau, c4, c6   -> 4 params on 5 observables -> dof=+1.

ETHICS MEML:
    [HONEST] tau is a fitted parameter, NOT stabilised.
    [HONEST] dof=+1: one genuine prediction. If chi2 > 10 -> model excluded.
    [HONEST] Mixing weights 2,4,6 for the same L with constant c_i is NOT
             modular invariant. This scan is a phenomenological test of an
             invalid modular model, not a viable construction.
    [EXACT]  CG coefficients from arXiv:1812.02158 (App. B).
    [OPEN]   Weight-6 basis completeness is NOT proven (only Y2 x Y4 used).
    [REPRODUCIBLE] Script prints all parameters, pulls, dof, and boundary checks.

Dependencies: numpy, scipy.
"""

import numpy as np
from scipy.optimize import differential_evolution

# ============================================================
# 1. Modular Forms (same as v10, no filterwarnings)
# ============================================================
def get_Y2(tau):
    """Weight 2 quintuplet Y_5."""
    q = np.exp(2j * np.pi * tau)
    q15 = np.exp(2j * np.pi * tau / 5.0)
    t1 = 1.0 + 6.0*q + 6.0*q**2 + 6.0*q**3
    t2 = np.sqrt(5.0) * q15 * (1.0 + 3.0*q + 3.0*q**2)
    t3 = np.sqrt(5.0) * q15**2 * (1.0 + 3.0*q + 3.0*q**2)
    t4 = np.sqrt(5.0) * q15**3 * (1.0 + 3.0*q)
    t5 = np.sqrt(5.0) * q15**4 * (1.0 + 3.0*q)
    return np.array([t1, t2, t3, t4, t5], dtype=complex)

def Y5x5_to_1(A, B):
    return A[0]*B[0] + A[1]*B[4] + A[2]*B[3] + A[3]*B[2] + A[4]*B[1]

def Y5x5_to_4(A, B):
    return np.array([
        np.sqrt(2)*A[0]*B[1] + 2*A[1]*B[0] - 2*A[3]*B[4] + A[4]*B[3],
        -np.sqrt(3)*A[0]*B[1] - np.sqrt(3)*A[1]*B[0] + np.sqrt(2)*A[3]*B[4] - np.sqrt(2)*A[4]*B[3],
        np.sqrt(3)*A[0]*B[1] - np.sqrt(3)*A[1]*B[0] + np.sqrt(2)*A[4]*B[3],
        np.sqrt(2)*A[0]*B[1] - 2*A[1]*B[0] + 2*A[3]*B[4] - A[4]*B[3]
    ], dtype=complex)

def Y5x5_to_5_S1(A, B):
    return np.array([
        2*A[0]*B[0] + A[1]*B[4] + A[2]*B[3] + A[3]*B[2] + A[4]*B[1],
        -np.sqrt(3)*A[0]*B[1] - np.sqrt(3)*A[1]*B[0] + np.sqrt(2)*A[2]*B[4] + np.sqrt(2)*A[4]*B[2],
        -np.sqrt(2)*A[0]*B[2] - 2*A[1]*B[1] - np.sqrt(2)*A[3]*B[0] + np.sqrt(2)*A[4]*B[3],
        np.sqrt(2)*A[0]*B[3] - np.sqrt(2)*A[1]*B[2] + np.sqrt(2)*A[2]*B[1] - np.sqrt(2)*A[3]*B[0],
        np.sqrt(2)*A[0]*B[4] + np.sqrt(2)*A[1]*B[3] + np.sqrt(2)*A[3]*B[1] + np.sqrt(2)*A[4]*B[0]
    ], dtype=complex)

def Y5x5_to_5_S2(A, B):
    return np.array([
        2*A[0]*B[1] - 2*A[1]*B[0] + A[2]*B[3] + A[3]*B[2],
        -2*A[0]*B[2] - np.sqrt(2)*A[1]*B[3] - np.sqrt(2)*A[3]*B[1] - 2*A[2]*B[0],
        np.sqrt(3)*A[0]*B[3] + np.sqrt(3)*A[3]*B[0] + A[1]*B[1],
        A[0]*B[4] + A[4]*B[0] + np.sqrt(6)*A[1]*B[2] + np.sqrt(6)*A[2]*B[1],
        -A[1]*B[4] - A[4]*B[1] + np.sqrt(6)*A[2]*B[3] + np.sqrt(6)*A[3]*B[2]
    ], dtype=complex)

def get_Y4(tau):
    Y2 = get_Y2(tau)
    return Y5x5_to_1(Y2, Y2), Y5x5_to_4(Y2, Y2), Y5x5_to_5_S1(Y2, Y2)

def get_Y6(tau):
    Y2 = get_Y2(tau)
    _, _, Y5_4 = get_Y4(tau)
    return Y5x5_to_5_S1(Y2, Y5_4), Y5x5_to_5_S2(Y2, Y5_4)

# ============================================================
# 2. Lepton CG
# ============================================================
def LL_3_to_5(L1, L2):
    return np.array([
        2*L1[0]*L2[0] - L1[1]*L2[2] - L1[2]*L2[1],
        -np.sqrt(3)*(L1[0]*L2[1] + L1[1]*L2[0]),
        np.sqrt(6)*L1[1]*L2[1],
        np.sqrt(6)*L1[2]*L2[2],
        -np.sqrt(3)*(L1[0]*L2[2] + L1[2]*L2[0])
    ], dtype=complex)

# ============================================================
# 3. Mass matrices for B1 and B2
# ============================================================
def get_M_B1(tau, c4, c6a):
    """B1: c2=1, c4 real, c6a real, c6b=0."""
    Y2 = get_Y2(tau)
    _, _, Y5_4 = get_Y4(tau)
    Y5_6_S1, _ = get_Y6(tau)
    def Q(x):
        LL = LL_3_to_5(x, x)
        return (1.0 * Y5x5_to_1(LL, Y2)
                + c4 * Y5x5_to_1(LL, Y5_4)
                + c6a * Y5x5_to_1(LL, Y5_6_S1))
    M = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            ei = np.zeros(3, complex); ei[i] = 1
            ej = np.zeros(3, complex); ej[j] = 1
            M[i, j] = 0.5 * (Q(ei + ej) - Q(ei) - Q(ej))
    return M

def get_M_B2(tau, c4, c6):
    """B2: c2=1, c4 real, c6a=c6b=c6 real."""
    Y2 = get_Y2(tau)
    _, _, Y5_4 = get_Y4(tau)
    Y5_6_S1, Y5_6_S2 = get_Y6(tau)
    def Q(x):
        LL = LL_3_to_5(x, x)
        return (1.0 * Y5x5_to_1(LL, Y2)
                + c4 * Y5x5_to_1(LL, Y5_4)
                + c6 * (Y5x5_to_1(LL, Y5_6_S1) + Y5x5_to_1(LL, Y5_6_S2)))
    M = np.zeros((3, 3), dtype=complex)
    for i in range(3):
        for j in range(3):
            ei = np.zeros(3, complex); ei[i] = 1
            ej = np.zeros(3, complex); ej[j] = 1
            M[i, j] = 0.5 * (Q(ei + ej) - Q(ei) - Q(ej))
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
    if c13 < 1e-12:
        return {k: np.nan for k in EXP}
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
    dm21_dm31 = dm21 / dm31 if abs(dm31) > 1e-30 else np.nan
    return {
        "s12sq": float(s12 ** 2), "s13sq": float(s13 ** 2),
        "s23sq": float(s23 ** 2), "sin_delta": float(sin_delta),
        "dm21_dm31": float(dm21_dm31)
    }

def chi2_from_params(p):
    for k in EXP:
        if not np.isfinite(p[k]):
            return 1e12
    c2 = 0.0
    for k in ["s12sq", "s13sq", "s23sq", "dm21_dm31"]:
        c2 += ((p[k] - EXP[k]["val"]) / EXP[k]["sigma"]) ** 2
    c2 += ((p["sin_delta"] - EXP["sin_delta"]["val"]) / EXP["sin_delta"]["sigma"]) ** 2
    return float(c2)

def print_breakdown(p, title):
    print(f"\n[{title}]")
    for k in ["s12sq", "s13sq", "s23sq", "dm21_dm31", "sin_delta"]:
        val, exp, err = p[k], EXP[k]["val"], EXP[k]["sigma"]
        print(f"  {k:10s}: model = {val:+.6f}, exp = {exp:+.6f}, pull = {(val-exp)/err:+.2f} sigma")

# ============================================================
# 5. Scans with |tau|>=1 penalty
# ============================================================
def scan_B1(seed=42):
    bounds = [(-0.5, 0.5), (0.5, 3.0), (-20.0, 20.0), (-20.0, 20.0)]
    def obj(x):
        tau = x[0] + 1j * x[1]
        if abs(tau) < 1.0:
            return 1e12
        try:
            M = get_M_B1(tau, x[2], x[3])
            p = pmns_from_mnu(M)
            return chi2_from_params(p)
        except Exception:
            return 1e12
    return differential_evolution(obj, bounds, seed=seed, maxiter=2000,
                                  popsize=50, polish=True, tol=1e-10)

def scan_B2(seed=42):
    bounds = [(-0.5, 0.5), (0.5, 3.0), (-20.0, 20.0), (-20.0, 20.0)]
    def obj(x):
        tau = x[0] + 1j * x[1]
        if abs(tau) < 1.0:
            return 1e12
        try:
            M = get_M_B2(tau, x[2], x[3])
            p = pmns_from_mnu(M)
            return chi2_from_params(p)
        except Exception:
            return 1e12
    return differential_evolution(obj, bounds, seed=seed, maxiter=2000,
                                  popsize=50, polish=True, tol=1e-10)

def report(name, res, model_func, N_obs=5, N_params=4):
    print("=" * 60)
    print(name)
    print("=" * 60)
    print(f"chi2 = {res.fun:.4f}")
    tau = res.x[0] + 1j * res.x[1]
    print(f"tau = {tau:.6f}, |tau| = {abs(tau):.6f}")
    print(f"params: {res.x}")
    if abs(tau) < 1.0:
        print("WARNING: tau outside fundamental domain |tau|>=1")
    dof = N_obs - N_params
    print(f"dof = {dof}")
    if dof <= 0:
        print("WARNING: under-determined, chi2=0 is trivial interpolation, NOT a prediction")
    M = model_func(tau, *res.x[2:])
    p = pmns_from_mnu(M)
    print_breakdown(p, "BREAKDOWN")
    print()

if __name__ == "__main__":
    print("Path B: dof=+1 configurations")
    print("Precondition: if best chi2 > 10 -> Path A (honest negative).")
    print("If best chi2 in [1,10] -> phenomenological fit (tau fitted).")
    print("NOTE: mixing weights 2,4,6 with constant c_i is NOT modular invariant.")
    print()
    res1 = scan_B1()
    report("B1: one weight-6 quintet (c6b=0), real coefficients", res1, get_M_B1)
    res2 = scan_B2()
    report("B2: both weight-6 quintets (c6a=c6b=c6), real coefficients", res2, get_M_B2)
    best = min(res1.fun, res2.fun)
    print("=" * 60)
    print(f"BEST chi2 = {best:.4f}")
    if best > 10:
        print("DECISION: chi2 > 10 -> Path A (honest negative).")
    elif best >= 1:
        print("DECISION: chi2 in [1,10] -> phenomenological fit.")
        print("HONEST LINE: tau is a fitted parameter, not stabilised;")
        print("this is a phenomenological fit of a non-modular-invariant model.")
    else:
        print("DECISION: chi2 < 1 -> suspicious; check dof and modular invariance.")