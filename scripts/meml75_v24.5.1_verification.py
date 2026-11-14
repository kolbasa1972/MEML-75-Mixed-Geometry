#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_v24.5.1_verification.py — MEML 75 v24.5.1 code-consistency check

Reference: MEML_75_v24.5.1.tex

Corrections vs v13.0 of this script:
  * §3.3 check of |Y6_1|/|Y4_3| at tau=e^{i pi/3} re-labelled as a
    REGRESSION TEST of the Y-form code. Those numbers are NOT in the
    manuscript (they belonged to the deleted Appendix G).
  * §10: d13^2 = 4 d12^2 is now verified NON-TRIVIALLY, by computing
    all three pairwise distances independently from the positions
    z1=0, z2=(1+tau)/3, z3=2(1+tau)/3.
  * §10: the "alpha ~ 10" check is now DIAGNOSTIC ONLY. The phrase
    "alpha ~ 10, spread ~ 1.7" was in a removed appendix and is not
    present in the v24.5.1 manuscript, so it must not be used as a
    pass/fail target. alpha_ab is printed for reference only.
  * App D: invariance under c_i -> k c_i is checked on the OBSERVABLES
    (relative tolerance 1e-10), not on chi^2 (which is at roundoff).
  * Unused variables removed (v_EW, eta_obs, g_star, M_Pl=1.22e19).
  * Planck-mass naming consolidated (M_PL_RED, M_PL).
  * Explicit checks of c^raw[i] and c^norm[i] against the manuscript.

Corrections in this revision (v24.5.1 alignment, no placeholders):
  * §7: the unconstrained 7-parameter quark fit chi2 ~ 1e-19 is an
    OVERFIT, not a test of the model. It is now reported as a
    diagnostic only, NOT as an [OK] check. The physical structural
    fit gives chi2 = 13.93 (paper) / 29.50 (PDG 2024 at 10%); see
    meml75_honest_pure2.py v4.1.
  * §9: the old count N_eff = 11, N_obs = 10, N_dof = -1 referred to
    the 7-parameter overfit. It is replaced by the honest v24.5.1
    count: N_model = 12, N_eff = 13 (with theta_i), N_obs = 16,
    N_dof = +3 (or +4 if c_d is fixed by the structural fit).
    Moduli are counted as free because pure GVW neither predicts
    nor stabilizes them (v24.5.1 Section 'Flux stabilization').
  * I_FACTOR = 0.366025 (theoretical, headline numbers in v24.5.1);
    checks are against m_KK = 2.0818e14 GeV, M_6 = 2.9631e16 GeV.
    The fitted value 0.3653 (used in some older scripts) gives
    2.0838e14 / 2.9646e16 (0.1% difference).
  * §10 b_u/c_u note is a documented limitation, not counted as PASS.
  * §9 and Summary print explicit SCOPE lines: this script checks
    code consistency and v24.5.1 bookkeeping. It does NOT verify
    the structural quark fit, the lepton fit, the GVW saddle, or
    the ALP/WGC numbers; those live in dedicated scripts.

v24.5.2 FIX (this revision):
  * §10 "Effective alpha and the b_u/c_u problem": the hardcoded
    alpha=10 (from the removed Appendix G) has been replaced by the
    COMPUTED alpha_ab = ln(b_u/a_u)/(d13^2 - d12^2) = 13.21.
    The conclusion is unchanged: with alpha_ab the required
    d_x^2 ~ 0.69 matches neither d12^2 = 0.298 nor d13^2 = 1.192,
    so c_u still needs a non-O(1) prefactor. But the arithmetic
    is now internally consistent (no hardcoded number that is
    not derived in the script).
"""
import numpy as np
from scipy.optimize import brentq
import time

# --- Physical constants (single source of truth) ---
PI        = np.pi
M_PL_RED  = 2.4350e18        # reduced Planck mass [GeV]
M_PL      = 1.2209e19        # unreduced Planck mass [GeV]
h_eV_s    = 4.135667696e-15  # Planck constant [eV s]
hbar_SI   = 1.054571817e-34  # [J s]
c_SI      = 2.99792458e8     # [m/s]
G_SI      = 6.67430e-11      # [m^3 kg^-1 s^-2]
eV_to_kg  = 1.782661921e-36  # 1 eV / c^2 in kg
M_SUN     = 1.989e30         # [kg]
alpha_EM  = 1.0 / 137.035999
chi_14    = 0.0750           # chi^{1/4} = 75 MeV in GeV

# --- Fitted quark parameters (raw, before a_u = 1 calibration) ---
TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j
COEFS_RAW = np.array([
    5.314403415527852e-05,   # a_u
    7.192000606749913,       # b_u
    0.038947583908142965,    # c_u
    0.00024324586524881046,  # a_1
    0.5030286858187134,      # a_2
    0.0013651716604568512,   # b
    0.002220219681204568,    # c
])
COEFS_NORM = COEFS_RAW / COEFS_RAW[0]   # a_u -> 1

# Theoretical value, used for the headline numbers in v24.5.1:
# m_KK = 2.0818e14 GeV, M_6 = 2.9631e16 GeV.
# The fitted value 0.3653 (some older scripts) gives
# 2.0838e14 / 2.9646e16, a 0.1% difference.
I_FACTOR = 0.366025

# --- A_4 modular form q-expansions ---
C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]

def Y2(tau):
    q   = np.exp(2j*PI*tau)
    q13 = np.exp(2j*PI*tau/3)
    q23 = np.exp(4j*PI*tau/3)
    y1 = 1+0j; b2 = 1+0j; b3 = 1+0j
    for n, c in enumerate(C1, 1): y1 += c*q**n
    for n, c in enumerate(C2, 1): b2 += c*q**n
    for n, c in enumerate(C3, 1): b3 += c*q**n
    return np.array([y1, -6*q13*b2, -18*q23*b3])

def Y4(tau):
    a, b, c = Y2(tau)
    return np.array([a*a - b*c, c*c - a*b, b*b - a*c])

def Y6(tau):
    a = Y2(tau); b = Y4(tau)
    return np.array([
        (2*a[0]*b[0] - a[1]*b[2] - a[2]*b[1])/3,
        (2*a[2]*b[2] - a[0]*b[1] - a[1]*b[0])/3,
        (2*a[1]*b[1] - a[2]*b[0] - a[0]*b[2])/3,
    ])

def build_M_quarks(tau_u, tau_d, coefs):
    a_u, b_u, c_u, a1, a2, b, c_d = coefs
    y2u, y4u, y6u = Y2(tau_u), Y4(tau_u), Y6(tau_u)
    y2d, y4d, y6d = Y2(tau_d), Y4(tau_d), Y6(tau_d)
    Mu = np.array([
        [a_u*y2u[0], b_u*y4u[2], c_u*y6u[1]],
        [a_u*y2u[2], b_u*y4u[1], c_u*y6u[0]],
        [a_u*y2u[1], b_u*y4u[0], c_u*y6u[2]],
    ], dtype=complex)
    Md = np.array([
        [a1*y6d[0], b*y2d[2], c_d*y2d[1]],
        [a1*y6d[2], b*y2d[1], c_d*y2d[0]],
        [a2*y2d[1], b*y2d[0], c_d*y4d[2]],
    ], dtype=complex)
    return Mu, Md

def quark_observables(Mu, Md):
    wu, Uu = np.linalg.eigh(Mu @ Mu.conj().T)
    wd, Ud = np.linalg.eigh(Md @ Md.conj().T)
    mu, mc, mt = np.sqrt(np.maximum(wu, 1e-60))
    md, ms, mb = np.sqrt(np.maximum(wd, 1e-60))
    V = Uu.conj().T @ Ud
    J = abs(np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0])))
    return {
        "m_u/m_t": mu/mt, "m_c/m_t": mc/mt,
        "m_d/m_b": md/mb, "m_s/m_b": ms/mb, "m_b/m_t": mb/mt,
        "V_us": abs(V[0,1]), "V_cb": abs(V[1,2]), "V_ub": abs(V[0,2]),
        "J": J,
    }

TARGETS = {
    "m_u/m_t": (1.27e-3/172.0,  0.30e-3/172.0),
    "m_c/m_t": (0.620/172.0,    0.020/172.0),
    "m_d/m_b": (2.70e-3/2.860,  0.20e-3/2.860),
    "m_s/m_b": (55.0e-3/2.860,  5.0e-3/2.860),
    "m_b/m_t": (2.860/172.0,    0.030/172.0),
    "V_us":    (0.22501, 0.00068),
    "V_cb":    (0.04053, 0.00042),
    "V_ub":    (0.00382, 0.00020),
    "J":       (3.080e-5, 0.130e-5),
}

def chi2_from_obs(obs):
    return sum(((obs[k] - TARGETS[k][0]) / TARGETS[k][1])**2 for k in TARGETS)

# --- Checker ---
class Checker:
    def __init__(self):
        self.passed = 0; self.failed = 0
    def check(self, label, computed, expected, rtol=1e-6, atol=0.0):
        if expected == 0:
            err = abs(computed); ok = err <= atol
            err_str = f"abs err {err:.2e}"
        else:
            err = abs(computed - expected) / abs(expected)
            ok  = err <= rtol
            err_str = f"rel err {err:.2e}"
        status = "OK  " if ok else "FAIL"
        print(f"  [{status}] {label:<38s} "
              f"{computed:>14.6e} vs {expected:>14.6e}  ({err_str})")
        if ok: self.passed += 1
        else:  self.failed += 1
        return ok
    def head(self, s):
        print(f"\n{'='*74}\n  {s}\n{'='*74}")

C = Checker()

def main():
    t0 = time.time()
    print("\n" + "█"*74)
    print("  MEML 75 — VERIFICATION (v13, corrected, aligned with v24.5.1)")
    print("  Reference: MEML 75 v24.5.5.tex")
    print("█"*74)

    # =====================================================================
    C.head("§3  Bulk coupling alpha_G = 2/(3 pi)")
    # =====================================================================
    alpha_G = 2.0/(3*PI)
    C.check("alpha_G", alpha_G, 0.2122065908, rtol=1e-9)

    # REGRESSION TEST of the Y-form code — NOT a manuscript claim.
    # |Y6_1|/|Y4_3| at tau=e^{i pi/3} appeared in the DELETED Appendix G.
    # We keep it only to confirm that Y6/Y4 implementations are stable.
    print("\n  [REGRESSION TEST — not a manuscript claim]")
    print("  |Y6_1|/|Y4_3| at tau=e^{i pi/3} was in the removed Appendix G;")
    print("  the value is kept here only as a code sanity check.")
    tau_333 = np.exp(1j*PI/3)
    C.check("|Y6_1|/|Y4_3| (code)", abs(Y6(tau_333)[0])/abs(Y4(tau_333)[2]),
            0.212125, rtol=1e-4)

    # =====================================================================
    C.head("§4  Suppression factor epsilon")
    # =====================================================================
    eps = (alpha_G/(4*PI))**2
    C.check("epsilon = (aG/(4 pi))^2", eps, 2.851662e-4, rtol=1e-4)
    C.check("epsilon = 1/(36 pi^4)",   1.0/(36*PI**4), eps, rtol=1e-12)

    # =====================================================================
    C.head("§5  m_KK and M_6 (theoretical I_FACTOR = 0.366025)")
    # =====================================================================
    m_KK = M_PL_RED / (np.sqrt(I_FACTOR) * np.exp(PI**2))
    M_6  = (3.0 * M_PL_RED**2 * m_KK**2)**0.25
    C.check("m_KK [GeV]", m_KK, 2.0818e14, rtol=0.01)
    C.check("M_6  [GeV]", M_6,  2.9631e16, rtol=0.01)

    # =====================================================================
    C.head("§6  ALP sector")
    # =====================================================================
    f_a = M_PL_RED * eps**2
    C.check("f_a [GeV]", f_a, 1.9801e11, rtol=0.01)

    m_a_GeV  = chi_14**2 / f_a
    m_a_eV   = m_a_GeV * 1e9
    nu_a_GHz = m_a_eV / h_eV_s / 1e9
    C.check("m_a [mu eV]", m_a_eV*1e6, 28.407, rtol=0.01)
    C.check("nu_a [GHz]",  nu_a_GHz,   6.8688, rtol=0.01)

    # Relic density: solve for theta_i (T-corrections neglected, per paper)
    def Omega_ah2(theta):
        f_anh = (np.log(np.e / (1 - (theta/PI)**2)))**(7/6)
        return 0.12 * theta**2 * f_anh * (f_a / 3e11)**(7/6)
    theta_i = brentq(lambda th: Omega_ah2(th) - 0.12, 0.5, 2.0)
    C.check("theta_i (fitted)", theta_i, 1.174, rtol=0.02)
    print(f"  Omega_ah2(theta_i) - 0.12 = {Omega_ah2(theta_i) - 0.12:.2e}"
          "  (sanity that brentq converged)")

    pref = alpha_EM / (2*PI*f_a)
    C.check("alpha_EM/(2 pi f_a)", pref, 5.866e-15, rtol=0.01)
    C.check("g_KSVZ [GeV^-1]", pref*(0   - 1.92), -1.126e-14, rtol=0.01)
    C.check("g_DFSZ [GeV^-1]", pref*(8/3 - 1.92),  4.379e-15, rtol=0.01)

    m_a_kg = m_a_eV * eV_to_kg
    M_BH   = hbar_SI * c_SI / (G_SI * m_a_kg)
    C.check("M_BH [kg]",    M_BH, 9.36e24, rtol=0.05)
    C.check("M_BH [M_sun]", M_BH/M_SUN, 4.7e-6, rtol=0.05)

    # =====================================================================
    C.head("§7  Quark fit: c^raw -> c^norm")
    # =====================================================================
    Mu, Md = build_M_quarks(TAU_U, TAU_D, COEFS_NORM)
    obs_q  = quark_observables(Mu, Md)
    chi2_q = chi2_from_obs(obs_q)
    # The unconstrained 7-parameter fit is an overfit (more parameters
    # than independent ratios) and is NOT a test of the model; v24.5.1
    # does not quote its chi^2 as evidence. We only print it as a
    # numerical sanity check of the matrix implementation.
    print(f"\n  [diagnostic only, NOT evidence] chi2_quarks (overfit) = {chi2_q:.3e}")
    print("  The physical structural fit (2-parameter) gives")
    print("  chi2 = 13.93 (paper targets) / 29.50 (PDG 2024 at 10%).")
    print("  Reproduced by meml75_honest_pure2.py v4.1.")

    # Explicit comparison with the values printed in the manuscript
    manuscript_raw  = np.array([5.31e-5, 7.19, 0.039, 2.43e-4,
                                0.503, 1.37e-3, 2.22e-3])
    manuscript_norm = np.array([1.0, 1.35e5, 7.33e2, 4.58,
                                9.47e3, 2.58e1, 4.18e1])
    print("\n  c^raw vs manuscript (2-sig-fig values):")
    for i, name in enumerate(["a_u","b_u","c_u","a_1","a_2","b","c"]):
        C.check(f"  c^raw[{name}]", COEFS_RAW[i], manuscript_raw[i], rtol=0.01)
    print("\n  c^norm vs manuscript (2-sig-fig values):")
    for i, name in enumerate(["a_u","b_u","c_u","a_1","a_2","b","c"]):
        C.check(f"  c^norm[{name}]", COEFS_NORM[i], manuscript_norm[i], rtol=0.01)

    # =====================================================================
    C.head("§10 Distances (NON-TRIVIAL: from positions z_i, not formula)")
    # =====================================================================
    # Fixed points on T^2/Z3: z1=0, z2=(1+tau)/3, z3=2(1+tau)/3
    # Kaehler distance: d_ij^2 = |z_i - z_j|^2 / Im(tau)
    def distances_from_positions(tau):
        z1 = 0.0
        z2 = (1 + tau)/3
        z3 = 2*(1 + tau)/3
        d12_sq = abs(z2 - z1)**2 / tau.imag
        d23_sq = abs(z3 - z2)**2 / tau.imag
        d13_sq = abs(z3 - z1)**2 / tau.imag
        return d12_sq, d23_sq, d13_sq

    def d12_from_formula(tau):
        return (1 + 2*tau.real + abs(tau)**2) / (9 * tau.imag)

    dist = {}
    for name, tau in [("tau_u", TAU_U), ("tau_d", TAU_D)]:
        d12, d23, d13 = distances_from_positions(tau)
        dist[name] = (d12, d23, d13)
        print(f"\n  {name}:  d12^2 = {d12:.6f}   "
              f"d23^2 = {d23:.6f}   d13^2 = {d13:.6f}")
        C.check(f"  d23^2/d12^2 ({name})", d23/d12, 1.0, rtol=1e-12)
        C.check(f"  d13^2/d12^2 ({name})", d13/d12, 4.0, rtol=1e-12)
        C.check(f"  d12^2 matches formula ({name})",
                d12, d12_from_formula(tau), rtol=1e-12)

    # =====================================================================
    C.head("§10 Effective alpha and the b_u/c_u problem")
    # =====================================================================
    d12u, _, d13u = dist["tau_u"]
    a_u, b_u, c_u = COEFS_RAW[0], COEFS_RAW[1], COEFS_RAW[2]

    # Effective alpha from (a_u, b_u), assuming a_u at d13^2, b_u at d12^2.
    # DIAGNOSTIC ONLY: the phrase "alpha ~ 10, spread ~ 1.7" was in a
    # removed appendix and is NOT in the v24.5.1 manuscript, so it must
    # not be used as a pass/fail target.
    alpha_ab = np.log(b_u/a_u) / (d13u - d12u)
    print(f"\n  alpha_ab = ln(b_u/a_u) / (d13^2 - d12^2) = {alpha_ab:.2f}")
    print("  [diagnostic only] no explicit manuscript target for alpha_ab;")
    print("  the 'alpha ~ 10, spread 1.7' phrase was in a removed appendix.")

    # Documented limitation: b_u/c_u cannot be explained by a simple
    # exp(-alpha d^2) with O(1) prefactor. This is a NOTE, not a test.
    # v24.5.2 FIX: use the computed alpha_ab, not the hardcoded
    # alpha=10 (which came from the removed Appendix G).
    ratio_bc = b_u / c_u
    print(f"\n  b_u / c_u = {ratio_bc:.2f}  (i.e., c_u is {1/ratio_bc:.4f} of b_u)")
    print(f"  Simple exp(-alpha d^2) with O(1) prefactor would give:")
    print(f"    b_u/c_u = exp(-alpha * (d_x - d12^2))   for some distance d_x")
    print(f"  With alpha_ab = {alpha_ab:.2f} (computed above) and "
          f"b_u/c_u = {ratio_bc:.1f}, we need")
    print(f"    d_x - d12^2 = ln({ratio_bc:.0f})/{alpha_ab:.2f} = "
          f"{np.log(ratio_bc)/alpha_ab:.3f}")
    print(f"  i.e. d_x^2 = {d12u + np.log(ratio_bc)/alpha_ab:.3f}.")
    print(f"  Available distances: d12^2 = {d12u:.3f}, d13^2 = {d13u:.3f}.")
    print(f"  NEITHER matches -> c_u needs a non-O(1) prefactor A_c,")
    print(f"  or is at a position not captured by the simple 3-point picture.")
    print("  [NOTE] This is a genuine limitation of the qualitative")
    print("         interpretation in §10. It does not affect the")
    print("         quantitative fit, but should be stated explicitly.")
    print(f"  (v24.5.2: using the computed alpha_ab instead of the old")
    print(f"   hardcoded alpha=10 leaves the conclusion unchanged: with")
    print(f"   alpha=10 one would need d_x^2 ≈ {d12u + np.log(ratio_bc)/10:.2f};")
    print(f"   with alpha_ab={alpha_ab:.2f} one needs "
          f"d_x^2 ≈ {d12u + np.log(ratio_bc)/alpha_ab:.2f}. Both match")
    print(f"   neither d12^2={d12u:.2f} nor d13^2={d13u:.2f}.)")
    # Not counted as pass or fail: this is a documented limitation.

    # =====================================================================
    C.head("App D Coefficient degeneracy: c_i -> k c_i (on OBSERVABLES)")
    # =====================================================================
    # Correct test: the OBSERVABLES (mass ratios, CKM elements) are
    # invariant under M_u -> k M_u, M_d -> k M_d. We verify this at the
    # observable level with relative tolerance 1e-10. Comparing chi^2 is
    # meaningless here because chi^2 is at roundoff noise.
    for k in [0.5, 1.0, 10.0, 1e3, 1e-3]:
        Muk, Mdk = build_M_quarks(TAU_U, TAU_D, k * COEFS_NORM)
        obs_k = quark_observables(Muk, Mdk)
        max_rel = max(abs(obs_k[key] - obs_q[key]) / abs(obs_q[key])
                      for key in TARGETS)
        status = "OK  " if max_rel < 1e-10 else "FAIL"
        print(f"  [{status}] k = {k:<8.1e}: max rel. deviation in observables "
              f"= {max_rel:.2e}")
        if max_rel < 1e-10: C.passed += 1
        else:               C.failed += 1

    # =====================================================================
    C.head("§9 Full balance (v24.5.1 honest counting)")
    # =====================================================================
    print("  [SCOPE] This block restates the v24.5.1 Table 'Full balance'.")
    print("          It is a bookkeeping convention, not a derivation from")
    print("          the script's own degrees of freedom. The physics input")
    print("          is only that pure GVW does not stabilize the moduli")
    print("          (see meml75_full_hessian.py), hence moduli are free.")
    # v24.5.1 Table 'Full balance': moduli are counted as free,
    # because pure GVW neither predicts nor stabilizes them
    # (Section 'Flux stabilization', Hessian is a saddle, runaway).
    N_quark_ratios = 0     # 3 pure integer + 1 weight-0 discrete
    N_quark_scale  = 1     # a_1
    N_c_d          = 1     # c_d (or 0 if fixed by structural fit)
    N_moduli       = 4     # tau_u, tau_d (2 real each)
    N_lepton_c     = 2     # c_1, c_2 (with c_3 = -c_2)
    N_tau_l        = 2     # tau_l (2 real)
    N_y_nu         = 1
    N_Delta        = 1     # conditional (curvaton)
    N_model = (N_quark_ratios + N_quark_scale + N_c_d + N_moduli
               + N_lepton_c + N_tau_l + N_y_nu + N_Delta)   # = 12
    N_alp  = 1             # theta_i
    N_eff  = N_model + N_alp                                    # = 13
    N_obs  = 16
    N_dof  = N_obs - N_eff                                      # = +3
    C.check("N_model (fitted)",     N_model, 12, rtol=0.0)
    C.check("N_eff (with theta_i)", N_eff,   13, rtol=0.0)
    C.check("N_obs",                N_obs,   16, rtol=0.0)
    C.check("N_dof_fit",            N_dof,    3, rtol=0.0)
    print(f"  Breakdown: quark_ratios=0, a_1=1, c_d=1, moduli=4,")
    print(f"             lepton c_i=2, tau_l=2, y_nu=1, Delta=1  -> 12")
    print(f"             + theta_i = 1  -> N_eff = 13")
    print(f"  With c_d fixed by the structural fit: N_eff = 12, N_dof = +4")
    print("  Moduli are free: pure GVW neither predicts nor stabilizes")
    print("  them (see v24.5.1 'Flux stabilization', Table of Hessians,")
    print("  and meml75_gvw_hessian_cs.py).")

    # =====================================================================
    C.head("Summary")
    # =====================================================================
    print(f"  Checks passed: {C.passed}")
    print(f"  Checks failed: {C.failed}")
    print(f"  Elapsed:       {time.time()-t0:.2f}s")
    print(f"\n  chi2_quarks = {chi2_q:.3e}  (overfit; NOT evidence — see §7)")
    print(f"  structural fit chi2 = 13.93 (paper) / 29.50 (PDG 2024, 10%)")
    print(f"  theta_i     = {theta_i:.4f}  (manuscript: ~1.174)")
    print(f"  alpha_ab    = {alpha_ab:.2f}  (diagnostic only; no manuscript target)")
    print(f"  N_eff       = {N_eff}  (manuscript v24.5.1: 13)")
    print(f"  N_dof_fit   = +{N_dof}  (manuscript v24.5.1: +3 or +4)")
    print("\n  SCOPE: this script checks code consistency and v24.5.1")
    print("  bookkeeping. It does NOT verify the structural quark fit,")
    print("  the lepton fit, the GVW saddle, or the ALP/WGC numbers;")
    print("  those are in meml75_honest_pure2.py,")
    print("  meml75_tau_l_global_scan_v2.py, meml75_gvw_hessian_cs.py,")
    print("  meml75_wgc_check.py.")
    if C.failed == 0:
        print("\n  ✓ ALL CHECKS PASSED")
    else:
        print(f"\n  ✗ {C.failed} CHECK(S) FAILED")

if __name__ == '__main__':
    main()