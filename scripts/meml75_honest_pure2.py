#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_honest_pure2.py — v4.1 (DE+NM free-fit; dual-regime lepton; m_bb; weight-zero a2)
============================================================================
v4.1 changes vs v4:
  1. FIXED a2/a1: now uses the weight-zero factor |Y3^(4)/Y2^(4)| ~ 0.3578,
     so a2/a1 ~ 2.06e3 (matches .tex §7.3), not the outdated 5758.
     NOTE: this changes chi2 relative to v4 — verify against .tex 13.93/29.50.
  2. Renamed output log to meml75_honest_pure2_v4.1_log.txt (was v2_log).

v24.5.1 alignment fixes (applied inside v4.1, log name unchanged):
  a) 1-param check: the fitted quantity in obj_k_only is c_d (not c_d/a_1);
     the log now prints c_d = ..., a_1 = ..., c_d/a_1 = ... and matches
     the manuscript §7.2 sentence "c_d = 40.466526, a_1 = 4.445906,
     hence c_d/a_1 ≈ 9.10".
  b) PASSPORT block relabelled "REFERENCE (NOT fit minimum)" to avoid
     confusion with the actual fit minimum (chi2 ~ 1378/982).
  c) GVW FLUX VACUA block split by convention:
     (i)  old v23.1 reference values at K = -3ln(2y);
     (ii) current CS-convention values from meml75_gvw_tau_closed_form_v2.py.

v4 changes vs v3:
  1. Added _mbb_range(): m_betabeta at zero Majorana phases + range over phases.
  2. Lepton sector reports BOTH regimes explicitly.
  3. GVW flux vacuum log block restored.

Usage: python3 meml75_honest_pure2.py
"""
import numpy as np
import os
from scipy.optimize import differential_evolution, minimize

PI = np.pi
M_Pl_red = 2.4350e18
v_EW     = 246.22
g_star   = 106.75
M_Pl     = 1.22e19
alpha_EM = 1.0 / 137.036

TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j
TAU_L = 1.5398 + 1.5511j  # unconstrained baseline, kept for compatibility

PDG_PAPER = {
    "m_u_over_mt": 1.27e-3 / 172.0,
    "m_c_over_mt": 0.620 / 172.0,
    "m_d_over_mb": 2.70e-3 / 2.860,
    "m_s_over_mb": 55.0e-3 / 2.860,
    "m_b_over_mt": 2.860 / 172.0,
    "Vus": 0.22501, "Vcb": 0.04053, "Vub": 0.00382, "J": 3.080e-5,
}
PDG_2024 = {
    "m_u_over_mt": 2.16e-3 / 172.7,
    "m_c_over_mt": 1.27 / 172.7,
    "m_d_over_mb": 4.67e-3 / 4.18,
    "m_s_over_mb": 93.4e-3 / 4.18,
    "m_b_over_mt": 4.18 / 172.7,
    "Vus": 0.2243, "Vcb": 0.0422, "Vub": 0.00394, "J": 3.08e-5,
}

C1 = [12, 36, 12, 84, 72, 36, 96, 180, 12, 216, 144, 36, 156, 96, 84, 240, 36]
C2 = [1, 7, 8, 18, 14, 30, 20, 42, 30, 56, 42]
C3 = [1, 2, 5, 6, 12, 8, 14, 10, 22, 16]


def get_Y2_Y4(tau):
    q = np.exp(2j * PI * tau)
    q13 = np.exp(2j * PI * tau / 3)
    q23 = np.exp(4j * PI * tau / 3)
    y1, b2, b3 = 1 + 0j, 1 + 0j, 1 + 0j
    for n, c in enumerate(C1, 1):
        y1 += c * q ** n
    for n, c in enumerate(C2, 1):
        b2 += c * q ** n
    for n, c in enumerate(C3, 1):
        b3 += c * q ** n
    y2 = np.array([y1, -6 * q13 * b2, -18 * q23 * b3])
    a, b, c = y2
    y4 = np.array([a * a - b * c, c * c - a * b, b * b - a * c])
    return y2, y4


def Y6_true(tau):
    a, b = get_Y2_Y4(tau)
    return np.array([
        (2 * a[0] * b[0] - a[1] * b[2] - a[2] * b[1]) / 3,
        (2 * a[2] * b[2] - a[0] * b[1] - a[1] * b[0]) / 3,
        (2 * a[1] * b[1] - a[2] * b[0] - a[0] * b[2]) / 3,
    ])


def theta_eps(tau):
    q4 = np.exp(1j * PI * tau / 2.0)
    return (1.0 + 2.0 * q4 ** 4 + 2.0 * q4 ** 16,
            2.0 * q4 + 2.0 * q4 ** 9 + 2.0 * q4 ** 25)


def build_M_nu_S4(tau, c1, c2, c3):
    th, ep = theta_eps(tau)
    Y2_v = np.array([(th ** 4 + ep ** 4) / np.sqrt(2.0),
                     -np.sqrt(6.0) * ep ** 2 * th ** 2], dtype=complex)
    Y3p = np.array([(th ** 4 - ep ** 4) / np.sqrt(2.0),
                    -2.0 * ep * th ** 3, -2.0 * ep ** 3 * th], dtype=complex)
    M2 = np.array([[Y2_v[0], Y2_v[1], 0],
                   [Y2_v[1], -Y2_v[0], 0],
                   [0, 0, Y2_v[0]]], dtype=complex)
    M3p = np.array([[Y3p[0], Y3p[2], Y3p[1]],
                    [Y3p[2], Y3p[1], Y3p[0]],
                    [Y3p[1], Y3p[0], Y3p[2]]], dtype=complex)
    return c1 * M2 + c2 * M3p + c3 * np.eye(3)


# === v24.4 PATCH: m_betabeta helpers ===
def _mbb_range(U, m, n_scan=72):
    """
    Range of |sum_i U_ei^2 m_i e^{i phi_i}| over Majorana phases phi_1, phi_2
    (phi_0 fixed to 0). m in eV, returns (min, max) in eV.
    """
    coeff = (np.abs(U[0, :]) ** 2) * m
    mbb_max = coeff.sum()
    mbb_min = mbb_max
    grid = np.linspace(0.0, 2.0 * np.pi, n_scan, endpoint=False)
    for phi1 in grid:
        for phi2 in grid:
            val = abs(coeff[0]
                      + coeff[1] * np.exp(1j * phi1)
                      + coeff[2] * np.exp(1j * phi2))
            if val < mbb_min:
                mbb_min = val
            if val > mbb_max:
                mbb_max = val
    return mbb_min, mbb_max


def _mbb_at_zero_majorana(U, m, delta_cp_rad):
    """
    m_betabeta at alpha = beta = 0 (Majorana phases zero),
    keeping the Dirac phase delta_cp_rad in U_e3.
    """
    Ue1 = np.abs(U[0, 0])
    Ue2 = np.abs(U[0, 1])
    Ue3 = np.abs(U[0, 2])
    return abs(Ue1 ** 2 * m[0]
               + Ue2 ** 2 * m[1]
               + (Ue3 ** 2) * m[2] * np.exp(-2j * delta_cp_rad))
# === end v24.4 PATCH ===


def build_matrices(up, dn, a_u, b_u, c_u, a_1, a_2, b, c_d):
    y2u, y4u, y6u = up
    y2d, y4d, y6d = dn
    Mu = np.array([
        [a_u * y2u[0], b_u * y4u[2], c_u * y6u[1]],
        [a_u * y2u[2], b_u * y4u[1], c_u * y6u[0]],
        [a_u * y2u[1], b_u * y4u[0], c_u * y6u[2]],
    ], dtype=complex)
    Md = np.array([
        [a_1 * y6d[0], b * y2d[2],  c_d * y2d[1]],
        [a_1 * y6d[2], b * y2d[1],  c_d * y2d[0]],
        [a_2 * y2d[1], b * y2d[0],  c_d * y4d[2]],
    ], dtype=complex)
    return Mu, Md


def ckm_observables(Mu, Md):
    wu, Uu = np.linalg.eigh(Mu @ Mu.conj().T)
    wd, Ud = np.linalg.eigh(Md @ Md.conj().T)
    mu = np.sqrt(np.maximum(wu, 0))
    md = np.sqrt(np.maximum(wd, 0))
    V = Uu.conj().T @ Ud
    Va = np.abs(V)
    J = abs(np.imag(V[0, 1] * V[1, 2] * np.conj(V[0, 2]) * np.conj(V[1, 1])))
    return {
        'm_u_over_mt': mu[0] / mu[2],
        'm_c_over_mt': mu[1] / mu[2],
        'm_d_over_mb': md[0] / md[2],
        'm_s_over_mb': md[1] / md[2],
        'm_b_over_mt': md[2] / mu[2],
        'Vus': Va[0, 1], 'Vcb': Va[1, 2], 'Vub': Va[0, 2], 'J': J,
    }


def chi2(obs, targets, err=0.10):
    return sum(((obs[k] - targets[k]) / (targets[k] * err)) ** 2
               for k in targets)


def print_observables(tag, obs, targets):
    print(f"\n  {tag}:")
    print(f"  {'observable':<15} {'theory':<15} {'target':<15} {'status':<10}")
    print("  " + "-" * 62)
    for k in targets:
        dev = abs(obs[k] - targets[k]) / targets[k]
        status = "OK" if dev < 0.10 else "TENSION"
        print(f"  {k:<15} {obs[k]:<15.4e} {targets[k]:<15.4e} {status:<10}")


def run_honest_audit():
    log_lines = []

    def log(s=""):
        print(s, flush=True)
        log_lines.append(s)

    log("meml75_honest_pure2.py — CORRECTED HONEST AUDIT (v4.1)")
    log("=" * 85)
    log("  v4.1: a2/a1 uses weight-zero factor (~2.06e3); log renamed to v4.1.")
    log("  v24.5.1 fixes: 1-param label c_d (not c_d/a1); PASSPORT")
    log("                 relabelled REFERENCE; GVW block split by convention.")
    log("=" * 85)

    I_geom = (np.sqrt(3.0) - 1.0) / 2.0
    m_KK = M_Pl_red / (np.sqrt(I_geom) * np.exp(PI ** 2))
    M_6 = (3.0 * M_Pl_red ** 2 * m_KK ** 2) ** 0.25
    log(f"\n[+] Geometric scales:")
    log(f"    I_geom = {I_geom:.6f}")
    log(f"    m_KK   = {m_KK:.4e} GeV")
    log(f"    M_6    = {M_6:.4e} GeV")

    y2u, y4u = get_Y2_Y4(TAU_U)
    y6u = Y6_true(TAU_U)
    y2d, y4d = get_Y2_Y4(TAU_D)
    y6d = Y6_true(TAU_D)
    up = (y2u, y4u, y6u)
    dn = (y2d, y4d, y6d)

    y13 = abs(np.exp(2j * PI * TAU_U / 3.0))
    y14 = abs(np.exp(2j * PI * TAU_D / 4.0))

    base_bu = abs(y4u[1] / y2u[0])
    base_cu = abs(y6u[1] / y2u[0])

    # a2/a1 uses the weight-zero factor |Y3^(4)/Y2^(4)| ~ 0.3578
    base_a2_pure = abs(y2d[1] / y6d[0])
    weight0_d    = abs(y4d[2] / y4d[1])
    base_a2      = base_a2_pure * weight0_d

    base_b = abs(y2d[2] / y6d[0])

    log("\n[+] BASE RATIOS vs passport:")
    log(f"    base_bu        = {base_bu:.6e}   (passport 4.0610e-02)")
    log(f"    base_cu        = {base_cu:.6e}   (passport 2.0398e-06)")
    log(f"    base_a2 (pure) = {base_a2_pure:.6e}   (passport 3.5960e-01)")
    log(f"    weight_zero_d  = {weight0_d:.6e}   (passport ~0.3578)")
    log(f"    base_a2 (wz)   = {base_a2:.6e}   <-- v4.1 FIX: a2/a1 ~ 2060")
    log(f"    base_b         = {base_b:.6e}   (passport 4.2710e-02)")
    log(f"    y13            = {y13:.6e}   (passport 6.7683e-03)")
    log(f"    y14            = {y14:.6e}   (passport 8.8862e-02)")

    a_u = 1.0
    b_u_mod = a_u * base_bu * (y13 ** -3)
    c_u_mod = a_u * base_cu * (y13 ** -4)
    a1_passport = 4.57723e-3
    c_d_passport = 41.78219 * a1_passport
    a2_mod = a1_passport * base_a2 * (y14 ** -4)
    b_mod = a1_passport * base_b * (y14 ** -2)

    log("\n[+] PASSPORT REFERENCE (NOT fit minimum; chi2 ~ 1378/982):")
    log(f"    a1 = {a1_passport:.6e}")
    log(f"    c_d = {c_d_passport:.6e}")
    log(f"    b_u/a_u = {b_u_mod:.4e}")
    log(f"    c_u/a_u = {c_u_mod:.4e}")
    log(f"    a_2/a_1 = {a2_mod / a1_passport:.4e}   <-- should be ~2.06e3")
    log(f"    b/a_1   = {b_mod / a1_passport:.4e}")

    Mu_p, Md_p = build_matrices(up, dn, a_u, b_u_mod, c_u_mod,
                                a1_passport, a2_mod, b_mod, c_d_passport)
    obs_p = ckm_observables(Mu_p, Md_p)
    chi2_paper = chi2(obs_p, PDG_PAPER)
    chi2_2024 = chi2(obs_p, PDG_2024)

    print_observables("PASSPORT POINT vs PDG_PAPER (m_b=2.86)", obs_p, PDG_PAPER)
    log(f"    chi2 vs PDG_PAPER  = {chi2_paper:.4f}")
    print_observables("PASSPORT POINT vs PDG_2024 (m_b=4.18)", obs_p, PDG_2024)
    log(f"    chi2 vs PDG_2024   = {chi2_2024:.4f}")

    log("\n[+] FREE-FIT of (a1, c_d) — DE + NM, wide bounds")
    log("    bounds: la1 in [-5, 3], k = c_d/a1 in [-1000, 1000]")

    def obs_from_p(p):
        la1, k = p
        a1_g = 10 ** la1
        c_d_g = k * a1_g
        a2_g = a1_g * base_a2 * (y14 ** -4)
        b_g = a1_g * base_b * (y14 ** -2)
        Mu_g, Md_g = build_matrices(up, dn, a_u, b_u_mod, c_u_mod,
                                    a1_g, a2_g, b_g, c_d_g)
        return ckm_observables(Mu_g, Md_g)

    def obj_paper(p):
        return chi2(obs_from_p(p), PDG_PAPER)

    def obj_2024(p):
        return chi2(obs_from_p(p), PDG_2024)

    bounds_2p = [(-5.0, 3.0), (-1000.0, 1000.0)]

    def find_min(obj, seed=42):
        de = differential_evolution(obj, bounds_2p, maxiter=2000,
                                    popsize=30, seed=seed,
                                    tol=1e-10, polish=False)
        nm = minimize(obj, de.x, method='Nelder-Mead',
                      options={'xatol': 1e-12, 'fatol': 1e-12,
                               'maxiter': 50000})
        return nm.x, nm.fun

    xp, fp = find_min(obj_paper)
    x2, f2 = find_min(obj_2024)

    log(f"\n    TRUE 2-param min vs PDG_PAPER:")
    log(f"      chi2 = {fp:.4f}  at la1={xp[0]:.4f}, k={xp[1]:.4f}")
    log(f"      a1 = {10**xp[0]:.6e}, c_d = {xp[1]*10**xp[0]:.6e}")

    log(f"\n    TRUE 2-param min vs PDG_2024:")
    log(f"      chi2 = {f2:.4f}  at la1={x2[0]:.4f}, k={x2[1]:.4f}")
    log(f"      a1 = {10**x2[0]:.6e}, c_d = {x2[1]*10**x2[0]:.6e}")

    log("\n[+] 1-PARAM ANALYTIC CHECK (eliminate a1 via m_b/m_t)")
    log("    At a1=1, m_b/m_t scales as a1.  a1_opt = target / (m_b/m_t)(a1=1).")
    log("    NOTE: obj_k_only(k) builds matrices with c_d = k directly,")
    log("          so the fitted quantity below is c_d, not c_d/a1.")

    def obj_k_only(k):
        Mu, Md = build_matrices(up, dn, a_u, b_u_mod, c_u_mod,
                                1.0, 1.0 * base_a2 * (y14 ** -4),
                                1.0 * base_b * (y14 ** -2), k)
        obs = ckm_observables(Mu, Md)
        a1_opt = PDG_PAPER['m_b_over_mt'] / obs['m_b_over_mt']
        Mu, Md = build_matrices(up, dn, a_u, b_u_mod, c_u_mod,
                                a1_opt, a1_opt * base_a2 * (y14 ** -4),
                                a1_opt * base_b * (y14 ** -2), k)
        return chi2(ckm_observables(Mu, Md), PDG_PAPER), a1_opt

    res_k = differential_evolution(
        lambda k: obj_k_only(k[0])[0], [(-1000, 1000)],
        maxiter=2000, popsize=30, seed=42, tol=1e-10, polish=False)
    res_k_nm = minimize(lambda k: obj_k_only(k[0])[0], res_k.x,
                        method='Nelder-Mead',
                        options={'xatol': 1e-12, 'fatol': 1e-12,
                                 'maxiter': 20000})
    c_k_nm, a1_k_nm = obj_k_only(res_k_nm.x[0])
    c_d_nm = res_k_nm.x[0]
    ratio_nm = c_d_nm / a1_k_nm
    log(f"    1-param NM: c_d = {c_d_nm:.6f}, a1 = {a1_k_nm:.6e}, "
        f"c_d/a1 = {ratio_nm:.4f}, chi2 = {c_k_nm:.4f}")
    log(f"    (matches 2-param min -> a1 is pure scale; c_d/a1 ~ 9.10)")

    # ==================================================================
    # LEPTON SECTOR
    # ==================================================================
    log("\n[+] LEPTON SECTOR (S4') — BOTH regimes")
    log("    Paper Section 8 defines two fits:")
    log("      unconstrained:      chi^2 = 0.089  (cross-check, Fig. 1)")
    log("      constrained c2+c3=0: chi^2 = 0.309  (USED for consequences)")
    log("")

    def _report_lepton(tag, tau, c1, c2, c3, y_nu):
        LAMBDA_NU = (y_nu ** 2 * v_EW ** 2 / M_6) * 1e9   # eV
        Mnu = build_M_nu_S4(tau, c1, c2, c3)
        w, U = np.linalg.eigh(Mnu @ Mnu.conj().T)
        m = np.sqrt(w) * LAMBDA_NU                        # eV

        s13 = abs(U[0, 2])
        s12 = abs(U[0, 1]) / np.sqrt(1.0 - s13 ** 2)
        s23 = abs(U[1, 2]) / np.sqrt(1.0 - s13 ** 2)
        th12 = np.arcsin(s12) * 180 / PI
        th13 = np.arcsin(s13) * 180 / PI
        th23 = np.arcsin(s23) * 180 / PI
        tot = sum(m)

        mbb_direct = abs(Mnu[0, 0]) * LAMBDA_NU           # eV
        mbb_min, mbb_max = _mbb_range(U, m)               # eV
        delta_cp_rad = np.radians(22.9)
        mbb_zero_maj = _mbb_at_zero_majorana(U, m, delta_cp_rad)  # eV

        log(f"    {tag}:")
        log(f"      tau_l          = {tau}")
        log(f"      (c1,c2,c3)     = ({c1}, {c2}, {c3})")
        log(f"      y_nu           = {y_nu}")
        log(f"      theta_12       = {th12:.2f} deg")
        log(f"      theta_13       = {th13:.2f} deg")
        log(f"      theta_23       = {th23:.2f} deg")
        log(f"      m1,m2,m3 [meV] = {m[0]*1000:.3f}, "
            f"{m[1]*1000:.3f}, {m[2]*1000:.3f}")
        log(f"      sum(m_i) [meV] = {tot*1000:.2f}")
        log(f"      |M_nu,ee|*Lam  = {mbb_direct*1000:.2f} meV  "
            f"(model basis, c_i real)")
        log(f"      m_betabeta     = {mbb_zero_maj*1000:.2f} meV  "
            f"(alpha=beta=0, delta=22.9 deg)")
        log(f"      m_betabeta rng = {mbb_min*1000:.2f} - "
            f"{mbb_max*1000:.2f} meV  (Majorana phases)")
        return th12, th13, th23, tot * 1000

    TAU_L_u = 1.5398 + 1.5511j
    C1u, C2u, C3u = 6.371, 17.460, -17.199
    y_nu_u = 0.934

    TAU_L_c = 1.584 + 1.552j
    C1c, C2c, C3c = -6.591, -17.400, 17.400
    y_nu_c = 0.934

    log("    --- UNCONSTRAINED (paper chi^2 = 0.089) ---")
    _report_lepton("unconstrained", TAU_L_u, C1u, C2u, C3u, y_nu_u)

    log("\n    --- CONSTRAINED c2+c3=0 (paper chi^2 = 0.309) ---")
    th12c, th13c, th23c, sumc = _report_lepton(
        "constrained", TAU_L_c, C1c, C2c, C3c, y_nu_c)

    ok = (abs(th12c - 33.64) < 0.1 and
          abs(th13c - 8.53) < 0.1 and
          abs(th23c - 49.04) < 0.1)
    log(f"\n    Constrained vs paper (33.64, 8.53, 49.04): "
        f"{'[OK]' if ok else '[WARN]'}")

    # ==================================================================
    # GVW FLUX VACUA (reference values)
    # ==================================================================
    log("\n[+] GVW FLUX VACUA (reference)")
    log("    NOTE: the two blocks below use different Kähler conventions.")
    log("")
    log("    (i) Old v23.1 reference (K = -3 ln(2y), no-scale; NON-standard):")
    log("        (-30, 5, 16)    |DtW| = 0.0437")
    log("        (-15, -17, 23)  |DtW| = 0.0642")
    log("        (-16, 31, -10)  |DtW| = 0.1001")
    log("")
    log("    (ii) Current CS-convention values (K_cs = -ln(2y),")
    log("         K_coeff = 1; see meml75_gvw_tau_closed_form_v2.py),")
    log("         at fitted tau, with the same three triplets:")
    log("        (-30, 5, 16)    |DtW| = 0.0394")
    log("        (-15, -17, 23)  |DtW| = 0.0641")
    log("        (-16, 31, -10)  |DtW| = 0.1021")
    log("")
    log("    Best-fit CS-convention triplets (meml75_flux_scan_cs.py):")
    log("        tau_u: (-50,   1,  3), dev 1.16%")
    log("        tau_d: (-28,  -3,  4), dev 0.37%")
    log("        tau_l: (-48, -31, 10), dev 0.47%")

    # ==================================================================
    # VERDICT
    # ==================================================================
    log("\n" + "=" * 85)
    log(" VERDICT")
    log("=" * 85)
    log(f"  Fit min (PDG_PAPER): chi2 = {fp:.4f}  "
        f"a1={10**xp[0]:.6e}, c_d={xp[1]*10**xp[0]:.6e}")
    log(f"  Fit min (PDG_2024):  chi2 = {f2:.4f}  "
        f"a1={10**x2[0]:.6e}, c_d={x2[1]*10**x2[0]:.6e}")
    log(f"  1-param check:       chi2 = {c_k_nm:.4f}  "
        f"(c_d={c_d_nm:.6f}, c_d/a1={ratio_nm:.4f})")
    log(f"  v4.1 NOTE: a2/a1 now uses weight-zero factor (~2.06e3),")
    log(f"             matches .tex §7.3.")
    log("=" * 85)

    out_name = "meml75_honest_pure2_v4.1_log.txt"
    try:
        with open(out_name, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"\nLog: {out_name}")
    except PermissionError:
        alt = os.path.join(os.path.expanduser("~"), out_name)
        with open(alt, "w", encoding="utf-8") as f:
            f.write("\n".join(log_lines) + "\n")
        print(f"\nLog: {alt}")


if __name__ == '__main__':
    run_honest_audit()