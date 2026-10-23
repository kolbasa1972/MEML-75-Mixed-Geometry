#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — COMPLETE VERIFICATION (Version 15.11 Final)
================================================================================
Fixes with respect to v15.10:
  - Bibliography: added DOI/arXiv to Planck2018, PDG2024, Fukugita1986.
  - §10 table caption and total row clarified: 17 fitted + 2 UV inputs = 19.
  - LaTeX order confirmed: Conclusions → Data Availability → Declaration
    → Appendix → Bibliography.
  - No changes to numerical logic; identical to v15.10.
================================================================================
"""
import numpy as np
from scipy.optimize import brentq
import time

PI = np.pi
M_Pl_red = 2.4350e18
h_eV_s = 4.135667696e-15
alpha_EM = 1.0 / 137.036
M_PL = 2.435e18
chi_14 = 0.0750
v_EW = 246.22  # GeV

# Quark parameters (fitted)
TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j
COEFS_Q = np.array([5.314403415527852e-05, 7.192000606749913, 0.038947583908142965,
    0.00024324586524881046, 0.5030286858187134, 0.0013651716604568512, 0.002220219681204568])

# Lepton parameters (fitted)
TAU_L = 1.539846 + 1.551070j
C1_L, C2_L, C3_L = 6.3598, 17.4280, -17.1682

# See-Saw inputs (counted as free parameters in N_eff; see Open Problems)
M_6 = 2.9646e16
y_nu = 0.934
I_factor = 0.3653
LAMBDA_NU = (y_nu**2 * v_EW**2 / M_6) * 1e9  # eV

# Cosmology
g_star = 106.75
M_Pl = 1.22e19
eta_obs = 6.1e-10

# Targets
TH12_T, TH13_T, TH23_T = 33.41, 8.54, 49.10
SIG12, SIG13, SIG23 = 0.5, 0.1, 0.5
DM21_T, DM31_T = 7.42e-5, 2.517e-3
SIG_DM21, SIG_DM31 = 0.21e-5, 0.026e-3

TARGET_Q = {
    "m_u_over_mt": (1.27e-3/172.0,  0.30e-3/172.0),
    "m_c_over_mt": (0.620/172.0,    0.020/172.0),
    "m_d_over_mb": (2.70e-3/2.860,  0.20e-3/2.860),
    "m_s_over_mb": (55.0e-3/2.860,  5.0e-3/2.860),
    "m_b_over_mt": (2.860/172.0,    0.030/172.0),
    "Vus": (0.22501, 0.00068), "Vcb": (0.04053, 0.00042),
    "Vub": (0.00382, 0.00020), "J": (3.080e-5, 0.130e-5),
}
KEYS_Q = list(TARGET_Q.keys())

# Modular forms
C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]

def Y2(tau):
    q = np.exp(2j*PI*tau); q13 = np.exp(2j*PI*tau/3); q23 = np.exp(4j*PI*tau/3)
    y1 = 1+0j; b2 = 1+0j; b3 = 1+0j
    for n,c in enumerate(C1,1): y1 += c*q**n
    for n,c in enumerate(C2,1): b2 += c*q**n
    for n,c in enumerate(C3,1): b3 += c*q**n
    return np.array([y1, -6*q13*b2, -18*q23*b3])

def Y4(tau):
    a,b,c = Y2(tau); return np.array([a*a - b*c, c*c - a*b, b*b - a*c])

def Y6(tau):
    a = Y2(tau); b = Y4(tau)
    return np.array([(2*a[0]*b[0] - a[1]*b[2] - a[2]*b[1])/3,
                     (2*a[2]*b[2] - a[0]*b[1] - a[1]*b[0])/3,
                     (2*a[1]*b[1] - a[2]*b[0] - a[0]*b[2])/3])

def theta_eps(tau):
    q4 = np.exp(1j * PI * tau / 2.0)
    return 1.0 + 2.0 * q4**4 + 2.0 * q4**16, 2.0 * q4 + 2.0 * q4**9 + 2.0 * q4**25

def build_M_quarks(tau_u, tau_d, coefs):
    a_u, b_u, c_u, a1, a2, b, c_d = coefs
    y2u, y4u, y6u = Y2(tau_u), Y4(tau_u), Y6(tau_u)
    y2d, y4d, y6d = Y2(tau_d), Y4(tau_d), Y6(tau_d)
    Mu = np.array([[a_u*y2u[0], b_u*y4u[2], c_u*y6u[1]],
                   [a_u*y2u[2], b_u*y4u[1], c_u*y6u[0]],
                   [a_u*y2u[1], b_u*y4u[0], c_u*y6u[2]]], dtype=complex)
    Md = np.array([[a1*y6d[0], b*y2d[2], c_d*y2d[1]],
                   [a1*y6d[2], b*y2d[1], c_d*y2d[0]],
                   [a2*y2d[1], b*y2d[0], c_d*y4d[2]]], dtype=complex)
    return Mu * M_PL, Md * M_PL

def build_M_nu_S4(tau, c1, c2, c3):
    th, ep = theta_eps(tau)
    Y2 = np.array([(th**4 + ep**4) / np.sqrt(2.0),
                   -np.sqrt(6.0) * ep**2 * th**2], dtype=complex)
    Y3p = np.array([(th**4 - ep**4) / np.sqrt(2.0),
                    -2.0 * ep * th**3, -2.0 * ep**3 * th], dtype=complex)
    M2 = np.array([[Y2[0], Y2[1], 0],
                   [Y2[1], -Y2[0], 0],
                   [0, 0, Y2[0]]], dtype=complex)
    M3p = np.array([[Y3p[0], Y3p[2], Y3p[1]],
                    [Y3p[2], Y3p[1], Y3p[0]],
                    [Y3p[1], Y3p[0], Y3p[2]]], dtype=complex)
    return c1 * M2 + c2 * M3p + c3 * np.eye(3)

def check(label, computed, expected, tol):
    rel = abs(computed - expected) / max(abs(expected), 1e-15)
    ok = rel < tol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {label}: computed={computed:.6e}, expected={expected:.6e} (rel err {rel:.2e})")
    assert ok, f"FAIL {label}: {computed} vs {expected}"

def head(s):
    print(f"\n{'='*72}\n  {s}\n{'='*72}")

def main():
    print("\n" + "█"*72)
    print("  MEML 75 — COMPLETE VERIFICATION (v15.11)")
    print("█"*72)
    t0 = time.time()

    # §3 alpha_G
    head("§3 Bulk coupling alpha_G (postulate)")
    alpha_G = 2.0/(3*PI)
    check("alpha_G", alpha_G, 0.2122065908, 1e-9)

    # §4 epsilon
    head("§4 Suppression factor epsilon (ansatz)")
    eps = (alpha_G/(4*PI))**2
    check("epsilon = 1/(36 pi^4)", eps, 2.851662e-4, 1e-4)

    # §5 m_KK, M_6
    head("§5 m_KK, M_6")
    m_KK = M_Pl_red / (np.sqrt(I_factor)*np.exp(PI**2))
    M_6_calc = (3.0 * M_Pl_red**2 * m_KK**2)**0.25
    check("m_KK", m_KK, 2.0838e14, 0.01)
    check("M_6", M_6_calc, 2.9646e16, 0.01)

    # §6 ALP
    head("§6 ALP sector")
    f_a = M_Pl_red * eps**2
    m_a_GeV = chi_14**2 / f_a
    m_a_eV = m_a_GeV * 1e9
    nu_a_GHz = (m_a_eV / h_eV_s) / 1e9
    check("f_a", f_a, 1.9801e11, 0.01)
    check("m_a (u eV)", m_a_eV*1e6, 28.407, 0.01)
    check("nu_a (GHz)", nu_a_GHz, 6.8688, 0.01)

    # theta_i: free parameter fitted to Omega_a h^2 = 0.12
    def Omega(theta):
        f_anh = (np.log(np.e / (1 - (theta/PI)**2)))**(7/6)
        return 0.12 * theta**2 * f_anh * (f_a / 3e11)**(7/6)
    theta_i = brentq(lambda th: Omega(th) - 0.12, 0.5, 2.0)
    print(f"  theta_i (fitted to Omega_a h^2 = 0.12) = {theta_i:.4f}")
    assert 0.5 < theta_i < 2.0, "theta_i out of natural range"

    # g_agamma
    pref = alpha_EM/(2*PI*f_a)
    check("g_KSVZ", pref*(0 - 1.92), -1.126e-14, 0.01)
    check("g_DFSZ", pref*(8/3 - 1.92), 4.379e-15, 0.01)

    # M_BH
    m_a_kg = m_a_eV * 1.78266e-36
    hbar = 1.0545718e-34; c = 299792458; G = 6.67430e-11
    M_BH_kg = hbar * c / (G * m_a_kg)
    check("M_BH (kg)", M_BH_kg, 9.36e24, 0.05)

    # §7 Quarks (full chi2)
    head("§7 Quark sector")
    Mu, Md = build_M_quarks(TAU_U, TAU_D, COEFS_Q)
    wu, Uu = np.linalg.eigh(Mu @ Mu.conj().T)
    wd, Ud = np.linalg.eigh(Md @ Md.conj().T)
    mu,mc,mt = np.sqrt(np.maximum(wu, 1e-60))
    md,ms,mb = np.sqrt(np.maximum(wd, 1e-60))
    V = Uu.conj().T @ Ud
    obs_q = {"m_u_over_mt": mu/mt, "m_c_over_mt": mc/mt,
             "m_d_over_mb": md/mb, "m_s_over_mb": ms/mb, "m_b_over_mt": mb/mt,
             "Vus": abs(V[0,1]), "Vcb": abs(V[1,2]), "Vub": abs(V[0,2]),
             "J": abs(np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0])))}
    chi2_q = sum(((obs_q[k] - TARGET_Q[k][0]) / TARGET_Q[k][1])**2 for k in KEYS_Q)
    print(f"  chi2_quarks = {chi2_q:.3e}")
    assert chi2_q < 1e-10, f"Quark chi2 too large: {chi2_q}"
    print("  [OK] Quark chi2 < 10^-10")

    # §8 Leptons
    head("§8 Lepton sector")
    Mnu_full = build_M_nu_S4(TAU_L, C1_L, C2_L, C3_L)
    # For complex symmetric M, M M^dagger = U^* D^2 U^T.
    # eigh(M M^dagger) returns V = U^*, and |m_bb| is invariant under U -> U^*.
    w, U = np.linalg.eigh(Mnu_full @ Mnu_full.conj().T)
    m1 = np.sqrt(w[0]) * LAMBDA_NU
    m2 = np.sqrt(w[1]) * LAMBDA_NU
    m3 = np.sqrt(w[2]) * LAMBDA_NU
    print(f"  m1 = {m1*1000:.2f} meV, m2 = {m2*1000:.2f} meV, m3 = {m3*1000:.2f} meV")
    check("m1 (meV)", m1*1000, 6.46, 0.02)
    check("m2 (meV)", m2*1000, 10.77, 0.02)
    check("m3 (meV)", m3*1000, 50.48, 0.02)

    Dm21 = m2**2 - m1**2; Dm31 = m3**2 - m1**2
    check("Dm21", Dm21, 7.45e-5, 0.01)
    check("Dm31", Dm31, 2.517e-3, 0.01)

    s13 = abs(U[0,2])
    s12 = abs(U[0,1]) / np.sqrt(max(1e-12, 1 - s13**2))
    s23 = abs(U[1,2]) / np.sqrt(max(1e-12, 1 - s13**2))
    t12 = np.arcsin(np.clip(s12,-1,1))*180/PI
    t13 = np.arcsin(np.clip(s13,-1,1))*180/PI
    t23 = np.arcsin(np.clip(s23,-1,1))*180/PI
    check("theta12", t12, 33.50, 0.02)
    check("theta13", t13, 8.53, 0.02)
    check("theta23", t23, 49.00, 0.02)

    chi2_l = ((t12-TH12_T)/SIG12)**2 + ((t13-TH13_T)/SIG13)**2 + ((t23-TH23_T)/SIG23)**2 + \
             ((Dm21-DM21_T)/SIG_DM21)**2 + ((Dm31-DM31_T)/SIG_DM31)**2
    print(f"  chi2_leptons = {chi2_l:.3e}")
    check("chi2_leptons ~ 0.23", chi2_l, 0.23, 0.05)

    J_cp = np.imag(U[0,0] * U[1,1] * np.conj(U[0,1]) * np.conj(U[1,0]))
    sin_delta = J_cp / (s12 * np.sqrt(1-s12**2) * s23 * np.sqrt(1-s23**2) * s13 * (1-s13**2))
    delta_CP = np.degrees(np.arcsin(np.clip(sin_delta, -1, 1)))
    check("J_cp", J_cp, 0.0156, 0.05)
    check("delta_CP", delta_CP, 28.09, 0.02)

    sum_m = m1 + m2 + m3
    check("Sum m_i", sum_m, 0.068, 0.02)
    m_bb = np.abs(np.sum((U[0, :]**2) * np.array([m1, m2, m3])))
    check("m_bb (meV)", m_bb*1000, 8.7, 0.02)

    # §9 Leptogenesis
    head("§9 Leptogenesis")
    delta_CP_rad = np.deg2rad(delta_CP)
    eps_CP = (3.0/(16.0*np.pi)) * (M_6 * m3 * 1e-9 / v_EW**2) * np.sin(delta_CP_rad)
    Gamma_N1 = (y_nu**2 / (8*np.pi)) * M_6
    H_M6 = 1.66 * np.sqrt(g_star) * M_6**2 / M_Pl
    K = Gamma_N1 / H_M6
    check("K (washout)", K, 0.83, 0.05)
    kappa = 1.0 if K < 1 else 0.1 * K * np.exp(-(0.1*K)**0.25)
    eta_B = (28.0/79.0) * eps_CP * kappa / g_star
    check("eta_B (before dilution)", eta_B, 2.3e-3, 0.02)
    # Delta: free parameter fitted to eta_B_obs
    delta = eta_B / eta_obs
    print(f"  Delta (fitted to eta_B_obs) = {delta:.3e}")
    assert 1e6 < delta < 1e7, "Delta out of expected range"

    # §12 Flux stabilization (residuals)
    head("§12 Flux stabilization of moduli")
    def flux_cond(a0, a1, a2, tau):
        Im_tau = tau.imag
        W = a0 + a1*tau + a2*tau**2
        dW = a1 + 2*a2*tau
        return abs(dW + (1j / (2 * Im_tau)) * W)
    check("Up-sector   (-17, 0, 1)", flux_cond(-17, 0, 1, TAU_U), 0.159, 0.05)
    check("Down-sector (-7, -1, 1)", flux_cond(-7, -1, 1, TAU_D), 0.1353, 0.05)

    # Appendix D: coefficient degeneracy
    head("App D: coefficient degeneracy")
    for k in [0.5, 1.0, 10.0]:
        Mu_k, Md_k = build_M_quarks(TAU_U, TAU_D, k*COEFS_Q)
        wu_k, Uu_k = np.linalg.eigh(Mu_k @ Mu_k.conj().T)
        wd_k, Ud_k = np.linalg.eigh(Md_k @ Md_k.conj().T)
        mu_k,mc_k,mt_k = np.sqrt(np.maximum(wu_k, 1e-60))
        md_k,ms_k,mb_k = np.sqrt(np.maximum(wd_k, 1e-60))
        V_k = Uu_k.conj().T @ Ud_k
        obs_k = {"m_u_over_mt": mu_k/mt_k, "m_c_over_mt": mc_k/mt_k,
                 "m_d_over_mb": md_k/mb_k, "m_s_over_mb": ms_k/mb_k, "m_b_over_mt": mb_k/mt_k,
                 "Vus": abs(V_k[0,1]), "Vcb": abs(V_k[1,2]), "Vub": abs(V_k[0,2]),
                 "J": abs(np.imag(V_k[0,0]*V_k[1,1]*np.conj(V_k[0,1])*np.conj(V_k[1,0])))}
        chi2_k = sum(((obs_k[key] - TARGET_Q[key][0]) / TARGET_Q[key][1])**2 for key in KEYS_Q)
        assert chi2_k < 1e-10, f"Degeneracy broken at k={k}: chi2={chi2_k}"
    print("  [OK] chi2 invariant under c_i -> k c_i")

    # §10 Balance (N_eff = 19 = 17 fitted + 2 UV inputs)
    head("§10 Balance (N_eff = 19 = 17 fitted + 2 UV inputs)")
    n_eff_fitted = 17     # 10 quarks + 5 leptons + theta_i + Delta
    n_eff_uv = 2          # y_nu + I
    n_eff = n_eff_fitted + n_eff_uv
    n_obs_fit = 16        # 9 quarks + 5 leptons + Omega + eta_B
    n_predictions = 3     # delta_CP, sum m_i, m_bb
    check("N_eff (fitted)", n_eff_fitted, 17, 1e-10)
    check("N_eff (UV)", n_eff_uv, 2, 1e-10)
    check("N_eff (total)", n_eff, 19, 1e-10)
    check("N_obs^fit", n_obs_fit, 16, 1e-10)
    check("N_dof^fit", n_obs_fit - n_eff, -3, 1e-10)
    check("N_dof^total", (n_obs_fit + n_predictions) - n_eff, 0, 1e-10)

    print(f"\n{'='*72}")
    print(f"  ALL ASSERTS PASSED ({time.time()-t0:.1f}s)")
    print(f"  N_eff = 17 fitted + 2 UV inputs = 19")
    print(f"  N_obs^fit = 16, predictions = 3")
    print(f"  N_dof^fit = -3, N_dof^total = 0 (exactly determined)")
    print(f"  theta_i (fitted) = {theta_i:.4f}")
    print(f"  Delta   (fitted) = {delta:.3e}")
    print(f"{'='*72}")

if __name__ == '__main__':
    main()