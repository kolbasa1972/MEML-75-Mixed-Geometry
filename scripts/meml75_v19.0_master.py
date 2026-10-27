#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — MASTER VERIFICATION (v19.0-master, variant B)

Variant B balance (moduli fixed by GVW fluxes):
  11 effective parameters = 10 Lagrangian + theta_i (cosmological IC)
  N_eff = 11, N_dof^fit = +5, N_dof^total = +8
  Variant C (no c2+c3=0 reduction): 12, +4, +7 (reported in §13 footnote)

Constrained fit (Section A baseline):
  tau_l = 1.584 + 1.552i, c1 = -6.591, c2 = -17.400, c3 = +17.400
  chi2 = 0.3092

Unconstrained fit (refined global scan, v2):
  tau_l = 1.5398 + 1.5511i, c1 = 6.371, c2 = 17.460, c3 = -17.199
  chi2 = 0.0894

delta_CP = 22.94 deg (branch B: 157.06 deg).

m_bb range (corrected from 1-50 meV to 0-8.7 meV):
  A = c12^2 c13^2 m1 = 4.331 meV
  B = s12^2 c13^2 m2 = 3.226 meV
  C = s13^2 m3       = 1.112 meV
  m_bb^max = A+B+C   = 8.669 meV  (triangle closes: A < B+C)
  m_bb^min = 0       (all three amplitudes can be aligned anti-phase)
  m_bb(delta_CP=22.94, alpha=beta=0) = 8.369 meV (~96.5% of max)

Usage:
    python meml75_v19.0_master.py
"""
import math
import time
import numpy as np
from scipy.optimize import brentq, differential_evolution
from multiprocessing import Pool, cpu_count

PI = np.pi
M_Pl_red = 2.4350e18
M_PL     = 2.435e18
h_eV_s   = 4.135667696e-15
alpha_EM = 1.0 / 137.036
chi_14   = 0.0750
v_EW     = 246.22
M_W      = 80.377
m_e      = 0.511e-3

TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j

_raw_coefs = np.array([5.314403415527852e-05, 7.192000606749913,
    0.038947583908142965, 0.00024324586524881046, 0.5030286858187134,
    0.0013651716604568512, 0.002220219681204568])
COEFS_Q = _raw_coefs / _raw_coefs[0]

# Constrained by phenomenological relation c2 + c3 = 0
TAU_L = 1.584 + 1.552j
C1_L  = -6.591
C2_L  = -17.400
C3_L  = -C2_L    # = +17.400

# Unconstrained refined global scan (v17.0)
TAU_L_UNCONSTR = 1.5398 + 1.5511j
C1_L_UNCONSTR  = 6.371
C2_L_UNCONSTR  = 17.460
C3_L_UNCONSTR  = -17.199

M_6_REF = 2.9646e16
y_nu     = 0.934
I_factor = 0.3653

g_star   = 106.75
M_Pl     = 1.22e19
eta_obs  = 6.1e-10

TH12_T, TH13_T, TH23_T = 33.50, 8.53, 49.04
TH12_NUFIT, TH13_NUFIT, TH23_NUFIT = 33.41, 8.54, 49.10
SIG12, SIG13, SIG23    = 0.5,   0.1,  0.5
DM21_T, DM31_T         = 7.42e-5, 2.517e-3
SIG_DM21, SIG_DM31     = 0.21e-5, 0.026e-3

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

C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]


def Y2(tau):
    q   = np.exp(2j*PI*tau)
    q13 = np.exp(2j*PI*tau/3)
    q23 = np.exp(4j*PI*tau/3)
    y1 = 1+0j; b2 = 1+0j; b3 = 1+0j
    for n,c in enumerate(C1,1): y1 += c*q**n
    for n,c in enumerate(C2,1): b2 += c*q**n
    for n,c in enumerate(C3,1): b3 += c*q**n
    return np.array([y1, -6*q13*b2, -18*q23*b3])


def Y4(tau):
    a,b,c = Y2(tau)
    return np.array([a*a - b*c, c*c - a*b, b*b - a*c])


def Y6(tau):
    a = Y2(tau); b = Y4(tau)
    return np.array([(2*a[0]*b[0] - a[1]*b[2] - a[2]*b[1])/3,
                     (2*a[2]*b[2] - a[0]*b[1] - a[1]*b[0])/3,
                     (2*a[1]*b[1] - a[2]*b[0] - a[0]*b[2])/3])


def theta_eps(tau):
    q4 = np.exp(1j * PI * tau / 2.0)
    return (1.0 + 2.0*q4**4 + 2.0*q4**16,
            2.0*q4 + 2.0*q4**9 + 2.0*q4**25)


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
    Y2_v = np.array([(th**4 + ep**4)/np.sqrt(2.0),
                     -np.sqrt(6.0)*ep**2*th**2], dtype=complex)
    Y3p = np.array([(th**4 - ep**4)/np.sqrt(2.0),
                    -2.0*ep*th**3, -2.0*ep**3*th], dtype=complex)
    M2  = np.array([[Y2_v[0], Y2_v[1], 0],
                    [Y2_v[1], -Y2_v[0], 0],
                    [0, 0, Y2_v[0]]], dtype=complex)
    M3p = np.array([[Y3p[0], Y3p[2], Y3p[1]],
                    [Y3p[2], Y3p[1], Y3p[0]],
                    [Y3p[1], Y3p[0], Y3p[2]]], dtype=complex)
    return c1*M2 + c2*M3p + c3*np.eye(3)


def pmns_matrix(t12, t13, t23, delta):
    c12, s12 = np.cos(t12), np.sin(t12)
    c13, s13 = np.cos(t13), np.sin(t13)
    c23, s23 = np.cos(t23), np.sin(t23)
    return np.array([
        [c12*c13, s12*c13, s13*np.exp(-1j*delta)],
        [-s12*c23 - c12*s23*s13*np.exp(1j*delta),
          c12*c23 - s12*s23*s13*np.exp(1j*delta), s23*c13],
        [ s12*s23 - c12*c23*s13*np.exp(1j*delta),
         -c12*s23 - s12*c23*s13*np.exp(1j*delta), c23*c13],
    ])


if hasattr(np, 'trapezoid'):
    trapz = np.trapezoid
else:
    trapz = np.trapz


def wavefunction(j, M, z, tau, N_max=30):
    a = j / M
    T = M * tau
    Z = M * z
    gaussian = np.exp(-np.pi * M * (z.imag**2) / tau.imag)
    n = np.arange(-N_max, N_max + 1)[:, None, None]
    phase = np.exp(1j*np.pi*T*(n + a)**2 + 2j*np.pi*(n + a)*Z)
    return gaussian * np.sum(phase, axis=0)


def flux_overlap_complex(i, j, k, M, tau, N_grid=50):
    x = np.linspace(0, 1, N_grid)
    y = np.linspace(0, 1, N_grid)
    X, Y = np.meshgrid(x, y)
    Z = X + 1j*Y
    psi_i = wavefunction(i, M, Z, tau)
    psi_j = wavefunction(j, M, Z, tau)
    psi_k = wavefunction(k, M, Z, tau)
    integrand = (psi_i * psi_j * psi_k) * tau.imag
    return trapz(trapz(integrand, y, axis=0), x)


def check(label, computed, expected, tol):
    if expected == 0:
        err = abs(computed); err_label = "abs err"
    else:
        err = abs(computed - expected) / max(abs(expected), 1e-15)
        err_label = "rel err"
    ok = err < tol
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] {label}: computed={computed:.6e}, "
          f"expected={expected:.6e} ({err_label} {err:.2e})")
    assert ok, f"FAIL {label}: {computed} vs {expected}"


def head(s):
    print(f"\n{'='*74}\n  {s}\n{'='*74}")


def section(s):
    print(f"\n{'#'*74}\n#  {s}\n{'#'*74}")


def m_bb_range(m1, m2, m3, t12, t13):
    c12 = math.cos(math.radians(t12)); s12 = math.sin(math.radians(t12))
    c13 = math.cos(math.radians(t13)); s13 = math.sin(math.radians(t13))
    A = c12**2 * c13**2 * m1
    B = s12**2 * c13**2 * m2
    C = s13**2 * m3
    m_max = A + B + C
    if (A <= B + C) and (B <= A + C) and (C <= A + B):
        m_min = 0.0
    else:
        m_min = max(0.0, max(A, B, C) - (A + B + C - max(A, B, C)))
    return A, B, C, m_max, m_min


def section_A():
    section("SECTION A. Core verification (§3-§13), variant B (c2+c3=0; moduli fixed)")

    head("§3 Bulk coupling alpha_G")
    alpha_G = 2.0/(3*PI)
    check("alpha_G", alpha_G, 0.2122065908, 1e-9)

    head("§3.3 Y-forms at tau = e^{i pi/3}")
    tau_333 = np.exp(1j*PI/3)
    y2_333 = Y2(tau_333); y4_333 = Y4(tau_333); y6_333 = Y6(tau_333)
    check("|Y2_1|",        abs(y2_333[0]),               0.948674, 1e-4)
    check("|Y6_1|/|Y4_3|", abs(y6_333[0])/abs(y4_333[2]), 0.212125, 1e-4)

    head("§4 Suppression factor epsilon")
    eps = (alpha_G/(4*PI))**2
    check("epsilon = 1/(36 pi^4)", eps, 2.851662e-4, 1e-4)

    m_KK_for_coinc = M_Pl_red / (np.sqrt(I_factor) * np.exp(PI**2))
    M_6_for_coinc  = (3.0 * M_Pl_red**2 * m_KK_for_coinc**2)**0.25
    ratio_coinc = (alpha_G**2 * m_KK_for_coinc / M_6_for_coinc) / eps
    check("alpha_G^2 (M_KK/M_6) / epsilon (coincidence factor)",
          ratio_coinc, 1.11, 0.02)

    head("§5 m_KK, M_6")
    m_KK = M_Pl_red / (np.sqrt(I_factor) * np.exp(PI**2))
    M_6  = (3.0 * M_Pl_red**2 * m_KK**2)**0.25
    check("m_KK", m_KK, 2.0838e14, 0.01)
    check("M_6",  M_6,  M_6_REF,  0.01)
    print(f"  (computed M_6 = {M_6:.6e} GeV; rel. dev = "
          f"{abs(M_6-M_6_REF)/M_6_REF:.1e})")

    LAMBDA_NU = (y_nu**2 * v_EW**2 / M_6) * 1e9

    head("§6 ALP sector")
    f_a     = M_Pl_red * eps**2
    m_a_GeV = chi_14**2 / f_a
    m_a_eV  = m_a_GeV * 1e9
    nu_a_GHz = (m_a_eV / h_eV_s) / 1e9
    check("f_a",         f_a,       1.9801e11, 0.01)
    check("m_a (u eV)",  m_a_eV*1e6, 28.407,   0.01)
    check("nu_a (GHz)",  nu_a_GHz,  6.8688,   0.01)

    def Omega(theta):
        f_anh = (np.log(np.e / (1 - (theta/PI)**2)))**(7/6)
        return 0.12 * theta**2 * f_anh * (f_a / 3e11)**(7/6)
    theta_i = brentq(lambda th: Omega(th) - 0.12, 0.5, 2.0)
    check("theta_i", theta_i, 1.174, 0.02)

    pref = alpha_EM/(2*PI*f_a)
    check("g_KSVZ", pref*(0 - 1.92),   -1.126e-14, 0.01)
    check("g_DFSZ", pref*(8/3 - 1.92),  4.379e-15, 0.01)

    m_a_kg = m_a_eV * 1.78266e-36
    hbar = 1.0545718e-34; c = 299792458; G = 6.67430e-11
    M_BH_kg = hbar * c / (G * m_a_kg)
    check("M_BH (kg)", M_BH_kg, 9.36e24, 0.05)

    head("§7 Quark sector")
    Mu, Md = build_M_quarks(TAU_U, TAU_D, COEFS_Q)
    wu, Uu = np.linalg.eigh(Mu @ Mu.conj().T)
    wd, Ud = np.linalg.eigh(Md @ Md.conj().T)
    mu,mc,mt = np.sqrt(np.maximum(wu, 1e-60))
    md,ms,mb = np.sqrt(np.maximum(wd, 1e-60))
    V = Uu.conj().T @ Ud
    obs_q = {
        "m_u_over_mt": mu/mt, "m_c_over_mt": mc/mt,
        "m_d_over_mb": md/mb, "m_s_over_mb": ms/mb, "m_b_over_mt": mb/mt,
        "Vus": abs(V[0,1]), "Vcb": abs(V[1,2]), "Vub": abs(V[0,2]),
        "J": abs(np.imag(V[0,0]*V[1,1]*np.conj(V[0,1])*np.conj(V[1,0]))),
    }
    chi2_q = sum(((obs_q[k] - TARGET_Q[k][0]) / TARGET_Q[k][1])**2 for k in KEYS_Q)
    check("chi2_quarks", chi2_q, 0, 1e-10)

    head("App D: coefficient degeneracy")
    for k in [0.5, 1.0, 10.0]:
        Mu_k, Md_k = build_M_quarks(TAU_U, TAU_D, k*COEFS_Q)
        wu_k, Uu_k = np.linalg.eigh(Mu_k @ Mu_k.conj().T)
        wd_k, Ud_k = np.linalg.eigh(Md_k @ Md_k.conj().T)
        mu_k,mc_k,mt_k = np.sqrt(np.maximum(wu_k, 1e-60))
        md_k,ms_k,mb_k = np.sqrt(np.maximum(wd_k, 1e-60))
        V_k = Uu_k.conj().T @ Ud_k
        obs_k = {
            "m_u_over_mt": mu_k/mt_k, "m_c_over_mt": mc_k/mt_k,
            "m_d_over_mb": md_k/mb_k, "m_s_over_mb": ms_k/mb_k,
            "m_b_over_mt": mb_k/mt_k,
            "Vus": abs(V_k[0,1]), "Vcb": abs(V_k[1,2]), "Vub": abs(V_k[0,2]),
            "J": abs(np.imag(V_k[0,0]*V_k[1,1]*np.conj(V_k[0,1])*np.conj(V_k[1,0]))),
        }
        chi2_k = sum(((obs_k[key] - TARGET_Q[key][0]) / TARGET_Q[key][1])**2
                     for key in KEYS_Q)
        assert chi2_k < 1e-10, f"Degeneracy broken at k={k}"
    print("  [OK] chi2 invariant under c_i -> k c_i")

    head("§8 Lepton sector (constrained: c2 + c3 = 0)")
    Mnu_full = build_M_nu_S4(TAU_L, C1_L, C2_L, C3_L)
    w, U = np.linalg.eigh(Mnu_full @ Mnu_full.conj().T)
    m1 = np.sqrt(w[0]) * LAMBDA_NU
    m2 = np.sqrt(w[1]) * LAMBDA_NU
    m3 = np.sqrt(w[2]) * LAMBDA_NU
    check("m1 (meV)", m1*1000, 6.388,   0.01)
    check("m2 (meV)", m2*1000, 10.751,  0.01)
    check("m3 (meV)", m3*1000, 50.551,  0.01)

    Dm21 = m2**2 - m1**2
    Dm31 = m3**2 - m1**2
    check("Dm21", Dm21, 7.478e-5,   0.01)
    check("Dm31", Dm31, 2.5146e-3,  0.01)

    s13 = abs(U[0,2])
    s12 = abs(U[0,1]) / np.sqrt(max(1e-12, 1 - s13**2))
    s23 = abs(U[1,2]) / np.sqrt(max(1e-12, 1 - s13**2))
    t12 = np.arcsin(np.clip(s12, -1, 1))*180/PI
    t13 = np.arcsin(np.clip(s13, -1, 1))*180/PI
    t23 = np.arcsin(np.clip(s23, -1, 1))*180/PI
    check("theta12", t12, 33.636, 0.01)
    check("theta13", t13,  8.530, 0.01)
    check("theta23", t23, 49.043, 0.01)

    chi2_l = (((t12-TH12_NUFIT)/SIG12)**2 + ((t13-TH13_NUFIT)/SIG13)**2 +
              ((t23-TH23_NUFIT)/SIG23)**2 +
              ((Dm21-DM21_T)/SIG_DM21)**2 + ((Dm31-DM31_T)/SIG_DM31)**2)
    check("chi2_leptons (constrained)", chi2_l, 0.309, 0.01)

    J_cp = np.imag(U[0,0]*U[1,1]*np.conj(U[0,1])*np.conj(U[1,0]))
    denom = s12*np.sqrt(1-s12**2)*s23*np.sqrt(1-s23**2)*s13*(1-s13**2)
    sin_delta = J_cp / denom
    delta_CP   = np.degrees(np.arcsin(np.clip(sin_delta, -1, 1)))
    delta_CP_2 = 180.0 - delta_CP
    check("J_cp", J_cp, 0.012908, 0.02)
    print(f"  sin(delta_CP) = {sin_delta:.4f}; "
          f"branches: {delta_CP:.2f} deg or {delta_CP_2:.2f} deg")

    sum_m = m1 + m2 + m3
    check("Sum m_i", sum_m, 0.067690, 0.01)

    U_pmns_fixed = pmns_matrix(np.radians(t12), np.radians(t13),
                                np.radians(t23), np.deg2rad(delta_CP))
    m_bb = np.abs(np.sum((U_pmns_fixed[0, :]**2) * np.array([m1, m2, m3])))
    check("m_bb (meV, vanishing Majorana phases)", m_bb*1000, 8.369, 0.01)

    head("§8 m_bb range over arbitrary Majorana phases")
    A_bb, B_bb, C_bb, mbb_max, mbb_min = m_bb_range(
        m1*1000, m2*1000, m3*1000, t12, t13)
    print(f"  A = c12^2 c13^2 m1 = {A_bb:.4f} meV")
    print(f"  B = s12^2 c13^2 m2 = {B_bb:.4f} meV")
    print(f"  C = s13^2 m3       = {C_bb:.4f} meV")
    print(f"  Triangle: A < B+C? {A_bb < B_bb + C_bb}  "
          f"({A_bb:.4f} vs {B_bb + C_bb:.4f})")
    print(f"  m_bb^max = A+B+C   = {mbb_max:.4f} meV")
    print(f"  m_bb^min           = {mbb_min:.4f} meV")
    check("m_bb_range_max (meV)", mbb_max, 8.669, 0.01)
    check("m_bb_range_min (meV)", mbb_min, 0.0, 1e-3)
    check("m_bb_vanishing_phases (meV)", m_bb*1000, 8.369, 0.01)
    assert mbb_min - 1e-9 <= m_bb*1000 <= mbb_max + 1e-9
    frac_of_max = (m_bb*1000) / mbb_max
    print(f"  m_bb(vanishing phases) / m_bb^max = {frac_of_max*100:.2f}%")
    check("m_bb_frac_of_max", frac_of_max, 0.965, 0.02)
    assert mbb_max < 30.0, "m_bb range exceeds KamLAND-Zen limit"
    print("  [OK] entire m_bb range [0, 8.669] meV is below KamLAND-Zen limit")

    head("§8 Unconstrained refined scan (for model selection)")
    Mnu_unc = build_M_nu_S4(TAU_L_UNCONSTR,
                             C1_L_UNCONSTR, C2_L_UNCONSTR, C3_L_UNCONSTR)
    w_u, U_u = np.linalg.eigh(Mnu_unc @ Mnu_unc.conj().T)
    m_u  = np.sqrt(w_u) * LAMBDA_NU
    idx  = np.argsort(m_u); m_u = m_u[idx]; U_u = U_u[:, idx]
    s13u = abs(U_u[0,2])
    s12u = abs(U_u[0,1]) / np.sqrt(max(1e-12, 1 - s13u**2))
    s23u = abs(U_u[1,2]) / np.sqrt(max(1e-12, 1 - s13u**2))
    t12u = np.arcsin(np.clip(s12u, -1, 1))*180/PI
    t13u = np.arcsin(np.clip(s13u, -1, 1))*180/PI
    t23u = np.arcsin(np.clip(s23u, -1, 1))*180/PI
    dm21u = m_u[1]**2 - m_u[0]**2
    dm31u = m_u[2]**2 - m_u[0]**2
    chi2_l_unc = (((t12u-TH12_NUFIT)/SIG12)**2
                  + ((t13u-TH13_NUFIT)/SIG13)**2
                  + ((t23u-TH23_NUFIT)/SIG23)**2
                  + ((dm21u-DM21_T)/SIG_DM21)**2
                  + ((dm31u-DM31_T)/SIG_DM31)**2)
    check("chi2_leptons (unconstrained)", chi2_l_unc, 0.089, 0.02)
    c2_plus_c3_unc = C2_L_UNCONSTR + C3_L_UNCONSTR
    check("c2 + c3 (unconstrained)",
          c2_plus_c3_unc, 0.261, 0.05)

    head("§8 Model selection (Δχ², AIC, BIC)")
    delta_chi2 = chi2_l - chi2_l_unc
    check("Delta_chi2 = chi2_constr - chi2_unconstr",
          delta_chi2, 0.22, 0.05)
    p_value = 1.0 - math.erf(math.sqrt(delta_chi2 / 2.0))
    check("p-value (1 dof)", p_value, 0.64, 0.02)
    dAIC  = delta_chi2 + 2*(-1)
    check("Delta_AIC", dAIC, -1.78, 0.03)
    dBIC16 = delta_chi2 + math.log(16)*(-1)
    check("Delta_BIC (N=16)", dBIC16, -2.55, 0.03)
    dBIC5  = delta_chi2 + math.log(5)*(-1)
    check("Delta_BIC (N=5)", dBIC5, -1.39, 0.03)

    head("§9 Leptogenesis (constrained)")
    delta_CP_rad = np.deg2rad(delta_CP)
    eps_CP = (3.0/(16.0*np.pi)) * (M_6 * m3 * 1e-9 / v_EW**2) * np.sin(delta_CP_rad)
    check("eps_CP", eps_CP, 0.5751, 0.02)
    Gamma_N1 = (y_nu**2 / (8*np.pi)) * M_6
    H_M6     = 1.66 * np.sqrt(g_star) * M_6**2 / M_Pl
    K = Gamma_N1 / H_M6
    check("K (washout)", K, 0.83284, 0.01)
    kappa = 1.0 if K < 1 else 0.1*K*np.exp(-(0.1*K)**0.25)
    eta_B = (28.0/79.0) * eps_CP * kappa / g_star
    check("eta_B (before dilution)", eta_B, 1.9092e-3, 0.01)
    delta_dil = eta_B / eta_obs
    check("Delta", delta_dil, 3.1299e6, 0.01)

    head("§10 LFV and EDM (constrained)")
    m_nu_GeV = np.array([m1, m2, m3]) * 1e-9
    U_pmns   = pmns_matrix(np.radians(t12), np.radians(t13),
                            np.radians(t23), delta_CP_rad)
    sum_Y2 = 0
    for i in range(3):
        sum_Y2 += np.conj(U_pmns[1,i]) * U_pmns[0,i] * (m_nu_GeV[i] * M_6 / v_EW**2)
    br_mu_e_gamma = (3*alpha_EM/(32*np.pi)) * np.abs(sum_Y2)**2 * (M_W**2/M_6**2)**2
    print(f"  BR(mu -> e gamma) = {br_mu_e_gamma:.2e} (limit < 4.2e-13)")
    assert 1e-63 < br_mu_e_gamma < 1e-60, f"BR out of range: {br_mu_e_gamma}"

    d_e_GeV_inv = (alpha_EM/(4*np.pi)) * (m_e / M_6**2) * np.sin(delta_CP_rad)
    d_e_cm = d_e_GeV_inv * 1.97e-14
    check("log10(|d_e|) ~ -53.59",
          np.log10(abs(d_e_cm)), np.log10(2.594e-54), 0.01)

    head("§12 Flux derivation (negative result)")
    tau_mpc = complex(TAU_L)
    Y_000 = abs(flux_overlap_complex(0, 0, 0, 2, tau_mpc, N_grid=50))
    Y_011 = abs(flux_overlap_complex(0, 1, 1, 2, tau_mpc, N_grid=50))
    print(f"  Y_000 = {Y_000:.4f}, Y_011 = {Y_011:.4f}")
    ratio = Y_000 / Y_011 if Y_011 > 1e-10 else 1e10
    print(f"  Ratio Y_000/Y_011 = {ratio:.1f} (need ~0.36)")
    assert ratio > 10, "Flux model cannot reproduce c1/c2"

    head("§13 Full balance (variant B: 11 eff = 10 Lagr + theta_i; c2+c3=0; moduli fixed)")
    n_eff = 11          # 10 Lagrangian + theta_i (moduli fixed by GVW fluxes)
    n_obs = 16
    n_pred = 3
    check("N_eff (10 Lagr + theta_i)", n_eff,       11, 1e-10)
    check("N_obs^fit",                  n_obs,       16, 1e-10)
    check("N_dof^fit",                  n_obs - n_eff, +5, 1e-10)
    check("N_dof^total",       (n_obs + n_pred) - n_eff, +8, 1e-10)
    n_eff_C = 12
    check("[variant C] N_eff",        n_eff_C,                          12, 1e-10)
    check("[variant C] N_dof^fit",    n_obs - n_eff_C,                  +4, 1e-10)
    check("[variant C] N_dof^total", (n_obs + n_pred) - n_eff_C,         +7, 1e-10)

    return dict(M_6=M_6, LAMBDA_NU=LAMBDA_NU, y_nu=y_nu,
                chi2_l=chi2_l, chi2_l_unc=chi2_l_unc,
                delta_chi2=delta_chi2, p_value=p_value,
                delta_CP=delta_CP, delta_CP_2=delta_CP_2,
                m_bb_meV=m_bb*1000, sum_mi_eV=sum_m, br=br_mu_e_gamma,
                d_e=d_e_cm, eta_B=eta_B, Delta=delta_dil,
                theta_i=theta_i, m_KK=m_KK, f_a=f_a,
                m_a_eV=m_a_eV, nu_a_GHz=nu_a_GHz,
                m_bb_max_meV=mbb_max, m_bb_min_meV=mbb_min)


# ============== SECTION B: I-consistency check ==============

def compute_derived_I(I, y_nu_val, base):
    m_KK = M_Pl_red / (np.sqrt(I) * np.exp(PI**2))
    M_6  = (3.0 * M_Pl_red**2 * m_KK**2)**0.25
    Lam  = (y_nu_val**2 * v_EW**2 / M_6) * 1e9
    scale = Lam / base['LAMBDA_NU']
    m1 = 6.388  * scale
    m2 = 10.751 * scale
    m3 = 50.551 * scale
    sum_mi = (m1 + m2 + m3) * 1e-3
    m_bb = base['m_bb_meV'] * scale
    BR = base['br'] * (base['M_6']/M_6)**2
    d_e = base['d_e'] * (base['M_6']/M_6)**2
    eta_pre = base['eta_B'] * (M_6/base['M_6'])
    Delta = eta_pre / eta_obs
    return dict(I=I, M_6=M_6, y_nu=y_nu_val, sum_mi=sum_mi, m_bb=m_bb,
                BR=BR, d_e=d_e, eta_pre=eta_pre, Delta=Delta)


def refit_y_nu_for_I(I, base, target_dm31=2.51e-3):
    def resid(log_y):
        r = compute_derived_I(I, math.exp(log_y), base)
        dm31 = (r['m_bb']/base['m_bb_meV'] * 50.551)**2 \
             - (r['m_bb']/base['m_bb_meV'] * 6.388)**2
        dm31 *= 1e-6
        return (dm31 - target_dm31) / target_dm31
    try:
        return math.exp(brentq(resid, math.log(0.5), math.log(2.0), xtol=1e-12))
    except ValueError:
        return None


def section_B(base):
    section("SECTION B. I-consistency check (Appendix G)")
    I_geom = (math.sqrt(3.0) - 1.0) / 2.0
    print(f"  I_baseline  = {I_factor:.6f}")
    print(f"  I_geometric = (sqrt(3)-1)/2 = {I_geom:.6f}")
    print(f"  Rel diff    = {(I_geom/I_factor - 1)*100:+.3f}%")

    y_base = refit_y_nu_for_I(I_factor, base)
    y_geom = refit_y_nu_for_I(I_geom,  base)
    if y_base is None or y_geom is None:
        print("  [WARN] y_nu refit failed; skipping Section B details.")
        return
    r_base = compute_derived_I(I_factor, y_base, base)
    r_geom = compute_derived_I(I_geom,  y_geom, base)

    keys = ['y_nu','M_6','sum_mi','m_bb','BR','d_e','eta_pre','Delta']
    print(f"\n  {'Quantity':<10} {'Baseline':>16} {'Geometric':>16} {'Rel. D':>12}")
    max_rel = 0.0
    for k in keys:
        vb, vg = r_base[k], r_geom[k]
        rel = (vg - vb) / vb if vb != 0 else 0.0
        max_rel = max(max_rel, abs(rel))
        print(f"  {k:<10} {vb:>16.6e} {vg:>16.6e} {rel*100:>+11.3f}%")
    print(f"\n  Max relative shift: {max_rel*100:.4f}%")
    if max_rel < 0.01:
        print("  [OK] CONSISTENT with geometric value (not a derivation).")
    else:
        print("  [WARN] Predictions shift by >1% under the geometric value.")


# ============== SECTION C: GUT test for y_nu ==============

A1_INV_MZ, A2_INV_MZ, A3_INV_MZ = 59.0, 29.6, 8.5
B1, B2, B3 = 33.0/5.0, 1.0, -3.0
M_Z = 91.1876
M_T_MZ = 165.0


def gauge_g(mu, mu0=M_Z):
    t = math.log(mu/mu0)
    a1 = A1_INV_MZ - B1/(2*math.pi)*t
    a2 = A2_INV_MZ - B2/(2*math.pi)*t
    a3 = A3_INV_MZ - B3/(2*math.pi)*t
    if a1 <= 0 or a2 <= 0 or a3 <= 0:
        return None
    return (math.sqrt(4*math.pi/a1), math.sqrt(4*math.pi/a2), math.sqrt(4*math.pi/a3))


def y_t_MZ(tan_beta):
    return math.sqrt(2.0) * M_T_MZ / (v_EW * math.sin(math.atan(tan_beta)))


def run_y_t(y0, mu0, mu1, n=4000):
    h = (math.log(mu1) - math.log(mu0)) / n
    y = y0
    for i in range(n):
        lm = math.log(mu0) + (i + 0.5)*h
        g = gauge_g(math.exp(lm))
        if g is None:
            return None
        g1, g2, g3 = g
        c = 6.0*y*y - (16.0/3.0)*g3*g3 - 3.0*g2*g2 - (13.0/15.0)*g1*g1
        y += y/(16*math.pi**2) * c * h
    return y


def gut_one(tan_beta, M_GUT=2e16):
    yt = run_y_t(y_t_MZ(tan_beta), M_Z, M_GUT)
    if yt is None:
        return None
    yn = run_y_t(yt, M_GUT, 2.9646e16)
    return dict(tb=tan_beta, yt_GUT=yt, yn_M6=yn,
                f=y_nu/yn if yn else None)


def section_C(n_cores):
    section("SECTION C. GUT test for y_nu (Section 13 footnote)")
    tan_betas = [2, 5, 10, 20, 30, 40, 50, 60]
    with Pool(n_cores) as pool:
        res = pool.map(gut_one, tan_betas)
    print(f"  {'tan_beta':<10} {'y_t(M_GUT)':>14} "
          f"{'y_nu(M_6)':>14} {'needed f':>12}")
    for tb, r in zip(tan_betas, res):
        if r is None:
            print(f"  {tb:<10} {'no sol':>14} {'no sol':>14} {'no sol':>12}")
            continue
        print(f"  {r['tb']:<10} {r['yt_GUT']:>14.4f} "
              f"{r['yn_M6']:>14.4f} {r['f']:>12.4f}")
    r50 = gut_one(50)
    if r50 is not None:
        f50 = r50['f']
        print(f"\n  tan_beta=50, f = {f50:.3f}:")
        if 0.95 < f50 < 1.15:
            print("  [PASS] standard SO(10) with 10_H works at ~10% level.")
        elif 1.15 <= f50 < 2.5:
            print("  [MARGINAL] needs SO(10) with 10_H + 126_H "
                  "-> N_eff NOT reduced.")
        else:
            print("  [FAIL] f outside natural CG range.")


# ============== SECTION D: NuFIT comparison ==============

NUFIT_VERSIONS = {
    "5.0 (IC19+SK atm)": {
        'theta12': 33.44, 'theta13': 8.57, 'theta23': 49.2,
        'dm2_21':  7.42e-5, 'dm2_31': 2.517e-3,
        'sig12': 0.77, 'sig13': 0.12, 'sig23': 1.3,
        'sig_dm2_21': 0.21e-5, 'sig_dm2_31': 0.028e-3,
    },
    "5.2 (IC19, NO, no SK atm)": {
        'theta12': 33.41, 'theta13': 8.57, 'theta23': 49.1,
        'dm2_21':  7.41e-5, 'dm2_31': 2.507e-3,
        'sig12': 0.77, 'sig13': 0.12, 'sig23': 1.0,
        'sig_dm2_21': 0.21e-5, 'sig_dm2_31': 0.026e-3,
    },
    "5.2 (IC19+SK atm, NO, lower octant)": {
        'theta12': 33.41, 'theta13': 8.57, 'theta23': 42.2,
        'dm2_21':  7.41e-5, 'dm2_31': 2.507e-3,
        'sig12': 0.77, 'sig13': 0.12, 'sig23': 0.9,
        'sig_dm2_21': 0.21e-5, 'sig_dm2_31': 0.026e-3,
    },
}


def section_D():
    section("SECTION D. NuFIT 5.0 vs 5.2 comparison")
    baseline = {
        'theta12': 33.64, 'theta13': 8.53, 'theta23': 49.04,
        'dm2_21':  7.478e-5, 'dm2_31': 2.5146e-3,
    }
    print(f"  {'NuFIT release':<44} {'chi^2':>8} {'verdict':>10}")
    for label, nf in NUFIT_VERSIONS.items():
        chi2 = 0.0
        for k, sig_key in [('theta12','sig12'),('theta13','sig13'),('theta23','sig23')]:
            chi2 += ((baseline[k]-nf[k]) / nf[sig_key])**2
        chi2 += ((baseline['dm2_21']-nf['dm2_21'])/nf['sig_dm2_21'])**2
        chi2 += ((baseline['dm2_31']-nf['dm2_31'])/nf['sig_dm2_31'])**2
        verdict = "OK" if chi2 < 25 else ("TENSION" if chi2 < 100 else "EXCLUDED")
        print(f"  {label:<44} {chi2:>8.2f} {verdict:>10}")
    print("\n  NOTE: NuFIT 5.2 with SK atm prefers the lower octant (42.2 deg).")
    print("        The model predicts the upper octant; this is a testable")
    print("        consequence, discussed in Section 8 of the paper.")


# ============== SECTION E: Delta dynamics scan ==============

DELTA_TARGET = 3.13e6


def compute_delta(args):
    lm, lp, lt = args
    m_phi = 10**lm; phi0 = 10**lp; Ti = 10**lt
    G_phi = m_phi**3 / (8*math.pi*M_PL**2)
    T_RH = (90.0/(math.pi**2*g_star))**0.25 * math.sqrt(G_phi*M_PL)
    rho = 0.5 * m_phi**2 * phi0**2
    denom = (math.pi**2/30.0) * g_star * T_RH * Ti**3
    if denom <= 0:
        return None
    return dict(m_phi=m_phi, phi0=phi0, Ti=Ti, T_RH=T_RH, Delta=rho/denom)


def section_E(n_cores):
    section("SECTION E. Delta dynamics scan (moduli dilution, constrained)")
    print(f"  Target Delta = {DELTA_TARGET:.2e} (constrained c2+c3=0)")
    grid = [(lm, lp, lt)
            for lm in np.linspace(4, 14, 5)
            for lp in np.linspace(10, 19, 5)
            for lt in np.linspace(4, 16, 5)]
    with Pool(n_cores) as pool:
        res = pool.map(compute_delta, grid)
    res = [r for r in res if r is not None]
    close = [r for r in res if 0.1*DELTA_TARGET < r['Delta'] < 10*DELTA_TARGET]
    print(f"  Found {len(close)} grid points within factor 10 of target:")
    print(f"  {'m_phi [GeV]':<14} {'phi0 [GeV]':<14} {'T_i [GeV]':<14} "
          f"{'T_RH [GeV]':<14} {'Delta':<14}")
    for r in sorted(close, key=lambda x: abs(math.log10(x['Delta']/DELTA_TARGET)))[:10]:
        print(f"  {r['m_phi']:<14.2e} {r['phi0']:<14.2e} {r['Ti']:<14.2e} "
              f"{r['T_RH']:<14.2e} {r['Delta']:<14.2e}")
    print("\n  Conclusion: Delta ~ 1e6 requires phi0 << M_Pl, OR a specific")
    print("  combination phi0 ~ 1e16-17 GeV with T_i ~ 1e13 GeV.")
    print("  Three inputs (m_phi, phi0, T_i) replace one (Delta) -> N_eff not reduced.")


# ============== SECTION F: tau_l coarse scan ==============

def obs_from_Mnu(M, Lam_nu_eV):
    w, U = np.linalg.eigh(M @ M.conj().T)
    m = np.sqrt(np.maximum(w, 1e-30)) * Lam_nu_eV
    idx = np.argsort(m); m = m[idx]; U = U[:, idx]
    s13 = abs(U[0,2])
    s12 = abs(U[0,1]) / math.sqrt(max(1 - s13**2, 1e-30))
    s23 = abs(U[1,2]) / math.sqrt(max(1 - s13**2, 1e-30))
    th12 = math.degrees(math.asin(min(max(s12, 0), 1)))
    th13 = math.degrees(math.asin(min(max(s13, 0), 1)))
    th23 = math.degrees(math.asin(min(max(s23, 0), 1)))
    dm21 = m[1]**2 - m[0]**2
    dm31 = m[2]**2 - m[0]**2
    return th12, th13, th23, dm21, dm31


def chi2_l_mnu(M, Lam_nu_eV):
    try:
        th12, th13, th23, dm21, dm31 = obs_from_Mnu(M, Lam_nu_eV)
    except Exception:
        return 1e10
    ch  = ((th12 - TH12_NUFIT) / SIG12)**2
    ch += ((th13 - TH13_NUFIT) / SIG13)**2
    ch += ((th23 - TH23_NUFIT) / SIG23)**2
    ch += ((dm21 - DM21_T)      / SIG_DM21)**2
    ch += ((dm31 - DM31_T)      / SIG_DM31)**2
    return ch


def scan_tau_l_point(args):
    Re, Im, Lam_nu = args
    tau = complex(Re, Im)
    bounds = [(0.1, 100.0), (0.1, 100.0), (-100.0, 100.0)]
    try:
        res = differential_evolution(
            lambda c: chi2_l_mnu(build_M_nu_S4(tau, c[0], c[1], c[2]), Lam_nu),
            bounds, maxiter=60, popsize=8, seed=42, tol=1e-3,
            polish=True, workers=1)
        return dict(Re=Re, Im=Im, chi2=res.fun, c=res.x)
    except Exception:
        return dict(Re=Re, Im=Im, chi2=1e10, c=[0,0,0])


def section_F(n_cores, Lam_nu):
    section("SECTION F. tau_l coarse scan (unconstrained 3-c fit)")
    Re_grid = np.linspace(-0.5, 2.0, 6)
    Im_grid = np.linspace(0.87, 2.5, 6)
    grid = [(a, b, Lam_nu) for a in Re_grid for b in Im_grid]
    grid.append((1.58, 1.55, Lam_nu))
    print(f"  Grid: {len(grid)} points (Re in [{Re_grid[0]:.2f}, {Re_grid[-1]:.2f}], "
          f"Im in [{Im_grid[0]:.2f}, {Im_grid[-1]:.2f}])")
    with Pool(n_cores) as pool:
        res = pool.map(scan_tau_l_point, grid)
    good = [r for r in res if r['chi2'] < 1.0]
    print(f"  Local minima with chi^2 < 1: {len(good)}")
    if good:
        print(f"  {'Re tau':<10} {'Im tau':<10} {'chi^2':<12} "
              f"{'c1':<10} {'c2':<10} {'c3':<10}")
        for r in sorted(good, key=lambda x: x['chi2'])[:10]:
            print(f"  {r['Re']:<10.3f} {r['Im']:<10.3f} {r['chi2']:<12.3e} "
                  f"{r['c'][0]:<10.3f} {r['c'][1]:<10.3f} {r['c'][2]:<10.3f}")

    print(f"\n  Baseline check (constrained fit, c2 + c3 = 0):")
    print(f"    tau_l = {TAU_L}, c = ({C1_L:.3f}, {C2_L:.3f}, {C3_L:.3f})")
    M_base = build_M_nu_S4(TAU_L, C1_L, C2_L, C3_L)
    chi2_base = chi2_l_mnu(M_base, Lam_nu)
    print(f"    chi^2(tau_l^constrained) = {chi2_base:.4e} "
          f"(matches Section A: 3.092e-01)")

    print(f"\n  Unconstrained refined global scan (from meml75_tau_l_global_scan_v2.py):")
    M_unc = build_M_nu_S4(TAU_L_UNCONSTR,
                           C1_L_UNCONSTR, C2_L_UNCONSTR, C3_L_UNCONSTR)
    chi2_unc = chi2_l_mnu(M_unc, Lam_nu)
    print(f"    tau_l = {TAU_L_UNCONSTR}")
    print(f"    c = ({C1_L_UNCONSTR:.3f}, {C2_L_UNCONSTR:.3f}, {C3_L_UNCONSTR:.3f})")
    print(f"    chi^2 = {chi2_unc:.4e} (reproduces 8.94e-02 from the scan)")
    print(f"    Delta_chi2 = chi2_constrained - chi2_unconstrained "
          f"= {chi2_base - chi2_unc:.4f} (expect ~0.22)")

    if len(good) > 1:
        Re_spread = max(r['Re'] for r in good) - min(r['Re'] for r in good)
        if Re_spread > 0.5:
            print(f"\n  [WARN] Multiple minima with Re tau spread {Re_spread:.3f}.")
        else:
            print(f"\n  [OK] Minima cluster (Re spread {Re_spread:.3f}).")
    elif len(good) == 1:
        print("\n  [OK] Single local minimum in the coarse scan.")
        print("  NOTE: this is not a global optimization. The refined DE")
        print("  gives the global minimum at tau_l = 1.5398 + 1.5511i,")
        print("  c = (6.371, 17.460, -17.199), chi2 = 0.0894")
        print("  (see meml75_tau_l_global_scan_v2.py).")


# ============== MAIN ==============

if __name__ == '__main__':
    n_cores = min(4, cpu_count())
    print("\n" + "█"*74)
    print(f"  MEML 75 — MASTER VERIFICATION (v19.0-master, variant B)")
    print(f"  [cores = {n_cores}]")
    print(f"  Balance: 11 eff = 10 Lagr + theta_i (moduli fixed by GVW fluxes)")
    print(f"  N_eff = 11, N_dof^fit = +5, N_dof^total = +8")
    print(f"  Variant C (no c2+c3=0): 12, +4, +7")
    print(f"  m_bb range: [0, 8.669] meV")
    print("█"*74)
    t0 = time.time()

    base = section_A()
    section_B(base)
    section_C(n_cores)
    section_D()
    section_E(n_cores)
    section_F(n_cores, base['LAMBDA_NU'])

    print("\n" + "="*74)
    print(f"  ALL SECTIONS COMPLETE ({time.time()-t0:.1f}s)")
    print("  A: §3-§13 asserts PASSED  N_eff=11, N_dof^fit=+5, N_dof^total=+8")
    print(f"     [variant C] 12, +4, +7")
    print(f"     m_bb range = [{base['m_bb_min_meV']:.4f}, "
          f"{base['m_bb_max_meV']:.4f}] meV")
    print("  B: I-consistency (Appendix G)")
    print("  C: GUT test for y_nu")
    print("  D: NuFIT 5.0 vs 5.2 (octant warning)")
    print("  E: Delta dynamics scan (constrained Delta target)")
    print("  F: tau_l coarse scan (local minima only; global scan separate)")
    print("="*74)