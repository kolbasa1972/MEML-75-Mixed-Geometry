#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Extended residual symmetry test (7 constraints).

Baseline (refined global scan, v2):
  tau_l = 1.5398 + 1.5511i, c = (6.371, 17.460, -17.199), chi2 = 0.0894.

Constraints (each reduces c1,c2,c3 to 2 free parameters):
  A) c1 + c2 + c3 = 0    c3 = -(c1+c2)
  B) c1 + c3 = 0          c3 = -c1
  C) c1 = c2              c2 = c1
  D) c1 + c2 = 0          c2 = -c1
  E) c2 + c3 = 0          c3 = -c2      <-- most promising (baseline: c2+c3 = 0.261)
  F) c1 = c3              c3 = c1
  G) c2 = c3              c3 = c2

For each constraint:
  Phase 1: fix tau_l = baseline, DE over 2 free c's
  Phase 2: full DE over (Re tau, Im tau, + 2 free c's)

Verdict thresholds:
  chi2_min < 0.5    -> PASS       (adopt)
  0.5 <= chi2 < 2.0 -> MARGINAL   (discuss chi2 cost)
  chi2 >= 2.0       -> FAIL       (do NOT adopt)

Expected outcome (v16.0):
  Only 'c2_plus_c3_zero' PASSes (chi2 ~ 0.309);
  the other six FAIL at chi2 > 38.

Balance (baseline = unconstrained 3-c fit = variant C of the article):
  Baseline (variant C, no reduction):
      N_Lagr = 18, N_eff = 19, N_obs = 16, N_pred = 3
      N_dof^fit = 16 - 19 = -3
      N_dof^total = (16+3) - 19 = 0
  Adopting c2 + c3 = 0 (variant B of the article):
      N_Lagr = 17, N_eff = 18
      N_dof^fit = 16 - 18 = -2
      N_dof^total = (16+3) - 18 = +1
  This matches Table in Section 13 of the article and master_verification_log.txt.

Phase 3 convention: m_bb is computed with the STANDARD PDG PMNS matrix
and vanishing Majorana phases, consistent with master.py (m_bb = 8.369 meV
for the c2+c3=0 fit), NOT with the raw eigh eigenvectors (whose arbitrary
phases would correspond to arbitrary Majorana phases).

Usage:
    python meml75_constraint_test_extended.py
"""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution
from multiprocessing import Pool, cpu_count

# ---------- Constants (matched to meml75_v16.0_master.py, refined scan v2) ----------
PI = np.pi
M_Pl_red = 2.4350e18
M_PL     = 2.435e18
v_EW     = 246.22
y_nu     = 0.934
I_factor = 0.3653

TAU_L_BASE = 1.5398 + 1.5511j
C1_L, C2_L, C3_L = 6.371, 17.460, -17.199

m_KK = M_Pl_red / (np.sqrt(I_factor) * np.exp(PI**2))
M_6  = (3.0 * M_Pl_red**2 * m_KK**2)**0.25
LAMBDA_NU = (y_nu**2 * v_EW**2 / M_6) * 1e9   # eV

TH12_NUFIT, TH13_NUFIT, TH23_NUFIT = 33.41, 8.54, 49.10
SIG12, SIG13, SIG23 = 0.5, 0.1, 0.5
DM21_T, DM31_T     = 7.42e-5, 2.517e-3
SIG_DM21, SIG_DM31 = 0.21e-5, 0.026e-3

BASELINE_CHI2 = 0.0894

CONSTRAINTS = ['sum_zero', 'c1_eq_minus_c3', 'c1_eq_c2',
               'c1_plus_c2_zero', 'c2_plus_c3_zero',
               'c1_eq_c3', 'c2_eq_c3']

CONSTRAINT_LABELS = {
    'sum_zero':         'c1 + c2 + c3 = 0',
    'c1_eq_minus_c3':   'c1 + c3 = 0',
    'c1_eq_c2':         'c1 = c2',
    'c1_plus_c2_zero':  'c1 + c2 = 0',
    'c2_plus_c3_zero':  'c2 + c3 = 0',
    'c1_eq_c3':         'c1 = c3',
    'c2_eq_c3':         'c2 = c3',
}


# ============================================================
# Modular forms
# ============================================================
def theta_eps(tau):
    q4 = np.exp(1j * PI * tau / 2.0)
    return (1.0 + 2.0*q4**4 + 2.0*q4**16,
            2.0*q4 + 2.0*q4**9 + 2.0*q4**25)


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
    """Standard PDG PMNS matrix, as in master.py."""
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


def apply_constraint(x, y, constraint):
    """x, y are the two free parameters; returns (c1, c2, c3)."""
    if constraint == 'sum_zero':
        c1, c2 = x, y
        c3 = -(c1 + c2)
    elif constraint == 'c1_eq_minus_c3':
        c1, c2 = x, y
        c3 = -c1
    elif constraint == 'c1_eq_c2':
        c1, c3 = x, y
        c2 = c1
    elif constraint == 'c1_plus_c2_zero':
        c1, c3 = x, y
        c2 = -c1
    elif constraint == 'c2_plus_c3_zero':
        c1, c2 = x, y
        c3 = -c2
    elif constraint == 'c1_eq_c3':
        c1, c2 = x, y
        c3 = c1
    elif constraint == 'c2_eq_c3':
        c1, c2 = x, y
        c3 = c2
    else:
        raise ValueError(f"Unknown constraint: {constraint}")
    return c1, c2, c3


def build_M_nu_constrained(tau, x, y, constraint):
    c1, c2, c3 = apply_constraint(x, y, constraint)
    return build_M_nu_S4(tau, c1, c2, c3)


def obs_from_Mnu(M, Lam_nu_eV):
    w, U = np.linalg.eigh(M @ M.conj().T)
    m_eV = np.sqrt(np.maximum(w, 1e-30)) * Lam_nu_eV   # eV
    idx = np.argsort(m_eV); m_eV = m_eV[idx]; U = U[:, idx]
    s13 = abs(U[0,2])
    s12 = abs(U[0,1]) / math.sqrt(max(1 - s13**2, 1e-30))
    s23 = abs(U[1,2]) / math.sqrt(max(1 - s13**2, 1e-30))
    th12 = math.degrees(math.asin(min(max(s12, 0), 1)))
    th13 = math.degrees(math.asin(min(max(s13, 0), 1)))
    th23 = math.degrees(math.asin(min(max(s23, 0), 1)))
    dm21 = m_eV[1]**2 - m_eV[0]**2
    dm31 = m_eV[2]**2 - m_eV[0]**2
    return th12, th13, th23, dm21, dm31, m_eV, U


def chi2_l_mnu(M, Lam_nu_eV):
    try:
        th12, th13, th23, dm21, dm31, _, _ = obs_from_Mnu(M, Lam_nu_eV)
    except Exception:
        return 1e10
    ch  = ((th12 - TH12_NUFIT) / SIG12)**2
    ch += ((th13 - TH13_NUFIT) / SIG13)**2
    ch += ((th23 - TH23_NUFIT) / SIG23)**2
    ch += ((dm21 - DM21_T)      / SIG_DM21)**2
    ch += ((dm31 - DM31_T)      / SIG_DM31)**2
    return ch


def predict(M, Lam_nu_eV):
    """Predictions using the STANDARD PDG PMNS matrix (vanishing Majorana
    phases), consistent with master.py and with the article."""
    th12, th13, th23, dm21, dm31, m_eV, U = obs_from_Mnu(M, Lam_nu_eV)

    # CP phase from the Jarlskog invariant (branch A: arcsin in [-90, 90])
    s13 = abs(U[0,2])
    s12 = abs(U[0,1]) / math.sqrt(max(1 - s13**2, 1e-30))
    s23 = abs(U[1,2]) / math.sqrt(max(1 - s13**2, 1e-30))
    J_cp = np.imag(U[0,0]*U[1,1]*np.conj(U[0,1])*np.conj(U[1,0]))
    denom = (s12*math.sqrt(max(1-s12**2, 1e-30))
             * s23*math.sqrt(max(1-s23**2, 1e-30))
             * s13 * (1-s13**2))
    if denom < 1e-15:
        delta_CP = float('nan')
    else:
        sin_d = J_cp / denom
        delta_CP = math.degrees(math.asin(min(max(sin_d, -1), 1)))

    # m_bb with STANDARD PMNS and vanishing Majorana phases
    U_std = pmns_matrix(math.radians(th12), math.radians(th13),
                        math.radians(th23), math.radians(delta_CP))
    m_bb_eV = abs(np.sum((U_std[0, :]**2) * m_eV))

    return dict(
        th12=th12, th13=th13, th23=th23, dm21=dm21, dm31=dm31,
        m1_meV=m_eV[0]*1e3, m2_meV=m_eV[1]*1e3, m3_meV=m_eV[2]*1e3,
        sum_mi_eV=m_eV.sum(), m_bb_meV=m_bb_eV*1e3,
        delta_CP=delta_CP, J_cp=J_cp,
    )


# ============================================================
# DE workers
# ============================================================
def de_fixed_tau_worker(args):
    seed, constraint = args
    bounds = [(-200.0, 200.0), (-200.0, 200.0)]
    def obj(p):
        return chi2_l_mnu(build_M_nu_constrained(TAU_L_BASE, p[0], p[1],
                                                  constraint), LAMBDA_NU)
    res = differential_evolution(obj, bounds, seed=seed, maxiter=200,
                                 popsize=15, tol=1e-8, polish=True)
    c1, c2, c3 = apply_constraint(res.x[0], res.x[1], constraint)
    return dict(seed=seed, constraint=constraint, mode='fixed',
                tau=TAU_L_BASE, c1=c1, c2=c2, c3=c3, chi2=res.fun)


def de_full_worker(args):
    seed, constraint = args
    bounds = [(-0.5, 2.0), (0.87, 2.5),
              (-200.0, 200.0), (-200.0, 200.0)]
    def obj(p):
        tau = complex(p[0], p[1])
        return chi2_l_mnu(build_M_nu_constrained(tau, p[2], p[3],
                                                  constraint), LAMBDA_NU)
    res = differential_evolution(obj, bounds, seed=seed, maxiter=250,
                                 popsize=15, tol=1e-8, polish=True)
    c1, c2, c3 = apply_constraint(res.x[2], res.x[3], constraint)
    return dict(seed=seed, constraint=constraint, mode='full',
                tau=complex(res.x[0], res.x[1]),
                c1=c1, c2=c2, c3=c3, chi2=res.fun)


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    n_cores = min(4, cpu_count())
    print("\n" + "█"*74)
    print(f"  MEML 75 — EXTENDED RESIDUAL SYMMETRY TEST   [cores = {n_cores}]")
    print("█"*74)
    t0 = time.time()

    # --- Baseline overview ---
    print(f"\nBaseline (unconstrained, refined global scan v2):")
    print(f"  tau_l    = {TAU_L_BASE}")
    print(f"  c1,c2,c3 = {C1_L}, {C2_L}, {C3_L}")
    M_base = build_M_nu_S4(TAU_L_BASE, C1_L, C2_L, C3_L)
    chi2_base = chi2_l_mnu(M_base, LAMBDA_NU)
    print(f"  chi2     = {chi2_base:.4e}  (master reference: {BASELINE_CHI2:.4e})")
    pred_base = predict(M_base, LAMBDA_NU)
    print(f"  m_bb     = {pred_base['m_bb_meV']:.3f} meV  "
          f"(standard PMNS, vanishing Majorana phases)")

    # --- Distance from baseline for each constraint ---
    print(f"\nHow far is each constraint from the baseline values?")
    print(f"  {'Constraint':<20} {'expression':<18} {'value':>10} {'|·|':>8}")
    distances = {}
    exprs = {
        'sum_zero':        (C1_L + C2_L + C3_L),
        'c1_eq_minus_c3':  (C1_L + C3_L),
        'c1_eq_c2':        (C1_L - C2_L),
        'c1_plus_c2_zero': (C1_L + C2_L),
        'c2_plus_c3_zero': (C2_L + C3_L),
        'c1_eq_c3':        (C1_L - C3_L),
        'c2_eq_c3':        (C2_L - C3_L),
    }
    for c in CONSTRAINTS:
        v = exprs[c]
        distances[c] = abs(v)
        print(f"  {c:<20} {CONSTRAINT_LABELS[c]:<18} {v:>+10.4f} {abs(v):>8.4f}")
    print(f"\n  [HINT] Smallest |·| = closest to baseline -> most likely to PASS.")
    best_a_priori = min(distances, key=distances.get)
    print(f"  Most promising a priori: '{best_a_priori}' "
          f"(|{CONSTRAINT_LABELS[best_a_priori]}| = {distances[best_a_priori]:.4f})")

    # --- Phase 1: fixed tau ---
    print(f"\n{'='*74}")
    print("PHASE 1. Fix tau_l = baseline, DE over 2 free c's")
    print(f"{'='*74}")
    tasks = [(seed, c) for c in CONSTRAINTS for seed in [42, 43, 44, 45]]
    with Pool(n_cores) as pool:
        res1 = pool.map(de_fixed_tau_worker, tasks)

    best1 = {}
    for r in res1:
        c = r['constraint']
        if c not in best1 or r['chi2'] < best1[c]['chi2']:
            best1[c] = r

    print(f"\n  {'Constraint':<20} {'chi2_min':>12} {'c1':>10} {'c2':>10} "
          f"{'c3':>10} {'verdict':>10}")
    for c in CONSTRAINTS:
        r = best1[c]
        verdict = 'OK' if r['chi2'] < 1.0 else ('MARGINAL' if r['chi2'] < 2.0 else 'FAIL')
        print(f"  {c:<20} {r['chi2']:>12.4e} {r['c1']:>10.3f} {r['c2']:>10.3f} "
              f"{r['c3']:>10.3f} {verdict:>10}")

    # --- Phase 2: full DE ---
    print(f"\n{'='*74}")
    print("PHASE 2. Full DE over (Re tau, Im tau, 2 free c's)")
    print(f"{'='*74}")
    with Pool(n_cores) as pool:
        res2 = pool.map(de_full_worker, tasks)

    best2 = {}
    for r in res2:
        c = r['constraint']
        if c not in best2 or r['chi2'] < best2[c]['chi2']:
            best2[c] = r

    print(f"\n  {'Constraint':<20} {'chi2_min':>12} {'Re tau':>10} {'Im tau':>10} "
          f"{'c1':>10} {'c2':>10} {'c3':>10} {'verdict':>10}")
    for c in CONSTRAINTS:
        r = best2[c]
        verdict = 'OK' if r['chi2'] < 1.0 else ('MARGINAL' if r['chi2'] < 2.0 else 'FAIL')
        print(f"  {c:<20} {r['chi2']:>12.4e} {r['tau'].real:>10.3f} "
              f"{r['tau'].imag:>10.3f} {r['c1']:>10.3f} {r['c2']:>10.3f} "
              f"{r['c3']:>10.3f} {verdict:>10}")

    # --- Phase 3: predictions for PASS / MARGINAL only ---
    print(f"\n{'='*74}")
    print("PHASE 3. Predictions (baseline vs best constrained fits that are")
    print("        PASS or MARGINAL, i.e. chi2 < 2.0)")
    print(f"        m_bb uses the STANDARD PDG PMNS matrix, vanishing Majorana")
    print("        phases (consistent with master.py and the article).")
    print(f"{'='*74}")

    def fmt_row(label, p):
        return (f"  {label:<22} "
                f"m1={p['m1_meV']:7.3f} m2={p['m2_meV']:7.3f} m3={p['m3_meV']:7.3f}  "
                f"sum={p['sum_mi_eV']:8.4f}  mbb={p['m_bb_meV']:7.3f}  "
                f"delta={p['delta_CP']:7.2f}")

    print(f"  {'':22} {'m1 [meV]':>8} {'m2 [meV]':>8} {'m3 [meV]':>8}  "
          f"{'sum [eV]':>9}  {'mbb [meV]':>9}  {'delta [deg]':>10}")
    print(fmt_row("Baseline", pred_base))

    passing_constraints = [c for c in CONSTRAINTS if best2[c]['chi2'] < 2.0]
    for c in passing_constraints:
        r = best2[c]
        M = build_M_nu_S4(r['tau'], r['c1'], r['c2'], r['c3'])
        pred = predict(M, LAMBDA_NU)
        print(fmt_row(f"{c}", pred))
        print(f"    chi2 = {r['chi2']:.4e}  (baseline {chi2_base:.4e})")
        for k in ['m1_meV','m2_meV','m3_meV','sum_mi_eV','m_bb_meV','delta_CP']:
            if abs(pred_base[k]) > 1e-12:
                rel = (pred[k] - pred_base[k]) / pred_base[k]
                print(f"      {k:<10} shift = {rel*100:+7.2f}%")

    if not passing_constraints:
        print("\n  (No constraint passed chi2 < 2.0. All FAIL.)")

    # --- Phase 4: final verdict ---
    print(f"\n{'='*74}")
    print("VERDICT")
    print(f"{'='*74}")
    print(f"  {'Constraint':<20} {'expr':<18} {'chi2':>12} {'verdict':<10}")
    for c in CONSTRAINTS:
        r = best2[c]
        if r['chi2'] < 0.5:
            v = 'PASS'
        elif r['chi2'] < 2.0:
            v = 'MARGINAL'
        else:
            v = 'FAIL'
        print(f"  {c:<20} {CONSTRAINT_LABELS[c]:<18} {r['chi2']:>12.4e} {v:<10}")

    n_pass = sum(1 for c in CONSTRAINTS if best2[c]['chi2'] < 0.5)
    n_marg = sum(1 for c in CONSTRAINTS if 0.5 <= best2[c]['chi2'] < 2.0)
    n_fail = len(CONSTRAINTS) - n_pass - n_marg
    print(f"\n  Summary: {n_pass} PASS, {n_marg} MARGINAL, {n_fail} FAIL")

    if n_pass > 0:
        print(f"\n  [RECOMMENDATION] Adopt a PASS constraint.")
        print(f"                   Baseline is the unconstrained 3-c fit (variant C):")
        print(f"                     N_Lagr = 18, N_eff = 19, N_dof^fit = -3, N_dof^total = 0")
        print(f"                   After imposing the PASS constraint (variant B):")
        print(f"                     N_Lagr: 18 -> 17 (with theta_i: N_eff: 19 -> 18)")
        print(f"                     N_dof^fit: -3 -> -2, N_dof^total: 0 -> +1")
        print(f"                   (matches Table in Section 13 and master_verification_log.txt)")
        pass_names = [c for c in CONSTRAINTS if best2[c]['chi2'] < 0.5]
        for c in pass_names:
            print(f"                   PASS: {c} ({CONSTRAINT_LABELS[c]}), "
                  f"chi2 = {best2[c]['chi2']:.4e}")
    elif n_marg > 0:
        print(f"\n  [RECOMMENDATION] Discuss a MARGINAL constraint in the text,")
        print(f"                   stating the chi2 cost. Do NOT reduce N_eff silently.")
    else:
        print(f"\n  [RECOMMENDATION] Keep 3 independent c_i. N_eff = 19 (variant C).")
        print(f"                   Report in §8: residual symmetries of the S4'")
        print(f"                   structure are incompatible with the data.")

    print(f"\n{'='*74}")
    print(f"  DONE ({time.time()-t0:.1f}s)")
    print(f"{'='*74}")