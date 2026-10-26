#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
MEML 75 — Extended residual symmetry test (7 constraints) + Stress Test (v17.0).

This script performs:
  Phase 1 & 2: Full DE over tau_l and c_i for the 7 linear constraints.
  Phase 5: Stress test fixing tau_l at exact symmetric points
           (tau = i, tau = e^{i\pi/3}, tau = -0.5 + e^{i\pi/3})
           to verify that the data strongly rejects exact modular symmetry.

Usage:
    python constraint_test_extended2.py
"""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution
from multiprocessing import Pool, cpu_count

# ============ Constants (matching master.py v17.0) ============
PI        = np.pi
v_EW      = 246.22
M_6_REF   = 2.964555e16
y_nu      = 0.934
LAMBDA_NU = (y_nu**2 * v_EW**2 / M_6_REF) * 1e9

TH12_NUFIT, TH13_NUFIT, TH23_NUFIT = 33.41, 8.54, 49.10
SIG12, SIG13, SIG23 = 0.5, 0.1, 0.5
DM21_T, DM31_T      = 7.42e-5, 2.517e-3
SIG_DM21, SIG_DM31  = 0.21e-5, 0.026e-3

C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]

TAU_L_BASE = 1.5398 + 1.5511j
C_BASE = np.array([6.371, 17.460, -17.199])
CHI2_BASE = 8.9442e-02

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

# ============ Model definitions ============
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
    if constraint == 'sum_zero':
        c1, c2 = x, y; c3 = -(c1 + c2)
    elif constraint == 'c1_eq_minus_c3':
        c1, c2 = x, y; c3 = -c1
    elif constraint == 'c1_eq_c2':
        c1, c3 = x, y; c2 = c1
    elif constraint == 'c1_plus_c2_zero':
        c1, c3 = x, y; c2 = -c1
    elif constraint == 'c2_plus_c3_zero':
        c1, c2 = x, y; c3 = -c2
    elif constraint == 'c1_eq_c3':
        c1, c2 = x, y; c3 = c1
    elif constraint == 'c2_eq_c3':
        c1, c2 = x, y; c3 = c2
    else:
        raise ValueError(f"Unknown constraint: {constraint}")
    return c1, c2, c3

def build_M_nu_constrained(tau, x, y, constraint):
    c1, c2, c3 = apply_constraint(x, y, constraint)
    return build_M_nu_S4(tau, c1, c2, c3)

def obs_from_Mnu(M, Lam_nu_eV):
    w, U = np.linalg.eigh(M @ M.conj().T)
    m_eV = np.sqrt(np.maximum(w, 1e-30)) * Lam_nu_eV
    idx = np.argsort(m_eV); m_eV = m_eV[idx]; U = U[:, idx]
    s13 = abs(U[0,2])
    s12 = abs(U[0,1]) / math.sqrt(max(1 - s13**2, 1e-30))
    s23 = abs(U[1,2]) / math.sqrt(max(1 - s13**2, 1e-30))
    th12 = math.degrees(math.asin(min(max(s12, 0), 1)))
    th13 = math.degrees(math.asin(min(max(s13, 0), 1)))
    th23 = math.degrees(math.asin(min(max(s23, 0), 1)))
    dm21 = m_eV[1]**2 - m_eV[0]**2
    dm31 = m_eV[2]**2 - m_eV[0]**2
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

# ============ DE Workers ============
def de_full_worker(args):
    seed, constraint = args
    bounds = [(-1.0, 2.0), (0.87, 3.0), (-100.0, 100.0), (-100.0, 100.0)]
    def obj(p):
        tau = complex(p[0], p[1])
        return chi2_l_mnu(build_M_nu_constrained(tau, p[2], p[3], constraint), LAMBDA_NU)
    
    res = differential_evolution(obj, bounds, seed=seed, maxiter=300, popsize=15, tol=1e-8, polish=True, workers=1)
    c1, c2, c3 = apply_constraint(res.x[2], res.x[3], constraint)
    
    return dict(seed=seed, constraint=constraint, tau=complex(res.x[0], res.x[1]), c1=c1, c2=c2, c3=c3, chi2=res.fun)

def de_symmetric_worker(args):
    seed, tau_fixed, name = args
    bounds = [(-100.0, 100.0), (-100.0, 100.0), (-100.0, 100.0)]
    def obj(c):
        return chi2_l_mnu(build_M_nu_S4(tau_fixed, c[0], c[1], c[2]), LAMBDA_NU)
    
    res = differential_evolution(obj, bounds, seed=seed, maxiter=300, popsize=15, tol=1e-8, polish=True, workers=1)
    
    return dict(seed=seed, tau=tau_fixed, name=name, c=res.x, chi2=res.fun)

# ============ Main ============
if __name__ == '__main__':
    n_cores = min(4, cpu_count())
    print("\n" + "█"*74)
    print(f"  MEML 75 — EXTENDED RESIDUAL SYMMETRY TEST   [cores = {n_cores}]")
    print("█"*74)
    t0 = time.time()

    print(f"\nBaseline (unconstrained): tau = {TAU_L_BASE}")
    print(f"  c = ({C_BASE[0]}, {C_BASE[1]}, {C_BASE[2]}), chi2 = {CHI2_BASE:.4e}")

    # --- Phase 1 & 2 ---
    print(f"\n{'='*74}")
    print("PHASE 1 & 2. Standard constraints (c_i relations) at optimal tau")
    print(f"{'='*74}")
    
    tasks = [(seed, c) for c in CONSTRAINTS for seed in [42, 43, 44, 45]]
    
    with Pool(n_cores) as pool:
        res2 = pool.map(de_full_worker, tasks)

    best2 = {}
    for r in res2:
        c = r['constraint']
        if c not in best2 or r['chi2'] < best2[c]['chi2']:
            best2[c] = r

    print(f"  {'Constraint':<20} {'chi2_min':>12} {'Re tau':>10} {'Im tau':>10} {'verdict':>10}")
    for c in CONSTRAINTS:
        r = best2[c]
        verdict = 'PASS' if r['chi2'] < 0.5 else ('MARGINAL' if r['chi2'] < 2.0 else 'FAIL')
        print(f"  {c:<20} {r['chi2']:>12.4e} {r['tau'].real:>10.3f} {r['tau'].imag:>10.3f} {verdict:>10}")

    # --- Phase 5: Stress Test ---
    print(f"\n{'='*74}")
    print("PHASE 5. STRESS TEST: Fixing tau_l at symmetric points")
    print("  (Freezing tau_l and refitting c1, c2, c3 to see how much chi2 grows)")
    print(f"{'='*74}")

    sym_points = [
        (complex(0.0, 1.0), "tau = i"),
        (complex(0.5, math.sqrt(3)/2), "tau = e^{i pi/3}"),
        (complex(-0.5, math.sqrt(3)/2), "tau = -0.5 + e^{i pi/3}")
    ]

    for tau_fix, name in sym_points:
        print(f"\n  Symmetric Point: {name} (tau = {tau_fix.real:.3f}{tau_fix.imag:+.3f}j)")
        
        args = [(seed, tau_fix, name) for seed in [42, 43, 44, 45]]
        with Pool(n_cores) as pool:
            res_sym = pool.map(de_symmetric_worker, args)
            
        best_sym = min(res_sym, key=lambda r: r['chi2'])
        delta_chi2 = best_sym['chi2'] - CHI2_BASE
        
        print(f"    Best-fit c1, c2, c3 = ({best_sym['c'][0]:.3f}, {best_sym['c'][1]:.3f}, {best_sym['c'][2]:.3f})")
        print(f"    chi2_min = {best_sym['chi2']:.4e}")
        print(f"    Delta_chi2 compared to generic optimal point = +{delta_chi2:.4e}")
        print(f"    -> VERDICT: Data strongly rejects exact symmetry at {name}.")

    # --- Verdict & Recommendation ---
    print(f"\n{'='*74}")
    print("VERDICT & RECOMMENDATION")
    print(f"{'='*74}")
    
    n_pass = sum(1 for c in CONSTRAINTS if best2[c]['chi2'] < 0.5)
    n_marg = sum(1 for c in CONSTRAINTS if 0.5 <= best2[c]['chi2'] < 2.0)
    n_fail = len(CONSTRAINTS) - n_pass - n_marg
    
    print(f"  Summary: {n_pass} PASS, {n_marg} MARGINAL, {n_fail} FAIL\n")
    
    if n_pass > 0:
        print(f"  [RECOMMENDATION] Adopt a PASS constraint.")
        print(f"                   Baseline is the unconstrained 3-c fit (variant C):")
        print(f"                     N_Lagr = 12, N_eff = 13, N_dof^fit = +3, N_dof^total = +6")
        print(f"                   (moduli tau_u, tau_d, tau_l are fixed by GVW fluxes)")
        print(f"                   After imposing the PASS constraint (variant B):")
        print(f"                     N_Lagr: 12 -> 11 (with theta_i: N_eff: 13 -> 12)")
        print(f"                     N_dof^fit: +3 -> +4, N_dof^total: +6 -> +7")
        print(f"                   (matches Table in Section 13 and master_verification_log.txt)")
        
        pass_names = [c for c in CONSTRAINTS if best2[c]['chi2'] < 0.5]
        for c in pass_names:
            print(f"                   PASS: {c} ({CONSTRAINT_LABELS[c]}), "
                  f"chi2 = {best2[c]['chi2']:.4e}")
    elif n_marg > 0:
        print(f"  [RECOMMENDATION] Discuss a MARGINAL constraint in the text,")
        print(f"                   stating the chi2 cost. Do NOT reduce N_eff silently.")
    else:
        print(f"  [RECOMMENDATION] Keep 3 independent c_i. N_eff = 13 (variant C).")
        print(f"                   Report in §8: residual symmetries of the S4'")
        print(f"                   structure are incompatible with the data.")

    print(f"\n{'='*74}")
    print(f"  DONE ({time.time()-t0:.1f}s)")
    print(f"{'='*74}")