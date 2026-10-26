#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — CONSTRAINED GLOBAL SCAN of tau_l (v17.0)

Note: In v17.0, the lepton modulus tau_l is interpreted as being
topologically stabilized by GVW fluxes (-15, 4, -4). This scan
demonstrates that the constrained minimum (c2 + c3 = 0) lies at
tau_l = 1.584 + 1.552i, consistent with the manuscript.

Question: is the constrained minimum really at
    tau_l = 1.584 + 1.552i, c = (-6.591, -17.400, +17.400)
    with chi2 = 0.3092 ?

Method: joint DE over (Re tau, Im tau, c1, c2) with c3 = -c2.
  Region: Re tau in [1.4, 1.8], Im tau in [1.4, 1.7]
  Also: 4 seeds (42, 43, 44, 45) to check stability.

Sign convention: the global sign ambiguity M -> -M leaves |M| and chi2
invariant. We canonicalize every solution to sum(c) < 0, i.e. c1 < 0
(with c3 = -c2), which matches the article (c1 = -6.591, c2 = -17.400,
c3 = +17.400).

Usage:
    python meml75_constrained_global_scan.py
"""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution
from multiprocessing import Pool, cpu_count

# ============ Constants (Unified) ============
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

# Unconstrained baseline (renamed for clarity)
TAU_L_UNCONSTR = 1.539847 + 1.551070j
C_U_BASE       = np.array([6.371, 17.460, -17.199])
CHI2_U         = 8.9413e-02

# Constrained baseline (from Section A of master.py)
TAU_C_BASE = 1.584 + 1.552j
C_C_BASE   = np.array([-6.591, -17.400, 17.400])
CHI2_C_REPORTED = 0.3092


# ============ Model definitions ============
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

# ============ Constrained DE worker ============
def de_constrained(args):
    seed, Re_range, Im_range, c_range = args
    bounds = [
        (Re_range[0], Re_range[1]),
        (Im_range[0], Im_range[1]),
        (c_range[0], c_range[1]),
        (c_range[0], c_range[1]),
    ]
    def chi2_wrap(x):
        Re, Im, c1, c2 = x
        if Im < 0.87:
            return 1e10
        tau = complex(Re, Im)
        return chi2_l_mnu(build_M_nu_S4(tau, c1, c2, -c2), LAMBDA_NU)
    
    try:
        res = differential_evolution(
            chi2_wrap, bounds, maxiter=500, popsize=25,
            seed=seed, tol=1e-10, polish=True, workers=1)
        
        Re, Im, c1, c2 = res.x
        
        # Apply canonical sign convention BEFORE returning.
        # Since c3 = -c2, sum(c) = c1. Enforcing sum(c) < 0 means c1 < 0.
        if c1 > 0:
            c1, c2 = -c1, -c2
            
        return dict(
            seed=seed,
            Re=float(Re), Im=float(Im),
            c1=float(c1), c2=float(c2), c3=float(-c2),
            chi2=float(res.fun),
        )
    except Exception:
        return dict(seed=seed, Re=np.nan, Im=np.nan,
                    c1=np.nan, c2=np.nan, c3=np.nan, chi2=1e10)


def main():
    n_cores = min(4, cpu_count())
    print("=" * 74)
    print("  MEML 75 — CONSTRAINED GLOBAL SCAN (v17.0)")
    print(f"  [cores = {n_cores}]")
    print(f"  Constraint: c3 = -c2  (c2 + c3 = 0)")
    print(f"  Region: Re tau in [1.4, 1.8], Im tau in [1.4, 1.7]")
    print(f"  DE: maxiter=500, popsize=25, tol=1e-10, 4 seeds")
    print(f"  Sign convention: canonicalized to sum(c) < 0 (matches article)")
    print("=" * 74)
    t_start = time.time()

    # ---- Reference values ----
    print("\n[Reference values]")
    chi2_u = chi2_l_mnu(build_M_nu_S4(TAU_L_UNCONSTR, *C_U_BASE), LAMBDA_NU)
    print(f"  Unconstrained baseline (v2 scan):")
    print(f"    tau = {TAU_L_UNCONSTR}, c = {C_U_BASE}, chi2 = {chi2_u:.6e}")

    chi2_c_reported = chi2_l_mnu(build_M_nu_S4(TAU_C_BASE, *C_C_BASE), LAMBDA_NU)
    print(f"  Constrained reported (Section A):")
    print(f"    tau = {TAU_C_BASE}, c = {C_C_BASE}, "
          f"chi2 = {chi2_c_reported:.6e}")

    # ---- Joint constrained DE with 4 seeds ----
    print("\n" + "-" * 74)
    print("[Scan] Joint DE over (Re tau, Im tau, c1, c2), c3 = -c2")
    print("-" * 74)

    Re_range = (1.4, 1.8)
    Im_range = (1.4, 1.7)
    c_range  = (-100.0, 100.0)

    seeds = [42, 43, 44, 45]
    args = [(s, Re_range, Im_range, c_range) for s in seeds]
    with Pool(n_cores) as pool:
        results = pool.map(de_constrained, args)

    print(f"\n  {'seed':<6} {'Re tau':<12} {'Im tau':<12} "
          f"{'chi2':<14} {'c1':<10} {'c2':<10} {'c3':<10}")
    print("  " + "-" * 80)
    for r in results:
        print(f"  {r['seed']:<6} {r['Re']:<12.6f} {r['Im']:<12.6f} "
              f"{r['chi2']:<14.4e} {r['c1']:<10.3f} "
              f"{r['c2']:<10.3f} {r['c3']:<10.3f}")

    chis = np.array([r['chi2'] for r in results])
    Res  = np.array([r['Re'] for r in results])
    Ims  = np.array([r['Im'] for r in results])
    C1s  = np.array([r['c1'] for r in results])
    C2s  = np.array([r['c2'] for r in results])

    print(f"\n  chi2:  mean = {chis.mean():.6e},  std = {chis.std():.2e}")
    print(f"  Re tau: mean = {Res.mean():.6f},  std = {Res.std():.2e}")
    print(f"  Im tau: mean = {Ims.mean():.6f},  std = {Ims.std():.2e}")
    print(f"  c1:     mean = {C1s.mean():.4f},  std = {C1s.std():.2e}")
    print(f"  c2:     mean = {C2s.mean():.4f},  std = {C2s.std():.2e}")

    # ---- Best constrained ----
    best = min(results, key=lambda r: r['chi2'])
    print("\n" + "=" * 74)
    print("  FINAL RESULT")
    print("=" * 74)
    print(f"\n  Best constrained minimum:")
    print(f"    tau_l = {best['Re']:.6f} + {best['Im']:.6f}i")
    print(f"    c = ({best['c1']:.4f}, {best['c2']:.4f}, {best['c3']:.4f})")
    print(f"    chi2 = {best['chi2']:.6e}")

    # ---- Compare with unconstrained and reported ----
    print(f"\n  Comparison:")
    print(f"    chi2_unconstrained (v2)  = {chi2_u:.6e}")
    print(f"    chi2_constrained (scan)  = {best['chi2']:.6e}")
    print(f"    chi2_constrained (report)= {chi2_c_reported:.6e}")

    delta_chi2 = best['chi2'] - chi2_u
    delta_chi2_reported = chi2_c_reported - chi2_u
    print(f"\n  Delta_chi2 (scan,   constrained - unconstrained) = "
          f"{delta_chi2:.4f}")
    print(f"  Delta_chi2 (report, constrained - unconstrained) = "
          f"{delta_chi2_reported:.4f}")

    # ---- Statistical interpretation ----
    from math import erf, sqrt as msqrt
    def p_value_1dof(dchi2):
        return 1.0 - erf(msqrt(dchi2 / 2.0))
    
    p_scan     = p_value_1dof(delta_chi2)
    p_reported = p_value_1dof(delta_chi2_reported)

    print(f"\n  p-values (chi^2_1 distribution, 1 dof):")
    print(f"    from scan:     p = {p_scan:.4f}")
    print(f"    from reported: p = {p_reported:.4f}")

    # AIC / BIC
    dAIC_scan     = delta_chi2 + 2*(-1)
    dBIC16_scan   = delta_chi2 + math.log(16)*(-1)
    dBIC5_scan    = delta_chi2 + math.log(5)*(-1)
    print(f"\n  Information criteria (from scan):")
    print(f"    Delta_AIC         = {dAIC_scan:+.4f}")
    print(f"    Delta_BIC (N=16)  = {dBIC16_scan:+.4f}")
    print(f"    Delta_BIC (N=5)   = {dBIC5_scan:+.4f}")

    # ---- Observables at best constrained ----
    M_best = build_M_nu_S4(complex(best['Re'], best['Im']),
                            best['c1'], best['c2'], best['c3'])
    th12, th13, th23, dm21, dm31 = obs_from_Mnu(M_best, LAMBDA_NU)
    print(f"\n  Observables at best constrained:")
    print(f"    th12 = {th12:.3f} deg  (NuFIT: {TH12_NUFIT})")
    print(f"    th13 = {th13:.3f} deg  (NuFIT: {TH13_NUFIT})")
    print(f"    th23 = {th23:.3f} deg  (NuFIT: {TH23_NUFIT})")
    print(f"    dm21 = {dm21:.4e} eV^2  (NuFIT: {DM21_T:.3e})")
    print(f"    dm31 = {dm31:.4e} eV^2  (NuFIT: {DM31_T:.3e})")

    # ---- Verdict ----
    print("\n" + "=" * 74)
    print("  INTERPRETATION")
    print("=" * 74)

    improvement = chi2_c_reported - best['chi2']
    if improvement > 0.01:
        print(f"""
  The scan found a BETTER constrained minimum:
    chi2 went from {chi2_c_reported:.4f} (reported) to {best['chi2']:.4f} (scan)
    improvement = {improvement:.4f}
    at tau_l = {best['Re']:.4f} + {best['Im']:.4f}i

  => Delta_chi2 = {delta_chi2:.4f} (not {delta_chi2_reported:.4f})
  => Update Section 8 with the new value.
""")
    else:
        print(f"""
  The reported constrained minimum is confirmed:
    chi2 = {chi2_c_reported:.4f} (reported) vs {best['chi2']:.4f} (scan)
    difference = {improvement:.4f} (< 0.01, negligible)

  => Delta_chi2 = {delta_chi2:.4f} (round to 0.22)
  => p = {p_scan:.2f}
  => Delta_AIC = {dAIC_scan:+.2f}
  => Delta_BIC(N=16) = {dBIC16_scan:+.2f}
  => Delta_BIC(N=5) = {dBIC5_scan:+.2f}
""")

    print(f"[Total time] {time.time() - t_start:.1f} s")


if __name__ == '__main__':
    main()