#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — GLOBAL SCAN of tau_l with joint DE (v17.0, corrected v2)

Fixes vs. v1:
  (1) Stage 2/3 use joint DE over (Re tau, Im tau, c1, c2, c3)
      — previously tau was fixed, so true minimum was missed
  (2) c averaging uses canonical sign — previously sign-flipped
      solutions averaged to half-scale
  (3) Pre-scan diagnostic at tau = 1.5398+1.5511i (reported unconstrained
      minimum, chi2 = 0.227) and other tau values

Runtime (4 cores): ~3-6 min.

Usage:
    python meml75_tau_l_global_scan_v2.py
"""
import json
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from multiprocessing import Pool, cpu_count

# ============ Constants (matching master.py v16.0) ============
PI        = np.pi
v_EW      = 246.22
M_6_REF   = 2.9646e16
y_nu      = 0.934
LAMBDA_NU = (y_nu**2 * v_EW**2 / M_6_REF) * 1e9

TH12_NUFIT, TH13_NUFIT, TH23_NUFIT = 33.41, 8.54, 49.10
SIG12, SIG13, SIG23 = 0.5, 0.1, 0.5
DM21_T, DM31_T      = 7.42e-5, 2.517e-3
SIG_DM21, SIG_DM31  = 0.21e-5, 0.026e-3

C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]


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


def canonicalize_c(c):
    """Fix sign ambiguity c <-> -c (M -> -M leaves |M| and chi2 invariant)."""
    c = np.asarray(c)
    if np.sum(c) < 0:
        return -c
    return c


# ============ Pre-scan diagnostic ============
def de_fixed_tau(args):
    Re, Im, seed = args
    tau = complex(Re, Im)
    bounds = [(-100, 100), (-100, 100), (-100, 100)]
    try:
        res = differential_evolution(
            lambda c: chi2_l_mnu(build_M_nu_S4(tau, c[0], c[1], c[2]),
                                 LAMBDA_NU),
            bounds, maxiter=300, popsize=15, seed=seed,
            tol=1e-8, polish=True, workers=1)
        return dict(Re=Re, Im=Im, seed=seed,
                    chi2=float(res.fun),
                    c=canonicalize_c(res.x).tolist())
    except Exception:
        return dict(Re=Re, Im=Im, seed=seed, chi2=1e10, c=[0.0, 0.0, 0.0])


# ============ Stage 1: NM at fixed tau ============
def nm_scan_point(args):
    Re, Im = args
    tau = complex(Re, Im)
    starts = [
        np.array([-6.591, -17.400, 17.400]),
        np.array([ 6.360,  17.430, -17.170]),
        np.array([-10.0,   20.0,  -20.0]),
    ]
    best = None
    for s0 in starts:
        try:
            res = minimize(
                lambda c: chi2_l_mnu(build_M_nu_S4(tau, c[0], c[1], c[2]),
                                     LAMBDA_NU),
                s0, method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-4, 'fatol': 1e-6})
            if best is None or res.fun < best.fun:
                best = res
        except Exception:
            pass
    if best is None:
        return dict(Re=float(Re), Im=float(Im), chi2=1e10,
                    c=[0.0, 0.0, 0.0])
    return dict(Re=float(Re), Im=float(Im), chi2=float(best.fun),
                c=canonicalize_c(best.x).tolist())


# ============ Stage 2/3: joint DE over (tau, c) ============
def joint_de(args):
    Re0, Im0, seed, dtau = args
    Re_min, Re_max = Re0 - dtau, Re0 + dtau
    Im_min = max(0.87, Im0 - dtau)
    Im_max = min(3.0, Im0 + dtau)
    bounds = [
        (Re_min, Re_max),
        (Im_min, Im_max),
        (-100, 100), (-100, 100), (-100, 100),
    ]
    def chi2_wrap(x):
        Re, Im, c1, c2, c3 = x
        if Im < 0.87:
            return 1e10
        tau = complex(Re, Im)
        return chi2_l_mnu(build_M_nu_S4(tau, c1, c2, c3), LAMBDA_NU)
    try:
        res = differential_evolution(
            chi2_wrap, bounds, maxiter=300, popsize=15, seed=seed,
            tol=1e-8, polish=True, workers=1)
        return dict(Re=float(res.x[0]), Im=float(res.x[1]),
                    chi2=float(res.fun),
                    c=canonicalize_c(res.x[2:]).tolist(), seed=seed)
    except Exception:
        return dict(Re=float(Re0), Im=float(Im0), chi2=1e10,
                    c=[0.0, 0.0, 0.0], seed=seed)


def cluster_by_tau(items, dRe, dIm):
    clusters = []
    for it in sorted(items, key=lambda x: x['chi2']):
        placed = False
        for cl in clusters:
            if (abs(cl['best']['Re'] - it['Re']) < dRe and
                abs(cl['best']['Im'] - it['Im']) < dIm):
                cl['members'].append(it)
                if it['chi2'] < cl['best']['chi2']:
                    cl['best'] = it
                placed = True
                break
        if not placed:
            clusters.append({'best': it, 'members': [it]})
    return clusters


def main():
    n_cores = min(4, cpu_count())
    print("=" * 74)
    print("  MEML 75 — GLOBAL SCAN of tau_l (v17.0, joint DE v2)")
    print(f"  [cores = {n_cores}]")
    print(f"  Region: Re tau in [-1.00, 2.00], Im tau in [0.87, 3.00]")
    print(f"  LAMBDA_NU = {LAMBDA_NU:.4e} eV")
    print(f"  NuFIT: th12={TH12_NUFIT}, th13={TH13_NUFIT}, th23={TH23_NUFIT}")
    print(f"         dm21={DM21_T:.3e}, dm31={DM31_T:.3e}")
    print("=" * 74)
    t_start = time.time()

    # ---- Calibration ----
    print("\n[Calibration] tau = 1.584+1.552i, c = (-6.591,-17.400,17.400)")
    chi2_cal = chi2_l_mnu(build_M_nu_S4(1.584+1.552j, -6.591, -17.400, 17.400),
                          LAMBDA_NU)
    print(f"  chi2 = {chi2_cal:.6f}   (expect 0.3092)")
    ok_cal = abs(chi2_cal - 0.3092) < 0.01
    print(f"  [{'OK' if ok_cal else 'WARN'}] calibration")

    # ---- Pre-scan diagnostic ----
    print("\n" + "-" * 74)
    print("[Pre-scan diagnostic] DE over c at fixed tau (4 seeds)")
    print("-" * 74)
    diag_taus = [
        (1.5398, 1.5511),  # claimed unconstrained minimum, chi2 = 0.227
        (1.5424, 1.5559),  # best grid point from v1 scan, chi2 = 0.481
        (1.5840, 1.5520),  # constrained baseline, chi2 = 0.309
        (1.4407, 1.5559),  # v1 second cluster
    ]
    diag_args = [(Re, Im, s)
                 for (Re, Im) in diag_taus for s in [42, 43, 44, 45]]
    with Pool(n_cores) as pool:
        diag = pool.map(de_fixed_tau, diag_args)

    print(f"\n  {'tau_l':<22} {'chi2 mean':<12} {'chi2 std':<12} "
          f"{'c1':<9} {'c2':<9} {'c3':<9}")
    print("  " + "-" * 80)
    for (Re, Im) in diag_taus:
        rs = [d for d in diag if abs(d['Re']-Re) < 1e-9
                                and abs(d['Im']-Im) < 1e-9]
        chis = np.array([r['chi2'] for r in rs])
        cs   = np.array([r['c'] for r in rs])
        print(f"  {Re:.4f}{Im:+.4f}i      "
              f"{chis.mean():<12.4e} {chis.std():<12.2e} "
              f"{cs.mean(axis=0)[0]:<9.3f} "
              f"{cs.mean(axis=0)[1]:<9.3f} "
              f"{cs.mean(axis=0)[2]:<9.3f}")

    print("""
  [Interpretation]
    chi2 = 0.227 at tau = 1.5398+1.5511i → pasp value reproducible
    chi2 = 0.48  at tau = 1.5398+1.5511i → pasp value NOT reproducible
                                           (joint tau+c fit needed)
""")

    # ---- Stage 1 ----
    print("-" * 74)
    print("[Stage 1] Grid 60x60 with Nelder-Mead (maxiter=500)")
    print("-" * 74)
    Re_grid = np.linspace(-1.0, 2.0, 60)
    Im_grid = np.linspace(0.87, 3.0, 60)
    grid = [(float(a), float(b)) for a in Re_grid for b in Im_grid]
    print(f"  Grid points: {len(grid)}")
    t0 = time.time()
    with Pool(n_cores) as pool:
        res1 = pool.map(nm_scan_point, grid)
    t_stage1 = time.time() - t0
    print(f"  Stage 1 done in {t_stage1:.1f} s")

    res1_sorted = sorted(res1, key=lambda x: x['chi2'])
    print(f"\n  Top 10 candidates:")
    print(f"  {'Re tau':<10} {'Im tau':<10} {'chi2':<12} "
          f"{'c1':<9} {'c2':<9} {'c3':<9}")
    for r in res1_sorted[:10]:
        print(f"  {r['Re']:<10.4f} {r['Im']:<10.4f} {r['chi2']:<12.4e} "
              f"{r['c'][0]:<9.3f} {r['c'][1]:<9.3f} {r['c'][2]:<9.3f}")

    cands = [r for r in res1 if r['chi2'] < 5.0]
    print(f"\n  Candidates with chi2 < 5: {len(cands)}")
    if not cands:
        print("  [WARN] No candidates. Stopping.")
        return

    # ---- Clustering ----
    print("\n" + "-" * 74)
    print("[Clustering] dRe = 0.10, dIm = 0.10")
    print("-" * 74)
    clusters = cluster_by_tau(cands, dRe=0.10, dIm=0.10)
    print(f"  Clusters: {len(clusters)}")
    for i, cl in enumerate(clusters):
        b = cl['best']
        print(f"    C{i+1}: tau = {b['Re']:+.4f}{b['Im']:+.4f}i, "
              f"chi2 = {b['chi2']:.4e}, N = {len(cl['members'])}")

    # ---- Stage 2: joint DE ----
    print("\n" + "-" * 74)
    print("[Stage 2] Joint DE over (tau, c), seed=42, dtau=0.15")
    print("           maxiter=300, popsize=15")
    print("-" * 74)
    s2_args = [(cl['best']['Re'], cl['best']['Im'], 42, 0.15)
               for cl in clusters]
    t0 = time.time()
    with Pool(n_cores) as pool:
        res2 = pool.map(joint_de, s2_args)
    t_stage2 = time.time() - t0
    print(f"  Stage 2 done in {t_stage2:.1f} s")

    print(f"\n  Refined minima (all clusters):")
    print(f"  {'Re tau':<12} {'Im tau':<12} {'chi2':<12} "
          f"{'c1':<9} {'c2':<9} {'c3':<9}")
    for r in sorted(res2, key=lambda x: x['chi2']):
        print(f"  {r['Re']:<12.6f} {r['Im']:<12.6f} {r['chi2']:<12.4e} "
              f"{r['c'][0]:<9.3f} {r['c'][1]:<9.3f} {r['c'][2]:<9.3f}")

    refined = [r for r in res2 if r['chi2'] < 2.0]
    print(f"\n  Minima with chi2 < 2: {len(refined)}")

    if refined:
        clusters2 = cluster_by_tau(refined, dRe=0.03, dIm=0.03)
        print(f"  After re-clustering: {len(clusters2)} distinct minima")
        refined = [cl['best'] for cl in clusters2]

    # ---- Stage 3: verification with 4 seeds ----
    print("\n" + "-" * 74)
    print("[Stage 3] Joint DE with 4 seeds (42-45) per minimum, dtau=0.10")
    print("-" * 74)
    verified = []
    for r in refined:
        s3_args = [(r['Re'], r['Im'], s, 0.10) for s in [42, 43, 44, 45]]
        with Pool(n_cores) as pool:
            vres = pool.map(joint_de, s3_args)
        chis = np.array([v['chi2'] for v in vres])
        Res  = np.array([v['Re'] for v in vres])
        Ims  = np.array([v['Im'] for v in vres])
        Cs   = np.array([v['c'] for v in vres])
        verified.append(dict(
            Re=float(Res.mean()), Im=float(Ims.mean()),
            Re_std=float(Res.std()), Im_std=float(Ims.std()),
            chi2=float(chis.mean()), chi2_std=float(chis.std()),
            c=Cs.mean(axis=0).tolist(),
            c_std=Cs.std(axis=0).tolist(),
            all_chi2=chis.tolist(),
        ))

    # ---- Final report ----
    print("\n" + "=" * 74)
    print("  FINAL RESULTS")
    print("=" * 74)
    print(f"\n  {'#':<3} {'Re tau':<12} {'Im tau':<12} "
          f"{'chi2':<12} {'std':<11} "
          f"{'c1':<9} {'c2':<9} {'c3':<9}")
    print("  " + "-" * 90)
    for i, v in enumerate(verified):
        print(f"  {i+1:<3} {v['Re']:<12.6f} {v['Im']:<12.6f} "
              f"{v['chi2']:<12.4e} {v['chi2_std']:<11.2e} "
              f"{v['c'][0]:<9.3f} {v['c'][1]:<9.3f} {v['c'][2]:<9.3f}")

    print(f"\n[Cross-check] Constrained baseline:")
    print(f"  tau = 1.584+1.552i, c = (-6.591,-17.400,17.400), "
          f"chi2 = {chi2_cal:.4e}")

    # Observables at global minimum
    if verified:
        v = verified[0]
        M = build_M_nu_S4(complex(v['Re'], v['Im']), *v['c'])
        th12, th13, th23, dm21, dm31 = obs_from_Mnu(M, LAMBDA_NU)
        print(f"\n[Observables at global minimum]")
        print(f"  th12 = {th12:.3f} deg   (NuFIT: {TH12_NUFIT})")
        print(f"  th13 = {th13:.3f} deg   (NuFIT: {TH13_NUFIT})")
        print(f"  th23 = {th23:.3f} deg   (NuFIT: {TH23_NUFIT})")
        print(f"  dm21 = {dm21:.4e} eV^2  (NuFIT: {DM21_T:.3e})")
        print(f"  dm31 = {dm31:.4e} eV^2  (NuFIT: {DM31_T:.3e})")
        print(f"  c2 + c3 = {v['c'][1] + v['c'][2]:+.4f}")

    # Verdict
    print()
    if len(verified) == 0:
        print("[VERDICT] No minimum with chi2 < 2 found.")
    elif len(verified) == 1:
        v = verified[0]
        print(f"[VERDICT] UNIQUE minimum: "
              f"tau = {v['Re']:.4f}{v['Im']:+.4f}i, chi2 = {v['chi2']:.4e}")
    else:
        print(f"[VERDICT] {len(verified)} distinct minima:")
        for i, v in enumerate(verified):
            tag = " (global)" if i == 0 else ""
            print(f"  #{i+1}: tau = {v['Re']:.4f}{v['Im']:+.4f}i, "
                  f"chi2 = {v['chi2']:.4e}{tag}")

    # Save JSON
    out = {
        'config': {
            'Re_range': [-1.0, 2.0],
            'Im_range': [0.87, 3.0],
            'grid': [60, 60],
            'nm_maxiter': 500,
            'de_maxiter': 300,
            'de_popsize': 15,
            'seeds': [42, 43, 44, 45],
            'Lambda_nu_eV': LAMBDA_NU,
        },
        'diagnostic': diag,
        'stage1_candidates': cands,
        'stage2_refined':    res2,
        'verified_minima':   verified,
        'constrained_baseline': {
            'tau_l': [1.584, 1.552],
            'c':     [-6.591, -17.400, 17.400],
            'chi2':  chi2_cal,
        },
        'timing': {
            'stage1_s': t_stage1,
            'stage2_s': t_stage2,
            'total_s':  time.time() - t_start,
        },
    }
    out_file = 'meml75_tau_l_global_scan_v2_results.json'
    with open(out_file, 'w') as f:
        json.dump(out, f, indent=2)
    print(f"\n[Saved] {out_file}")
    print(f"[Total time] {time.time() - t_start:.1f} s")


if __name__ == '__main__':
    main()