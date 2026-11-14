#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_close_tails.py — closes the two open tails for MEML 75 v24.4.

Tail 1: chi2_min(a2/a1) landscape vs the 3-parameter fit (a1,c_d,a2/a1 free).
Tail 2: chi2 spread over the four weight-zero ratios r in [2055,2068].

Run next to meml75_honest_pure2.py:
  python3 meml75_close_tails.py
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize
from meml75_honest_pure2 import (
    TAU_U, TAU_D, PDG_PAPER, PDG_2024,
    get_Y2_Y4, Y6_true, build_matrices, ckm_observables,
)

y2u, y4u = get_Y2_Y4(TAU_U); y6u = Y6_true(TAU_U)
y2d, y4d = get_Y2_Y4(TAU_D); y6d = Y6_true(TAU_D)
up, dn = (y2u, y4u, y6u), (y2d, y4d, y6d)
y13 = abs(np.exp(2j*np.pi*TAU_U/3.0))
y14 = abs(np.exp(2j*np.pi*TAU_D/4.0))

a_u = 1.0
b_u_mod = a_u * abs(y4u[1]/y2u[0]) * y13**-3
c_u_mod = a_u * abs(y6u[1]/y2u[0]) * y13**-4
b_unit  = abs(y2d[2]/y6d[0]) * y14**-2

def _obs(la1, k, r_a2):
    a1 = 10.0**la1; c_d = k * a1; a2 = a1 * r_a2; b = a1 * b_unit
    Mu, Md = build_matrices(up, dn, a_u, b_u_mod, c_u_mod, a1, a2, b, c_d)
    return ckm_observables(Mu, Md)

def _chi2(obs, sets):
    return sum(((obs[key]-t[key])/(t[key]*e))**2 for t, e in sets for key in t)

def fit_2p(r, sets, seed=42):
    obj = lambda p: _chi2(_obs(p[0], p[1], r), sets)
    de = differential_evolution(obj, [(-5,3),(-1000,1000)], maxiter=1500,
        popsize=25, seed=seed, tol=1e-10, polish=False)
    nm = minimize(obj, de.x, method='Nelder-Mead',
        options={'xatol':1e-11,'fatol':1e-11,'maxiter':40000})
    return nm.x, nm.fun

def fit_3p(sets, seed=42):
    obj = lambda p: _chi2(_obs(p[0], p[1], 10.0**p[2]), sets)
    de = differential_evolution(obj, [(-5,3),(-1000,1000),(2.0,4.0)],
        maxiter=2000, popsize=30, seed=seed, tol=1e-10, polish=False)
    nm = minimize(obj, de.x, method='Nelder-Mead',
        options={'xatol':1e-11,'fatol':1e-11,'maxiter':50000})
    return nm.x, nm.fun

def report(tag, sets):
    p3, chi3 = fit_3p(sets)
    r_star = 10.0**p3[2]
    p2, chi_c = fit_2p(2060.5, sets)
    dchi = chi_c - chi3
    grid = np.linspace(2055.0, 2068.0, 14)
    chis = np.array([fit_2p(r, sets)[1] for r in grid])
    print(f"\n=== {tag} ===")
    print(f"  3-param min: chi2={chi3:.4f}  at a1={10**p3[0]:.4e}, "
          f"c_d={p3[1]*10**p3[0]:.4e}, r*={r_star:.1f}")
    print(f"  r=2060.5   : chi2={chi_c:.4f}  ->  Delta chi2={dchi:+.4f} "
          f"({np.sqrt(max(dchi,0)):.2f} sigma)")
    print(f"  tail2 spread over [2055,2068]: {chis.max()-chis.min():.4f} "
          f"(argmin r={grid[chis.argmin()]:.1f})")
    return chi3, r_star, chi_c, dchi, chis.max()-chis.min()

sets = {
    "paper @ 10%":    [(PDG_PAPER, 0.10)],
    "PDG2024 @ 5%":   [(PDG_2024,  0.05)],
    "PDG2024 @ 10%":  [(PDG_2024,  0.10)],
    "combined":       [(PDG_PAPER, 0.10), (PDG_2024, 0.05)],
}
results = {tag: report(tag, s) for tag, s in sets.items()}

print("\n=== PASSPORT v24.4 additions ===")
for tag, (chi3, r_star, chi_c, dchi, spread) in results.items():
    print(f"  {tag}: chi2_3p={chi3:.4f}, r*={r_star:.1f}, "
          f"chi2(2060.5)={chi_c:.4f}, dchi={dchi:+.4f}, "
          f"{np.sqrt(max(dchi,0)):.2f}sigma, tail2_spread={spread:.4f}")