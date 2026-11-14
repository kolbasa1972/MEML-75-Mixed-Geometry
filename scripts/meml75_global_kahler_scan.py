#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 (v22.0) — Global Kähler Scan over All Four Ratios
Minimizes the sum of log-errors for b_u/a_u, c_u/a_u, a2/a1, b/a1.
Now includes the modular forms at tau_u and tau_d.
"""
import numpy as np

TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j

TARGET_BU = 1.3533e5
TARGET_CU = 7.3287e2
TARGET_A2 = 2.07e3
TARGET_B  = 5.61

# --- Modular Forms Coefficients ---
C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]

def Y2(tau):
    q   = np.exp(2j*np.pi*tau)
    q13 = np.exp(2j*np.pi*tau/3)
    q23 = np.exp(4j*np.pi*tau/3)
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

def distances(tau):
    d12_sq = (1 + 2*tau.real + abs(tau)**2) / (9 * tau.imag)
    return d12_sq, 4*d12_sq

def kahler(tau, k, alpha, point):
    d12_sq, d13_sq = distances(tau)
    weight = (2 * tau.imag)**(-k)
    if point == 0: return weight
    if point == 1: return weight * np.exp(-alpha * d12_sq)
    if point == 2: return weight * np.exp(-alpha * d13_sq)

def all_ratios(k1, k2, k3, alpha):
    # Up-sector
    K0_u = kahler(TAU_U, k1, alpha, 0)
    K1_u = kahler(TAU_U, k2, alpha, 1)
    K2_u = kahler(TAU_U, k3, alpha, 2)
    y2_u = Y2(TAU_U); y4_u = Y4(TAU_U); y6_u = Y6(TAU_U)
    
    bu = (K0_u / K1_u) * abs(y4_u[1] / y2_u[0])
    cu = (K0_u / K2_u) * abs(y6_u[1] / y2_u[0])

    # Down-sector
    K0_d = kahler(TAU_D, k1, alpha, 0)
    K1_d = kahler(TAU_D, k2, alpha, 1)
    K2_d = kahler(TAU_D, k3, alpha, 2)
    y2_d = Y2(TAU_D); y6_d = Y6(TAU_D)
    
    a2 = ((K0_d / K2_d)**1.5) * abs(y2_d[0] / y6_d[0])
    bd = (K0_d / K1_d) * abs(y2_d[2] / y6_d[0])

    return bu, cu, a2, bd

print("=" * 70)
print(" MEML 75 — Global Kähler Scan (All Four Ratios + Modular Forms)")
print("=" * 70)

best = None
best_err = np.inf

for k1 in range(1, 16):
    for k2 in range(1, 26):
        for k3 in range(1, 16):
            for alpha in np.linspace(0, 25, 100):
                bu, cu, a2, bd = all_ratios(k1, k2, k3, alpha)
                if not all(np.isfinite(x) and x > 0 for x in [bu, cu, a2, bd]):
                    continue
                err = (abs(np.log10(bu/TARGET_BU)) +
                       abs(np.log10(cu/TARGET_CU)) +
                       abs(np.log10(a2/TARGET_A2)) +
                       abs(np.log10(bd/TARGET_B)))
                if err < best_err:
                    best_err = err
                    best = (k1, k2, k3, alpha, bu, cu, a2, bd)

k1, k2, k3, alpha, bu, cu, a2, bd = best
print(f"\nBest global fit (all four ratios):")
print(f"  k1={k1}, k2={k2}, k3={k3}, alpha={alpha:.2f}")
print(f"  b_u/a_u = {bu:.4e} (target {TARGET_BU:.4e})")
print(f"  c_u/a_u = {cu:.4e} (target {TARGET_CU:.4e})")
print(f"  a2/a1   = {a2:.4e} (target {TARGET_A2:.4e})")
print(f"  b/a1    = {bd:.4e} (target {TARGET_B:.4e})")
print(f"  Total log-error = {best_err:.4f}")

print(f"\nBoundary check: k1={k1} (scan up to 15), k2={k2} (up to 25), k3={k3} (up to 15)")

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)
if best_err > 3:
    print(f" [OK] Global log-error = {best_err:.2f} > 3.")
    print(" No single parameter set reproduces all four ratios.")
else:
    print(f" [WARN] Global log-error = {best_err:.2f} < 3.")
    print(" The Kähler metric can accommodate all four ratios with enough freedom.")
print("=" * 70)