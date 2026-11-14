#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 (v22.0) — Kähler Scan (Corrected Normalization + Modular Forms)
Physical normalization: Y_phys = Y_mod / sqrt(K_i K_j K_k).
Field assignments:
- a_u ~ (Q_1, u_1, H) at point 0  -> a_u ~ Y2[0] * K_0^{-3/2}
- b_u ~ (Q_2, u_2, H) with Q_2,u_2 at point 1, H at point 0
       -> b_u ~ Y4[1] * K_1^{-1} K_0^{-1/2}
- c_u ~ (Q_3, u_3, H) with Q_3,u_3 at point 2, H at point 0
       -> c_u ~ Y6[1] * K_2^{-1} K_0^{-1/2}
Therefore:
- b_u/a_u = (Y4[1]/Y2[0]) * (K_0/K_1)
- c_u/a_u = (Y6[1]/Y2[0]) * (K_0/K_2)

v24.5.1 alignment fix:
  The VERDICT block now uses the GLOBAL log-error = up_err + down_err,
  matching v24.5.1 §14 Item 6 ("Up-sector OK; down-sector off by 10.7
  orders. Global log-error 8.39"). Note that the quoted 8.39 comes
  from meml75_global_kahler_scan.py, which JOINTLY optimises all four
  ratios; the up-first-then-down procedure used here gives a larger
  combined value (~14). Both values are negative results; the script
  prints both up- and down-errors separately so the source of the
  discrepancy is visible.
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

def up_ratios(k1, k2, k3, alpha):
    K0 = kahler(TAU_U, k1, alpha, 0)
    K1 = kahler(TAU_U, k2, alpha, 1)
    K2 = kahler(TAU_U, k3, alpha, 2)

    y2_u = Y2(TAU_U); y4_u = Y4(TAU_U); y6_u = Y6(TAU_U)

    # b_u/a_u = (Y4[1]/Y2[0]) * (K0/K1)
    rb = (K0 / K1) * abs(y4_u[1] / y2_u[0])
    # c_u/a_u = (Y6[1]/Y2[0]) * (K0/K2)
    rc = (K0 / K2) * abs(y6_u[1] / y2_u[0])

    return rb, rc

def down_ratios(k1, k2, k3, alpha):
    K0 = kahler(TAU_D, k1, alpha, 0)
    K1 = kahler(TAU_D, k2, alpha, 1)
    K2 = kahler(TAU_D, k3, alpha, 2)

    y2_d = Y2(TAU_D); y6_d = Y6(TAU_D)

    # a_2/a_1 = (Y2[0]/Y6[0]) * (K0/K2)^{3/2}
    rd_a2 = ((K0 / K2)**1.5) * abs(y2_d[0] / y6_d[0])
    # b/a_1 = (Y2[2]/Y6[0]) * (K0/K1)
    rd_b  = (K0 / K1) * abs(y2_d[2] / y6_d[0])

    return rd_a2, rd_b

print("=" * 70)
print(" MEML 75 — Kähler Scan (Corrected Normalization + Modular Forms)")
print("=" * 70)

best = None
best_err = np.inf

# Extended range to 25 to avoid boundary artifacts
for k1 in range(1, 16):
    for k2 in range(1, 26):
        for k3 in range(1, 16):
            for alpha in np.linspace(0, 25, 150):
                rb, rc = up_ratios(k1, k2, k3, alpha)
                if not (np.isfinite(rb) and np.isfinite(rc)):
                    continue
                err = abs(np.log10(rb/TARGET_BU)) + abs(np.log10(rc/TARGET_CU))
                if err < best_err:
                    best_err = err
                    best = (k1, k2, k3, alpha, rb, rc)

k1, k2, k3, alpha, rb, rc = best
print(f"\nBest up-sector fit (extended range):")
print(f"  k1={k1}, k2={k2}, k3={k3}, alpha={alpha:.2f}")
print(f"  b_u/a_u = {rb:.4e} (target {TARGET_BU:.4e})")
print(f"  c_u/a_u = {rc:.4e} (target {TARGET_CU:.4e})")
print(f"  Up-sector log-error = {best_err:.4f}")

rd_a2, rd_b = down_ratios(k1, k2, k3, alpha)
down_dev_a2 = abs(np.log10(rd_a2/TARGET_A2))
down_dev_b  = abs(np.log10(rd_b/TARGET_B))
down_err    = down_dev_a2 + down_dev_b
up_err      = best_err
global_err  = up_err + down_err

print(f"\nDown-sector with same parameters:")
print(f"  a2/a1 = {rd_a2:.4e} (target {TARGET_A2:.4e})")
print(f"  b/a1  = {rd_b:.4e} (target {TARGET_B:.4e})")
print(f"  Deviation a2/a1: {np.log10(rd_a2/TARGET_A2):+.2f} orders")
print(f"  Deviation b/a1:  {np.log10(rd_b/TARGET_B):+.2f} orders")
print(f"  Down-sector log-error = {down_err:.4f}")

print(f"\nBoundary check: k2={k2} (scan up to 25)")

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)
print(f" Up-sector log-error:   {up_err:.3f}")
print(f" Down-sector log-error: {down_err:.3f}")
print(f" Global log-error:      {global_err:.3f}")
print(" (threshold for acceptable simultaneous fit: global log-error < 0.5)")
if global_err < 0.5:
    print(" [OK] Simultaneous fit found.")
else:
    print(" [NEGATIVE] No simultaneous fit; down-sector is off.")
    print("            This is the negative result reported in")
    print("            v24.5.1 Section 14, Item 6.")
print("=" * 70)