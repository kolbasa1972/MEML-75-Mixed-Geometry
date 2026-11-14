#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 (v20.0) — Magnetized Overlap Test v2 (Strict Selection Rule)
Checks physical zero-mode range for fluxes (3, -4, 1).
"""
import numpy as np

M_Q, M_u, M_H = 3, -4, 1
TAU = -0.15779440852968135 + 2.3851782590052113j
tau1, tau2 = TAU.real, TAU.imag

print("=" * 70)
print(" MEML 75 — Magnetized Overlap Test v2 (Strict Selection Rule)")
print("=" * 70)
print(f"Fluxes: M_Q={M_Q}, M_u={M_u}, M_H={M_H}")
print(f"Physical range: I in [0,{abs(M_Q)-1}], J in [0,{abs(M_u)-1}], K in [0,{abs(M_H)-1}]")
print(f"Selection rule: I/|M_Q| + J/|M_u| + K/|M_H| in Z")

def selection_rule(i, j, k):
    val = i/abs(M_Q) + j/abs(M_u) + k/abs(M_H)
    return abs(val - round(val)) < 1e-9

def yukawa_overlap(i, j, k, N_max=50):
    if not selection_rule(i, j, k):
        return 0.0
    alpha = i/M_Q + j/M_u + k/M_H
    n = np.arange(-N_max, N_max + 1)
    real_exp = -np.pi * tau2 * (alpha + n)**2
    imag_exp = 2j * np.pi * tau1 * (alpha + n)**2
    return np.abs(np.sum(np.exp(real_exp + imag_exp)))

allowed = []
for I in range(abs(M_Q)):
    for J in range(abs(M_u)):
        for K in range(abs(M_H)):
            val = yukawa_overlap(I, J, K)
            if val > 1e-9:
                allowed.append((I, J, K, val))

print(f"\nAllowed non-zero couplings:")
for idx in allowed:
    print(f"  Y_{idx[0]}{idx[1]}{idx[2]} = {idx[3]:.4e}")

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)
if len(allowed) == 1:
    print(" [OK] Only Y_000 survives. Mass matrix is rank-1.")
    print(" No hierarchy can be generated from these fluxes.")
else:
    print(f" [WARN] {len(allowed)} couplings survive.")
print("=" * 70)