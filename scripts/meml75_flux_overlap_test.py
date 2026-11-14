#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 (v20.0) — Selection rule test for fluxes (-5, -7, +12)
Demonstrates that almost all Yukawa couplings vanish in one physical period.
"""
import numpy as np

# Fluxes
M_Q = -5
M_u = -7
M_H = 12

print("=" * 70)
print(" MEML 75 — Flux Selection Rule Test (One Period)")
print("=" * 70)
print(f"Fluxes: M_Q={M_Q}, M_u={M_u}, M_H={M_H} (Sum = {M_Q+M_u+M_H})")

allowed = []
total = 0

# Scan strictly over one physical period (the number of zero modes)
for I in range(abs(M_Q)):
    for J in range(abs(M_u)):
        for K in range(abs(M_H)):
            total += 1
            # Selection rule: I/|M1| + J/|M2| + K/|M3| in Z
            val = I/abs(M_Q) + J/abs(M_u) + K/abs(M_H)
            if abs(val - round(val)) < 1e-9:
                allowed.append((I, J, K))

print(f"\nTotal combinations checked (one period): {total}")
print(f"Allowed by selection rule: {len(allowed)}")
print("Allowed indices (I, J, K):")
for idx in allowed:
    print(f"  Y_{idx[0]}{idx[1]}{idx[2]}")

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)
if len(allowed) == 1:
    print(" [OK] Only Y_000 survives. No hierarchy can be generated from these fluxes.")
    print(" This provides a strong argument for the necessity of A4 modular forms.")
else:
    print(f" [WARN] {len(allowed)} couplings survive, check further.")
print("=" * 70)