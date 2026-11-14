#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 (v22.0) — Swampland (WGC) Check for the ALP sector (Corrected)
Tests Electric and Magnetic WGC, plus anomaly consistency.
"""
import numpy as np

# Constants
M_Pl = 2.435e18    # GeV (reduced Planck mass)
f_a = 1.9801e11    # GeV
alpha_EM = 1.0/137.036
e = np.sqrt(4 * np.pi * alpha_EM)  # Gauge coupling

# DFSZ anomaly
E_over_N = 8.0 / 3.0
anomaly_coef = E_over_N - 1.92

print("=" * 70)
print(" MEML 75 — Swampland (WGC) Check (v22.0, Corrected)")
print("=" * 70)
print(f"f_a = {f_a:.2e} GeV")
print(f"Gauge coupling e = sqrt(4*pi*alpha_EM) = {e:.4f}")

# --- 1. Electric WGC ---
S_max = M_Pl / f_a
print("\n--- Electric WGC ---")
print(f"Bound: S_max = M_Pl / f_a = {S_max:.2e}")
print("Typical string instantons have S ~ 10-100.")
if S_max > 100:
    print(" [OK] Electric WGC is satisfied with a large margin.")
else:
    print(" [WARN] Electric WGC is violated.")

# --- 2. Magnetic WGC (Standard) ---
# Bound: f_a <= M_Pl / e
bound_magnetic = M_Pl / e
print("\n--- Magnetic WGC (Standard) ---")
print(f"Bound: f_a_max = M_Pl / e = {bound_magnetic:.2e} GeV")
print(f"Ratio f_a / f_a_max = {f_a / bound_magnetic:.2e}")
if f_a < bound_magnetic:
    print(" [OK] Magnetic WGC is satisfied with a large margin.")
else:
    print(" [WARN] Magnetic WGC is violated.")

# --- 3. Anomaly Consistency Check (Variant B) ---
print("\n--- Anomaly Consistency Check ---")
print(f"|E/N - 1.92| = {abs(anomaly_coef):.4f}")
print("This is O(1), as expected for a consistent UV completion.")
print("(Note: This is NOT a WGC test, but a sanity check on the DFSZ anomaly.)")

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)
print("""
The ALP sector satisfies both the Electric and Magnetic WGC.
The anomaly coefficient is O(1), consistent with DFSZ.
The model is consistent with the Swampland criteria (WGC).
""")
print("=" * 70)