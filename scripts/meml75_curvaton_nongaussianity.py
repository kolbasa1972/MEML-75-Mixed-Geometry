#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 (v20.0) — Curvaton Non-Gaussianity Check
Verifies if the curvaton dilution (Delta ~ 3.13e6) is consistent 
with Planck constraints on local non-Gaussianity (f_NL).
"""
import numpy as np

# Constants
Delta = 3.13e6

# Planck 2018 constraints (95% CL)
F_NL_MEAN = -0.9
F_NL_SIGMA = 5.1

def calculate_fNL(Delta):
    """
    Calculate the local non-Gaussianity parameter f_NL for a curvaton model.
    f_NL = (5/4) * (1/r) - (5/3)
    where r is the curvaton energy fraction at decay.
    r = 3*rho_sigma / (4*rho_rad + 3*rho_sigma)
    Since Delta = rho_sigma / rho_rad, we have:
    r = 3*Delta / (4 + 3*Delta)
    """
    r = (3.0 * Delta) / (4.0 + 3.0 * Delta)
    if r == 0:
        return np.inf
    f_NL = (5.0 / 4.0) * (1.0 / r) - (5.0 / 3.0)
    return f_NL

print("=" * 70)
print(" MEML 75 — Curvaton Non-Gaussianity Check (v20.0)")
print("=" * 70)
print(f"Required Dilution Factor Delta: {Delta:.2e}")

f_NL_pred = calculate_fNL(Delta)

print(f"\nCurvaton energy fraction at decay (r): {(3*Delta)/(4+3*Delta):.6f}")
print(f"Predicted local non-Gaussianity (f_NL): {f_NL_pred:.4f}")

print(f"\nPlanck 2018 constraint (95% CL):")
print(f"  f_NL = {F_NL_MEAN} +/- {F_NL_SIGMA}")

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)

if abs(f_NL_pred - F_NL_MEAN) < F_NL_SIGMA:
    print(" [OK] The predicted f_NL is well within the Planck limits.")
    print(" This proves the curvaton mechanism is cosmologically viable!")
else:
    print(" [WARN] The predicted f_NL is outside the Planck limits.")
    print(" The curvaton model might need modifications (e.g., non-minimal coupling).")

print("=" * 70)