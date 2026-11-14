#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 (v20.0) — Curvaton entropy dilution (Corrected v2)
Calculates the required curvaton amplitude using the correct
entropy scaling: Delta = (rho_sigma / rho_rad)^(3/4).
"""
import numpy as np

# Constants
M_Pl = 2.435e18  # GeV
Delta_target = 3.13e6
T_RH = 1e9       # GeV (Standard reheating)
g_star = 106.75
m_sigma_fixed = 1e6  # GeV (Natural moduli scale)

print("=" * 70)
print(" MEML 75 — Curvaton Dilution (Corrected exponent 3/4)")
print("=" * 70)

# Correct relation: Delta = (rho_sigma / rho_rad)^(3/4)
# Therefore, rho_sigma / rho_rad = Delta^(4/3)
ratio_target = Delta_target**(4.0 / 3.0)

# Radiation energy density at reheating
rho_rad = (np.pi**2 / 30.0) * g_star * (T_RH**4)

# Curvaton energy density: rho_sigma = 0.5 * m_sigma^2 * sigma_0^2
# Solving for sigma_0:
sigma_0 = np.sqrt(2.0 * ratio_target * rho_rad / (m_sigma_fixed**2))

print(f"Target Delta: {Delta_target:.2e}")
print(f"Required energy ratio (rho_sigma/rho_rad): {ratio_target:.2e}")
print(f"Fixed Curvaton mass m_sigma: {m_sigma_fixed:.2e} GeV")
print(f"CORRECTED initial amplitude sigma_0: {sigma_0:.2e} GeV")
print(f"sigma_0 / M_Pl: {sigma_0 / M_Pl:.4f}")

# Check isocurvature constraint
H_inf_limit_new = 3e11 * np.sqrt(Delta_target)
print(f"\nOriginal ALP isocurvature limit on H_inf: 3.0e11 GeV")
print(f"Relaxed ALP isocurvature limit on H_inf: {H_inf_limit_new:.2e} GeV")

if H_inf_limit_new > 1e13:
    print(" [OK] Starobinsky inflation (H_inf ~ 10^13 GeV) is fully allowed.")
else:
    print(" [WARN] Tension remains.")

print("=" * 70)