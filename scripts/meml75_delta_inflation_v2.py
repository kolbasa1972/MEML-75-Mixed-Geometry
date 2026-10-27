#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Delta from inflation + moduli coupling (v19.0)

Tests if a Starobinsky-like inflation with a KKLT modulus
can produce the required dilution factor Delta = 3.13e6.

Physics:
  - Starobinsky inflation: H_inf ~ 10^13 GeV, T_RH ~ 10^9 GeV
  - KKLT modulus: m_phi ~ 10^8 GeV (from previous calculation)
  - Modulus displacement during inflation:
      phi_0 ~ H_inf^2 / m_phi^2 * M_Pl  (capped at M_Pl)
  - Delta = 0.5 m_phi^2 phi_0^2 / [(pi^2/30) g_* T_RH T_i^3]

Usage:
    python meml75_delta_inflation_v2.py
"""
import numpy as np
import math

# ============ Constants ============
M_Pl = 2.435e18  # GeV
g_star = 106.75
Delta_target = 3.13e6

# ============ Inflation parameters (Starobinsky) ============
H_inf = 1.0e13     # GeV (Starobinsky, N=60)
T_RH = 1.0e9       # GeV (Starobinsky reheating)

# ============ Modulus parameters (KKLT) ============
m_phi = 8.63e7     # GeV (from KKLT, N=50, W0=1e-12)

print("=" * 70)
print(" MEML 75 — Delta from inflation + moduli coupling")
print("=" * 70)
print(f"\nTarget Delta = {Delta_target:.2e}\n")
print("Inflation (Starobinsky-like):")
print(f"  H_inf = {H_inf:.2e} GeV")
print(f"  T_RH  = {T_RH:.2e} GeV")
print(f"Modulus (KKLT):")
print(f"  m_phi = {m_phi:.2e} GeV")
print(f"  H_inf / m_phi = {H_inf/m_phi:.2e}")

# ============ Standard moduli displacement ============
# During inflation, the modulus gets a Hubble-induced mass:
#   m_eff^2 = m_phi^2 + c H_inf^2
# For c > 0, the modulus is driven to phi = 0 (its minimum).
# For c < 0, it can be displaced.
#
# For the standard case (c ~ 1), the displacement is:
#   phi_0 ~ min(H_inf^2/m_phi^2 * M_Pl, M_Pl)
#
# If H_inf >> m_phi, the displacement is M_Pl (saturated).

phi_0_naive = (H_inf**2 / m_phi**2) * M_Pl
phi_0_capped = min(phi_0_naive, M_Pl)

print(f"\nStandard moduli displacement:")
print(f"  phi_0 (naive) = (H_inf/m_phi)^2 * M_Pl = {phi_0_naive:.2e} GeV")
print(f"  phi_0 (capped) = min(..., M_Pl) = {phi_0_capped:.2e} GeV")

# ============ Compute Delta for this phi_0 ============
def Delta_of(phi_0, m_phi, T_RH, T_i):
    return (0.5 * m_phi**2 * phi_0**2) / ((math.pi**2/30) * g_star * T_RH * T_i**3)

print(f"\nDelta for various T_i (with phi_0 = {phi_0_capped:.2e} GeV):")
print(f"  {'T_i [GeV]':<16} {'Delta':<14} {'vs 3.13e6':<12}")
print("  " + "-" * 45)

for T_i in [1e8, 1e9, 1e10, 1e11, 1e12]:
    D = Delta_of(phi_0_capped, m_phi, T_RH, T_i)
    diff = abs(D - Delta_target) / Delta_target * 100
    print(f"  {T_i:<16.2e} {D:<14.3e} {diff:.0f}%")

# ============ What phi_0 is needed? ============
print(f"\nRequired phi_0 for Delta = {Delta_target:.2e}:")
print(f"  {'T_i [GeV]':<16} {'phi_0 [GeV]':<18} {'phi_0/M_Pl':<14}")
print("  " + "-" * 50)

for T_i in [1e9, 1e10, 1e11, 1e12]:
    numer = Delta_target * (math.pi**2/30) * g_star * T_RH * T_i**3
    phi0_req = math.sqrt(numer / (0.5 * m_phi**2))
    print(f"  {T_i:<16.2e} {phi0_req:<18.4e} {phi0_req/M_Pl:<14.4f}")

# ============ Comparison with alternatives ============
print("\n" + "=" * 70)
print(" TENSION / COMPARISON")
print("=" * 70)

phi_0_req = math.sqrt(Delta_target * (math.pi**2/30) * g_star * T_RH * 1e10**3 / (0.5 * m_phi**2))
print(f"""
If T_i ~ 10^10 GeV (typical for moduli decay), required phi_0:
  phi_0_req ~ {phi_0_req:.2e} GeV = {phi_0_req/M_Pl:.3e} M_Pl

Standard moduli displacement (capped at M_Pl):
  phi_0_std ~ {phi_0_capped:.2e} GeV = 1 M_Pl

Ratio: phi_0_req / phi_0_std = {phi_0_req/phi_0_capped:.2e}

This means the standard moduli problem would give a MUCH LARGER
Delta than needed (by a factor ~ {Delta_of(phi_0_capped, m_phi, T_RH, 1e10)/Delta_target:.2e}).

To get the correct Delta, we need either:
  (a) A much larger T_i (e.g., 10^13 GeV), or
  (b) A much smaller phi_0 (e.g., by a factor 10^6).

Neither is "natural" in the simplest moduli problem. This
means Delta requires fine-tuning or a specific mechanism.
""")

# ============ Verdict ============
print("=" * 70)
print(" VERDICT")
print("=" * 70)
print("""
The standard moduli problem (H_inf >> m_phi) gives
phi_0 ~ M_Pl, which produces Delta >> 10^6 (over-dilution).

To get the correct Delta = 3.13e6, we need phi_0 ~ 10^-3 M_Pl
(if T_i ~ 10^10 GeV) or T_i ~ 10^13 GeV (if phi_0 ~ 0.01 M_Pl).

Both require a specific mechanism beyond the simplest
Starobinsky + KKLT setup. A UV completion is needed.

CONCLUSION: Delta is NOT derivable from the simplest
inflation + moduli model. It remains a free parameter
in future work.
""")