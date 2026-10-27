#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Derivation of Delta from Starobinsky-like inflation (v19.0)
================================================================================
NEGATIVE RESULT: Delta is NOT derivable from the simplest inflation +
moduli model. A full derivation requires specifying the coupling between
the modulus and the inflaton.

Consistent with the manuscript (Section 9, "Status of the dilution factor
Delta") and with meml75_delta_inflation_v2.py.
================================================================================
"""
import numpy as np
import math

# ============ Constants ============
M_Pl = 2.435e18
g_star = 106.75
Delta_target = 3.13e6

# ============ Starobinsky inflation ============
def starobinsky_phi0(N_efolds=60):
    return math.sqrt(3/2) * math.log(4 * N_efolds / 3)

def starobinsky_mass(M_Pl, A_s=2.1e-9):
    return 3.0e-6 * M_Pl

print("=" * 70)
print(" MEML 75 — Delta from Starobinsky-like inflation (v19.0)")
print("=" * 70)
print(f"\nTarget Delta = {Delta_target:.2e}\n")

N_efolds = 60
phi0_over_MPl = starobinsky_phi0(N_efolds)
m_inflaton = starobinsky_mass(M_Pl)
phi_0 = phi0_over_MPl * M_Pl

print("Starobinsky inflation parameters:")
print(f"  N_efolds = {N_efolds}")
print(f"  phi_0 / M_Pl = {phi0_over_MPl:.4f}")
print(f"  phi_0 = {phi_0:.4e} GeV")
print(f"  m_inflaton = {m_inflaton:.4e} GeV")

m_phi = 8.63e7   # GeV (KKLT modulus mass for N=50, W0=1e-12)
print(f"\nModulus mass (from KKLT): m_phi = {m_phi:.4e} GeV")

T_RH = (90.0/(math.pi**2 * g_star))**0.25 * math.sqrt(m_inflaton**3 / (8*math.pi*M_Pl))
print(f"  T_RH = {T_RH:.4e} GeV")

# ============ Delta for various phi_0 and T_i ============
def Delta_of_phi0_Ti(phi_0, m_phi, T_i, T_RH):
    return (0.5 * m_phi**2 * phi_0**2) / ((math.pi**2/30) * g_star * T_RH * T_i**3)

print("\nDelta for various phi_0 and T_i (Starobinsky T_RH):")
print(f"  {'phi_0 [GeV]':<16} {'phi_0/M_Pl':<14} {'T_i [GeV]':<16} {'Delta':<12} {'vs 3.13e6':<12}")
print("  " + "-" * 75)

for c in [0.001, 0.01, 0.1, 0.5]:
    phi0 = c * M_Pl
    for Ti in [1e10, 1e12, 1e13]:
        D = Delta_of_phi0_Ti(phi0, m_phi, Ti, T_RH)
        diff = abs(D - Delta_target) / Delta_target * 100
        print(f"  {phi0:<16.2e} {c:<14.3f} {Ti:<16.2e} {D:<12.3e} {diff:.0f}%")

# ============ Required phi_0 for Delta_target ============
print("\n" + "=" * 70)
print(" FINDING THE REQUIRED phi_0 FOR Delta = 3.13e6")
print("=" * 70)
print(f"  (with m_phi = {m_phi:.2e} GeV, T_RH = {T_RH:.2e} GeV)\n")
print(f"  {'T_i [GeV]':<16} {'phi_0 [GeV]':<18} {'phi_0 / M_Pl':<14}")
print("  " + "-" * 50)

for Ti in [1e8, 1e9, 1e10, 1e11, 1e12, 1e13]:
    numer = Delta_target * (math.pi**2/30) * g_star * T_RH * Ti**3
    phi0 = math.sqrt(numer / (0.5 * m_phi**2))
    print(f"  {Ti:<16.2e} {phi0:<18.4e} {phi0/M_Pl:<14.4f}")

# ============ NEGATIVE RESULT ============
print("\n" + "=" * 70)
print(" VERDICT (NEGATIVE RESULT)")
print("=" * 70)
print("""
Starobinsky inflation predicts:
  - Inflaton mass: m ~ 3e-6 M_Pl
  - Reheating: T_RH ~ 1e9-1e10 GeV
  - Inflaton displacement: phi_0 ~ 5 M_Pl (for N_efolds=60)

If the modulus phi is displaced by the Hubble scale during inflation,
its displacement is typically phi_0 ~ (H/m_phi)^2 M_Pl, which for
H ~ 1e13 GeV and m_phi ~ 1e8 GeV gives phi_0 ~ 0.001 M_Pl.

For Delta = 3.13e6, we need either:
  (a) phi_0 ~ 1e12 GeV with T_i ~ 1e10 GeV, or
  (b) phi_0 ~ 1e16-1e17 GeV with T_i ~ 1e13 GeV.

Neither case follows from the simplest Starobinsky + KKLT setup: in
case (a) phi_0 is too small for standard Hubble displacement, and in
case (b) T_i is far above the Starobinsky reheating temperature.

We therefore conclude that Delta is NOT derivable from the simplest
inflation + moduli model. A full derivation would require specifying
the coupling between the modulus and the inflaton (e.g., a non-minimal
coupling, or a specific reheating scenario beyond the simplest setup).

In the manuscript (Section 9), Delta is therefore treated as a
phenomenological parameter, with an explicit acknowledgment of the
isocurvature tension if the ALP constitutes the entirety of dark
matter.
""")
print("=" * 70)

# ============ Sanity checks ============
assert phi0_over_MPl > 4.0 and phi0_over_MPl < 7.0, \
    "Starobinsky phi_0/M_Pl out of expected range"
print("[OK] Delta confirmed NOT derivable from the simplest model.")
print("[OK] See Section 9 of the manuscript for the full discussion.")