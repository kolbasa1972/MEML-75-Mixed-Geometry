#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 (v20.0) — Curvaton Decay via Shift Symmetry
Calculates the branching ratio Gamma(sigma -> aa) / Gamma(sigma -> SM)
assuming the ALP has a shift symmetry, making the interaction derivative.
"""
import numpy as np

# Constants
M_Pl = 2.435e18 # GeV
m_sigma = 1e6   # GeV
f_a = 1.9801e11 # GeV

# Standard Model decay rate (gravitational)
Gamma_SM = (m_sigma**3) / (8 * np.pi * M_Pl**2)

# ALP mass in GeV
m_a_GeV = 28.407e-6 / 1e9 

# Decay rate with shift symmetry (suppressed by (m_a/m_sigma)^4)
Gamma_aa_shift = (m_sigma**3 / (32 * np.pi * f_a**2)) * (m_a_GeV / m_sigma)**4

# Branching ratio
r_decay_shift = Gamma_aa_shift / Gamma_SM

print("=" * 70)
print(" MEML 75 — Curvaton Decay with Shift Symmetry (v20.0)")
print("=" * 70)
print(f"ALP mass m_a: {m_a_GeV:.2e} GeV")
print(f"Curvaton mass m_sigma: {m_sigma:.2e} GeV")
print(f"Ratio m_a / m_sigma: {m_a_GeV / m_sigma:.2e}")

print(f"\nBranching ratio with shift symmetry (r_decay):")
print(f"  r_decay ~ {r_decay_shift:.2e}")

print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)

if r_decay_shift < 1e-2:
    print(" [OK] The shift symmetry strongly suppresses the decay into ALPs!")
    print(" The curvaton decays almost entirely into SM particles.")
    print(" The entropy dilution mechanism works perfectly!")
else:
    print(" [WARN] Still problematic. Needs further suppression.")
print("=" * 70)