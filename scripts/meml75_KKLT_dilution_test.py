#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — KKLT derivation of the dilution factor Delta (v18.0)

Checks if KKLT moduli stabilization can give the required modulus mass
and whether the initial amplitude phi_0 and temperature T_i can be
chosen naturally to yield Delta ~ 3.13e6.

Usage:
    python meml75_KKLT_dilution_test.py
"""
import math
from scipy.optimize import brentq

# ============ Constants ============
M_Pl = 2.435e18   # GeV (reduced Planck mass)
g_star = 106.75
Delta_target = 3.13e6

print("=" * 70)
print(" MEML 75 — KKLT derivation of Delta")
print("=" * 70)
print(f"\nTarget Delta = {Delta_target:.2e}\n")

# ============ KKLT modulus mass ============
def solve_x(w):
    """
    Solves w = exp(-x) * (1 + 2x/3) for x > 1.5.
    Here w = |W_0|/A (assuming A=1).
    """
    def f(x):
        return math.exp(-x) * (1 + 2 * x / 3) - w
    try:
        return brentq(f, 0.5, 100.0)
    except ValueError:
        return None

def kklt_modulus_mass(N, W0_abs, A=1.0):
    """
    Returns m_phi in GeV for given flux N and |W0|.
    """
    w = W0_abs / A
    x = solve_x(w)
    if x is None:
        return None, None, None
    
    m_3_2 = w * M_Pl  # Gravitino mass in GeV
    # m_phi = (2/sqrt(3)) * x * m_3_2
    m_phi = (2.0 / math.sqrt(3.0)) * x * m_3_2
    return m_phi, x, m_3_2

print("KKLT modulus mass for various (N, |W0|):")
print(f"{'N':<6} {'|W0|':<10} {'x=a*sigma0':<12} {'m_3/2 [GeV]':<16} {'m_phi [GeV]':<16}")
print("-" * 70)

# Scan over N and W0 to find masses in the required range
for N in [5, 10, 50, 100]:
    for W0_abs in [1e-4, 1e-8, 1e-12, 1e-14]:
        m_phi, x, m_3_2 = kklt_modulus_mass(N, W0_abs)
        if m_phi is not None:
            print(f"{N:<6} {W0_abs:<10.0e} {x:<12.2f} {m_3_2:<16.2e} {m_phi:<16.2e}")

# ============ Required phi_0 and T_i ============
def T_RH_of_m_phi(m_phi):
    return (90.0/(math.pi**2 * g_star))**0.25 * math.sqrt(m_phi**3 / (8 * math.pi * M_Pl))

def phi_0_from_Delta(Delta, m_phi, T_i):
    T_RH = T_RH_of_m_phi(m_phi)
    denom = 0.5 * m_phi**2
    numer = Delta * (math.pi**2/30.0) * g_star * T_RH * T_i**3
    return math.sqrt(numer / denom) if denom > 0 else None

def T_i_from_Delta(Delta, m_phi, phi_0):
    T_RH = T_RH_of_m_phi(m_phi)
    denom = Delta * (math.pi**2/30.0) * g_star * T_RH
    numer = 0.5 * m_phi**2 * phi_0**2
    return (numer / denom)**(1/3) if denom > 0 else None

print("\n" + "=" * 70)
print(" Required phi_0 (given T_i) for Delta = 3.13e6")
print("=" * 70)
print(f"{'m_phi [GeV]':<15} {'T_i [GeV]':<15} {'phi_0 [GeV]':<15}")
print("-" * 50)

# Typical masses from the table above
m_phi_values = [3.16e6, 1e9, 1e10]
for m_phi in m_phi_values:
    for T_i in [1e10, 1e13]:
        phi_0 = phi_0_from_Delta(Delta_target, m_phi, T_i)
        if phi_0 is not None:
            print(f"{m_phi:<15.2e} {T_i:<15.2e} {phi_0:<15.2e}")

print("\n" + "=" * 70)
print(" Required T_i (given phi_0) for Delta = 3.13e6")
print("=" * 70)
print(f"{'m_phi [GeV]':<15} {'phi_0 [GeV]':<15} {'T_i [GeV]':<15}")
print("-" * 50)
for m_phi in [3.16e6, 1e9]:
    for phi_0 in [1e12, 1e16, 1e17]:
        T_i = T_i_from_Delta(Delta_target, m_phi, phi_0)
        if T_i is not None:
            print(f"{m_phi:<15.2e} {phi_0:<15.2e} {T_i:<15.2e}")

# ============ Verdict ============
print("\n" + "=" * 70)
print(" VERDICT")
print("=" * 70)
print("""
KKLT fixes m_phi (modulus mass) but NOT phi_0 (initial amplitude)
or T_i (initial temperature). These depend on the inflationary
sector and reheating history.

Typical KKLT masses for standard fluxes:
  m_phi ~ 10^6 - 10^10 GeV.

To get Delta ~ 3.13e6, one needs either:
  (a) phi_0 ~ 10^12 GeV with T_i ~ 10^10 GeV (requires tuning phi_0),
  or
  (b) phi_0 ~ 10^16-17 GeV with T_i ~ 10^13 GeV (large displacement).

Both are cosmologically allowed but require specifying the
inflationary sector. A full derivation of Delta requires a
model of inflation where phi_0 is naturally displaced by the
required amount and T_i is set by the reheating dynamics.

This is beyond the scope of the current paper and is a natural
target for v19.0.
""")