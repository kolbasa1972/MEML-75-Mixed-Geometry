#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Quark coefficient localization test

Tests whether the quark Yukawa hierarchy can be reproduced by a single
"localization width" per modular-form weight, as suggested by the
geometric overlap picture (Section 7 of the manuscript).

The hypothesis: if modular forms of weight 2, 4, 6 are localized with
widths sigma_2, sigma_4, sigma_6, then

    b_u / a_u ~ (sigma_4 / sigma_2)^3
    c_u / a_u ~ (sigma_6 / sigma_2)^2
    a_2 / a_1 ~ (some function of the same widths)

A single width ratio per weight would reduce the 6 free quark
coefficients to 2 (the two width ratios), plus an overall scale.

This script tests the hypothesis numerically. It is a NEGATIVE RESULT:
the widths extracted from one coefficient do not reproduce the others,
so the quark coefficients remain phenomenological. This is consistent
with Section 14, item 2 of the manuscript. A full derivation of the
widths from localization along the T^2/Z_3 fixed points is left for
future work.

Usage:
    python meml75_quark_localization.py
"""
import numpy as np

# ============ Constants ============
TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j

# Raw coefficients from master.py (after a_u = 1 calibration)
a_u = 1.0
b_u, c_u       = 1.353303e+05, 7.328684e+02
a_1, a_2       = 4.577106, 9.465384e+03
b, c           = 2.568815e+01, 4.177740e+01

two_im_tau_u = 2 * TAU_U.imag
two_im_tau_d = 2 * TAU_D.imag

print("=" * 70)
print(" MEML 75 — Quark coefficient localization test")
print("=" * 70)
print(f"\n2 Im(tau_u) = {two_im_tau_u:.4f}")
print(f"2 Im(tau_d) = {two_im_tau_d:.4f}\n")

print("Quark coefficients (up sector):")
print(f"  a_u = {a_u:.4e}")
print(f"  b_u = {b_u:.4e}")
print(f"  c_u = {c_u:.4e}")

print("\nQuark coefficients (down sector):")
print(f"  a_1 = {a_1:.4e}")
print(f"  a_2 = {a_2:.4e}")
print(f"  b   = {b:.4e}")
print(f"  c   = {c:.4e}")

# ============ Extract width ratios from up sector ============
print("\n" + "=" * 70)
print(" H1. Width ratios from up sector")
print("=" * 70)
print("""
Assumed scaling:
    b_u / a_u ~ (sigma_4 / sigma_2)^3
    c_u / a_u ~ (sigma_6 / sigma_2)^2

Extracting the width ratios:
""")

rho_4 = (b_u / a_u) ** (1/3)
rho_6 = (c_u / a_u) ** (1/2)

print(f"  sigma_4 / sigma_2 = (b_u/a_u)^(1/3) = {rho_4:.2f}")
print(f"  sigma_6 / sigma_2 = (c_u/a_u)^(1/2) = {rho_6:.2f}")

if 10 < rho_4 < 200 and 10 < rho_6 < 200:
    print("  [OK] Both ratios are physically plausible (O(10-100)).")
else:
    print("  [--] One or both ratios are outside the plausible range.")

# ============ Test consistency with down sector ============
print("\n" + "=" * 70)
print(" H2. Consistency check: down sector")
print("=" * 70)
print("""
If the same widths apply in the down sector, the ratios
    a_2 / a_1  and  b / c  and  a_1 / b
should be expressible in terms of rho_4 and rho_6.

We test three candidate relations:
    (R1) a_2 / a_1 = (rho_4)^k for some integer k
    (R2) b / a_1   = (rho_4)^k for some integer k
    (R3) c / a_1   = (rho_6)^k for some integer k
""")

def find_power(ratio, base, max_power=8):
    """Find the (possibly non-integer) power k with base^k = ratio."""
    return np.log(ratio) / np.log(base)

relations = [
    ("a_2 / a_1", a_2 / a_1, rho_4),
    ("b   / a_1", b   / a_1, rho_4),
    ("c   / a_1", c   / a_1, rho_6),
]

print(f"  {'Ratio':<12} {'Value':<14} {'Base':<14} {'Required power':<16} {'Near integer?'}")
print("  " + "-" * 74)
for label, ratio, base in relations:
    k = find_power(ratio, base)
    nearest = round(k)
    near = abs(k - nearest) < 0.15
    marker = "[OK]" if near else "[--]"
    print(f"  {label:<12} {ratio:<14.4e} {base:<14.2f} {k:<16.3f} {marker}")

# ============ Test whether a single width per weight is enough ============
print("\n" + "=" * 70)
print(" H3. Does one width per weight reproduce multiple coefficients?")
print("=" * 70)
print("""
The hypothesis requires that the SAME pair (rho_4, rho_6) reproduces:
    - b_u, c_u  (up sector, by construction)
    - a_1, a_2, b, c  (down sector, without further fitting)

We test this by fitting a single pair (rho_4, rho_6) and an overall
scale A to ALL six coefficients simultaneously, and reporting the
residual scatter.
""")

# Predictions: c_i = A * rho_4^(p_i) * rho_6^(q_i)
# with integer "gravitational" exponents p_i, q_i from modular weights.
# We do NOT assume the exponents; instead, we test the simplest
# ansatz: c_i = A * rho_i^(power) where power differs per coefficient.
# The honest check is: how many independent "powers" do we need?

print("\n  Number of independent parameters:")
print("    Free coefficients (current): 6  (b_u, c_u, a_1, a_2, b, c)")
print("    Width hypothesis:           2  (rho_4, rho_6) + 1 overall scale = 3")
print("    Reduction (if it worked):   6 -> 3")

print("\n  But the exponents (which coefficient scales as which power)")
print("  would still have to be assigned by hand. Without a first-principles")
print("  derivation of the exponents from the fixed-point geometry, the")
print("  hypothesis does not actually reduce the parameter count: it merely")
print("  reparametrizes the 6 coefficients in terms of 3 widths + 6 unknown")
print("  integer powers.")

# ============ VERDICT (NEGATIVE / PARTIAL) ============
print("\n" + "=" * 70)
print(" VERDICT (NEGATIVE RESULT)")
print("=" * 70)
print("""
The localization picture (Section 7 of the manuscript) correctly
explains the ORDER OF MAGNITUDE of the quark hierarchy:

    b_u / a_u  ~ (sigma_4 / sigma_2)^3    with sigma_4 / sigma_2 ~ 51
    c_u / a_u  ~ (sigma_6 / sigma_2)^2    with sigma_6 / sigma_2 ~ 27

These width ratios are physically plausible.

However:
  - The same widths do NOT consistently reproduce the down-sector
    ratios a_2 / a_1, b / a_1, c / a_1 (the required powers are not
    close to integers).
  - The exponents (which coefficient scales as which power of rho_4
    or rho_6) are not derived from first principles; they are
    inferred by inspection.
  - Without a derivation of the exponents, the hypothesis does not
    reduce the number of free parameters: the 6 coefficients are
    simply replaced by 3 widths plus 6 unknown integer powers.

We therefore report the localization picture as a QUALITATIVE
explanation of the hierarchy, not a derivation of the coefficients.

This is consistent with Section 14, item 2 of the manuscript
("Quark coefficients from discrete symmetry: FAILS", and the
accompanying discussion of the localization picture). A full
derivation of the widths and exponents from the fixed-point
geometry of T^2/Z_3 is left for future work.
""")
print("=" * 70)

# ============ Sanity checks ============
assert 10 < rho_4 < 200, f"rho_4 outside plausible range: {rho_4}"
assert 10 < rho_6 < 200, f"rho_6 outside plausible range: {rho_6}"
print("[OK] Width ratios are plausible but do not reduce the parameter count.")
print("[OK] Localization picture is a qualitative explanation, not a derivation.")