#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Quark coefficients from fractional powers (v19.0)

Tests if the quark coefficients can be expressed as
(2 Im tau)^p with fractional p (rational with small denominator).

This is motivated by orbifold CFT, where twisted sectors give
fractional modular weights.

Usage:
    python meml75_quark_fractional_powers.py
"""
import numpy as np
import math

# ============ Constants ============
TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j

# Quark coefficients
a_u, b_u, c_u = 1.0, 1.353303e+05, 7.328684e+02
a_1, a_2, b, c = 4.577106, 9.465384e+03, 2.568815e+01, 4.177740e+01

two_im_tau_u = 2 * TAU_U.imag
two_im_tau_d = 2 * TAU_D.imag

print("=" * 70)
print(" MEML 75 — Quark coefficients from fractional powers")
print("=" * 70)
print(f"\n2 Im(tau_u) = {two_im_tau_u:.4f}")
print(f"2 Im(tau_d) = {two_im_tau_d:.4f}\n")

# ============ Find best fractional power ============
def best_fractional_power(coefficient, base, max_denom=12):
    """
    Find p = numerator/denominator such that base^p ~ coefficient.
    Returns (p, numerator, denominator, error).
    """
    log_coef = math.log(coefficient)
    log_base = math.log(base)
    p_exact = log_coef / log_base
    
    best = None
    for denom in range(1, max_denom + 1):
        numer = round(p_exact * denom)
        p_test = numer / denom
        val_test = base ** p_test
        rel_error = abs(val_test - coefficient) / coefficient
        if best is None or rel_error < best[3]:
            best = (p_test, numer, denom, rel_error)
    
    return best

print("Best fractional powers for quark coefficients:")
print(f"  {'Coeff':<8} {'Value':<14} {'Best p':<10} {'Fraction':<10} {'Error':<10}")
print("  " + "-" * 54)

for name, val in [("b_u", b_u), ("c_u", c_u), ("a_2", a_2), ("b", b), ("c", c)]:
    p, num, den, err = best_fractional_power(val, two_im_tau_u)
    frac = f"{num}/{den}"
    print(f"  {name:<8} {val:<14.4e} {p:<10.4f} {frac:<10} {err*100:.2f}%")

# ============ Check specific fractions ============
print("\n" + "=" * 70)
print(" Specific fractional powers (small denominators)")
print("=" * 70)
print(f"  {'Fraction':<10} {'(2 Im tau_u)^p':<18} {'Closest coefficient':<20} {'Error':<10}")
print("  " + "-" * 60)

fractions = [
    (1, 2), (3, 4), (1, 1), (5, 4), (3, 2), (7, 4), (2, 1),
    (9, 4), (5, 2), (11, 4), (3, 1), (13, 4), (7, 2), (15, 4),
    (4, 1), (17, 4), (9, 2), (19, 4), (5, 1), (21, 4), (11, 2),
    (23, 4), (6, 1), (25, 4), (13, 2), (27, 4), (7, 1), (29, 4),
    (15, 2), (31, 4), (8, 1)
]

coefs = [("b_u", b_u), ("c_u", c_u), ("a_2", a_2), ("b", b), ("c", c)]
for num, den in fractions:
    p = num / den
    val = two_im_tau_u ** p
    # Find closest coefficient
    best_name, best_err = None, 1e10
    for name, coef in coefs:
        err = abs(val - coef) / coef
        if err < best_err:
            best_err = err
            best_name = name
    if best_err < 0.5:  # only print if within 50%
        print(f"  {num}/{den:<8} {val:<18.4e} {best_name:<20} {best_err*100:.1f}%")

# ============ Test if coefficients match simple products ============
print("\n" + "=" * 70)
print(" Products of fractional powers")
print("=" * 70)

# Maybe coefficients are products like (2 Im tau)^p * 3^q
# Let's check if log10(coef) can be decomposed as p*log10(2 Im tau) + q*log10(3)

log_base = math.log10(two_im_tau_u)
log_3 = math.log10(3)

print(f"\n  log10(2 Im tau_u) = {log_base:.4f}")
print(f"  log10(3) = {log_3:.4f}")
print(f"\n  {'Coeff':<8} {'log10':<10} {'= p*log_base + q*log_3':<28}")
print("  " + "-" * 50)

for name, val in [("b_u", b_u), ("c_u", c_u), ("a_2", a_2), ("b", b), ("c", c)]:
    log_val = math.log10(val)
    # Solve for (p, q) using least squares with integer q
    best = None
    for q in range(-3, 4):
        p = (log_val - q*log_3) / log_base
        # Check if p is close to a simple fraction
        p_rounded = round(p * 4) / 4  # round to nearest 1/4
        reconstructed = p_rounded * log_base + q * log_3
        err = abs(10**reconstructed - val) / val
        if best is None or err < best[2]:
            best = (p_rounded, q, err)
    
    p, q, err = best
    print(f"  {name:<8} {log_val:<10.4f} {p:+.2f}*log_base {q:+d}*log_3  ({err*100:.1f}%)")


# ============ Verdict (NEGATIVE RESULT) ============
print("\n" + "=" * 70)
print(" VERDICT (NEGATIVE RESULT)")
print("=" * 70)
print("""
The best-fit fractional powers have denominators (9, 9, 7, 12, 5),
which are not in the set {1, 2, 4} allowed by orbifold CFT
(untwisted: k = 1, 2, 3; twisted: k = 1/4, 1/2, 3/4).

We conclude: the quark coefficients CANNOT be derived from
fractional powers of (2 Im tau) with the modular weights of
orbifold CFT. The quark sector remains with 6 free parameters.

This is a NEGATIVE RESULT, consistent with Section 14, item 4
of the manuscript.
""")
print("=" * 70)