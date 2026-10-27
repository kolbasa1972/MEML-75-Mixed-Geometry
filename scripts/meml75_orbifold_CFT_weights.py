#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Orbifold CFT with fractional modular weights (v19.0)
================================================================================
Reports a NEGATIVE RESULT: the observed c1/c2 cannot be derived from
fractional modular weights of the Z_4 orbifold in the minimal setup.

Consistent with the manuscript (Section 8, "Attempted derivation of the
ratio c1/c2"; Section 14, item 6), which reports c1/c2 ≈ 1.37 as a
negative result and treats c1, c2 as free parameters.
================================================================================
"""
import numpy as np

PI = np.pi

# Unconstrained fit values (used for the CFT check)
TAU_L = 1.540 + 1.551j
Im_tau = TAU_L.imag
two_Im_tau = 2 * Im_tau

# Observed c1/c2 from the two fits
C1_OVER_C2_UNCONSTRAINED = 6.371 / 17.460   # ≈ 0.365
C1_OVER_C2_CONSTRAINED   = 6.591 / 17.400   # ≈ 0.379

# ---------- S4' level-4 modular forms (via theta and eps) ----------
def theta_eps(tau):
    q4 = np.exp(1j * PI * tau / 2.0)
    theta = 1.0 + 2.0 * q4**4 + 2.0 * q4**16
    eps   = 2.0 * q4 + 2.0 * q4**9 + 2.0 * q4**25
    return theta, eps

def Y2_S4(tau):
    th, ep = theta_eps(tau)
    return np.array([(th**4 + ep**4) / np.sqrt(2.0),
                     -np.sqrt(6.0) * ep**2 * th**2], dtype=complex)

def Y3p_S4(tau):
    th, ep = theta_eps(tau)
    return np.array([(th**4 - ep**4) / np.sqrt(2.0),
                     -2.0 * ep * th**3,
                     -2.0 * ep**3 * th], dtype=complex)

Y2 = Y2_S4(TAU_L)
Y3p = Y3p_S4(TAU_L)

Y_2_eff  = abs(Y2[0])
Y_3p_eff = np.linalg.norm(Y3p)
bare_ratio = Y_2_eff / Y_3p_eff

print("=" * 70)
print(" MEML 75 — Orbifold CFT with fractional modular weights (v19.0)")
print("=" * 70)
print(f"\ntau_l = {TAU_L}")
print(f"2 Im(tau_l) = {two_Im_tau:.4f}")
print(f"Observed unconstrained c1/c2 = {C1_OVER_C2_UNCONSTRAINED:.4f}")
print(f"Observed constrained   c1/c2 = {C1_OVER_C2_CONSTRAINED:.4f}\n")

# ---------- Modular weights of Z_4 twisted sectors ----------
def modular_weight_z4(k_twist, N=4):
    v = k_twist / N
    return N * v * (1 - v)

print("Modular weights of Z_4 twisted sectors:")
print(f"  {'Sector':<12} {'Twist v':<10} {'k_eff':<10}")
print("  " + "-" * 35)
for k in [0, 1, 2, 3]:
    v = k / 4.0
    print(f"  theta^{k:<8} {v:<10.2f} {modular_weight_z4(k):<10.4f}")

print(f"\n|Y_2_eff|    = {Y_2_eff:.6f}")
print(f"||Y_3'_eff|| = {Y_3p_eff:.6f}")
print(f"Bare ratio |Y_2|/||Y_3'|| = {bare_ratio:.6f}")

# ---------- Kähler-normalized ratio ----------
def kahler_ratio(k_2, k3p):
    return bare_ratio * (two_Im_tau)**((k3p - k_2) / 2)

print("\nKähler-normalized ratio for various (k_2, k_3'):")
print(f"  {'k_2':<8} {'k3p':<8} {'Delta k':<14} {'c1/c2':<12}")
print("  " + "-" * 50)

print("  [Integer weights]")
for k_2, k3p in [(2, 4), (4, 4), (2, 6), (4, 6)]:
    print(f"  {k_2:<8} {k3p:<8} {k3p - k_2:<14} {kahler_ratio(k_2, k3p):<12.4f}")

print("\n  [Fractional weights, Delta k = 3/4]")
for k_2, k3p in [(0, 0.75), (0.25, 1.0), (0.5, 1.25), (0.75, 1.5), (1.0, 1.75)]:
    print(f"  {k_2:<8} {k3p:<8} {k3p - k_2:<14} {kahler_ratio(k_2, k3p):<12.4f}")

# ---------- NEGATIVE RESULT ----------
ratio_pred = kahler_ratio(0, 0.75)
dev_unc = abs(ratio_pred - C1_OVER_C2_UNCONSTRAINED) / C1_OVER_C2_UNCONSTRAINED * 100
dev_con = abs(ratio_pred - C1_OVER_C2_CONSTRAINED)   / C1_OVER_C2_CONSTRAINED   * 100

print("\n" + "=" * 70)
print(" PHYSICAL INTERPRETATION (NEGATIVE RESULT)")
print("=" * 70)
print(f"""
For the natural assignment (untwisted 2, theta-twisted 3'),
the Kähler-normalized ratio is:

    c1/c2 ≈ {ratio_pred:.4f}

Rounded to two decimal places, this is 1.37, consistent with the
manuscript (Section 8 and Section 14, item 6).

This is INCONSISTENT with:
  - the unconstrained fit c1/c2 ≈ {C1_OVER_C2_UNCONSTRAINED:.3f}
    (deviation {dev_unc:.0f}%),
  - the constrained   fit c1/c2 ≈ {C1_OVER_C2_CONSTRAINED:.3f}
    (deviation {dev_con:.0f}%).

We therefore report this as a NEGATIVE RESULT. The ratio c1/c2
CANNOT be derived from the fractional modular weights of the
Z_4 orbifold in the minimal setup.

A viable derivation would require additional structure, such as:
  - non-uniform wavefunction localization along the twisted sectors,
  - a different Z_4 projection with Wilson lines,
  - or a different S_4' representation assignment for the 3'.

The observed c1/c2 in the paper is therefore treated as a free
phenomenological parameter.
""")
print("=" * 70)

# ---------- Sanity checks ----------
assert abs(ratio_pred - 1.3678) < 0.01, \
    f"Predicted ratio drifted: {ratio_pred} (expected ≈ 1.368)"
assert abs(ratio_pred - C1_OVER_C2_UNCONSTRAINED) > 0.5, \
    "Prediction unexpectedly matches unconstrained fit"
assert abs(ratio_pred - C1_OVER_C2_CONSTRAINED) > 0.5, \
    "Prediction unexpectedly matches constrained fit"
print("[OK] Sanity checks passed: negative result confirmed.")
print(f"[OK] c1/c2 ≈ {ratio_pred:.2f} vs required ≈ 0.36 (both fits).")