#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Quark coefficient reduction test (v19.0)

Tests four hypotheses to reduce the 6 quark coefficients
(b_u, c_u, a_1, a_2, b, c) to a smaller set:

  H1. Kähler normalization:
      b_u/a_u ~ (2 Im tau_u)^(-1), c_u/a_u ~ (2 Im tau_u)^(-2)
  H2. Discrete symmetry:
      b_u = b, c_u = c, a_1 = a_2 (or similar)
  H3. GUT relation (SU(5)):
      m_b(M_GUT) = m_tau(M_GUT) -> relates b (down) to c_2 (lepton)
  H4. "Nice number" pattern:
      Are coefficients simple combinations of Im(tau)?

All four fail — this is consistent with the manuscript's
"Summary of negative results" (Section 14, items 1-2 and the GUT cross-check H3 discussed in Section 7).

Usage:
    python meml75_quark_reduction_test.py
"""
import numpy as np
import math

# ============ Constants ============
TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j
M_Pl = 2.435e18
M_GUT = 2e16
v_EW = 246.22

# Raw coefficients from master.py
_raw_coefs = np.array([5.314403415527852e-05, 7.192000606749913,
    0.038947583908142965, 0.00024324586524881046, 0.5030286858187134,
    0.0013651716604568512, 0.002220219681204568])
COEFS_Q = _raw_coefs / _raw_coefs[0]

a_u, b_u, c_u, a_1, a_2, b, c = COEFS_Q

# Signed value of c_2 from the constrained fit (Section 8 of the paper)
c_2_fit = -17.400

print("=" * 70)
print(" MEML 75 — Quark coefficient reduction test (v19.0)")
print("=" * 70)
print(f"\n2 Im(tau_u) = {2*TAU_U.imag:.4f}")
print(f"2 Im(tau_d) = {2*TAU_D.imag:.4f}\n")

print("Quark coefficients (from master.py):")
print(f"  a_u = {a_u:.6e}")
print(f"  b_u = {b_u:.6e}")
print(f"  c_u = {c_u:.6e}")
print(f"  a_1 = {a_1:.6e}")
print(f"  a_2 = {a_2:.6e}")
print(f"  b   = {b:.6e}")
print(f"  c   = {c:.6e}")

# ============ H1: Kähler normalization ============
print("\n" + "=" * 70)
print(" H1. Kähler normalization")
print("=" * 70)

ratio_b_over_a = b_u / a_u
ratio_c_over_a = c_u / a_u

print(f"  Observed b_u/a_u = {ratio_b_over_a:.4e}")
print(f"  Observed c_u/a_u = {ratio_c_over_a:.4e}")
print(f"  Observed log10(b_u/a_u) = {math.log10(ratio_b_over_a):.2f}")
print(f"  Observed log10(c_u/a_u) = {math.log10(ratio_c_over_a):.2f}")

two_im_tau_u = 2 * TAU_U.imag
print(f"\n  (2 Im tau_u)^n for n = -3 to +3:")
for n in range(-3, 4):
    val = two_im_tau_u ** n
    print(f"    n = {n:+d}: (2 Im tau)^n = {val:.4e}")

n_b = math.log(ratio_b_over_a) / math.log(two_im_tau_u)
n_c = math.log(ratio_c_over_a) / math.log(two_im_tau_u)
print(f"\n  Required n for b_u/a_u: n = {n_b:.2f}")
print(f"  Required n for c_u/a_u: n = {n_c:.2f}")

if abs(n_b - round(n_b)) < 0.2 and abs(n_c - round(n_c)) < 0.2:
    print("  [OK] H1 works! Integer powers of (2 Im tau_u).")
else:
    print("  [--] H1 FAILS. The powers are not integers (n_b ~ %.1f, n_c ~ %.1f)." % (n_b, n_c))

# ============ H2: Discrete symmetry ============
print("\n" + "=" * 70)
print(" H2. Discrete symmetry (b_u = b, c_u = c, etc.)")
print("=" * 70)

tests = [
    ("b_u = b", b_u, b),
    ("c_u = c", c_u, c),
    ("a_1 = a_2", a_1, a_2),
    ("b_u = c_u", b_u, c_u),
    ("b = c", b, c),
]

print(f"  {'Relation':<16} {'Left':<14} {'Right':<14} {'Ratio':<10} {'Deviation':<10}")
print("  " + "-" * 66)
for name, left, right in tests:
    ratio = left / right if right != 0 else float('inf')
    dev = abs(ratio - 1) * 100
    print(f"  {name:<16} {left:<14.4e} {right:<14.4e} {ratio:<10.4f} {dev:.1f}%")

print("\n  Verdict:")
for name, left, right in tests:
    ratio = left / right if right != 0 else float('inf')
    if abs(ratio - 1) < 0.1:
        print(f"    [OK] {name}: ratio = {ratio:.3f} (within 10%)")
    else:
        print(f"    [--] {name}: ratio = {ratio:.3f} (NOT close to 1)")

# ============ H3: GUT relation (FIXED: signed value) ============
print("\n" + "=" * 70)
print(" H3. GUT relation (SU(5): m_b = m_tau at M_GUT)")
print("=" * 70)

ratio_h3 = b / c_2_fit

print("  At M_GUT, m_b = m_tau (SU(5) relation)")
print(f"  m_b(M_GUT) ~ b * |Y^(2)_down(tau_d)|")
print(f"  m_tau(M_GUT) ~ c_2 * |Y_3'(tau_l)|")
print(f"  This would require b * |Y_d| = c_2 * |Y_l|")
print()
print(f"  With fitted values: b = {b:.2f}, c_2 = {c_2_fit:.3f} (signed)")
print(f"  Signed ratio b/c_2 = {ratio_h3:+.3f}")
print()
print(f"  [--] H3 is NOT a simple relation (|ratio| approx {abs(ratio_h3):.1f}, not 1)")
print("       The fit already includes m_b and m_tau, so this is a")
print("       cross-check, not an independent derivation.")

# ============ H4: "Nice number" pattern ============
print("\n" + "=" * 70)
print(" H4. 'Nice number' pattern (simple combinations of Im(tau))")
print("=" * 70)

print(f"  log10(2 Im tau_u) = {math.log10(two_im_tau_u):.4f}")
print(f"  log10(2 Im tau_d) = {math.log10(2*TAU_D.imag):.4f}")

print(f"\n  {'Coefficient':<14} {'log10':<10} {'= n*log10(2 Im tau_u)?':<28}")
print("  " + "-" * 54)
for name, val in [("b_u", b_u), ("c_u", c_u), ("a_2", a_2), ("b", b), ("c", c)]:
    if val > 0:
        log_val = math.log10(val)
        n_ratio = log_val / math.log10(two_im_tau_u)
        is_int = abs(n_ratio - round(n_ratio)) < 0.2
        marker = "[OK]" if is_int else "[--]"
        print(f"  {name:<14} {log_val:<10.2f} {n_ratio:<10.2f} {marker}")

print("\n  (If a coefficient matches n * log10(2 Im tau), it could be")
print("   derived from the Kähler normalization or modular weights.)")

# ============ Summary ============
print("\n" + "=" * 70)
print(" SUMMARY (NEGATIVE RESULT)")
print("=" * 70)
print("""
H1 (Kähler normalization): FAILS.
   The powers n_b ~ 7.56, n_c ~ 4.22 are not integers.

H2 (Discrete symmetry): FAILS.
   No pairwise equality holds. Smallest deviation is b/c ~ 0.61;
   largest is b_u/a_1 ~ 3e4.

H3 (GUT relation): NOT A PHYSICAL TEST (see disclaimer above).
   The signed ratio b/c_2 = -1.476 is not close to 1.

H4 (Nice number pattern): PARTIALLY WORKS but inconclusive.
   Two coefficients (a_2, b) fall near integer multiples, but
   c_u and c fall between multiples. This is consistent with
   chance given the small number of tests.

CONCLUSION:
   None of the four hypotheses reduces the quark sector from
   6 free coefficients. The quark coefficients are therefore
   phenomenological, consistent with Section 14 (items 1-2) and the GUT cross-check H3 discussed in Section 7 of
   the manuscript.
""")
print("=" * 70)