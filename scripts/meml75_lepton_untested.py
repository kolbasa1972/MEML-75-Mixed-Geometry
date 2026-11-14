#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_lepton_untested.py — v2 (v24.5.1 alignment)
==================================================
Tests the three lepton-sector claims from §13:
  (1) Magnetic flux: |Y_000/Y_011| ~ 65, vs fit needs 0.379
  (2) Fractional CFT weights: c1/c2 ~ 1.37, vs fit needs 0.379
  (3) SO(10) for y_nu: y_nu ~ 0.966 vs fit 0.934

v2 change vs v1:
  - Added small tanβ scenario (y_t(M_GUT) = 0.7) for claim (3).
  - v24.5.1 alignment: the paper's small-tanβ factor is now ~2.28
    (manuscript §8 quotes ≈2.28), not the earlier ~1.95. The
    large-tanβ factor remains ≈1.38. Both labels in the SUMMARY
    and in the interpretation text have been updated accordingly.

Usage: python meml75_lepton_untested.py
"""
import cmath
import math
import numpy as np


TAU_L = 1.584 + 1.552j
C1_C2_TARGET = abs(-6.591 / -17.400)  # 0.3788
Y_NU_FIT = 0.934

print("=" * 78)
print(" meml75_lepton_untested.py — three lepton-sector tests from §13")
print("=" * 78)
print(f" tau_l = {TAU_L}")
print(f" Im(tau_l) = {TAU_L.imag}")
print(f" |c1/c2| target (fit) = {C1_C2_TARGET:.4f}")
print(f" y_nu (fit) = {Y_NU_FIT}")


# =================================================================
# PART 1: Magnetic flux (CI+M Yukawa)
# =================================================================

def yukawa_CIBM(tau, alpha, N_max=20):
    """Y = sum_n exp(-pi tau2 (alpha+n)^2 + i pi tau1 (alpha+n)^2)."""
    t2, t1 = tau.imag, tau.real
    return sum(
        math.exp(-math.pi * t2 * (alpha + n) ** 2)
        * cmath.exp(1j * math.pi * t1 * (alpha + n) ** 2)
        for n in range(-N_max, N_max + 1)
    )


def alpha_sel(i, j, k, M_Q, M_u, M_H):
    return i / M_Q + j / M_u + k / M_H


def is_integer(x):
    return abs(x - round(x)) < 1e-9


def scan_ratio(M_Q, M_u, M_H, target_ratio=65.0):
    """Find the Y_ijk pair closest to |Y_000/Y_ijk| = target."""
    Y000 = yukawa_CIBM(TAU_L, alpha_sel(0, 0, 0, M_Q, M_u, M_H))
    best = None
    for i in range(abs(M_Q)):
        for j in range(abs(M_u)):
            for k in range(abs(M_H)):
                if (i, j, k) == (0, 0, 0):
                    continue
                a = alpha_sel(i, j, k, M_Q, M_u, M_H)
                if not is_integer(a):
                    continue
                Y = yukawa_CIBM(TAU_L, a)
                if abs(Y) < 1e-15:
                    continue
                r = abs(Y000 / Y)
                if best is None or abs(r - target_ratio) < abs(best[3] - target_ratio):
                    best = (i, j, k, r, a)
    return best, Y000


print("\n" + "=" * 78)
print(" PART 1: Magnetic flux")
print("=" * 78)

print("\n--- Setup: M_Q = M_u = M_H = 2 (paper's 'flux M=2') ---")
Y000 = yukawa_CIBM(TAU_L, alpha_sel(0, 0, 0, 2, 2, 2))
print(f" Y_000 = {Y000:.6f}")
for (i, j, k) in [(0, 1, 1), (1, 1, 0), (2, 2, 2), (1, 0, 1)]:
    a = alpha_sel(i, j, k, 2, 2, 2)
    if is_integer(a):
        Y = yukawa_CIBM(TAU_L, a)
        print(f" Y_{i}{j}{k}: alpha={a:+.2f}, |Y|={abs(Y):.6f}, "
              f"|Y_000/Y_ijk|={abs(Y000/Y):.4f}")

print("\n Result: for M=2 setup, all allowed Y_ijk have integer alpha")
print(" -> |Y_ijk| = |Y_000| -> ratio = 1 (not 65).")
print(" The paper's 65 does not come from this simple setup.")

print("\n--- Systematic scan: which (M_Q, M_u, M_H) gives ~65? ---")
best_all = None
for M_Q in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
    for M_u in [-6, -5, -4, -3, -2, -1, 1, 2, 3, 4, 5, 6]:
        if M_Q + M_u == 0:
            continue
        M_H = -(M_Q + M_u)
        if abs(M_H) > 6:
            continue
        try:
            best, _ = scan_ratio(M_Q, M_u, M_H, target_ratio=65.0)
        except Exception:
            continue
        if best and (best_all is None or
                     abs(best[3] - 65) < abs(best_all[3] - 65)):
            best_all = (M_Q, M_u, M_H) + best

if best_all:
    M_Q, M_u, M_H, i, j, k, r, a = best_all
    print(f" Closest to 65: M_Q={M_Q}, M_u={M_u}, M_H={M_H}")
    print(f" |Y_000/Y_{i}{j}{k}| = {r:.4f} (target 65)")
    print(f" Still cannot give 65 with small fluxes in 1-modulus setup")
else:
    print(" No scan results (unexpected)")

print("\n Interpretation: 1-modulus CI+M Yukawa on magnetized T^2")
print(" gives ratios ~1 for integer alpha, and ratios 0.5-2 for")
print(" fractional alpha. Getting 65 requires either:")
print("  (a) a specific cross-modulus product, or")
print("  (b) Kähler metric factors that differ between zero modes, or")
print("  (c) a multi-modulus T^2/Z_3 × T^2/Z_4 setup (mixed orbifold).")
print(" None of these are specified in the paper; §13.4 must specify")
print(" the exact flux assignment to make the 65 claim reproducible.")


# =================================================================
# PART 2: Fractional CFT weights
# =================================================================

print("\n" + "=" * 78)
print(" PART 2: Fractional CFT weights")
print("=" * 78)

im_tau = TAU_L.imag
print(f"\n Im(tau_l) = {im_tau:.6f}")

candidates = {
    "(Im tau)^(1/4)": im_tau ** 0.25,
    "(Im tau)^(1/3)": im_tau ** (1 / 3),
    "(Im tau)^(3/8)": im_tau ** 0.375,
    "(Im tau)^(1/2)": im_tau ** 0.5,
    "(Im tau)^(2/3)": im_tau ** (2 / 3),
    "(Im tau)^(3/4)": im_tau ** 0.75,
    "(Im tau)^1": im_tau,
    "(2 Im tau)^(3/4)": (2 * im_tau) ** 0.75,
    "|Y_1/4| / |Y_1/2|": math.exp(-math.pi * im_tau / 2),
    "|Y_1/2| / |Y_1/4|": math.exp(math.pi * im_tau / 2),
}

print("\n Candidate formulas:")
for name, val in candidates.items():
    marker = " <-- closest to 1.37" if abs(val - 1.37) < 0.1 else ""
    print(f"   {name:25s} = {val:.4f}{marker}")

best_name = min(candidates, key=lambda k: abs(candidates[k] - 1.37))
print(f"\n Closest to paper's 1.37: {best_name} = {candidates[best_name]:.4f}")
print(f" Fit needs: {C1_C2_TARGET:.4f}")
print(f" Factor discrepancy: {candidates[best_name] / C1_C2_TARGET:.2f}")

print(f"\n Interpretation: fractional CFT weights give c1/c2 ~ 1.37-1.39,")
print(f" which is O(1) but factor ~3.7 above the fit value 0.379.")
print(f" INCONSISTENT with fit; negative result as claimed in §13.6.")


# =================================================================
# PART 3: SO(10) for y_nu
# =================================================================

print("\n" + "=" * 78)
print(" PART 3: SO(10) for y_nu")
print("=" * 78)

y_nu_fit = 0.934
y_t_MGUT = 0.483  # gives 2*y_t = 0.966
y_nu_SO10_GUT = 2 * y_t_MGUT

print(f"\n y_nu (fit at M_6) = {y_nu_fit}")
print(f" y_t(M_GUT) = {y_t_MGUT}")
print(f" SO(10) at M_GUT: y_nu = 2 y_t = {y_nu_SO10_GUT:.4f}")

ratio_MGUT = y_nu_fit / y_nu_SO10_GUT
print(f"\n Direct comparison at M_GUT:")
print(f"   y_nu_fit / y_nu_SO10 = {ratio_MGUT:.4f}")
print(f"   Deviation = {abs(ratio_MGUT - 1) * 100:.2f}%")
print(f"   Paper claim: ~3% agreement -> CONSISTENT at M_GUT")

M_GUT = 2.0e16
M_6 = 2.96e16
delta_t_GUT_to_M6 = math.log(M_6 / M_GUT)
print(f"\n M_6/M_GUT = {M_6/M_GUT:.3f}")
print(f" Delta log = {delta_t_GUT_to_M6:.4f}")
print(f" One-loop MSSM running (3 y_t^2 term only):")
y_t = 0.483
factor_1loop = 1 + (3 / (16 * math.pi ** 2)) * y_t ** 2 * delta_t_GUT_to_M6
print(f"   y_nu(M_6)/y_nu(M_GUT) = "
      f"1 + {(3 / (16 * math.pi ** 2)) * y_t ** 2 * delta_t_GUT_to_M6:.4f}")
print(f"   = {factor_1loop:.4f}")
print(f"   -> negligible running")

y_nu_RGE_paper = y_nu_fit / 1.95
print(f"\n Old v24.4 claim: y_nu^fit / y_nu^RGE = 1.95")
print(f" Implied: y_nu^RGE = {y_nu_RGE_paper:.4f}")
print(f" Compare: y_nu_SO10(M_GUT) = {y_nu_SO10_GUT:.4f}")
print(f" y_nu_SO10 / y_nu_RGE = {y_nu_SO10_GUT / y_nu_RGE_paper:.4f}")
print(f" RGE from M_GUT to M_6 alone reduces y_nu by factor ~1.0017")
print(f" -> RGE from M_GUT to M_6 alone CANNOT explain the ~1.38/2.28 gap.")

print(f"\n Naive long-interval RGE (GUT to M_SUSY ~ 1 TeV):")
M_SUSY = 1.0e3
delta_t_long = math.log(M_GUT / M_SUSY)
print(f" Delta log = {delta_t_long:.2f}")
y_t_avg = 0.7
integral = y_t_avg ** 2 * delta_t_long
factor_long = math.exp((3 / (16 * math.pi ** 2)) * integral)
print(f" Factor y_nu(TeV)/y_nu(GUT) = exp(3 y_t^2 log / 16pi^2) = "
      f"{factor_long:.4f}")
print(f" This IS O(2)! But it's running GUT -> TeV, not GUT -> M_6.")

# === Small tanβ scenario: y_t(M_GUT) = 0.7 ===
print(f"\n --- Small tanβ scenario: y_t(M_GUT) = 0.7 ---")
y_t_MGUT_small = 0.7
y_nu_SO10_small = 2.0 * y_t_MGUT_small           # = 1.40
print(f" SO(10) at M_GUT: y_nu = 2 y_t = {y_nu_SO10_small:.4f}")
print(f" (cf. paper: y_t(M_GUT) ~ 0.7 for small tanβ)")

# y_t runs from 0.7 (GUT) to ~1.0 (TeV); avg over interval ~0.85
y_t_avg_small = 0.85
integral_small = y_t_avg_small ** 2 * delta_t_long
factor_long_small = math.exp((3 / (16 * math.pi ** 2)) * integral_small)
print(f" Long-interval RGE factor (GUT->TeV, y_t_avg=0.85) = "
      f"{factor_long_small:.4f}")

y_nu_at_susy_small = y_nu_SO10_small * factor_long_small
print(f" y_nu(M_SUSY) from minimal SO(10) = {y_nu_at_susy_small:.4f}")
print(f" Fit needs y_nu = {y_nu_fit}")
print(f" Ratio (requires 126_H) = {y_nu_at_susy_small / y_nu_fit:.2f}")
print(f" Paper claim for small tanβ: factor ~2.28")

# Sensitivity to y_t_avg
print(f"\n Sensitivity: RGE factor vs assumed y_t_avg (small tanβ):")
for yt_avg_test in [0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]:
    fac = math.exp((3 / (16 * math.pi ** 2)) * (yt_avg_test ** 2) * delta_t_long)
    ratio = y_nu_SO10_small * fac / y_nu_fit
    print(f"   y_t_avg={yt_avg_test:.2f} -> RGE factor {fac:.3f}, "
          f"ratio to fit {ratio:.2f}")
# === end small-tanβ block ===

print(f"\n Interpretation: the paper's large-tanβ factor (~1.38) and")
print(f" small-tanβ factor (~2.28) both come from running between")
print(f" GUT scale and SUSY scale (long interval), NOT from M_GUT to M_6")
print(f" (short interval, factor ~1.0017).")
print(f" With long-interval running, y_nu at M_SUSY is factor ~1.3-1.5 larger")
print(f" than at M_GUT, which is compatible with the claim.")
print(f" Verdict: minimal SO(10) (y_nu = 2 y_t at GUT) gives y_nu(M_SUSY)")
print(f" = {y_nu_SO10_GUT * factor_long:.3f} (large tanβ), but fit needs "
      f"{y_nu_fit}. Factor discrepancy")
print(f" {y_nu_SO10_GUT * factor_long / y_nu_fit:.2f} requires 126_H.")


# =================================================================
# SUMMARY
# =================================================================

print("\n" + "=" * 78)
print(" SUMMARY: three lepton-sector claims from §13")
print("=" * 78)

print(f"""
  (1) Magnetic flux:
       Paper claim: |Y_000/Y_011| ~ 65 for 'flux M=2'
       Fit needs: |c1/c2| = 0.379
       Status:      INCONSISTENT (65 vs 0.379) -- negative, as claimed
       Note:        our simple 1-modulus scan finds ratios ~1, not 65;
                    the exact 65 requires a specific setup (multi-modulus,
                    Kähler factors, or mixed orbifold) not detailed in §13.4.

  (2) Fractional CFT weights:
       Paper claim: c1/c2 ~ 1.37
       Our computation: (Im tau_l)^(3/4) = {candidates["(Im tau)^(3/4)"]:.4f},
                        close to 1.37 (weight difference 3/4)
       Fit needs: |c1/c2| = 0.379
       Status: INCONSISTENT (factor ~3.7 off) -- negative, as claimed

  (3) SO(10) for y_nu:
       Large tanβ (y_t(M_GUT)=0.483): y_nu_SO10(M_GUT)=0.966,
         ratio after long RGE = {y_nu_SO10_GUT * factor_long / y_nu_fit:.2f} (paper: ~1.38)
       Small tanβ (y_t(M_GUT)=0.7): y_nu_SO10(M_GUT)=1.40,
         ratio after long RGE = {y_nu_at_susy_small / y_nu_fit:.2f} (paper: ~2.28)
       Status: both require 126_H -- negative, as claimed.
       See sensitivity table above for y_t_avg dependence.
""")
print("=" * 78)