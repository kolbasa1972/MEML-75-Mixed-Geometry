#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_delta_cp.py
==================
Compute the Dirac CP phase delta_CP of the PMNS matrix from the
constrained lepton fit of Section 8 of the paper.

Method:
  1. Build M_nu from the constrained c_i and tau_l.
  2. Diagonalize M_nu M_nu^dagger to get U (PMNS-like).
  3. Extract angles theta_12, theta_13, theta_23 from |U_ij|.
  4. Compute Jarlskog invariant J = Im(U_e1 U_mu2 U_e2* U_mu1*).
  5. Recover sin(delta_CP) = J / (c12 s12 c23 s23 c13^2 s13).
  6. Report delta_CP candidates in [0, 360).

If |sin(delta_CP)| ~ 0.39, then delta_CP ~ 23 or 157 degrees,
confirming the abstract claim. Otherwise, the abstract is wrong.

Usage: python meml75_delta_cp.py
"""
import numpy as np


# ==================== Constrained lepton fit (Section 8) ====================
TAU_L = 1.584 + 1.552j
C1 = -6.591
C2 = -17.400
C3 = 17.400  # = -C2

# ==================== Alternative: unconstrained (for comparison) =========
TAU_L_UNCONSTR = 1.5398 + 1.5511j
C1_UNCONSTR =  6.371
C2_UNCONSTR = 17.460
C3_UNCONSTR = -17.199


def theta_eps(tau):
    q4 = np.exp(1j * np.pi * tau / 2.0)
    return (1.0 + 2.0 * q4 ** 4 + 2.0 * q4 ** 16,
            2.0 * q4 + 2.0 * q4 ** 9 + 2.0 * q4 ** 25)


def build_M_nu_S4(tau, c1, c2, c3):
    th, ep = theta_eps(tau)
    Y2_v = np.array([(th ** 4 + ep ** 4) / np.sqrt(2.0),
                     -np.sqrt(6.0) * ep ** 2 * th ** 2], dtype=complex)
    Y3p = np.array([(th ** 4 - ep ** 4) / np.sqrt(2.0),
                    -2.0 * ep * th ** 3, -2.0 * ep ** 3 * th], dtype=complex)
    M2 = np.array([[Y2_v[0], Y2_v[1], 0],
                   [Y2_v[1], -Y2_v[0], 0],
                   [0, 0, Y2_v[0]]], dtype=complex)
    M3p = np.array([[Y3p[0], Y3p[2], Y3p[1]],
                    [Y3p[2], Y3p[1], Y3p[0]],
                    [Y3p[1], Y3p[0], Y3p[2]]], dtype=complex)
    return c1 * M2 + c2 * M3p + c3 * np.eye(3)


def extract_pmns(M_nu, label):
    """Given M_nu, compute angles, J, delta_CP."""
    w, U = np.linalg.eigh(M_nu @ M_nu.conj().T)
    m = np.sqrt(np.maximum(w, 1e-30))

    s13 = abs(U[0, 2])
    s12 = abs(U[0, 1]) / np.sqrt(max(1.0 - s13 ** 2, 1e-30))
    s23 = abs(U[1, 2]) / np.sqrt(max(1.0 - s13 ** 2, 1e-30))
    c13 = np.sqrt(max(1.0 - s13 ** 2, 1e-30))
    c12 = np.sqrt(max(1.0 - s12 ** 2, 1e-30))
    c23 = np.sqrt(max(1.0 - s23 ** 2, 1e-30))

    J = np.imag(U[0, 1] * U[1, 2] * np.conj(U[0, 2]) * np.conj(U[1, 1]))
    denom = c12 * s12 * c23 * s23 * c13 ** 2 * s13

    print(f"\n--- {label} ---")
    print(f"  theta_12 = {np.degrees(np.arcsin(s12)):.2f} deg")
    print(f"  theta_13 = {np.degrees(np.arcsin(s13)):.2f} deg")
    print(f"  theta_23 = {np.degrees(np.arcsin(s23)):.2f} deg")
    print(f"  J        = {J:+.6e}")
    print(f"  denom    = {denom:+.6e}")

    if abs(denom) < 1e-15:
        print("  WARNING: sin(delta_CP) undefined (denom -> 0)")
        return None

    sin_d = J / denom
    print(f"  sin(delta_CP) = {sin_d:+.4f}")

    if abs(sin_d) > 1.0:
        print(f"  WARNING: |sin(delta_CP)| = {abs(sin_d):.4f} > 1 (numerical issue)")
        return None

    d1 = float(np.degrees(np.arcsin(sin_d)))
    if d1 < 0:
        d1 += 360.0
    d2 = 180.0 - d1
    print(f"  delta_CP = {d1:.1f} deg  or  {d2:.1f} deg")
    print(f"  |delta_CP| in [0,180]: min({d1:.1f}, {d2:.1f}) = "
          f"{min(d1, d2 % 360):.1f} deg")

    return sin_d, d1, d2


def main():
    print("=" * 74)
    print(" meml75_delta_cp.py — Dirac CP phase from the lepton fit")
    print("=" * 74)

    # Constrained (used for consequences)
    M_c = build_M_nu_S4(TAU_L, C1, C2, C3)
    res_c = extract_pmns(M_c, "CONSTRAINED (c2+c3=0), used in paper")

    # Unconstrained (cross-check)
    M_u = build_M_nu_S4(TAU_L_UNCONSTR, C1_UNCONSTR, C2_UNCONSTR, C3_UNCONSTR)
    res_u = extract_pmns(M_u, "UNCONSTRAINED (cross-check)")

    # Comparison with paper claim
    print("\n" + "=" * 74)
    print(" VERDICT vs PAPER CLAIM ('delta_CP = 23 or 157 deg')")
    print("=" * 74)
    target_sin = np.sin(np.radians(23.0))
    print(f"  Paper claim: delta_CP ~ 23 or 157 deg  =>  |sin| ~ {abs(target_sin):.4f}")
    print(f"  (Note: sin(23) = sin(157), so |sin| alone is ambiguous)")

    if res_c is None:
        print("\n  [FAIL] Constrained fit does not give a well-defined delta_CP.")
    else:
        sin_c, d1_c, d2_c = res_c
        match = abs(abs(sin_c) - abs(target_sin)) < 0.05
        status = "[OK]" if match else "[FAIL]"
        print(f"\n  Constrained: sin(delta) = {sin_c:+.4f}, "
              f"delta = {d1_c:.1f} or {d2_c:.1f} deg")
        print(f"  {status} match to paper claim (tolerance 0.05 in |sin|)")

        if not match:
            print("\n  ACTION REQUIRED: abstract claim '23 or 157 deg' is NOT")
            print("  reproduced by the constrained fit. Either")
            print("    (a) the abstract is wrong and must be corrected, or")
            print("    (b) the value 23/157 came from an older fit/script")
            print("        that is not the current one.")

    print("\n" + "=" * 74)
    print(" NOTE on Majorana phases")
    print("=" * 74)
    print("""
  The model gives M_nu = c1 M2 + c2 M3' + c3 I with complex c_i.
  Diagonalizing M_nu M_nu^dagger gives U (PMNS-like) and m_i.
  The Dirac phase delta_CP is extracted from the Jarlskog invariant J.
  Majorana phases alpha, beta do NOT affect oscillations and do not
  enter J; they are not fixed by the model as stated. The value of
  delta_CP is a well-defined prediction (given c_i and tau_l); the
  Majorana phases are not.

  If the user wants to claim alpha, beta as predictions, additional
  structure (e.g., specific phases from the orbifold) is required.
""")
    print("=" * 74)


if __name__ == "__main__":
    main()