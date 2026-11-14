#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 (v22.1) — Exact S4' modular forms (CORRECTED, v2)
===========================================================
FIX: variable 'err' was overwritten in the loop; renamed to
'val_err' (validation) and 'pair_err' (pairwise ratio).
"""
import numpy as np

TAU_L = 1.584 + 1.552j
TARGET_C1_C2 = abs(-6.591 / -17.400)  # = 0.3788

def eta(tau, N_max=200):
    q = np.exp(2j * np.pi * tau)
    prod = 1 + 0j
    for n in range(1, N_max):
        prod *= (1 - q**n)
    return q**(1/24) * prod

def dlog_eta(tau, N_max=200):
    q = np.exp(2j * np.pi * tau)
    E2 = 1.0 + 0j
    for n in range(1, N_max):
        qn = q**n
        E2 -= 24 * n * qn / (1 - qn)
    return 1j * np.pi / 12 * E2

def Y(a, tau, N_max=200):
    if abs(sum(a)) > 1e-12:
        raise ValueError("sum(a) must be 0")
    args = [
        (2, 1, 2),   # tau + 1/2   <-- FIXED
        (4, 0, 1),   # 4*tau
        (1, 0, 4),   # tau/4
        (1, 1, 4),   # (tau+1)/4
        (1, 2, 4),   # (tau+2)/4
        (1, 3, 4),   # (tau+3)/4
    ]
    total = 0j
    for ai, (num, b, denom) in zip(a, args):
        arg = (num * tau + b) / denom
        total += ai * (num / denom) * dlog_eta(arg, N_max)
    return total

omega = np.exp(2j * np.pi / 3)

def S4_forms(tau, N_max=200):
    Y1 = Y([1, 1, omega,   omega**2, omega,   omega**2], tau, N_max)
    Y2 = Y([1, 1, omega**2, omega,   omega**2, omega  ], tau, N_max)
    Y3 = Y([1, -1, -1, -1, 1, 1], tau, N_max)
    Y4 = Y([1, -1, -omega**2, -omega,   omega**2, omega  ], tau, N_max)
    Y5 = Y([1, -1, -omega,   -omega**2, omega,   omega**2], tau, N_max)
    return Y1, Y2, Y3, Y4, Y5

if __name__ == "__main__":
    print("=" * 72)
    print(" MEML 75 (v22.1) — Exact S4' forms (CORRECTED, v2)")
    print("=" * 72)

    # 1. Validation
    print(f"\n[1] Validation at tau = i:")
    Y1i, Y2i, Y3i, Y4i, Y5i = S4_forms(1j, N_max=300)
    Y1_paper = 0.7107 + 1.231j
    val_err = abs(Y1i - Y1_paper) / abs(Y1_paper)   # <-- RENAMED
    print(f"    Y1(i) = {Y1i.real:.6f} + {Y1i.imag:.6f}i")
    print(f"    paper = {Y1_paper.real:.4f} + {Y1_paper.imag:.4f}i")
    print(f"    relative error = {val_err:.2e}")
    if val_err < 1e-4:
        print("    [OK] Validation passed")
    else:
        print("    [FAIL] Validation failed")

    # 2. Forms at tau_l
    print(f"\n[2] S4' forms at tau_l = {TAU_L}:")
    Y1, Y2, Y3, Y4, Y5 = S4_forms(TAU_L, N_max=300)
    for name, val in zip(["Y1", "Y2", "Y3", "Y4", "Y5"],
                          [Y1, Y2, Y3, Y4, Y5]):
        print(f"    {name} = {val:.4f}  (|{name}| = {abs(val):.4f})")

    # 3. Pairwise ratios
    print(f"\n[3] Target |c1/c2| = {TARGET_C1_C2:.4f}")
    print(f"    {'ratio':<12} {'|Yi/Yj|':>12} {'err (orders)':>14}")
    print("    " + "-" * 40)
    best_err = np.inf
    best_pair = None
    for i in range(5):
        for j in range(5):
            if i == j:
                continue
            r = abs([Y1, Y2, Y3, Y4, Y5][i] / [Y1, Y2, Y3, Y4, Y5][j])
            pair_err = abs(np.log10(r / TARGET_C1_C2))   # <-- RENAMED
            if pair_err < best_err:
                best_err = pair_err
                best_pair = (i + 1, j + 1, r)
            print(f"    |Y{i+1}/Y{j+1}|     {r:>12.4f} {pair_err:>14.4f}")

    # 4. Verdict (FIXED: uses val_err, not err)
    print(f"\n[4] Best ratio: |Y{best_pair[0]}/Y{best_pair[1]}| = {best_pair[2]:.4f}")
    print(f"    Target: {TARGET_C1_C2:.4f}")
    print(f"    Error: {best_err:.4f} orders (factor {10**best_err:.2f})")

    print("\n" + "=" * 72)
    print(" VERDICT")
    print("=" * 72)
    if val_err > 1e-4:                          # <-- FIXED
        print(" [FAIL] Validation failed — check eta arguments")
    elif best_err < 0.1:
        print(f" [OK] Good match: |Y{best_pair[0]}/Y{best_pair[1]}| ≈ {best_pair[2]:.4f}")
    else:
        print(f" [NEGATIVE] No S4' ratio matches target.")
        print(f"            Best: |Y{best_pair[0]}/Y{best_pair[1]}| = {best_pair[2]:.4f}")
        print(f"            Error: factor {10**best_err:.2f}")
        print(f"            c1, c2 remain free parameters.")
    print("=" * 72)