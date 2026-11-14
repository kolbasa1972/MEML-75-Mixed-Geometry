#!/usr/bin/env python3
"""
meml75_leptogenesis.py — CP asymmetry and pre-dilution baryon asymmetry

Computes:
  1. Davidson-Ibarra upper bound on epsilon_CP
  2. Required epsilon_CP to reproduce the observed eta_B after dilution
  3. eta_B^pre (before dilution)
  4. Required dilution Delta
  5. Consistency checks

Reference formulas:
  eta_B^pre ≈ (28/79) * epsilon_CP * kappa / g_*
  |epsilon_CP| ≤ (3/16pi) * M_R * (m3 - m1) / v^2    (Davidson-Ibarra)

Inputs (from meml75_honest_pure2.py):
  M_R ~ M_6 = 2.9631e16 GeV
  m1 = 6.39 meV, m2 = 10.75 meV, m3 = 50.55 meV
  v = 246.22 GeV, kappa = 1, g_* = 106.75

Observed baryon asymmetry (Planck 2018):
  eta_B^obs = (6.12 ± 0.04) × 10^-10
"""
import numpy as np

# ---- Inputs ----
M_R = 2.9631e16       # GeV (M_6 reference scale)
m1_eV = 6.39e-3       # eV
m2_eV = 10.75e-3      # eV
m3_eV = 50.55e-3      # eV
v_GeV = 246.22        # GeV
kappa = 1.0           # washout factor (order unity)
g_star = 106.75       # SM relativistic dof

# Conversions
eV_to_GeV = 1e-9
m1 = m1_eV * eV_to_GeV
m2 = m2_eV * eV_to_GeV
m3 = m3_eV * eV_to_GeV

# Observed baryon asymmetry
eta_B_obs = 6.12e-10

# Reference values from the paper (v23.1, §9)
DELTA_TARGET = 3.13e6
EPSILON_CP_PAPER = 0.57
ETA_B_PRE_PAPER = 1.9e-3


def eta_B_pre(eps_CP, kappa=1.0, g_star=106.75):
    """Pre-dilution baryon asymmetry."""
    return (28.0 / 79.0) * eps_CP * kappa / g_star


def davidson_ibarra_bound(M_R, m3, m1, v):
    """Upper bound on |epsilon_CP|."""
    return (3.0 / (16.0 * np.pi)) * M_R * (m3 - m1) / v**2


def main():
    print("=" * 78)
    print(" meml75_leptogenesis.py — CP asymmetry and eta_B^pre")
    print("=" * 78)
    print(f" Inputs:")
    print(f"   M_R (heavy neutrino scale) = {M_R:.4e} GeV")
    print(f"   m1 = {m1_eV*1000:.2f} meV, m2 = {m2_eV*1000:.2f} meV, "
          f"m3 = {m3_eV*1000:.2f} meV")
    print(f"   v = {v_GeV} GeV, kappa = {kappa}, g_* = {g_star}")

    # ---- Step 1: Davidson-Ibarra bound ----
    print("\n[Step 1] Davidson-Ibarra upper bound on |epsilon_CP|")
    bound = davidson_ibarra_bound(M_R, m3, m1, v_GeV)
    print(f"  |epsilon_CP| <= (3/16pi) * M_R * (m3 - m1) / v^2")
    print(f"                = {bound:.4f}")

    # ---- Step 2: eta_B^pre from paper epsilon ----
    print("\n[Step 2] Pre-dilution baryon asymmetry (from paper epsilon_CP = 0.57)")
    eta_pre_paper = eta_B_pre(EPSILON_CP_PAPER)
    print(f"  eta_B^pre = (28/79) * {EPSILON_CP_PAPER} * {kappa} / {g_star}")
    print(f"            = {eta_pre_paper:.4e}")
    print(f"  Paper: eta_B^pre = {ETA_B_PRE_PAPER:.4e}  "
          f"[{'OK' if abs(eta_pre_paper - ETA_B_PRE_PAPER)/ETA_B_PRE_PAPER < 0.05 else 'MISMATCH'}]")

    # ---- Step 3: required epsilon_CP from eta_B_obs ----
    print("\n[Step 3] Required epsilon_CP to reproduce eta_B^obs")
    eta_pre_req = eta_B_obs * DELTA_TARGET
    eps_req = eta_pre_req * g_star / ((28.0 / 79.0) * kappa)
    print(f"  eta_B^pre (required) = eta_B^obs * Delta")
    print(f"                       = {eta_B_obs:.4e} * {DELTA_TARGET:.4e}")
    print(f"                       = {eta_pre_req:.4e}")
    print(f"  epsilon_CP (required) = eta_B^pre * g_* / ((28/79) * kappa)")
    print(f"                        = {eps_req:.4f}")
    print(f"  Paper: epsilon_CP = {EPSILON_CP_PAPER}  "
          f"[{'OK' if abs(eps_req - EPSILON_CP_PAPER)/EPSILON_CP_PAPER < 0.05 else 'MISMATCH'}]")

    # ---- Step 4: check against bound ----
    print("\n[Step 4] Consistency check: required epsilon vs Davidson-Ibarra bound")
    ok = eps_req <= bound
    print(f"  Required: {eps_req:.4f}")
    print(f"  Bound:    {bound:.4f}")
    print(f"  [{'OK' if ok else 'VIOLATION'}] "
          f"Required epsilon_CP is {'below' if ok else 'above'} the DI bound")

    # ---- Step 5: required Delta from paper epsilon ----
    print("\n[Step 5] Required dilution Delta (from paper epsilon_CP)")
    Delta_req = eta_pre_paper / eta_B_obs
    print(f"  Delta = eta_B^pre / eta_B^obs = {eta_pre_paper:.4e} / {eta_B_obs:.4e}")
    print(f"        = {Delta_req:.4e}")
    print(f"  Paper: Delta = {DELTA_TARGET:.4e}  "
          f"[{'OK' if abs(Delta_req - DELTA_TARGET)/DELTA_TARGET < 0.05 else 'MISMATCH'}]")

    # ---- Verdict ----
    print("\n" + "=" * 78)
    print(" VERDICT")
    print("=" * 78)

    checks = [
        ("eta_B^pre matches paper", abs(eta_pre_paper - ETA_B_PRE_PAPER) / ETA_B_PRE_PAPER < 0.05),
        ("epsilon_CP matches paper", abs(eps_req - EPSILON_CP_PAPER) / EPSILON_CP_PAPER < 0.05),
        ("epsilon_CP below DI bound", eps_req <= bound),
        ("Delta matches paper", abs(Delta_req - DELTA_TARGET) / DELTA_TARGET < 0.05),
    ]
    all_ok = True
    for label, ok in checks:
        print(f"  [{'OK' if ok else 'FAIL'}] {label}")
        if not ok:
            all_ok = False

    print()
    if all_ok:
        print("  All checks passed. Numbers in §9 (leptogenesis) are internally consistent.")
        print("  Note: epsilon_CP ~ 0.57 is LARGE (O(1)), requiring near-maximal CP phases.")
        print("  This is allowed by the DI bound but is a strong requirement.")
    else:
        print("  Some checks failed. Re-examine §9 inputs.")

    print("=" * 78)


if __name__ == "__main__":
    main()