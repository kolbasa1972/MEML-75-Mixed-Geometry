#!/usr/bin/env python3
"""
meml75_flux_scan_cs.py — Standard Kähler convention flux scan

Scans integer fluxes (a0, a1, a2) in [-50, 50]^3 for each of the three
moduli, using the STANDARD Type IIB Kähler potential for complex-structure
moduli:

    K_cs(tau) = -ln(-i(tau - tau_bar)) = -ln(2y)
    K_tau = i / (2y)

The GVW condition D_tau W = 0 has the closed-form solution:

    x = -a1 / (2*a2)
    y^2 = (1/3) * (-a0/a2 + a1^2/(4 a2^2))

(compare with K = -3 ln(2y), where y^2 = -a0/a2 + a1^2/(4 a2^2) = 3 y^2_CS)

This script tests whether the flux triplets (-30, 5, 16), (-15, -17, 23),
(-16, 31, -10) used in v23.1 (which work at K = -3 ln) reproduce the fit
values at K_cs, and searches for alternative triplets that do.

Output:
    - best triplet and deviation from fit for each modulus
    - full list of triplets with deviation < 10%
    - comparison with the v23.1 triplets
"""
import numpy as np

# ---- Fit values (from meml75_honest_pure2.py) ----
TAU_FIT = {
    "u": -0.1577944085 + 2.3851782590j,
    "d": +0.3693905617 + 1.5410496057j,
    "l": 1.5398 + 1.5511j,
}

# ---- Fluxes used in v23.1 (which work at K = -3 ln) ----
FLUX_V23 = {
    "u": (-30,   5,  16),
    "d": (-15, -17,  23),
    "l": (-16,  31, -10),
}


def solve_gvw_standard(a0, a1, a2):
    """Solve D_tau W = 0 with K_cs = -ln(2y)."""
    if a2 == 0:
        return None
    x = -a1 / (2.0 * a2)
    y2 = (1.0 / 3.0) * (-a0 / a2 + a1**2 / (4.0 * a2**2))
    if y2 <= 0:
        return None
    return complex(x, np.sqrt(y2))


def solve_gvw_nonstandard(a0, a1, a2):
    """Solve D_tau W = 0 with K = -3 ln(2y)."""
    if a2 == 0:
        return None
    x = -a1 / (2.0 * a2)
    y2 = -3.0 * a0 / a2 + 3.0 * a1**2 / (4.0 * a2**2)
    if y2 <= 0:
        return None
    return complex(x, np.sqrt(y2))


def relative_deviation(tau_gvw, tau_fit):
    return abs(tau_gvw - tau_fit) / abs(tau_fit)


def scan_modulus(tau_fit, flux_range=(-50, 50), max_dev=0.10):
    """Scan all fluxes, return list of candidates with dev < max_dev."""
    lo, hi = flux_range
    candidates = []
    for a0 in range(lo, hi + 1):
        for a1 in range(lo, hi + 1):
            for a2 in range(lo, hi + 1):
                if a2 == 0:
                    continue
                tau = solve_gvw_standard(a0, a1, a2)
                if tau is None:
                    continue
                dev = relative_deviation(tau, tau_fit)
                if dev < max_dev:
                    candidates.append((dev, (a0, a1, a2), tau))
    candidates.sort(key=lambda x: x[0])
    return candidates


def main():
    print("=" * 78)
    print(" meml75_flux_scan_cs.py")
    print(" Standard Type IIB Kähler convention for complex-structure moduli")
    print(" K_cs = -ln(-i(tau - tau_bar)) = -ln(2y),  K_tau = i/(2y)")
    print("=" * 78)

    # ---- Part 1: show the v23.1 fluxes at k=3 and at k=1 ----
    print("\n[Part 1] Fluxes used in v23.1: what happens at k=3 vs k=1?")
    print(f"{'modulus':<10}{'fluxes':<20}{'tau^GVW (k=3)':<30}{'tau^GVW (k=1)':<30}{'fit'}")
    print("-" * 110)
    for name in ["u", "d", "l"]:
        flux = FLUX_V23[name]
        tau_3 = solve_gvw_nonstandard(*flux)
        tau_1 = solve_gvw_standard(*flux)
        fit = TAU_FIT[name]
        print(f"{name:<10}{str(flux):<20}"
              f"{tau_3.real:+.4f}{tau_3.imag:+.4f}i         "
              f"{tau_1.real:+.4f}{tau_1.imag:+.4f}i         "
              f"{fit.real:+.4f}{fit.imag:+.4f}i")

    print("\nDeviation from fit (relative):")
    print(f"{'modulus':<10}{'v23.1 fluxes @ k=3':<30}{'v23.1 fluxes @ k=1'}")
    print("-" * 70)
    for name in ["u", "d", "l"]:
        flux = FLUX_V23[name]
        tau_3 = solve_gvw_nonstandard(*flux)
        tau_1 = solve_gvw_standard(*flux)
        dev_3 = relative_deviation(tau_3, TAU_FIT[name])
        dev_1 = relative_deviation(tau_1, TAU_FIT[name])
        print(f"{name:<10}{dev_3:>20.4e}      {dev_1:>20.4e}")

    # ---- Part 2: scan for best fluxes at k=1 ----
    print("\n[Part 2] Search for fluxes at k=1 (standard convention)")
    print("         Range: [-50, 50]^3, accept deviation < 10%")
    print()

    for name in ["u", "d", "l"]:
        print("-" * 78)
        print(f" Searching for tau_{name}")
        print("-" * 78)
        cands = scan_modulus(TAU_FIT[name], max_dev=0.10)
        if not cands:
            print(f"  NO triplet reproduces tau_{name} to better than 10%")
            continue

        print(f"  Top 5 triplets with deviation < 10%:")
        print(f"  {'dev':<12}{'fluxes':<25}{'tau^GVW'}")
        for dev, flux, tau in cands[:5]:
            print(f"  {dev:<12.4e}{str(flux):<25}"
                  f"{tau.real:+.4f}{tau.imag:+.4f}i")
        print(f"  Total triplets with dev < 10%: {len(cands)}")

    # ---- Part 3: verdict ----
    print("\n" + "=" * 78)
    print(" VERDICT")
    print("=" * 78)

    all_found = {}
    for name in ["u", "d", "l"]:
        cands = scan_modulus(TAU_FIT[name], max_dev=0.10)
        all_found[name] = cands

    n_ok = sum(1 for name in ["u", "d", "l"]
               if len(all_found[name]) > 0)

    if n_ok == 3:
        print("\n  All three moduli reproduced by at least one flux triplet")
        print("  at the standard convention K_cs = -ln(2y).")
        print("\n  Best triplets:")
        for name in ["u", "d", "l"]:
            if all_found[name]:
                dev, flux, tau = all_found[name][0]
                print(f"    tau_{name}: flux {flux}, "
                      f"tau = {tau.real:+.4f}{tau.imag:+.4f}i, "
                      f"dev = {dev:.2e}")
        print("\n  => The claim in v24.0 §11 that the standard convention")
        print("     does not reproduce the fit at better than 10% is")
        print("     FALSE. The scan finds suitable triplets at k=1.")
        print("  => Section 11 of v24.0 must be updated.")
    elif n_ok == 0:
        print("\n  No flux triplet reproduces any of the fit values at k=1.")
        print("  The honest-negative claim in §11 is confirmed.")
    else:
        print(f"\n  {n_ok}/3 moduli reproduced at k=1.")
        print("  Mixed result: §11 needs nuanced revision.")
    print("=" * 78)


if __name__ == "__main__":
    main()