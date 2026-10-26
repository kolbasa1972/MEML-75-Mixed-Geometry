#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — FLUX CONVENTIONS CHECK (v17.0)

Tests the "old" fluxes (-17, 0, 1) and (-7, -1, 1) that were
previously cited in an earlier draft (Section 12) in three different
Kaehler conventions, to understand whether the value |D_tau W| ~ 0.16
can be reproduced.

Conventions:
    "T"   : K = -3 ln(-i(tau - tau_bar))  -> K_tau = -3i/(tau - tau_bar)
    "1D"  : K = -1 ln(-i(tau - tau_bar))  -> K_tau = -i/(tau - tau_bar)
    "none": no Kaehler term (dW only, non-covariant)

Also performs an extended scan for tau_u in [-50, 50] (K_T convention).

Usage:
    python meml75_flux_conventions.py
"""
import numpy as np
import math


def D_tau_W(tau, a0, a1, a2, K_convention="T"):
    W = a0 + a1 * tau + a2 * tau**2
    dW = a1 + 2 * a2 * tau

    if K_convention == "T":
        K_tau = -3j / (tau - np.conj(tau))
        D = dW + K_tau * W
    elif K_convention == "1D":
        K_tau = -1j / (tau - np.conj(tau))
        D = dW + K_tau * W
    elif K_convention == "none":
        D = dW
    else:
        raise ValueError(K_convention)

    return np.abs(D)


def main():
    print("="*70)
    print(" Check old fluxes from the article (Section 12) in 3 conventions")
    print("="*70)

    checks = [
        ("tau_u", complex(-0.158, 2.385), (-17, 0, 1)),
        ("tau_d", complex( 0.369, 1.541), (-7, -1, 1)),
        ("tau_u", complex(-0.158, 2.385), (-1, 0, 0)),
        ("tau_l", complex( 1.540, 1.551), (-5, -3, 1)),
        ("tau_l", complex( 1.540, 1.551), (-15, 4, -4)),
    ]

    for name, tau, flux in checks:
        print(f"\n{name} = {tau.real:+.3f}{tau.imag:+.3f}i   fluxes = {flux}")
        for conv in ["T", "1D", "none"]:
            val = D_tau_W(tau, *flux, K_convention=conv)
            print(f"   |D_tau W| (K_{conv:4}) = {val:.5f}")

    print("\n" + "="*70)
    print(" Extended scan for tau_u in [-50, 50], best 10 candidates (K_T)")
    print("="*70)

    tau_u = complex(-0.158, 2.385)
    results = []
    for a0 in range(-50, 51):
        for a1 in range(-50, 51):
            for a2 in range(-50, 51):
                if a0 == 0 and a1 == 0 and a2 == 0:
                    continue
                g = math.gcd(math.gcd(abs(a0), abs(a1)), abs(a2))
                if g > 1:
                    continue

                val = D_tau_W(tau_u, a0, a1, a2, K_convention="T")
                results.append((val, (a0, a1, a2)))

    results.sort(key=lambda x: x[0])
    for val, flux in results[:10]:
        print(f"   |D_tau W| = {val:.5f}   fluxes = {str(flux):20}")


if __name__ == '__main__':
    main()