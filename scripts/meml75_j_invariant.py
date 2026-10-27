#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — J-INVARIANT CHECK (v17.0)

Computes the modular j-invariant
    j(tau) = 1728 * E4(tau)^3 / (E4(tau)^3 - E6(tau)^2)
for the three fitted moduli, to check whether they are algebraic
(CM) points or generic points in the fundamental domain.

A real j-invariant close to an integer would indicate a CM point,
which would allow analytic (non-fitted) stabilization of the modulus.
A complex j-invariant indicates a generic point.

Usage:
    python meml75_j_invariant.py
"""
import numpy as np


def j_invariant(tau, N_terms=10):
    """
    Compute the j-invariant via q-expansion of Eisenstein series E4, E6.
    Converges rapidly for Im(tau) > 1.
    """
    q = np.exp(2j * np.pi * tau)

    sum_E4 = 0.0 + 0.0j
    sum_E6 = 0.0 + 0.0j

    for n in range(1, N_terms + 1):
        qn = q**n
        term = qn / (1.0 - qn)
        sum_E4 += (n**3) * term
        sum_E6 += (n**5) * term

    E4 = 1.0 + 240.0 * sum_E4
    E6 = 1.0 - 504.0 * sum_E6

    j = 1728.0 * (E4**3) / (E4**3 - E6**2)
    return j


def main():
    taus = {
        "tau_l (lepton)":   complex(-0.46, 1.55),
        "tau_u (up-quark)": complex(-0.158, 2.385),
        "tau_d (down-quark)": complex(0.369, 1.541),
    }

    print("j-invariants of fitted moduli:\n")

    for name, tau in taus.items():
        j_val = j_invariant(tau)
        j_real = j_val.real
        j_imag = j_val.imag
        nearest_int = round(j_real)
        diff = abs(j_real - nearest_int)

        print(f"{name:20} : j = {j_real:.6f} + {j_imag:.6f}i")

        if abs(j_imag) < 1e-5 and diff < 0.01:
            print("     [OK] Real j-invariant near integer -> CM point!")
        elif abs(j_imag) < 1e-5:
            print("     [~]  Real but not integer -> algebraic, not CM.")
        else:
            print("     [--] Complex j-invariant (generic point).")
        print()


if __name__ == '__main__':
    main()