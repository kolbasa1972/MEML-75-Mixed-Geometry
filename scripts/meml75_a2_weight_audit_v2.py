#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_a2_weight_audit_v2.py

Weight audit фактора |Y3^(4)/Y2^(4)|.

Изменения относительно v1:
  - Y6_triplet приведён к конвенции honest_pure2 (для консистентности
    с gvw_v2 и honest_pure2). На численные выводы не влияет, т.к.
    в аудите используется только Y6[0] — он инвариантен к перестановке 1<->2.
  - safe_log (fallback), первая строка лога = имя скрипта.

Показывает:
  * |Y3^(4)/Y2^(4)| = 0.358, модулярный вес 0 — не меняет вес.
  * Тот же фактор даётся четырьмя weight-0 отношениями — НЕ уникален.
  * y14^-4 = exp(2*pi*Im tau) — не кэлерова степень (best real n = 8.60).
  * Total weight не меняется от base_a2 (вес -4).

Запуск: python meml75_a2_weight_audit_v2.py
"""
from __future__ import annotations

import cmath
import math
import os
import sys

from meml75_logutil import safe_log

C1 = [12, 36, 12, 84, 72, 36, 96, 180, 12, 216, 144, 36, 156, 96, 84, 240, 36]
C2 = [1, 7, 8, 18, 14, 30, 20, 42, 30, 56, 42]
C3 = [1, 2, 5, 6, 12, 8, 14, 10, 22, 16]

TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j


def _series(tau: complex, coefs: list[int]) -> complex:
    q = cmath.exp(2j * math.pi * tau)
    return sum(c * q ** (k + 1) for k, c in enumerate(coefs))


def Y1(tau: complex) -> complex:
    return 1.0 + _series(tau, C1)


def Y2(tau: complex) -> complex:
    q13 = cmath.exp(2j * math.pi * tau / 3.0)
    return -6.0 * q13 * (1.0 + _series(tau, C2))


def Y3(tau: complex) -> complex:
    q23 = cmath.exp(4j * math.pi * tau / 3.0)
    return -18.0 * q23 * (1.0 + _series(tau, C3))


def Y2_triplet(tau: complex) -> tuple[complex, complex, complex]:
    return (Y1(tau), Y2(tau), Y3(tau))


def Y4_triplet(tau: complex) -> tuple[complex, complex, complex]:
    y1, y2, y3 = Y2_triplet(tau)
    return (y1 * y1 - y2 * y3,
            y3 * y3 - y1 * y2,
            y2 * y2 - y1 * y3)


def Y6_triplet(tau: complex) -> tuple[complex, complex, complex]:
    """Y6 в конвенции honest_pure2."""
    y1, y2, y3 = Y2_triplet(tau)
    Y4 = Y4_triplet(tau)
    return (
        (2 * y1 * Y4[0] - y2 * Y4[2] - y3 * Y4[1]) / 3.0,
        (2 * y3 * Y4[2] - y1 * Y4[1] - y2 * Y4[0]) / 3.0,
        (2 * y2 * Y4[1] - y3 * Y4[0] - y1 * Y4[2]) / 3.0,
    )


def abs_ratio(num: complex, den: complex) -> float:
    return abs(num) / abs(den)


def best_real_power(base: float, target: float,
                    lo: float = -20.0, hi: float = 20.0) -> float:
    """n, минимизирующее |base**n - target| по вещественному n."""
    best_n, best_err = lo, float("inf")
    steps = 40000
    for i in range(steps + 1):
        n = lo + (hi - lo) * i / steps
        try:
            err = abs(base ** n - target)
        except (OverflowError, ZeroDivisionError):
            continue
        if err < best_err:
            best_err, best_n = err, n
    a, b = best_n - (hi - lo) / steps, best_n + (hi - lo) / steps
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    for _ in range(200):
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        if abs(base ** c - target) < abs(base ** d - target):
            b = d
        else:
            a = c
    return 0.5 * (a + b)


def main() -> int:
    with safe_log("meml75_a2_weight_audit_v2_log.txt",
                  script=os.path.basename(__file__)) as log:
        w = log.write

        w("\n=== WEIGHT AUDIT: factor |Y3^(4)/Y2^(4)| ===\n")
        w(" Y6 convention: honest_pure2 (Y6[0] is same in both conventions,\n")
        w("  so numerical results of the audit are unchanged from v1)\n\n")

        y4d = Y4_triplet(TAU_D)
        y2d = Y2_triplet(TAU_D)
        y6d = Y6_triplet(TAU_D)

        base_a2 = abs_ratio(y2d[1], y6d[0])
        w(f"\nbase_a2 = |Y2[1]/Y6[0]| = {base_a2:.6e}   (passport 3.590602e-01)\n")
        w("  modular weight: 2 - 6 = -4\n")

        y14 = abs(cmath.exp(2j * math.pi * TAU_D)) ** 0.25
        y14_m4 = y14 ** -4
        w(f"\ny14      = |q_d|^(1/4)      = {y14:.6e}   (passport 8.886161e-02)\n")
        w(f"y14^-4   = exp(2*pi*Im tau)  = {y14_m4:.6e}   (passport 1.6038e+04)\n")
        w("  modular weight: NON-MODULAR\n")

        two_im = 2.0 * TAU_D.imag
        n_best = best_real_power(two_im, y14_m4)
        w(f"  best real n for (2*Im tau)^n = {n_best:.3f}   (passport 8.60)\n")
        w(f"  integer? {'YES' if abs(n_best - round(n_best)) < 1e-3 else 'NO'}\n")

        factor = abs_ratio(y4d[2], y4d[1])
        w(f"\n|Y3^(4)/Y2^(4)| = {factor:.6f}   (passport 0.357816)\n")
        w("  modular weight: 4 - 4 = 0\n")

        cand = base_a2 * y14_m4 * factor
        w(f"\ncandidate = base_a2 * y14^-4 * factor = {cand:.1f}   "
          f"(passport 2060.5)\n")

        family = [
            ("|Y3^(4)/Y2^(4)|", y4d[2], y4d[1], factor, 2060.5),
            ("|Y4_2/Y6_1|",     y4d[1], y6d[0], None, 2061.9),
            ("|Y4_3/Y2_2|",     y4d[2], y2d[1], None, 2054.7),
            ("|Y2_2/Y6_1|",     y2d[1], y6d[0], None, 2067.65),
        ]
        w("\n=== FAMILY 0.358: four weight-0 ratios ===\n")
        w(f"{'ratio':<18}{'value':>12}{'base*factor':>16}{'passport':>12}\n")
        for name, num, den, pre, ref in family:
            val = pre if pre is not None else abs_ratio(num, den)
            proj = base_a2 * y14_m4 * val
            w(f"{name:<18}{val:>12.6f}{proj:>16.1f}{ref:>12.1f}\n")

        delta = 0.01
        f2 = abs_ratio(Y4_triplet(TAU_D + delta)[2],
                       Y4_triplet(TAU_D + delta)[1])
        sens = abs(f2 - factor) / factor * 100.0
        w(f"\nsensitivity to dtau = {delta}: {sens:.4f} %   (passport 0.02%)\n")

        w("\n=== TOTAL WEIGHT ===\n")
        w("  base_a2            : -4\n")
        w("  y14^-4             : non-modular\n")
        w("  |Y3^(4)/Y2^(4)|    :  0\n")
        w("  TOTAL              : -4  (does not change from base_a2)\n")
        w("\nverdict: structural motivation ABSENT — weight 0, factor not unique,\n")
        w("         y14^-4 is not a Kaehler power (best real n = %.2f).\n" % n_best)

        w(f"\nlog written to: {log.path}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())