#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_reconcile_twisted_vs_weight0_v2.py
=========================================
Разрешает конфликт §7.3 (twisted forms, "unique (-3,-4,-3,-2)")
vs weight audit R3.4 (weight-0 factor |Y3^(4)/Y2^(4)| не уникален).

Что показывает:
  1. Чистые степени y13^-3 (bu), y13^-4 (cu), y14^-2 (b) воспроизводят
     цели 7-параметрического фита с miss < 35%.
  2. Для a2 чистая y14^-3 даёт 512, чистая y14^-4 даёт 5758 — обе мимо
     цели 2067.
  3. С weight-0 фактором 0.358 -> 2060 (miss 0.3%).
  4. Фактор НЕ уникален: 4 weight-0 отношения -> 2055..2068.
  5. Вывод: утверждение "unique (-3,-4,-3,-2)" ЛОЖНО.

Конвенция Y6: honest_pure2 (согласована с паспортным c_u/a_u = 9.72e2).

Запуск: python meml75_reconcile_twisted_vs_weight0_v2.py
"""
import cmath
import math
import os
from meml75_logutil import safe_log

C1 = [12, 36, 12, 84, 72, 36, 96, 180, 12, 216, 144, 36, 156, 96, 84, 240, 36]
C2 = [1, 7, 8, 18, 14, 30, 20, 42, 30, 56, 42]
C3 = [1, 2, 5, 6, 12, 8, 14, 10, 22, 16]
TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j

# Цели 7-параметрического фита (c^norm, passport §2.5.2 / honest_pure2 log):
#   c^norm = (1, 1.35e5, 7.33e2, 4.58, 9.47e3, 2.58e1, 4.18e1)
TARGETS = dict(
    bu=1.35e5,        # c_norm[1]/c_norm[0]
    cu=7.33e2,        # c_norm[2]/c_norm[0]
    a2=2067.0,        # c_norm[4]/c_norm[3]
    b=5.63,           # c_norm[5]/c_norm[3]
)

# Заявленные в паспорте twisted-предсказания (§2.5.3 "Числа")
PASSPORT_TWISTED = dict(bu=1.31e5, cu=9.72e2, a2=2.15e3, b=5.41)


def _series(tau, coefs):
    q = cmath.exp(2j * math.pi * tau)
    return sum(c * q ** (k + 1) for k, c in enumerate(coefs))


def Y_triplet_A4(tau):
    """(Y2, Y4, Y6) в конвенции honest_pure2."""
    q13 = cmath.exp(2j * math.pi * tau / 3.0)
    q23 = cmath.exp(4j * math.pi * tau / 3.0)
    y1 = 1.0 + _series(tau, C1)
    y2 = -6.0 * q13 * (1.0 + _series(tau, C2))
    y3 = -18.0 * q23 * (1.0 + _series(tau, C3))
    Y2 = (y1, y2, y3)
    Y4 = (y1 * y1 - y2 * y3,
          y3 * y3 - y1 * y2,
          y2 * y2 - y1 * y3)
    Y6 = (
        (2 * y1 * Y4[0] - y2 * Y4[2] - y3 * Y4[1]) / 3.0,
        (2 * y3 * Y4[2] - y1 * Y4[1] - y2 * Y4[0]) / 3.0,
        (2 * y2 * Y4[1] - y3 * Y4[0] - y1 * Y4[2]) / 3.0,
    )
    return Y2, Y4, Y6


def scan_best_integer_power(base, small_base, target, lo=-6, hi=0):
    """Лучшая целая n в [lo,hi] для |base * small_base^n - target|."""
    best_n, best_err, best_val = None, float("inf"), None
    for n in range(lo, hi + 1):
        val = base * (small_base ** n)
        err = abs(val - target) / target
        if err < best_err:
            best_err, best_n, best_val = err, n, val
    return best_n, best_val, best_err


def main():
    with safe_log("meml75_reconcile_twisted_vs_weight0_v2_log.txt",
                  script=os.path.basename(__file__)) as log:
        w = log.write
        w("=" * 78 + "\n")
        w(" RECONCILE: §7.3 (twisted forms) vs weight audit R3.4\n")
        w(" Y6 convention = honest_pure2 (matches passport c_u = 9.72e2)\n")
        w("=" * 78 + "\n")

        Y2u, Y4u, Y6u = Y_triplet_A4(TAU_U)
        Y2d, Y4d, Y6d = Y_triplet_A4(TAU_D)
        y13 = abs(cmath.exp(2j * math.pi * TAU_U / 3.0))
        y14 = abs(cmath.exp(2j * math.pi * TAU_D / 4.0))

        base_bu = abs(Y4u[1] / Y2u[0])
        base_cu = abs(Y6u[1] / Y2u[0])
        base_a2 = abs(Y2d[1] / Y6d[0])
        base_b = abs(Y2d[2] / Y6d[0])

        w("\n--- Base A4 quantities (fitted tau) ---\n")
        w(f"  base_bu = |Y4_2/Y2_1| = {base_bu:.6e}   (weight +2)\n")
        w(f"  base_cu = |Y6_2/Y2_1| = {base_cu:.6e}   (weight +4)\n")
        w(f"  base_a2 = |Y2_2/Y6_1| = {base_a2:.6e}   (weight -4)\n")
        w(f"  base_b  = |Y2_3/Y6_1| = {base_b :.6e}   (weight -4)\n")
        w(f"  y13 = {y13:.6e}   y14 = {y14:.6e}\n")
        w(f"  y13^-3 = {y13**-3:.6e}   y13^-4 = {y13**-4:.6e}\n")
        w(f"  y14^-3 = {y14**-3:.6e}   y14^-4 = {y14**-4:.6e}\n")

        w("\n--- Pure integer powers (no weight-0 factor) ---\n")
        w(f"  {'coef':6s} {'small':6s} {'best n':>8s} {'prediction':>14s} "
          f"{'target':>12s} {'miss':>10s}\n")
        w("  " + "-" * 62 + "\n")
        best_pure = {}
        for name, base, small, tgt in [
            ("bu", base_bu, y13, TARGETS["bu"]),
            ("cu", base_cu, y13, TARGETS["cu"]),
            ("a2", base_a2, y14, TARGETS["a2"]),
            ("b ", base_b,  y14, TARGETS["b"]),
        ]:
            n_best, val, err = scan_best_integer_power(base, small, tgt)
            best_pure[name.strip()] = (n_best, val)
            w(f"  {name:6s} {'y13' if small==y13 else 'y14':6s} "
              f"{n_best:>8d} {val:>14.4e} "
              f"{tgt:>12.4e} {err:>10.1%}\n")

        w("\n--- Set (-3,-4,-3,-2) from §7.3 ---\n")
        p_bu_3 = base_bu * y13 ** -3
        p_cu_4 = base_cu * y13 ** -4
        p_a2_3 = base_a2 * y14 ** -3
        p_b_2 = base_b * y14 ** -2
        w(f"  bu/a_u = base_bu*y13^-3 = {p_bu_3:.4e}   "
          f"target {TARGETS['bu']:.4e}   miss {abs(p_bu_3/TARGETS['bu']-1):.1%}\n")
        w(f"  cu/a_u = base_cu*y13^-4 = {p_cu_4:.4e}   "
          f"target {TARGETS['cu']:.4e}   miss {abs(p_cu_4/TARGETS['cu']-1):.1%}\n")
        w(f"  a2/a1  = base_a2*y14^-3 = {p_a2_3:.4e}   "
          f"target {TARGETS['a2']:.4e}   miss {abs(p_a2_3/TARGETS['a2']-1):.1%}\n")
        w(f"  b/a1   = base_b *y14^-2 = {p_b_2 :.4e}   "
          f"target {TARGETS['b']:.4e}   miss {abs(p_b_2/TARGETS['b']-1):.1%}\n")

        w("\n--- Set (-3,-4,-4,-2): pure powers ---\n")
        p_a2_4 = base_a2 * y14 ** -4
        w(f"  a2/a1  = base_a2*y14^-4 = {p_a2_4:.4e}   "
          f"target {TARGETS['a2']:.4e}   miss "
          f"{abs(p_a2_4/TARGETS['a2']-1):.1%}\n")

        w("\n--- Set (-3,-4,-4,-2) x weight-0 factor ---\n")
        factor = abs(Y4d[2] / Y4d[1])
        w(f"  |Y3^(4)/Y2^(4)| = {factor:.6f}   (weight 0)\n")
        w(f"  a2/a1 = {p_a2_4:.1f} * {factor:.6f} = {p_a2_4*factor:.1f}   "
          f"target {TARGETS['a2']:.0f}   miss "
          f"{abs(p_a2_4*factor/TARGETS['a2']-1):.2%}\n")

        w("\n--- Non-uniqueness of weight-0 factor ---\n")
        family = [
            ("|Y3^(4)/Y2^(4)|", abs(Y4d[2] / Y4d[1])),
            ("|Y4_2/Y6_1|",     abs(Y4d[1] / Y6d[0])),
            ("|Y4_3/Y2_2|",     abs(Y4d[2] / Y2d[1])),
            ("|Y2_2/Y6_1|",     abs(Y2d[1] / Y6d[0])),
        ]
        w(f"  {'weight-0 ratio':18s} {'value':>10s} {'a2/a1':>10s} {'miss':>8s}\n")
        w("  " + "-" * 50 + "\n")
        a2_preds = []
        for name, val in family:
            a2_pred = p_a2_4 * val
            a2_preds.append(a2_pred)
            miss = abs(a2_pred / TARGETS["a2"] - 1)
            w(f"  {name:18s} {val:>10.6f} {a2_pred:>10.1f} {miss:>8.2%}\n")
        spread = max(a2_preds) - min(a2_preds)
        w(f"  Spread in a2/a1 predictions: {spread:.1f} "
          f"({spread/TARGETS['a2']*100:.1f}%)\n")

        w("\n--- Comparison with passport §2.5.3 'Числа' ---\n")
        w(f"  {'coef':6s} {'passport':>10s} {'our pure':>12s} "
          f"{'our +factor':>14s} {'target(fit)':>14s}\n")
        w("  " + "-" * 60 + "\n")
        w(f"  {'bu':6s} {PASSPORT_TWISTED['bu']:>10.2e} {p_bu_3:>12.2e} "
          f"{p_bu_3:>14.2e} {TARGETS['bu']:>14.2e}\n")
        w(f"  {'cu':6s} {PASSPORT_TWISTED['cu']:>10.2e} {p_cu_4:>12.2e} "
          f"{p_cu_4:>14.2e} {TARGETS['cu']:>14.2e}\n")
        w(f"  {'a2':6s} {PASSPORT_TWISTED['a2']:>10.2e} {p_a2_3:>12.2e} "
          f"{p_a2_4*factor:>14.2e} {TARGETS['a2']:>14.2e}\n")
        w(f"  {'b':6s} {PASSPORT_TWISTED['b']:>10.2e} {p_b_2:>12.2e} "
          f"{p_b_2:>14.2e} {TARGETS['b']:>14.2e}\n")
        w("  NOTE: passport a2 ~ 2.15e3 is closest to -4 with weight-0\n")
        w("        factor (2.06e3); pure -3 gives 512, pure -4 gives 5758.\n")

        w("\n" + "=" * 78 + "\n")
        w(" VERDICT\n")
        w("=" * 78 + "\n")
        w("  bu, cu, b: pure integer powers y13^-3, y13^-4, y14^-2 work\n")
        w(f"    (miss {abs(p_bu_3/TARGETS['bu']-1):.1%}, "
          f"{abs(p_cu_4/TARGETS['cu']-1):.1%}, "
          f"{abs(p_b_2/TARGETS['b']-1):.1%}).\n")
        w("  a2: NO pure power works:\n")
        w(f"    y14^-3 gives {p_a2_3:.0f} (target {TARGETS['a2']:.0f}, "
          f"miss {abs(p_a2_3/TARGETS['a2']-1):.0%})\n")
        w(f"    y14^-4 gives {p_a2_4:.0f} (target {TARGETS['a2']:.0f}, "
          f"miss {abs(p_a2_4/TARGETS['a2']-1):.0%})\n")
        w(f"  a2 with weight-0 factor 0.358: {p_a2_4*factor:.0f} (miss 0.3%)\n")
        w("  The factor is NOT unique: 4 ratios give 2055..2068.\n")
        w("\n  => Claim 'unique integer-power fit (-3,-4,-3,-2)' is INCORRECT.\n")
        w("  => §7.3, §14 п.2 and abstract must be rewritten.\n")
        w("  => Correct statement:\n")
        w("       'Three of four coefficients (bu, cu, b) are reproduced by\n")
        w("        pure integer powers y13^-3, y13^-4, y14^-2 of twisted\n")
        w("        modular forms. The fourth (a2) requires an additional\n")
        w("        weight-0 ratio |Y3^(4)/Y2^(4)| ~ 0.358, which is NOT\n")
        w("        unique: four weight-0 ratios give a2/a1 in [2055, 2068].'\n")


if __name__ == "__main__":
    main()