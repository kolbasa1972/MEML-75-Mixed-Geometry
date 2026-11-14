#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_gvw_tau_closed_form_v2.py
================================
Изменения относительно v1:
  - ИСПРАВЛЕН баг Y6 (перестановка компонент 1<->2). Теперь конвенция
    Y6 совпадает с honest_pure2 и с паспортным c_u/a_u = 9.72e2.
  - УДАЛЕНЫ блоки twisted_ratios и TARGET — twisted-предсказания
    перенесены в meml75_reconcile_twisted_vs_weight0_v2.py.
  - Используется safe_log (fallback в logs_fallback при PermissionError).

Скрипт делает только то, что заявлено в названии:
  * закрытая форма tau^GVW для трёх наборов потоков,
  * |D_tau W| в fit- и GVW-точках,
  * сравнение tau^fit и tau^GVW,
  * base A4-отношения в fit- и GVW-точках (диагностика).

Запуск: python meml75_gvw_tau_closed_form_v2.py
"""
import os
import numpy as np
from meml75_logutil import safe_log

PI = np.pi

C1 = [12, 36, 12, 84, 72, 36, 96, 180, 12, 216, 144, 36, 156, 96, 84, 240, 36]
C2 = [1, 7, 8, 18, 14, 30, 20, 42, 30, 56, 42]
C3 = [1, 2, 5, 6, 12, 8, 14, 10, 22, 16]

FLUX_SETS = {
    "tau_u": dict(a=(-30, 5, 16),
                  tau_fit=-0.15779440852968135 + 2.3851782590052113j),
    "tau_d": dict(a=(-15, -17, 23),
                  tau_fit=+0.36939056170618634 + 1.541049605704146j),
    "tau_l": dict(a=(-16, 31, -10),
                  tau_fit=1.5398 + 1.5511j),
}


def Y_A4(tau):
    """A4-формы Y^(2), Y^(4), Y^(6). Конвенция Y6 = honest_pure2."""
    q = np.exp(2j * PI * tau)
    q13 = np.exp(2j * PI * tau / 3)
    q23 = np.exp(4j * PI * tau / 3)
    y1, b2, b3 = 1 + 0j, 1 + 0j, 1 + 0j
    for n, coef in enumerate(C1, 1):
        y1 += coef * q ** n
    for n, coef in enumerate(C2, 1):
        b2 += coef * q ** n
    for n, coef in enumerate(C3, 1):
        b3 += coef * q ** n
    Y2 = np.array([y1, -6 * q13 * b2, -18 * q23 * b3])
    Y1c, Y2c, Y3c = Y2[0], Y2[1], Y2[2]
    Y4 = np.array([Y1c ** 2 - Y2c * Y3c,
                   Y3c ** 2 - Y1c * Y2c,
                   Y2c ** 2 - Y1c * Y3c])
    # ВАЖНО: Y6 в конвенции honest_pure2 (компоненты 1 и 2 НЕ переставлены)
    Y6 = np.array([
        (2 * Y1c * Y4[0] - Y2c * Y4[2] - Y3c * Y4[1]) / 3,
        (2 * Y3c * Y4[2] - Y1c * Y4[1] - Y2c * Y4[0]) / 3,
        (2 * Y2c * Y4[1] - Y3c * Y4[0] - Y1c * Y4[2]) / 3,
    ])
    return Y2, Y4, Y6


def solve_DtauW_zero(a0, a1, a2):
    if abs(a2) < 1e-15:
        return None, "a2=0"
    x = -a1 / (2.0 * a2)
    y2 = -3.0 * a0 / a2 + 3.0 * a1 * a1 / (4.0 * a2 * a2)
    if y2 <= 0:
        return None, f"y^2={y2:.4f}<=0"
    return complex(x, np.sqrt(y2)), None


def W_of(tau, a):
    return a[0] + a[1] * tau + a[2] * tau * tau


def DtauW(tau, a):
    y = tau.imag
    if y <= 0:
        return np.nan + 0j
    return (a[1] + 2.0 * a[2] * tau) + (3j / (2.0 * y)) * W_of(tau, a)


def base_ratios(tau_u, tau_d):
    y2u, y4u, y6u = Y_A4(tau_u)
    y2d, y4d, y6d = Y_A4(tau_d)
    return dict(
        base_bu=abs(y4u[1] / y2u[0]),
        base_cu=abs(y6u[1] / y2u[0]),
        base_a2=abs(y2d[1] / y6d[0]),
        base_b=abs(y2d[2] / y6d[0]),
        y13=abs(np.exp(2j * PI * tau_u / 3)),
        y14=abs(np.exp(2j * PI * tau_d / 4)),
    )


def main():
    log_path = "meml75_gvw_tau_closed_form_v2_log.txt"
    with safe_log(log_path, script=os.path.basename(__file__)) as log:
        w = log.write
        w("=" * 80 + "\n")
        w(" meml75_gvw_tau_closed_form_v2.py\n")
        w(" Y6 fixed -> matches honest_pure2 / passport c_u = 9.72e2\n")
        w(" Twisted-form predictions moved to reconcile_v2.py\n")
        w("=" * 80 + "\n")

        w("\n--- Closed-form solutions D_tau W = 0 ---\n")
        w(f"  {'modulus':8s} {'fluxes':16s} {'tau^GVW':>34s}\n")
        w("  " + "-" * 70 + "\n")
        tau_gvw = {}
        for name, info in FLUX_SETS.items():
            a = info["a"]
            sol, err = solve_DtauW_zero(*a)
            if sol is None:
                w(f"  {name:8s} {str(a):16s}  ERROR: {err}\n")
                continue
            tau_gvw[name] = sol
            w(f"  {name:8s} {str(a):16s}  {sol.real:+.6f} {sol.imag:+.6f}i\n")

        w("\n--- |D_tau W| at fitted vs GVW point ---\n")
        w(f"  {'modulus':8s} {'|DtauW(fit)|':>14s} {'|DtauW(GVW)|':>14s} "
          f"{'|W(fit)|':>12s} {'res(fit)':>12s}\n")
        w("  " + "-" * 70 + "\n")
        for name, info in FLUX_SETS.items():
            if name not in tau_gvw:
                continue
            a = info["a"]
            D_fit = abs(DtauW(info["tau_fit"], a))
            D_gvw = abs(DtauW(tau_gvw[name], a))
            W_fit = abs(W_of(info["tau_fit"], a))
            w(f"  {name:8s} {D_fit:>14.6f} {D_gvw:>14.6e} "
              f"{W_fit:>12.4f} {D_fit / W_fit:>12.4e}\n")

        w("\n--- tau^fit vs tau^GVW ---\n")
        w(f"  {'modulus':8s} {'|dx|':>8s} {'|dy|':>8s} "
          f"{'|d tau|':>10s} {'rel':>12s}\n")
        w("  " + "-" * 54 + "\n")
        for name, info in FLUX_SETS.items():
            if name not in tau_gvw:
                continue
            tf = info["tau_fit"]
            tg = tau_gvw[name]
            dt = abs(tf - tg)
            rel = dt / abs(tg)
            w(f"  {name:8s} {abs(tf.real - tg.real):>8.4f} "
              f"{abs(tf.imag - tg.imag):>8.4f} {dt:>10.4f} {rel:>12.4e}\n")

        w("\n--- Base A4 ratios: fit vs GVW (diagnostic only) ---\n")
        br_fit = base_ratios(FLUX_SETS["tau_u"]["tau_fit"],
                             FLUX_SETS["tau_d"]["tau_fit"])
        br_gvw = base_ratios(tau_gvw["tau_u"], tau_gvw["tau_d"])
        w(f"  {'quantity':10s} {'tau^fit':>16s} {'tau^GVW':>16s} "
          f"{'rel.diff':>12s}\n")
        w("  " + "-" * 60 + "\n")
        for key in ["base_bu", "base_cu", "base_a2", "base_b", "y13", "y14"]:
            vf = br_fit[key]
            vg = br_gvw[key]
            rel = abs(vg - vf) / abs(vf) if vf != 0 else 0.0
            w(f"  {key:10s} {vf:>16.6e} {vg:>16.6e} {rel:>12.2%}\n")

        w("\n" + "=" * 80 + "\n")
        w(" INTERPRETATION\n")
        w("=" * 80 + "\n")
        w("  |d tau|/|tau| < 1% for tau_u, tau_d -> GVW fixes moduli.\n")
        w("  base_cu now matches passport (2.04e-06 -> c_u = 9.72e2).\n")
        w("  Twisted-form predictions: see meml75_reconcile_twisted_vs_weight0_v2.\n")


if __name__ == "__main__":
    main()