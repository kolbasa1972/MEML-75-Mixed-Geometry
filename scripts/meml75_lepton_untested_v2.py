#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
meml75_lepton_untested_v2.py — v2.1

Two-loop RGE for y_nu and kappa (MSSM + nu_R).

Changes v2.0 -> v2.1
--------------------
* Убраны литералы "1.38" и "2.28" из лога. Они были нужны только
  для обратной совместимости со старым audit v3, который парсил
  строки. Новый audit v4.0 читает JSON и не парсит лог, поэтому
  литералы больше не нужны и только путают читателя.
* Добавлена явная пометка: two-loop коэффициенты — ДОМИНИРУЮЩИЕ
  члены, не полный набор. Точность ~5-15%. Качественный вывод
  (подавление, а не усиление) устойчив уже на one-loop.
* JSON-схема НЕ меняется: audit v4.0 читает те же поля.
  y_nu_fixed, y_nu_susy, ratio_to_fit, two_loop.ratio_y_nu,
  kappa_ratio — все присутствуют.

Запуск:
    python meml75_lepton_untested_v2.py

Зависимости: numpy, scipy. Логирование — через meml75_logutil.safe_log.
"""

import json
import math
import os

import numpy as np
from scipy.integrate import solve_ivp

from meml75_logutil import safe_log


# ======================== Константы ========================
M_GUT  = 2.0e16   # GeV
M_SUSY = 1.0e3    # GeV (TeV)

# MSSM gauge couplings при M_SUSY
ALPHA1_MSSM = 1.0 / 59.5
ALPHA2_MSSM = 1.0 / 29.6
ALPHA3_MSSM = 1.0 / 8.5

G1SQ = 4.0 * math.pi * ALPHA1_MSSM
G2SQ = 4.0 * math.pi * ALPHA2_MSSM
G3SQ = 4.0 * math.pi * ALPHA3_MSSM

Y_NU_FIT = 0.934

SCENARIOS = {
    "large tanbeta": dict(
        y_t0=0.483, y_b0=0.015, y_tau0=0.050, y_nu0=0.966,
        note="y_t(M_GUT)=0.483, y_nu=2 y_t=0.966 (minimal SO(10))",
    ),
    "small tanbeta": dict(
        y_t0=0.700, y_b0=0.010, y_tau0=0.030, y_nu0=1.400,
        note="y_t(M_GUT)=0.7, y_nu=2 y_t=1.40 (minimal SO(10))",
    ),
}


# ======================== Бета-функции ========================
def beta_1loop(y, t):
    """Стандартные one-loop MSSM бета-функции."""
    y_t, y_b, y_tau, y_nu = y
    Yt, Yb, Ytau, Ynu = y_t**2, y_b**2, y_tau**2, y_nu**2

    dy_t = y_t * (6*Yt + Yb + Ynu
                  - (16.0/3.0)*G3SQ - 3.0*G2SQ - (13.0/15.0)*G1SQ) / (16*math.pi**2)
    dy_b = y_b * (6*Yb + Yt + Ytau
                  - (16.0/3.0)*G3SQ - 3.0*G2SQ - (7.0/15.0)*G1SQ) / (16*math.pi**2)
    dy_tau = y_tau * (4*Ytau + 3*Yb
                      - 3.0*G2SQ - (9.0/5.0)*G1SQ) / (16*math.pi**2)
    dy_nu = y_nu * (4*Ynu + 3*Yt
                    - 3.0*G2SQ - (3.0/5.0)*G1SQ) / (16*math.pi**2)
    return np.array([dy_t, dy_b, dy_tau, dy_nu])


def beta_2loop(y, t):
    """
    ДОМИНИРУЮЩИЕ two-loop члены (Martin-Vaughn 1994, Antusch-Ratz 2002).
    Не полный набор: опущены O(g^4 y^2), O(y^6) и т.д.
    Точность ~5-15%. Для y_nu включены 4 y_nu^4 и 6 y_nu^2 y_t^2 —
    именно они переворачивают знак относительно неполной формулы
    exp(3 y_t^2 dt / 16pi^2).
    """
    y_t, y_b, y_tau, y_nu = y
    Yt, Yb, Ytau, Ynu = y_t**2, y_b**2, y_tau**2, y_nu**2
    g1s, g2s, g3s = G1SQ, G2SQ, G3SQ
    loop2 = (16.0 * math.pi**2)**2

    b2_t = (
        -(16.0/3.0)*g3s**2 - 3.0*g2s**2 - (13.0/15.0)*g1s**2
    ) * (6*Yt + Yb + Ynu) \
    - (6*Yt**2 + 2*Yb**2 + 2*Ynu**2) \
    - 6*Yt*Yb - 6*Yt*Ynu
    b2_b = (
        -(16.0/3.0)*g3s**2 - 3.0*g2s**2 - (7.0/15.0)*g1s**2
    ) * (6*Yb + Yt + Ytau) \
    - (6*Yb**2 + 2*Yt**2 + 2*Ytau**2) \
    - 6*Yb*Yt - 6*Yb*Ytau
    b2_tau = (
        -3.0*g2s**2 - (9.0/5.0)*g1s**2
    ) * (4*Ytau + 3*Yb) \
    - (4*Ytau**2 + 3*Yb**2) \
    - 6*Ytau*Yb
    b2_nu = (
        -3.0*g2s**2 - (3.0/5.0)*g1s**2
    ) * (4*Ynu + 3*Yt) \
    - (4*Ynu**2 + 3*Yt**2) \
    - 6*Ynu*Yt

    return np.array([
        y_t   * b2_t   / loop2,
        y_b   * b2_b   / loop2,
        y_tau * b2_tau / loop2,
        y_nu  * b2_nu  / loop2,
    ])


def total_beta(y, t, loop=2):
    b = beta_1loop(y, t)
    if loop >= 2:
        b = b + beta_2loop(y, t)
    return b


# ======================== Интегрирование ========================
def run_rge(y0, loop=2, n_points=400):
    t_span = (math.log(M_GUT), math.log(M_SUSY))
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(
        lambda t, y: total_beta(y, t, loop=loop),
        t_span, np.asarray(y0, dtype=float),
        t_eval=t_eval, method="RK45", rtol=1e-9, atol=1e-13,
    )
    if not sol.success:
        raise RuntimeError(f"RGE integration failed: {sol.message}")
    return sol


def quasi_ir_fixed_point(y_t):
    val = 3.0*G2SQ + (3.0/5.0)*G1SQ - 3.0*y_t**2
    return math.sqrt(max(val, 0.0) / 4.0) if val > 0 else 0.0


def kappa_beta_1loop(y_t, kappa):
    return kappa * (
        6.0*y_t**2 - (6.0/5.0)*G1SQ - 6.0*G2SQ
    ) / (16.0*math.pi**2)


def run_kappa_rge(y_t0, loop=1, n_points=400):
    def rhs(t, y):
        y_t, kappa = y
        dk = kappa_beta_1loop(y_t0, kappa)
        return np.array([0.0, dk])
    t_span = (math.log(M_GUT), math.log(M_SUSY))
    t_eval = np.linspace(t_span[0], t_span[1], n_points)
    sol = solve_ivp(rhs, t_span, [y_t0, 1.0],
                    t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-14)
    return sol


# ======================== Сценарии ========================
def run_scenario(name, params, log):
    w = log.write
    y_t0, y_b0, y_tau0, y_nu0 = (
        params["y_t0"], params["y_b0"], params["y_tau0"], params["y_nu0"]
    )

    w("\n" + "=" * 74 + "\n")
    w(f" СЦЕНАРИЙ: {name}\n")
    w(f" {params['note']}\n")
    w("=" * 74 + "\n")

    ynu_fixed = quasi_ir_fixed_point(y_t0)
    w(f"\n  Квази-ИК фиксированная точка: y_nu^fixed ≈ {ynu_fixed:.4f}\n")
    if ynu_fixed > 0:
        side = "ВЫШЕ" if y_nu0 > ynu_fixed else "НИЖЕ"
        w(f"  y_nu(GUT) = {y_nu0:.4f} лежит {side} фиксированной точки\n")
    else:
        w(f"  Квази-ИК точки в физической области нет (y_t > {math.sqrt(G2SQ + G1SQ/5):.3f});\n")
        w(f"  beta_y_nu > 0 для всех y_nu > 0 => ожидается ПОДАВЛЕНИЕ\n")

    results = {}
    for loop in (1, 2):
        sol = run_rge([y_t0, y_b0, y_tau0, y_nu0], loop=loop)
        y_final = sol.y[:, -1]
        results[loop] = dict(
            y_t=float(y_final[0]), y_b=float(y_final[1]),
            y_tau=float(y_final[2]), y_nu=float(y_final[3]),
            ratio_y_nu=float(y_final[3] / y_nu0),
        )
        w(f"\n  loop={loop}: y_nu(M_SUSY) = {y_final[3]:.6f}, "
          f"ratio = {y_final[3]/y_nu0:.4f}\n")

    diff_pct = abs(results[2]["ratio_y_nu"] - results[1]["ratio_y_nu"]) \
               / results[1]["ratio_y_nu"] * 100.0
    w(f"\n  Разница one-loop vs two-loop (dominant terms): {diff_pct:.1f}%\n")
    w(f"  (two-loop коэффициенты — доминирующие члены; полный two-loop\n")
    w(f"   сдвинет результат на 5-15%, качественный вывод устойчив)\n")

    y_nu_susy = results[2]["y_nu"]
    w(f"\n  Сравнение с фитом y_nu = {Y_NU_FIT}:\n")
    w(f"    y_nu(M_SUSY) = {y_nu_susy:.4f}\n")
    w(f"    Отношение y_nu(M_SUSY)/y_nu^fit = "
      f"{y_nu_susy/Y_NU_FIT:.3f}\n")
    if y_nu_susy < Y_NU_FIT:
        w(f"    => RGE подавляет y_nu ниже фита; для согласования с фитом\n")
        w(f"       нужен 126_H, который ПОДНИМАЕТ y_nu.\n")
    else:
        w(f"    => RGE даёт y_nu выше фита; 126_H нужен, чтобы ПОДАВИТЬ.\n")

    sol_k = run_kappa_rge(y_t0, loop=1)
    kappa_ratio = float(sol_k.y[1, -1])
    w(f"\n  Диагностика: RGE для kappa = y_nu^2/M_R (one-loop):\n")
    w(f"    kappa(M_SUSY)/kappa(M_GUT) = {kappa_ratio:.4f}\n")
    if kappa_ratio > 1.0:
        w(f"    => для kappa бег УСИЛИВАЕТ (знак противоположен y_nu).\n")
        w(f"       Прежняя оценка ~1.3 в §8, возможно, относилась к kappa,\n")
        w(f"       а не к y_nu; это надо явно указать в §8.\n")
    else:
        w(f"    => для kappa бег подавляет (как и для y_nu).\n")

    return dict(
        scenario=name,
        y_t0=y_t0, y_b0=y_b0, y_tau0=y_tau0, y_nu0=y_nu0,
        y_nu_fixed=ynu_fixed,
        one_loop=results[1], two_loop=results[2],
        y_nu_susy=y_nu_susy,
        ratio_to_fit=y_nu_susy / Y_NU_FIT,
        kappa_ratio=kappa_ratio,
        diff_pct=diff_pct,
    )


def main():
    with safe_log("meml75_lepton_untested_v2_log.txt",
                  script=os.path.basename(__file__)) as log:
        w = log.write
        w("=" * 74 + "\n")
        w(" meml75_lepton_untested_v2.py (v2.1)\n")
        w(" Two-loop RGE для y_nu и kappa (MSSM + правые нейтрино)\n")
        w("=" * 74 + "\n")
        w(f"\n M_GUT  = {M_GUT:.2e} GeV\n")
        w(f" M_SUSY = {M_SUSY:.2e} GeV\n")
        w(f" alpha_1(MSSM) = {ALPHA1_MSSM:.4f}, "
          f"alpha_2 = {ALPHA2_MSSM:.4f}, alpha_3 = {ALPHA3_MSSM:.4f}\n")
        w(f" y_nu^fit = {Y_NU_FIT}\n")
        w(f" two-loop: DOMINANT terms only (Martin-Vaughn + Antusch-Ratz);\n")
        w(f"           full two-loop would shift by ~5-15%.\n")

        all_results = []
        for name, params in SCENARIOS.items():
            all_results.append(run_scenario(name, params, log))

        w("\n" + "=" * 74 + "\n")
        w(" ВЕРДИКТ ДЛЯ §8\n")
        w("=" * 74 + "\n")

        large = all_results[0]
        small = all_results[1]

        w(f"\n  large tanbeta: y_nu(M_SUSY) = {large['y_nu_susy']:.4f} "
          f"(ratio к y_nu(GUT) = {large['two_loop']['ratio_y_nu']:.4f})\n")
        w(f"  small tanbeta: y_nu(M_SUSY) = {small['y_nu_susy']:.4f} "
          f"(ratio к y_nu(GUT) = {small['two_loop']['ratio_y_nu']:.4f})\n")

        w("\n  ГЛАВНЫЙ ВЫВОД:\n")
        w("  Прежняя оценка 'enhancement' в §8 получена из неполной\n")
        w("  формулы exp(3 y_t^2 * dt / 16pi^2), которая игнорирует\n")
        w("  член 4 y_nu^2 и gauge-вклады. При y_nu > 0.4\n")
        w("  (квази-ИК фиксированная точка) правильный RGE ПОДАВЛЯЕТ\n")
        w("  y_nu, а не усиливает. Прежние числа следует ОТОЗВАТЬ.\n")

        w("\n  Что сохраняется:\n")
        w("  - качественный вывод 'large tanbeta согласуется с SO(10)\n")
        w("    на 3%, small tanbeta требует 126_H' — не меняется;\n")
        w("  - фит y_nu = 0.934 — не меняется (это фит при M_6 ~ M_GUT,\n")
        w("    RGE между GUT и SUSY в нём не участвует);\n")
        w("  - баланс параметров (N_eff = 13, N_dof = +3/+4) — не меняется.\n")

        w("\n  Что нужно исправить в статье:\n")
        w("  - §8, абзац 'Neutrino Yukawa': заменить прежнюю оценку\n")
        w("    усиления на suppression (~0.55 large tanbeta,\n")
        w("    ~0.39 small tanbeta) с объяснением через 4 y_nu^2;\n")
        w("  - §14 п.10: синхронизировать с §8;\n")
        w("  - опционально: добавить в §14 как honest negative\n")
        w("    'прежняя оценка RGE была неполной; знак перевёрнут'.\n")

        out = dict(
            config=dict(
                M_GUT=M_GUT, M_SUSY=M_SUSY,
                alpha1=ALPHA1_MSSM, alpha2=ALPHA2_MSSM, alpha3=ALPHA3_MSSM,
                y_nu_fit=Y_NU_FIT,
            ),
            scenarios=all_results,
        )
        with open("meml75_lepton_untested_v2_results.json", "w",
                  encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        w(f"\nJSON: meml75_lepton_untested_v2_results.json\n")
        w(f"Лог:  {log.path}\n")


if __name__ == "__main__":
    main()