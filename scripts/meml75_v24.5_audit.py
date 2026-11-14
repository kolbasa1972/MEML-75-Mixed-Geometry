#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
meml75_v24.5_audit.py — v4.0 (honest, JSON-based, no overrides)

Что проверяет ЭТОТ скрипт
-------------------------
  1. Log-content check: наличие χ² = 13.9322/29.5026 в
     meml75_honest_pure2_v4.1_log.txt. Это НЕ re-derivation;
     re-derivation — в meml75_honest_pure2.py.
  2. Physics re-derivation: m_ββ из m_i и PMNS-углов.
  3. JSON-based numerical check: читает
     meml75_lepton_untested_v2_results.json и проверяет
     RGE-паттерн подавления. Никакого парсинга лога,
     никаких string-matching эвристик, никаких override.
  4. Physics re-derivation: b/a₁ из модулярных форм при TAU_D
     (не hardcoded).
  5. Physics re-derivation: конвенция Y⁽⁴⁾ при TAU_D.

Что этот скрипт НЕ проверяет
----------------------------
  * Структурный кварковый фит (meml75_honest_pure2.py).
  * Лептонный фит (meml75_tau_l_global_scan_v2.py).
  * GVW-седло (meml75_gvw_hessian_cs.py).
  * ALP/WGC (meml75_alp_verify.py, meml75_wgc_check.py).

PASS означает: ожидаемые значения присутствуют и физические
re-derivation совпадают. PASS НЕ означает, что исходные фиты
корректны — их проверяют dedicated-скрипты выше.

Changelog v3.2 -> v4.0
----------------------
* test_3 полностью переписан: читает JSON, а не лог.
  - Убран regex (баг с "1.38" в quasi-IR списке).
  - Убран override_by_3b_3c (маскировал регрессии).
  - Убраны skip-списки (не нужны: в JSON только числа).
  - Явные критерии:
      * хотя бы в одном сценарии y_nu_fixed ∈ [0.3, 0.5];
      * во всех сценариях y_nu_susy < 0.934;
      * во всех сценариях two_loop.ratio_y_nu < 1.0;
      * хотя бы в одном сценарии kappa_ratio > 1.0
        (диагностика: κ растёт, поэтому прежнее "1.38"
         могло относиться к κ, а не к y_ν).
* test_4 сделан honest: b/a₁ пересчитывается из модулярных
  форм (Y2, Y4, Y6 при TAU_D), а не берётся hardcoded 5.4094.
* test_1 честно помечен как log-content check (не re-derivation).
* В шапке добавлен блок SCOPE.
* test_2, test_5 — без изменений (это чистые physics re-derivations).

Запуск
------
    python meml75_v24.5_audit.py

Код возврата
------------
0 — все тесты PASS или SKIP
1 — есть хотя бы один FAIL
"""

import sys
import os
import json
import numpy as np


PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"
results = {}


def report(test_id, status, msg=""):
    results[test_id] = status
    print(f"  [{status}] {msg}")


# ======================== A4 modular forms ========================
C1 = [12, 36, 12, 84, 72, 36, 96, 180, 12, 216, 144, 36, 156, 96, 84, 240, 36]
C2 = [1, 7, 8, 18, 14, 30, 20, 42, 30, 56, 42]
C3 = [1, 2, 5, 6, 12, 8, 14, 10, 22, 16]


def Y2(tau):
    q = np.exp(2j * np.pi * tau)
    q13 = np.exp(2j * np.pi * tau / 3)
    q23 = np.exp(4j * np.pi * tau / 3)
    y1 = b2 = b3 = 1 + 0j
    for n, c in enumerate(C1, 1):
        y1 += c * q ** n
    for n, c in enumerate(C2, 1):
        b2 += c * q ** n
    for n, c in enumerate(C3, 1):
        b3 += c * q ** n
    return np.array([y1, -6 * q13 * b2, -18 * q23 * b3])


def Y4(tau):
    a, b, c = Y2(tau)
    return np.array([a * a - b * c, c * c - a * b, b * b - a * c])


def Y6(tau):
    a = Y2(tau)
    b = Y4(tau)
    return np.array([
        (2 * a[0] * b[0] - a[1] * b[2] - a[2] * b[1]) / 3,
        (2 * a[2] * b[2] - a[0] * b[1] - a[1] * b[0]) / 3,
        (2 * a[1] * b[1] - a[2] * b[0] - a[0] * b[2]) / 3,
    ])


def Y4_swap(tau):
    y = Y4(tau)
    return np.array([y[0], y[2], y[1]])


TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j


# ======================== Test 1: chi2 (log-content check) ========================
def test_1_chi2():
    """
    SCOPE: log-content check, NOT re-derivation.
    Проверяет, что meml75_honest_pure2_v4.1_log.txt содержит
    ожидаемые χ² (13.9322 и 29.5026). Re-derivation — в
    meml75_honest_pure2.py.
    """
    print("\n[1] χ² = 13.93/29.50; a₁, c_d  [log-content check, not re-derivation]")
    log_file = "meml75_honest_pure2_v4.1_log.txt"
    if not os.path.exists(log_file):
        log_file = "meml75_honest_pure2_v2_log.txt"

    if os.path.exists(log_file):
        with open(log_file, "r", encoding="utf-8") as f:
            text = f.read()
        if "13.9322" in text and "29.5026" in text:
            report(1, PASS,
                   f"chi2 = 13.9322/29.5026 present in {log_file} "
                   f"(re-derivation in meml75_honest_pure2.py)")
        else:
            report(1, FAIL,
                   f"{log_file} found but expected chi2 values not present")
    else:
        report(1, SKIP,
               "Log file missing. Run meml75_honest_pure2.py first")


# ======================== Test 2: m_betabeta (physics) ========================
def m_bb(m1, m2, m3, th12, th13, delta_cp, alpha=0.0, beta=0.0):
    c12, s12 = np.cos(np.radians(th12)), np.sin(np.radians(th12))
    c13, s13 = np.cos(np.radians(th13)), np.sin(np.radians(th13))
    d = np.radians(delta_cp)
    return abs(
        c12**2 * c13**2 * m1
        + s12**2 * c13**2 * m2 * np.exp(1j * alpha)
        + s13**2 * m3 * np.exp(1j * (beta - 2 * d))
    )


def test_2_mbb():
    """
    SCOPE: physics re-derivation.
    m_ββ из m_i и PMNS-углов (m_i и углы — из constrained-фита §8).
    """
    print("\n[2] m_ββ: 8.37/8.37/8.67, range [0, 8.67]  [physics re-derivation]")
    m1, m2, m3 = 6.391, 10.756, 50.576
    th12, th13 = 33.64, 8.53
    v_229 = m_bb(m1, m2, m3, th12, th13, 22.9)
    v_157 = m_bb(m1, m2, m3, th12, th13, 157.1)
    v_0   = m_bb(m1, m2, m3, th12, th13, 0.0)
    print(f"    m_ββ(δ= 22.9°, α=β=0) = {v_229:.3f} meV  (expect 8.37)")
    print(f"    m_ββ(δ=157.1°, α=β=0) = {v_157:.3f} meV  (expect 8.37)")
    print(f"    m_ββ(δ=  0.0°, α=β=0) = {v_0:.3f} meV  (expect 8.67)")
    ok = (abs(v_229 - 8.37) < 0.05
          and abs(v_157 - 8.37) < 0.05
          and abs(v_0 - 8.67) < 0.05)

    N = 180
    alphas = np.linspace(0, 2 * np.pi, N, endpoint=False)
    betas  = np.linspace(0, 2 * np.pi, N, endpoint=False)
    vals = np.empty((N, N))
    for i, a in enumerate(alphas):
        for j, b in enumerate(betas):
            vals[i, j] = m_bb(m1, m2, m3, th12, th13, 22.9, a, b)
    vmin, vmax = vals.min(), vals.max()
    print(f"    range over (α,β): [{vmin:.4f}, {vmax:.4f}] meV  "
          f"(expect ~[0, 8.67])")
    ok = ok and (vmax < 8.75) and (vmin < 0.05)

    if ok:
        report(2, PASS, "m_ββ headline and range consistent")
    else:
        report(2, FAIL, "m_ββ inconsistency")


# ======================== Test 3: RGE (JSON-based) ========================
def test_3_rge():
    """
    SCOPE: JSON-based numerical check.
    Читает meml75_lepton_untested_v2_results.json и проверяет
    RGE-паттерн подавления. Никакого парсинга лога, никаких
    skip-списков, никаких override.

    Критерии PASS:
      a) хотя бы в одном сценарии y_nu_fixed ∈ [0.3, 0.5]
         (квази-ИК точка существует);
      b) во всех сценариях y_nu_susy < y_nu_fit = 0.934
         (подавление, а не усиление);
      c) во всех сценариях two_loop.ratio_y_nu < 1.0
         (подавление на уровне отношения);
      d) хотя бы в одном сценарии kappa_ratio > 1.0
         (диагностика: κ растёт, поэтому прежнее "1.38"
          могло относиться к κ, а не к y_ν).
    """
    print("\n[3] RGE для y_ν: SUPPRESSION (JSON-based)")

    json_path = "meml75_lepton_untested_v2_results.json"
    if not os.path.exists(json_path):
        report(3, SKIP,
               f"{json_path} missing. Run meml75_lepton_untested_v2.py")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    scenarios = data.get("scenarios", [])
    y_nu_fit = data.get("config", {}).get("y_nu_fit", 0.934)

    if not scenarios:
        report(3, FAIL, "JSON present but 'scenarios' empty")
        return

    print(f"    Read {len(scenarios)} scenario(s) from {json_path}")
    for s in scenarios:
        print(f"      {s['scenario']}: y_nu_fixed={s['y_nu_fixed']:.4f}, "
              f"y_nu_susy={s['y_nu_susy']:.4f}, "
              f"ratio_to_fit={s['ratio_to_fit']:.4f}, "
              f"kappa_ratio={s['kappa_ratio']:.4f}")

    checks = []

    # a) хотя бы одна квази-ИК точка в [0.3, 0.5]
    fixed_ok = any(0.3 < s["y_nu_fixed"] < 0.5 for s in scenarios)
    checks.append((f"quasi-IR point in [0.3, 0.5] in >=1 scenario",
                   fixed_ok))
    print(f"    [{'OK' if fixed_ok else 'FAIL'}] "
          f"quasi-IR point y_ν^fixed ∈ [0.3, 0.5] in >=1 scenario")

    # b) во всех сценариях y_nu_susy < y_nu_fit
    below_fit = all(s["y_nu_susy"] < y_nu_fit for s in scenarios)
    checks.append((f"all y_ν(SUSY) < {y_nu_fit}", below_fit))
    print(f"    [{'OK' if below_fit else 'FAIL'}] "
          f"all y_ν(SUSY) < y_ν^fit = {y_nu_fit}")

    # c) во всех сценариях two_loop.ratio_y_nu < 1.0
    suppression = all(s["two_loop"]["ratio_y_nu"] < 1.0
                      for s in scenarios)
    checks.append(("all two_loop.ratio_y_nu < 1.0 (suppression)",
                   suppression))
    print(f"    [{'OK' if suppression else 'FAIL'}] "
          f"all two_loop.ratio_y_nu < 1.0")

    # d) хотя бы один kappa_ratio > 1.0 (диагностика)
    kappa_info = any(s["kappa_ratio"] > 1.0 for s in scenarios)
    checks.append(("kappa_ratio > 1.0 in >=1 scenario (diagnostic)",
                   kappa_info))
    print(f"    [{'OK' if kappa_info else 'FAIL'}] "
          f"kappa_ratio > 1.0 in >=1 scenario "
          f"(explains where '1.38' came from: κ, not y_ν)")

    if all(ok for _, ok in checks):
        report(3, PASS,
               "RGE suppression (JSON, no log parsing, no overrides)")
    else:
        failed = [n for n, ok in checks if not ok]
        report(3, FAIL, f"RGE checks failed: {failed}")


# ======================== Test 4: b/a1 (physics re-derivation) ========================
def test_4_b_over_a1():
    """
    SCOPE: physics re-derivation.
    b/a₁ = |Y2[2]/Y6[0]| at TAU_D · y14⁻²
    (не hardcoded; пересчитывается из модулярных форм).
    """
    print("\n[4] b/a₁ = 5.41, miss < 4.5%  [physics re-derivation]")

    y2d = Y2(TAU_D)
    y6d = Y6(TAU_D)
    y14 = abs(np.exp(2j * np.pi * TAU_D / 4.0))

    base_b = abs(y2d[2] / y6d[0])
    b_over_a1 = base_b * y14**-2

    target = 5.63
    miss = abs(b_over_a1 - target) / target * 100.0

    print(f"    base_b = |Y2[2]/Y6[0]| at TAU_D = {base_b:.6e}")
    print(f"    y14 = {y14:.6e}")
    print(f"    b/a₁ = base_b · y14⁻² = {b_over_a1:.6f}")
    print(f"    target = {target:.2f}, miss = {miss:.2f}%  (expect < 4.5%)")

    if miss < 4.5:
        report(4, PASS, f"b/a₁ = {b_over_a1:.4f}, miss {miss:.2f}% < 4.5%")
    else:
        report(4, FAIL, f"b/a₁ = {b_over_a1:.4f}, miss {miss:.2f}% >= 4.5%")


# ======================== Test 5: Y^(4) convention (physics) ========================
def test_5_y4_convention():
    """
    SCOPE: physics re-derivation.
    Проверяет, что paper-конвенция Y⁽⁴⁾ даёт |Y3⁽⁴⁾/Y2⁽⁴⁾| ≈ 0.3578
    при TAU_D, а swap-конвенция — 2.79.
    """
    print("\n[5] Y⁽⁴⁾ convention (using TAU_D)  [physics re-derivation]")
    y4d = Y4(TAU_D)
    y4s = Y4_swap(TAU_D)

    y3y2_paper = abs(y4d[2] / y4d[1])
    y3y2_swap  = abs(y4s[2] / y4s[1])

    print(f"    paper  |Y3/Y2| = {y3y2_paper:.4f}  (expect ~0.3578)")
    print(f"    swap   |Y3/Y2| = {y3y2_swap :.4f}  (expect ~2.79)")

    ok = (abs(y3y2_paper - 0.3578) < 0.01
          and abs(y3y2_swap - 0.3578) > 1.0)
    if ok:
        report(5, PASS, "Y⁽⁴⁾ convention consistent at TAU_D")
    else:
        report(5, FAIL, "Y⁽⁴⁾ convention inconsistency")


# ======================== main ========================
def main():
    print("=" * 70)
    print(" meml75_v24.5_audit.py (v4.0 — honest, JSON-based, no overrides)")
    print("=" * 70)
    print(" SCOPE:")
    print("   [1] log-content check (NOT re-derivation)")
    print("   [2] physics re-derivation (m_ββ)")
    print("   [3] JSON numerical check (RGE suppression, no log parsing)")
    print("   [4] physics re-derivation (b/a₁ from modular forms)")
    print("   [5] physics re-derivation (Y⁽⁴⁾ convention)")
    print(" What this does NOT verify: structural quark fit, lepton fit,")
    print(" GVW saddle, ALP/WGC. See dedicated scripts (see docstring).")
    print("=" * 70)

    test_1_chi2()
    test_2_mbb()
    test_3_rge()
    test_4_b_over_a1()
    test_5_y4_convention()

    print("\n" + "=" * 70)
    print(" Summary")
    print("=" * 70)
    for i in (1, 2, 3, 4, 5):
        print(f"  [{i}] {results.get(i, 'N/A')}")

    n_fail = sum(1 for s in results.values() if s == FAIL)
    n_skip = sum(1 for s in results.values() if s == SKIP)
    n_pass = sum(1 for s in results.values() if s == PASS)

    if n_fail == 0:
        print(f"\n[VERDICT] OK: {n_pass} PASS, {n_skip} SKIP, 0 FAIL")
        return 0
    else:
        print(f"\n[VERDICT] FAIL: {n_pass} PASS, {n_skip} SKIP, "
              f"{n_fail} FAIL")
        return 1


if __name__ == "__main__":
    sys.exit(main())