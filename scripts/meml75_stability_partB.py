#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_stability_partB.py — проверка стабильности 6 кандидатов из Part B.

Part B нашёл 6 комбинаций (Y_a/Y_b)*(Y_c/Y_d)*(Y_e/Y_f)*y14^n
с miss < 2% для c_d/a_1 = 9.102. Этот скрипт проверяет:
стабильны ли они при вариации tau_d?

Критерий вывода (а не совпадения): miss остаётся < 2% при dtau = ±1%.
Если miss растёт до > 5% при dtau = ±1% — кандидат нестабилен.

Запуск: python3 meml75_stability_partB.py
"""

import cmath
import math

# ======================== A4 forms ========================
C1 = [12, 36, 12, 84, 72, 36, 96, 180, 12, 216, 144, 36, 156, 96, 84, 240, 36]
C2 = [1, 7, 8, 18, 14, 30, 20, 42, 30, 56, 42]
C3 = [1, 2, 5, 6, 12, 8, 14, 10, 22, 16]
TAU_D = 0.36939056170618634 + 1.541049605704146j
TARGET_CD = 9.1020


def series(tau, coefs):
    q = cmath.exp(2j * math.pi * tau)
    return sum(c * q ** (k + 1) for k, c in enumerate(coefs))


def Y2(tau):
    q13 = cmath.exp(2j * math.pi * tau / 3)
    q23 = cmath.exp(4j * math.pi * tau / 3)
    y1 = 1.0 + series(tau, C1)
    y2 = -6.0 * q13 * (1.0 + series(tau, C2))
    y3 = -18.0 * q23 * (1.0 + series(tau, C3))
    return (y1, y2, y3)


def Y4(tau):
    a, b, c = Y2(tau)
    return (a * a - b * c, c * c - a * b, b * b - a * c)


def Y6(tau):
    a = Y2(tau)
    b = Y4(tau)
    return (
        (2 * a[0] * b[0] - a[1] * b[2] - a[2] * b[1]) / 3,
        (2 * a[2] * b[2] - a[0] * b[1] - a[1] * b[0]) / 3,
        (2 * a[1] * b[1] - a[2] * b[0] - a[0] * b[2]) / 3,
    )


def y14_at(tau):
    return abs(cmath.exp(2j * math.pi * tau / 4))


def miss_pct(pred, target):
    return abs(pred - target) / target * 100.0


def get_form(name, Y2v, Y4v, Y6v):
    """name = '2_2' -> Y2v[1], '6_3' -> Y6v[2], и т.д."""
    weight = int(name.split('_')[0])
    idx = int(name.split('_')[1]) - 1
    Y = {2: Y2v, 4: Y4v, 6: Y6v}[weight]
    return Y[idx]


def evaluate_candidate(candidate, tau):
    """candidate = {'ratios': [(num, den), ...], 'n': int}"""
    Y2v = Y2(tau)
    Y4v = Y4(tau)
    Y6v = Y6(tau)
    y14 = y14_at(tau)

    pred = 1.0
    for (num_name, den_name) in candidate['ratios']:
        num_val = get_form(num_name, Y2v, Y4v, Y6v)
        den_val = get_form(den_name, Y2v, Y4v, Y6v)
        if abs(den_val) < 1e-30:
            return float('nan')
        pred *= abs(num_val / den_val)
    pred *= y14 ** candidate['n']
    return pred


# ======================== 6 кандидатов из Part B ========================
CANDIDATES = [
    {
        'name': '2_2/6_1 * 4_3/6_3 * 4_3/6_3',
        'ratios': [('2_2', '6_1'), ('4_3', '6_3'), ('4_3', '6_3')],
        'n': -1,
        'base_miss': 0.415,
    },
    {
        'name': '2_2/6_3 * 4_3/6_1 * 4_3/6_3',
        'ratios': [('2_2', '6_3'), ('4_3', '6_1'), ('4_3', '6_3')],
        'n': -1,
        'base_miss': 0.415,
    },
    {
        'name': '2_2/6_1 * 2_1/6_1 * 4_3/6_3',
        'ratios': [('2_2', '6_1'), ('2_1', '6_1'), ('4_3', '6_3')],
        'n': -1,
        'base_miss': 0.713,
    },
    {
        'name': '2_2/6_1 * 2_1/6_3 * 4_3/6_1',
        'ratios': [('2_2', '6_1'), ('2_1', '6_3'), ('4_3', '6_1')],
        'n': -1,
        'base_miss': 0.713,
    },
    {
        'name': '2_2/6_3 * 2_1/6_1 * 4_3/6_1',
        'ratios': [('2_2', '6_3'), ('2_1', '6_1'), ('4_3', '6_1')],
        'n': -1,
        'base_miss': 0.713,
    },
    {
        'name': '2_2/6_1 * 2_1/6_1 * 2_1/6_1',
        'ratios': [('2_2', '6_1'), ('2_1', '6_1'), ('2_1', '6_1')],
        'n': -1,
        'base_miss': 1.012,
    },
]


def main():
    print("=" * 78)
    print(" meml75_stability_partB.py")
    print(" Проверка стабильности 6 кандидатов из Part B (A4, 3 twisted ratio)")
    print("=" * 78)
    print(f" tau_d = {TAU_D}")
    print(f" TARGET c_d/a_1 = {TARGET_CD}")
    print(f" Критерий: вывод, если miss < 2% сохраняется при dtau = ±1%")
    print()

    variations = [0.0, 0.005, -0.005, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05]

    # Сначала проверяем базовую точку (d=0)
    print("-" * 78)
    print(" Проверка базовой точки (d=0) — воспроизводим ли мы Part B?")
    print("-" * 78)
    for i, cand in enumerate(CANDIDATES, 1):
        pred = evaluate_candidate(cand, TAU_D)
        miss = miss_pct(pred, TARGET_CD)
        match = "OK" if abs(miss - cand['base_miss']) < 0.05 else "MISMATCH"
        print(f"  [{i}] {cand['name']}")
        print(f"       pred = {pred:.4f}, miss = {miss:.3f}% "
              f"(Part B: {cand['base_miss']:.3f}%)  [{match}]")
    print()

    # Затем стабильность
    print("-" * 78)
    print(" Стабильность при вариации tau_d")
    print("-" * 78)

    results = []
    for i, cand in enumerate(CANDIDATES, 1):
        print(f"\n [{i}] {cand['name']}")
        print(f"     {'d_tau':>8} {'Re':>10} {'Im':>10} {'pred':>10} {'miss':>8}")
        print("     " + "-" * 52)

        misses = []
        for d in variations:
            tau_var = TAU_D * (1 + d)
            pred = evaluate_candidate(cand, tau_var)
            miss = miss_pct(pred, TARGET_CD)
            misses.append(miss)
            print(f"     {d*100:+7.1f}% {tau_var.real:10.4f} "
                  f"{tau_var.imag:10.4f} {pred:10.4f} {miss:7.3f}%")

        # Критерий: miss при ±1% и ±2%
        misses_arr = dict(zip(variations, misses))
        m_pos1 = misses_arr[0.01]
        m_neg1 = misses_arr[-0.01]
        m_pos2 = misses_arr[0.02]
        m_neg2 = misses_arr[-0.02]
        m_max = max(m_pos1, m_neg1)

        # Стабилен, если при ±1% miss < 2%
        stable_1pct = (m_pos1 < 2.0 and m_neg1 < 2.0)
        stable_2pct = (m_pos2 < 5.0 and m_neg2 < 5.0)

        verdict = "STABLE" if stable_1pct else ("MARGINAL" if stable_2pct else "UNSTABLE")
        results.append({
            'name': cand['name'],
            'base': cand['base_miss'],
            'm_pos1': m_pos1, 'm_neg1': m_neg1,
            'm_pos2': m_pos2, 'm_neg2': m_neg2,
            'max_at_1pct': m_max,
            'verdict': verdict,
        })
        print(f"     -> при dtau=±1%: miss = {m_pos1:.2f}% / {m_neg1:.2f}%, "
              f"max = {m_max:.2f}%")
        print(f"     -> при dtau=±2%: miss = {m_pos2:.2f}% / {m_neg2:.2f}%")
        print(f"     -> ВЕРДИКТ: {verdict}")

    # Итог
    print("\n" + "=" * 78)
    print(" ИТОГ")
    print("=" * 78)
    print(f"\n {'#':>2} {'base miss':>10} {'miss@+1%':>10} "
          f"{'miss@-1%':>10} {'max@1%':>8} {'verdict':>10}")
    print(" " + "-" * 60)
    for i, r in enumerate(results, 1):
        print(f" {i:>2d} {r['base']:>9.3f}% {r['m_pos1']:>9.3f}% "
              f"{r['m_neg1']:>9.3f}% {r['max_at_1pct']:>7.3f}% "
              f"{r['verdict']:>10}")

    n_stable = sum(1 for r in results if r['verdict'] == 'STABLE')
    n_marginal = sum(1 for r in results if r['verdict'] == 'MARGINAL')
    n_unstable = sum(1 for r in results if r['verdict'] == 'UNSTABLE')

    print(f"\n STABLE: {n_stable}, MARGINAL: {n_marginal}, UNSTABLE: {n_unstable}")

    print("\n" + "-" * 78)
    print(" ВЕРДИКТ")
    print("-" * 78)
    if n_stable == 0:
        print(" НИ ОДИН кандидат не стабилен при dtau = ±1%.")
        print(" => Числовые совпадения, не выводы.")
        print(" => c_d/a_1 остаётся свободным. Вопрос закрыт.")
    elif n_stable == 1:
        stable_r = [r for r in results if r['verdict'] == 'STABLE'][0]
        print(f" ОДИН стабильный кандидат: {stable_r['name']}")
        print(f" miss при dtau=±1%: {stable_r['m_pos1']:.2f}% / "
              f"{stable_r['m_neg1']:.2f}%")
        print(f" Требуется дальнейшая проверка (struct motivation, weight).")
    else:
        print(f" {n_stable} стабильных кандидатов.")
        print(f" Много кандидатов = нумерология. Проверять по одному.")
    print("=" * 78)


if __name__ == "__main__":
    main()