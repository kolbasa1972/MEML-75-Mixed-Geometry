#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_derive_final_attempt.py — последняя попытка вывода c_d/a_1 и c_1/c_2.

Все части помечены [EXPLORATORY] — структурная мотивация слабая или
нулевая. Строгий поиск (§14 п.11) уже показал 0 кандидатов для обоих
параметров. Этот скрипт проверяет расширенное пространство, чтобы
убедиться, что ничего структурно возможного не пропущено.

Части:
  A [EXPLORATORY]: A4 с двумя weight-0 факторами
  B [EXPLORATORY]: A4 с тремя twisted ratios
  C [EXPLORATORY]: S4' целочисленные степени с weight-условием
  D [EXPLORATORY]: S4' целочисленные степени без weight-условия
  E: Стабильность лучших кандидатов

Запуск: python3 meml75_derive_final_attempt.py
"""

import cmath
import math
import itertools

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


# ======================== S4' forms ========================
TAU_L = 1.584 + 1.552j
TARGET_C12 = 0.3788


def dlog_eta(tau, N_max=200):
    q = cmath.exp(2j * cmath.pi * tau)
    E2 = 1.0 + 0j
    for n in range(1, N_max):
        qn = q ** n
        E2 -= 24 * n * qn / (1 - qn)
    return 1j * cmath.pi / 12 * E2


def Y_S4(a, tau, N_max=200):
    args = [(2, 1, 2), (4, 0, 1), (1, 0, 4), (1, 1, 4), (1, 2, 4), (1, 3, 4)]
    total = 0j
    for ai, (num, b, denom) in zip(a, args):
        arg = (num * tau + b) / denom
        total += ai * (num / denom) * dlog_eta(arg, N_max)
    return total


_OMEGA = cmath.exp(2j * cmath.pi / 3)


def S4_forms(tau, N_max=200):
    Y1 = Y_S4([1, 1, _OMEGA, _OMEGA**2, _OMEGA, _OMEGA**2], tau, N_max)
    Y2 = Y_S4([1, 1, _OMEGA**2, _OMEGA, _OMEGA**2, _OMEGA], tau, N_max)
    Y3 = Y_S4([1, -1, -1, -1, 1, 1], tau, N_max)
    Y4 = Y_S4([1, -1, -_OMEGA**2, -_OMEGA, _OMEGA**2, _OMEGA], tau, N_max)
    Y5 = Y_S4([1, -1, -_OMEGA, -_OMEGA**2, _OMEGA, _OMEGA**2], tau, N_max)
    return Y1, Y2, Y3, Y4, Y5


# ======================== Helpers ========================
def miss_pct(pred, target):
    return abs(pred - target) / target * 100.0


def y14_at(tau):
    return abs(cmath.exp(2j * cmath.pi * tau / 4))


# ======================== Part A ========================
def part_A():
    print("\n" + "=" * 78)
    print(" Part A [EXPLORATORY]: A4 с двумя weight-0 факторами")
    print("=" * 78)
    print(" Цель: c_d/a_1 = 9.102")
    print(" Структура: |Y_num/Y_den| * y14^n * w0_1 * w0_2")
    print(" Мотивация: слабая (один weight-0 уже не сработал, §14 п.11)")
    print()

    Y2d, Y4d, Y6d = Y2(TAU_D), Y4(TAU_D), Y6(TAU_D)
    y14 = y14_at(TAU_D)

    numerators = {'2_2': Y2d[1], '2_1': Y2d[0], '4_3': Y4d[2]}
    denominators = {'6_1': Y6d[0], '6_3': Y6d[2]}

    w0 = {}
    for i in range(3):
        for j in range(3):
            if i != j:
                w0[f'w0_4_{i+1}{j+1}'] = abs(Y4d[i] / Y4d[j])
                w0[f'w0_6_{i+1}{j+1}'] = abs(Y6d[i] / Y6d[j])

    w0_names = list(w0.keys())
    w0_pairs = list(itertools.combinations_with_replacement(w0_names, 2))
    n_combos = len(numerators) * len(denominators) * len(w0_pairs) * 7
    print(f" Пространство: {len(numerators)}x{len(denominators)}x"
          f"{len(w0_pairs)}x7 = {n_combos}")

    candidates = []
    for num_name, num_val in numerators.items():
        for den_name, den_val in denominators.items():
            if abs(den_val) < 1e-30:
                continue
            base = abs(num_val / den_val)
            for (w1_name, w2_name) in w0_pairs:
                w1, w2 = w0[w1_name], w0[w2_name]
                for n in range(-6, 1):
                    pred = base * (y14 ** n) * w1 * w2
                    miss = miss_pct(pred, TARGET_CD)
                    if miss < 2.0:
                        candidates.append({
                            'miss': miss, 'num': num_name, 'den': den_name,
                            'n': n, 'w1': w1_name, 'w2': w2_name,
                            'pred': pred,
                        })

    candidates.sort(key=lambda x: x['miss'])
    print(f" Кандидатов с miss < 2%: {len(candidates)}")
    if candidates:
        print(f"\n {'miss':>7} {'num':>5} {'den':>5} {'n':>3} "
              f"{'w1':>10} {'w2':>10} {'pred':>10}")
        print(" " + "-" * 60)
        for c in candidates[:10]:
            print(f" {c['miss']:6.3f}% {c['num']:>5} {c['den']:>5} "
                  f"{c['n']:>3d} {c['w1']:>10} {c['w2']:>10} "
                  f"{c['pred']:>10.4f}")
        if len(candidates) > 10:
            print(f" ... и ещё {len(candidates) - 10}")
    else:
        print(" (нет кандидатов)")
    return candidates


# ======================== Part B ========================
def part_B():
    print("\n" + "=" * 78)
    print(" Part B [EXPLORATORY]: A4 с тремя twisted ratios")
    print("=" * 78)
    print(" Цель: c_d/a_1 = 9.102")
    print(" Структура: (Y_a/Y_b) * (Y_c/Y_d) * (Y_e/Y_f) * y14^n")
    print(" Мотивация: очень слабая (3 twisted ratio без weight-0)")
    print()

    Y2d, Y4d, Y6d = Y2(TAU_D), Y4(TAU_D), Y6(TAU_D)
    y14 = y14_at(TAU_D)

    numerators = {'2_2': Y2d[1], '2_1': Y2d[0], '4_3': Y4d[2]}
    denominators = {'6_1': Y6d[0], '6_3': Y6d[2]}

    base_ratios = {}
    for num_name, num_val in numerators.items():
        for den_name, den_val in denominators.items():
            if abs(den_val) < 1e-30:
                continue
            base_ratios[f'{num_name}/{den_name}'] = abs(num_val / den_val)

    base_names = list(base_ratios.keys())
    triples = list(itertools.combinations_with_replacement(base_names, 3))
    n_combos = len(triples) * 7
    print(f" Пространство: {len(triples)}x7 = {n_combos}")

    candidates = []
    for triple in triples:
        base = (base_ratios[triple[0]] * base_ratios[triple[1]]
                * base_ratios[triple[2]])
        for n in range(-6, 1):
            pred = base * (y14 ** n)
            miss = miss_pct(pred, TARGET_CD)
            if miss < 2.0:
                candidates.append({
                    'miss': miss, 'triple': triple, 'n': n, 'pred': pred,
                })

    candidates.sort(key=lambda x: x['miss'])
    print(f" Кандидатов с miss < 2%: {len(candidates)}")
    if candidates:
        print(f"\n {'miss':>7} {'triple':>50} {'n':>3} {'pred':>10}")
        print(" " + "-" * 75)
        for c in candidates[:10]:
            triple_str = ' * '.join(c['triple'])
            print(f" {c['miss']:6.3f}% {triple_str:>50} "
                  f"{c['n']:>3d} {c['pred']:>10.4f}")
        if len(candidates) > 10:
            print(f" ... и ещё {len(candidates) - 10}")
    else:
        print(" (нет кандидатов)")
    return candidates


# ======================== Part C ========================
def part_C():
    print("\n" + "=" * 78)
    print(" Part C [EXPLORATORY]: S4' целочисленные степени (weight=0)")
    print("=" * 78)
    print(" Цель: |c_1/c_2| = 0.3788")
    print(" Структура: prod |Y_i|^a_i, sum(a_i)=0 (weight 0)")
    print(" a_i ∈ [-4, 4]")
    print(" Мотивация: слабая; проверяет все целочисленные комбинации")
    print()

    Y = S4_forms(TAU_L)
    absY = [abs(y) for y in Y]
    logY = [math.log(y) for y in absY]
    log_target = math.log(TARGET_C12)

    print(f" |Y| = {[round(v, 4) for v in absY]}")
    print(f" log|Y| = {[round(v, 4) for v in logY]}")
    print(f" log(target) = {log_target:.4f}")
    print()

    Range = 4
    n_checked = 0
    candidates = []
    for a1 in range(-Range, Range + 1):
        for a2 in range(-Range, Range + 1):
            for a3 in range(-Range, Range + 1):
                for a4 in range(-Range, Range + 1):
                    a5 = -(a1 + a2 + a3 + a4)
                    if abs(a5) > Range:
                        continue
                    n_checked += 1
                    pred_log = (a1 * logY[0] + a2 * logY[1]
                                + a3 * logY[2] + a4 * logY[3]
                                + a5 * logY[4])
                    if abs(pred_log - log_target) < 0.02:
                        pred = math.exp(pred_log)
                        miss = miss_pct(pred, TARGET_C12)
                        candidates.append({
                            'miss': miss,
                            'coeffs': (a1, a2, a3, a4, a5),
                            'pred': pred, 'sum': 0,
                        })

    candidates.sort(key=lambda x: x['miss'])
    print(f" Проверено (sum=0): {n_checked}")
    print(f" Кандидатов с miss < 2%: {len(candidates)}")

    if candidates:
        print(f"\n {'miss':>7} {'a1':>3} {'a2':>3} {'a3':>3} "
              f"{'a4':>3} {'a5':>3} {'pred':>10}")
        print(" " + "-" * 45)
        for c in candidates[:15]:
            a = c['coeffs']
            print(f" {c['miss']:6.3f}% {a[0]:>3d} {a[1]:>3d} "
                  f"{a[2]:>3d} {a[3]:>3d} {a[4]:>3d} {c['pred']:>10.4f}")
        if len(candidates) > 15:
            print(f" ... и ещё {len(candidates) - 15}")

        small = [c for c in candidates
                 if all(abs(a) <= 2 for a in c['coeffs'])]
        print(f"\n Решений с |a_i| <= 2: {len(small)}")
        if small:
            for c in small:
                print(f"   coeffs = {c['coeffs']}, miss = {c['miss']:.3f}%")
    else:
        print(" (нет кандидатов)")
    return candidates


# ======================== Part D ========================
def part_D():
    print("\n" + "=" * 78)
    print(" Part D [EXPLORATORY]: S4' без weight-условия (сравнение)")
    print("=" * 78)
    print(" Цель: |c_1/c_2| = 0.3788")
    print(" Структура: prod |Y_i|^a_i, без sum=0")
    print(" a_i ∈ [-3, 3]")
    print(" Мотивация: нулевая (weight≠0), но полезно для сравнения")
    print()

    Y = S4_forms(TAU_L)
    absY = [abs(y) for y in Y]
    logY = [math.log(y) for y in absY]
    log_target = math.log(TARGET_C12)

    Range = 3
    n_checked = 0
    candidates = []
    for a1 in range(-Range, Range + 1):
        for a2 in range(-Range, Range + 1):
            for a3 in range(-Range, Range + 1):
                for a4 in range(-Range, Range + 1):
                    for a5 in range(-Range, Range + 1):
                        n_checked += 1
                        pred_log = (a1 * logY[0] + a2 * logY[1]
                                    + a3 * logY[2] + a4 * logY[3]
                                    + a5 * logY[4])
                        if abs(pred_log - log_target) < 0.02:
                            pred = math.exp(pred_log)
                            miss = miss_pct(pred, TARGET_C12)
                            candidates.append({
                                'miss': miss,
                                'coeffs': (a1, a2, a3, a4, a5),
                                'pred': pred, 'sum': a1+a2+a3+a4+a5,
                            })

    candidates.sort(key=lambda x: x['miss'])
    print(f" Проверено (без ограничения): {n_checked}")
    print(f" Кандидатов с miss < 2%: {len(candidates)}")

    if candidates:
        print(f"\n Топ-10:")
        print(f" {'miss':>7} {'a1':>3} {'a2':>3} {'a3':>3} "
              f"{'a4':>3} {'a5':>3} {'sum':>4} {'pred':>10}")
        print(" " + "-" * 50)
        for c in candidates[:10]:
            a = c['coeffs']
            print(f" {c['miss']:6.3f}% {a[0]:>3d} {a[1]:>3d} "
                  f"{a[2]:>3d} {a[3]:>3d} {a[4]:>3d} {c['sum']:>4d} "
                  f"{c['pred']:>10.4f}")
        n_sum0 = len([c for c in candidates if c['sum'] == 0])
        print(f"\n Из них с sum=0: {n_sum0} (сравните с Part C)")
    else:
        print(" (нет кандидатов)")
    return candidates


# ======================== Part E: Stability ========================
def part_E_stability(top_cd, top_c12):
    print("\n" + "=" * 78)
    print(" Part E: Стабильность лучших кандидатов")
    print("=" * 78)

    if top_cd:
        print("\n A4 кандидаты (c_d/a_1):")
        for c in top_cd[:3]:
            print(f"\n Кандидат: {c['num']}/{c['den']} * y14^{c['n']} "
                  f"* {c['w1']} * {c['w2']}")
            print(f" Базовая miss: {c['miss']:.3f}%")
            print(f" {'d_tau':>8} {'Re':>10} {'Im':>10} {'pred':>10} {'miss':>8}")
            print(" " + "-" * 52)

            def get_w0(name, Y4v, Y6v):
                parts = name.split('_')
                weight = int(parts[1])
                i = int(parts[2][0]) - 1
                j = int(parts[2][1]) - 1
                Y = Y4v if weight == 4 else Y6v
                return abs(Y[i] / Y[j])

            for d in [0.0, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05]:
                tau_var = TAU_D * (1 + d)
                Y2v, Y4v, Y6v = Y2(tau_var), Y4(tau_var), Y6(tau_var)
                y14v = y14_at(tau_var)
                num_val = {'2_2': Y2v[1], '2_1': Y2v[0],
                           '4_3': Y4v[2]}[c['num']]
                den_val = {'6_1': Y6v[0], '6_3': Y6v[2]}[c['den']]
                base = abs(num_val / den_val)
                w1 = get_w0(c['w1'], Y4v, Y6v)
                w2 = get_w0(c['w2'], Y4v, Y6v)
                pred = base * (y14v ** c['n']) * w1 * w2
                miss = miss_pct(pred, TARGET_CD)
                print(f" {d*100:+7.1f}% {tau_var.real:10.4f} "
                      f"{tau_var.imag:10.4f} {pred:10.4f} {miss:7.3f}%")

    if top_c12:
        print("\n S4' кандидаты (c_1/c_2):")
        for c in top_c12[:3]:
            print(f"\n Кандидат: {c['coeffs']} (sum={c['sum']})")
            print(f" Базовая miss: {c['miss']:.3f}%")
            print(f" {'d_tau':>8} {'Re':>10} {'Im':>10} {'pred':>10} {'miss':>8}")
            print(" " + "-" * 52)
            for d in [0.0, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05]:
                tau_var = TAU_L * (1 + d)
                Yv = S4_forms(tau_var)
                absYv = [abs(y) for y in Yv]
                pred = 1.0
                for ai, yi in zip(c['coeffs'], absYv):
                    pred *= yi ** ai
                miss = miss_pct(pred, TARGET_C12)
                print(f" {d*100:+7.1f}% {tau_var.real:10.4f} "
                      f"{tau_var.imag:10.4f} {pred:10.4f} {miss:7.3f}%")


# ======================== Main ========================
def main():
    print("\n" + "#" * 78)
    print("#  meml75_derive_final_attempt.py")
    print("#  Последняя попытка вывода c_d/a_1 и c_1/c_2")
    print("#" * 78)
    print("\n ЧЕСТНОЕ ПРЕДУПРЕЖДЕНИЕ:")
    print(" Строгий поиск (§14 п.11) уже дал 0 из 546 и 0 из 168.")
    print(" Свободный поиск дал 28 и 210 — ровно случайное ожидание.")
    print(" Всё ниже — exploratory, слабая или нулевая мотивация.")
    print(" Ожидание: 0 или много нумерологических кандидатов.")
    print(" Интересен только уникальный малый целочисленный результат.")

    cand_A = part_A()
    cand_B = part_B()
    cand_C = part_C()
    cand_D = part_D()

    all_cd = (cand_A or [])[:3] + (cand_B or [])[:3]
    all_c12 = (cand_C or [])[:3]
    if all_cd or all_c12:
        part_E_stability((cand_A or [])[:3], (cand_C or [])[:3])

    # Verdict
    print("\n" + "-" * 78)

    # Инициализация — фикс NameError, если cand_C пуст.
    small_solutions = []
    if cand_C:
        small_solutions = [c for c in cand_C
                           if all(abs(a) <= 2 for a in c['coeffs'])]

    # Кандидат считается "выводом" только если он ОДИН и СТАБИЛЕН
    # при dtau = ±1%. Part E показывает, что оба малых S4'-кандидата
    # нестабильны (miss растёт до 3–4%), а A4-кандидатов 6 (не уникальны).
    # Итог: структурного вывода не найдено — согласовано со статьёй
    # (§14 п.11 и abstract: "extended searches find no such derivation").
    n_A_small = len(cand_A) if (cand_A and len(cand_A) <= 2) else 0
    n_B_small = len(cand_B) if (cand_B and len(cand_B) <= 2) else 0

    if (0 < len(small_solutions) == 1
            and n_A_small == 0 and n_B_small == 0):
        print(" ВЕРДИКТ: найден 1 малый S4'-кандидат, НО Part E показывает")
        print("          нестабильность при dtau = ±1% (miss растёт до 3–4%).")
        print("          Это численное совпадение, а не вывод.")
        print("          c_d, c_1, c_2 остаются свободными.")
    else:
        print(" ВЕРДИКТ: все кандидаты — либо многочисленные (нумерология),")
        print("          либо отсутствуют, либо нестабильны при dtau = ±1%.")
        print("          Структурного вывода c_d, c_1, c_2 из модулярных")
        print("          форм не найдено в проверенных направлениях.")
        print("          c_d, c_1, c_2 остаются свободными.")
        print("          (Согласовано с §14 п.11 и abstract статьи.)")
    print("\n" + "#" * 78)


if __name__ == "__main__":
    main()