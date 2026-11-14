#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_derive_c1c2_strict.py — структурно мотивированный поиск c_1/c_2.

Ограничения:
  - Одно отношение: дублет/триплет или триплет/дублет (12 комбинаций).
  - Произведение: (отношение внутри дублета) × (отношение внутри триплета)
    (2 × 6 = 12 комбинаций).
  - Степени: y14^(-n), n in [-6, 0].
  - Никаких произведений трёх отношений, никаких произвольных weight-0.

Проверка стабильности: dtau_l = ±1%, ±2%, ±5%.

Запуск: python3 meml75_derive_c1c2_strict.py
"""
import numpy as np
import itertools

TAU_L = 1.584 + 1.552j
TARGET = 0.3788
TOL = 0.02  # 2%

def eta(tau, N_max=200):
    q = np.exp(2j*np.pi*tau)
    prod = 1 + 0j
    for n in range(1, N_max):
        prod *= (1 - q**n)
    return q**(1/24) * prod

def dlog_eta(tau, N_max=200):
    q = np.exp(2j*np.pi*tau)
    E2 = 1.0 + 0j
    for n in range(1, N_max):
        qn = q**n
        E2 -= 24 * n * qn / (1 - qn)
    return 1j*np.pi/12 * E2

def Y(a, tau, N_max=200):
    args = [(2,1,2),(4,0,1),(1,0,4),(1,1,4),(1,2,4),(1,3,4)]
    total = 0j
    for ai, (num, b, denom) in zip(a, args):
        arg = (num*tau + b)/denom
        total += ai * (num/denom) * dlog_eta(arg, N_max)
    return total

omega = np.exp(2j*np.pi/3)

def S4_forms(tau, N_max=200):
    Y1 = Y([1, 1, omega, omega**2, omega, omega**2], tau, N_max)
    Y2 = Y([1, 1, omega**2, omega, omega**2, omega], tau, N_max)
    Y3 = Y([1, -1, -1, -1, 1, 1], tau, N_max)
    Y4 = Y([1, -1, -omega**2, -omega, omega**2, omega], tau, N_max)
    Y5 = Y([1, -1, -omega, -omega**2, omega, omega**2], tau, N_max)
    return Y1, Y2, Y3, Y4, Y5

def evaluate(tau):
    Y1, Y2, Y3, Y4, Y5 = S4_forms(tau)
    Y = {'1': Y1, '2': Y2, '3': Y3, '4': Y4, '5': Y5}
    absY = {k: abs(v) for k, v in Y.items()}
    y14 = abs(np.exp(2j*np.pi*tau/4))

    doublet = ['1', '2']
    triplet = ['3', '4', '5']

    # Одиночные отношения: дублет/триплет и триплет/дублет
    single_ratios = {}
    for d in doublet:
        for t in triplet:
            single_ratios[f'{d}/{t}'] = absY[d]/absY[t]
            single_ratios[f'{t}/{d}'] = absY[t]/absY[d]

    # Произведения: (отношение внутри дублета) × (отношение внутри триплета)
    intra_doublet = {}
    if '1' in absY and '2' in absY:
        intra_doublet['1/2'] = absY['1']/absY['2']
        intra_doublet['2/1'] = absY['2']/absY['1']
    intra_triplet = {}
    for i, j in itertools.permutations(triplet, 2):
        intra_triplet[f'{i}/{j}'] = absY[i]/absY[j]

    products = {}
    for kd, vd in intra_doublet.items():
        for kt, vt in intra_triplet.items():
            products[f'({kd})*({kt})'] = vd * vt

    results = []
    for name, base in list(single_ratios.items()) + list(products.items()):
        for n in range(-6, 1):
            pred = base * (y14**n)
            miss = abs(pred - TARGET)/TARGET
            results.append((miss, name, n, pred))
    results.sort(key=lambda x: x[0])
    return results

print("="*78)
print(" meml75_derive_c1c2_strict.py — структурно мотивированный поиск")
print("="*78)
print(f" tau_l = {TAU_L}")
print(f" TARGET = {TARGET}, TOL = {TOL*100:.0f}%")
print(f" Пространство: {12 + 12} отношений × 7 степеней = {24*7} комбинаций")
print()

results = evaluate(TAU_L)
hits = [r for r in results if r[0] < TOL]
print(f"Кандидатов с miss < {TOL*100:.0f}%: {len(hits)} (из {len(results)})")
print()

if hits:
    print(f"{'miss':>8} {'ratio':>30} {'n':>3} {'pred':>10}")
    print("-"*55)
    for miss, name, n, pred in hits[:10]:
        print(f"{miss*100:7.3f}% {name:>30} {n:>3d} {pred:10.4f}")
else:
    print("  (нет кандидатов)")
    print("\n  Топ-5 ближайших:")
    for miss, name, n, pred in results[:5]:
        print(f"  miss={miss*100:6.2f}% {name} * y14^{n} = {pred:.4f}")

if hits:
    best = hits[0]
    miss0, name, n, pred0 = best
    print("\n" + "="*78)
    print(" ПРОВЕРКА СТАБИЛЬНОСТИ лучшего кандидата:")
    print(f"  {name} * y14^({n}) = {pred0:.4f}")
    print("="*78)
    stable = True
    for dtau_frac in [0.0, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05]:
        tau_var = TAU_L * (1 + dtau_frac)
        res_var = evaluate(tau_var)
        match = [r for r in res_var if r[1]==name and r[2]==n]
        if match:
            m, _, _, p = match[0]
            print(f"  dtau = {dtau_frac*100:+6.1f}%: pred = {p:.4f}, miss = {m*100:6.3f}%")
            if m > 0.05:
                stable = False
    if stable:
        print("\n [OK] Кандидат стабилен при dtau = ±5%")
        print(" => Возможен реальный вывод c_1/c_2.")
    else:
        print("\n [FAIL] Нестабилен: miss растёт до >5% при dtau = ±5%.")
        print(" => Числовое совпадение, не вывод.")

print("\n" + "="*78)
print(" ВЕРДИКТ")
print("="*78)
if hits:
    print(" Есть структурно мотивированные кандидаты. Требуют проверки.")
else:
    print(" НИ ОДНА структурно мотивированная комбинация не даёт 0.3788")
    print(" с miss < 2%. => c_1, c_2 остаются свободными.")
    print(" => Честный негативный результат.")