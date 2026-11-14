#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_derive_c1c2.py — попытка вывести c_1/c_2 из S4' модулярных форм.

Стратегия:
  Перебираем произведения |Y_i/Y_j|^k и комбинации с weight-0
  отношениями. Ищем |c_1/c_2| = 0.3788 с miss < 10%.

Запуск: python3 meml75_derive_c1c2.py
"""
import numpy as np
from itertools import product

TAU_L = 1.584 + 1.552j
TARGET = 0.3788
TOL = 0.10

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

Y1, Y2, Y3, Y4, Y5 = S4_forms(TAU_L)
Y = {'1': Y1, '2': Y2, '3': Y3, '4': Y4, '5': Y5}
absY = {k: abs(v) for k, v in Y.items()}

print("="*78)
print(" meml75_derive_c1c2.py — c_1/c_2 from S4' modular forms")
print("="*78)
print(f" tau_l = {TAU_L}")
for k in '12345':
    print(f" |Y_{k}| = {absY[k]:.6f}")
print(f" Target |c_1/c_2| = {TARGET}")
print()

# weight-0 отношения: любые |Y_i/Y_j| (все Y имеют один вес в S4' конвенции)
ratios = {}
for i in '12345':
    for j in '12345':
        if i != j:
            ratios[f'{i}/{j}'] = absY[i]/absY[j]

# Одиночные отношения
print("Одиночные отношения:")
singles = [(k, v, abs(v-TARGET)/TARGET) for k, v in ratios.items()]
singles.sort(key=lambda x: x[2])
for k, v, m in singles[:5]:
    print(f"  |Y_{k}| = {v:.4f}, miss = {m*100:.2f}%")

# Произведения двух отношений
print("\nПроизведения двух отношений (weight 0):")
candidates = []
for (k1, v1), (k2, v2) in product(ratios.items(), repeat=2):
    pred = v1 * v2
    miss = abs(pred - TARGET)/TARGET
    if miss < TOL:
        candidates.append((miss, f"{k1} * {k2}", pred))

# Произведения трёх отношений
for (k1, v1), (k2, v2), (k3, v3) in product(ratios.items(), repeat=3):
    pred = v1 * v2 * v3
    miss = abs(pred - TARGET)/TARGET
    if miss < TOL:
        candidates.append((miss, f"{k1} * {k2} * {k3}", pred))

candidates.sort(key=lambda x: x[0])
print(f"Кандидатов с miss < {TOL*100:.0f}% (2-3 отношения): {len(candidates)}")
for miss, expr, pred in candidates[:15]:
    print(f"  miss = {miss*100:6.2f}%, {expr} = {pred:.4f}")

# Проверка: |Y_4 Y_2 / Y_5^2| — "естественный" кандидат
print("\n" + "="*78)
print(" Естественные кандидаты:")
print("="*78)
natural = [
    ("|Y_4/Y_5|", absY['4']/absY['5']),
    ("|Y_4/Y_5| * |Y_2/Y_1|", absY['4']/absY['5'] * absY['2']/absY['1']),
    ("|Y_4 Y_2 / Y_5^2|", absY['4']*absY['2']/absY['5']**2),
    ("|Y_4 Y_3 / Y_5^2|", absY['4']*absY['3']/absY['5']**2),
    ("|Y_4 Y_1 / Y_5^2|", absY['4']*absY['1']/absY['5']**2),
    ("|Y_4 / (Y_5 * Y_1)|", absY['4']/(absY['5']*absY['1'])),
    ("|Y_4^2 / (Y_5 Y_2)|", absY['4']**2/(absY['5']*absY['2'])),
]
for name, val in natural:
    miss = abs(val - TARGET)/TARGET
    mark = " <-- HIT" if miss < TOL else ""
    print(f"  {name:30s} = {val:.4f}, miss = {miss*100:6.2f}%{mark}")

print("\n" + "="*78)
print(" ВЕРДИКТ")
print("="*78)
if candidates:
    best = candidates[0]
    print(f" ЛУЧШИЙ КАНДИДАТ: {best[1]} = {best[2]:.4f} (miss {best[0]*100:.2f}%)")
    print("   => Возможен вывод c_1/c_2 из S4' форм.")
    print("   => Проверить стабильность и неслучайность.")
else:
    print(" НИ ОДНО произведение 2-3 отношений не даёт 0.3788 с miss < 10%.")
    print(" => c_1/c_2 не сводится к простому произведению S4' отношений.")
    print(" => c_1, c_2 остаются свободными.")