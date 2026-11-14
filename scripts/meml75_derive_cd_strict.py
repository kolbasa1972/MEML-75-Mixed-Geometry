#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_derive_cd_strict.py — структурно мотивированный поиск c_d/a_1.

Ограничения:
  - Числитель: только Y_2^(2), Y_1^(2), Y_3^(4) (позиции c_d в M_d).
  - Знаменатель: только Y_1^(6), Y_3^(6) (позиции a_1 в M_d).
  - weight-0 факторы: только |Y_i^(4)/Y_j^(4)| и |Y_i^(6)/Y_j^(6)|, i≠j.
  - Степени: y14^(-n), n in [-6, 0].

Проверка стабильности: варьируем tau_d на ±1%, ±2%, смотрим miss.

Запуск: python3 meml75_derive_cd_strict.py
"""
import cmath, math

C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]
TAU_D = 0.36939056170618634 + 1.541049605704146j
TARGET = 9.1020
TOL = 0.02  # ужесточаем: 2%, не 10%

def series(tau, coefs):
    q = cmath.exp(2j*math.pi*tau)
    return sum(c * q**(k+1) for k, c in enumerate(coefs))

def Y2(tau):
    q13 = cmath.exp(2j*math.pi*tau/3)
    q23 = cmath.exp(4j*math.pi*tau/3)
    y1 = 1.0 + series(tau, C1)
    y2 = -6.0 * q13 * (1.0 + series(tau, C2))
    y3 = -18.0 * q23 * (1.0 + series(tau, C3))
    return (y1, y2, y3)

def Y4(tau):
    a, b, c = Y2(tau)
    return (a*a - b*c, c*c - a*b, b*b - a*c)

def Y6(tau):
    a = Y2(tau); b = Y4(tau)
    return (
        (2*a[0]*b[0] - a[1]*b[2] - a[2]*b[1])/3,
        (2*a[2]*b[2] - a[0]*b[1] - a[1]*b[0])/3,
        (2*a[1]*b[1] - a[2]*b[0] - a[0]*b[2])/3,
    )

def evaluate(tau):
    Y2d = Y2(tau); Y4d = Y4(tau); Y6d = Y6(tau)
    y14 = abs(cmath.exp(2j*math.pi*tau/4))
    numerators = {'2_2': Y2d[1], '2_1': Y2d[0], '4_3': Y4d[2]}
    denominators = {'6_1': Y6d[0], '6_3': Y6d[2]}
    w0_factors = {}
    for i in range(3):
        for j in range(3):
            if i != j:
                w0_factors[f'w0_4_{i+1}{j+1}'] = abs(Y4d[i]/Y4d[j])
                w0_factors[f'w0_6_{i+1}{j+1}'] = abs(Y6d[i]/Y6d[j])
    results = []
    for num_name, num_val in numerators.items():
        for den_name, den_val in denominators.items():
            if abs(den_val) < 1e-30:
                continue
            base = abs(num_val/den_val)
            for n in range(-6, 1):
                for w0_name, w0_val in [('none', 1.0)] + list(w0_factors.items()):
                    pred = base * (y14**n) * w0_val
                    miss = abs(pred - TARGET)/TARGET
                    results.append((miss, num_name, den_name, n, w0_name, pred))
    results.sort(key=lambda x: x[0])
    return results

# Основной поиск
print("="*78)
print(" meml75_derive_cd_strict.py — структурно мотивированный поиск")
print("="*78)
print(f" tau_d = {TAU_D}")
print(f" TARGET = {TARGET}, TOL = {TOL*100:.0f}%")
print()

results = evaluate(TAU_D)
hits = [r for r in results if r[0] < TOL]
print(f"Кандидатов с miss < {TOL*100:.0f}%: {len(hits)} (из {len(results)})")
print()
if hits:
    print(f"{'miss':>8} {'num':>5} {'den':>5} {'n':>3} {'w0':>10} {'pred':>10}")
    print("-"*50)
    for miss, num, den, n, w0, pred in hits[:10]:
        print(f"{miss*100:7.3f}% {num:>5} {den:>5} {n:>3d} {w0:>10} {pred:10.4f}")
else:
    print("  (нет кандидатов)")
    print("\n  Топ-5 ближайших:")
    for miss, num, den, n, w0, pred in results[:5]:
        print(f"  miss={miss*100:6.2f}% {num}/{den} * y14^{n} * {w0} = {pred:.4f}")

# Проверка стабильности для лучшего кандидата
if hits:
    best = hits[0]
    miss0, num, den, n, w0, pred0 = best
    print("\n" + "="*78)
    print(" ПРОВЕРКА СТАБИЛЬНОСТИ лучшего кандидата:")
    print(f"  {num}/{den} * y14^({n}) * {w0} = {pred0:.4f}")
    print("="*78)
    print(f" {'d_tau':>8} {'Re':>10} {'Im':>10} {'pred':>10} {'miss':>8}")
    print("-"*50)
    stable = True
    for dtau_frac in [0.0, 0.01, -0.01, 0.02, -0.02, 0.05, -0.05]:
        tau_var = TAU_D * (1 + dtau_frac)
        res_var = evaluate(tau_var)
        # ищем ту же комбинацию
        match = [r for r in res_var if r[1]==num and r[2]==den and r[3]==n and r[4]==w0]
        if match:
            m, _, _, _, _, p = match[0]
            print(f" {dtau_frac*100:+7.1f}% {tau_var.real:10.4f} {tau_var.imag:10.4f} {p:10.4f} {m*100:7.3f}%")
            if m > 0.05:  # 5%
                stable = False
    print()
    if stable:
        print(" [OK] Кандидат стабилен при dtau = ±5%")
        print(" => Возможен реальный вывод c_d/a_1.")
    else:
        print(" [FAIL] Кандидат нестабилен: miss растёт до >5% при dtau = ±5%.")
        print(" => Числовое совпадение, не вывод.")