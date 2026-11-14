#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_derive_cd.py — попытка вывести c_d/a_1 из A4 модулярных форм.

Стратегия:
  Перебираем все отношения |Y_i^(w) / Y_j^(w')| при w,w' in {2,4,6},
  умножаем на y14^(-n), n in [-8,0], опционально на weight-0 факторы
  |Y_k^(4)/Y_l^(4)| и |Y_k^(6)/Y_l^(6)|. Ищем c_d/a_1 = 9.10 с miss < 10%.

Запуск: python3 meml75_derive_cd.py
"""
import cmath, math, itertools

C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]
TAU_D = 0.36939056170618634 + 1.541049605704146j
TARGET = 9.1020   # c_d/a_1 из honest_pure2
TOL = 0.10        # 10% miss

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

Y2d = Y2(TAU_D); Y4d = Y4(TAU_D); Y6d = Y6(TAU_D)
y14 = abs(cmath.exp(2j*math.pi*TAU_D/4))

print("="*78)
print(" meml75_derive_cd.py — c_d/a_1 from A4 modular forms")
print("="*78)
print(f" tau_d = {TAU_D}")
print(f" y14 = {y14:.6e}, y14^-1 = {y14**-1:.4f}, y14^-2 = {y14**-2:.2f}, "
      f"y14^-3 = {y14**-3:.1f}, y14^-4 = {y14**-4:.1f}")
print(f" |Y^(2)| = {[abs(v) for v in Y2d]}")
print(f" |Y^(4)| = {[abs(v) for v in Y4d]}")
print(f" |Y^(6)| = {[abs(v) for v in Y6d]}")
print(f" Target c_d/a_1 = {TARGET}")
print()

# Собираем все одиночные отношения
forms = {'2_1': Y2d[0], '2_2': Y2d[1], '2_3': Y2d[2],
         '4_1': Y4d[0], '4_2': Y4d[1], '4_3': Y4d[2],
         '6_1': Y6d[0], '6_2': Y6d[1], '6_3': Y6d[2]}

# weight-0 факторы (отношения форм одного веса)
weight0 = {}
for w, arr in [('4', Y4d), ('6', Y6d)]:
    for i in range(3):
        for j in range(3):
            if i != j:
                weight0[f'w0_{w}_{i+1}{j+1}'] = abs(arr[i]/arr[j])

candidates = []
for num_name, num_val in forms.items():
    for den_name, den_val in forms.items():
        if num_name == den_name or abs(den_val) < 1e-30:
            continue
        base = abs(num_val/den_val)
        for n in range(-8, 1):
            for w0_name, w0_val in [('none', 1.0)] + list(weight0.items()):
                pred = base * (y14**n) * w0_val
                miss = abs(pred - TARGET)/TARGET
                if miss < TOL:
                    candidates.append((miss, num_name, den_name, n, w0_name, pred))

candidates.sort(key=lambda x: x[0])
print(f"Кандидатов с miss < {TOL*100:.0f}%: {len(candidates)}")
print()
print(f"{'miss':>8} {'num':>6} {'den':>6} {'n':>3} {'w0':>12} {'pred':>10}")
print("-"*60)
for miss, num, den, n, w0, pred in candidates[:20]:
    print(f"{miss*100:7.2f}% {num:>6} {den:>6} {n:>3d} {w0:>12} {pred:10.4f}")

# Сводка: что дают "естественные" кандидаты
print("\n" + "="*78)
print(" Естественные кандидаты (Y_1^(2) в числителе, Y_1^(6) в знаменателе):")
print("="*78)
X = abs(Y2d[0]/Y6d[0])
print(f" |Y_1^(2)/Y_1^(6)| = {X:.6e}")
for n in range(-6, 1):
    pred = X * (y14**n)
    miss = abs(pred - TARGET)/TARGET * 100
    mark = " <-- HIT" if miss < TOL*100 else ""
    print(f"   n = {n:+d}: pred = {pred:10.4f}, miss = {miss:7.2f}%{mark}")

print("\n" + "="*78)
print(" ВЕРДИКТ")
print("="*78)
if candidates:
    best = candidates[0]
    print(f" ЛУЧШИЙ КАНДИДАТ: |{best[1]}/{best[2]}| * y14^({best[3]}) * {best[4]}")
    print(f"   = {best[5]:.4f} (miss {best[0]*100:.2f}% от {TARGET})")
    print("   => Возможен вывод c_d/a_1 из модулярных форм.")
    print("   => Требуется проверить, стабилен ли результат и не случаен ли он.")
else:
    print(" НИ ОДНО отношение не даёт 9.10 с miss < 10%.")
    print(" => c_d/a_1 не сводится к простому отношению A4-форм.")
    print(" => c_d остаётся свободным параметром.")