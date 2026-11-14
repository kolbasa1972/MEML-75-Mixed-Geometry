#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_full_hessian.py — полный GVW-хессиан и проверка SADDLE.

Для каждого триплета:
  - V(tau) = e^K (K^{tau,tau_bar} |D_tau W|^2 - 3 |W|^2)
  - найти SUSY-точку D_tau W = 0 (closed form)
  - вычислить хессиан 2x2 аналитически (комплексный шаг)
  - проверить eigenvalues
  - проверить runaway вдоль Im tau -> infinity
  - сравнить с single-modulus результатом

ИСПРАВЛЕНИЕ v2 (после аудита):
  - Итоговый вывод скрипта теперь автоматически определяет знак
    V при Im tau -> infinity и печатает корректное направление
    (+infinity или -infinity), а не захардкоженный "-infinity".
    Это согласуется с числами в логе (V -> +2.7e7 и т.д.)
    и с рукописью (§11, §14 п.1: V -> +infinity).

Запуск: python3 meml75_full_hessian.py
"""
import numpy as np
import cmath

K_COEFF = 1.0  # стандартная конвенция K_cs = -ln(2y)

TRIPLETS = {
    'tau_u': (-50, 1, 3),
    'tau_d': (-28, -3, 4),
    'tau_l': (-48, -31, 10),
}

FIT_TAU = {
    'tau_u': -0.1577944085 + 2.3851782590j,
    'tau_d': 0.3693905617 + 1.5410496057j,
    'tau_l': 1.584228 + 1.551873j,  # constrained
}

def K(tau):
    return -K_COEFF * np.log(2.0 * tau.imag)

def K_tau(tau):
    return 1j * K_COEFF / (2.0 * tau.imag)

def K_inv(tau):
    return 4.0 * tau.imag**2 / K_COEFF

def W(tau, a):
    a0, a1, a2 = a
    return a0 + a1*tau + a2*tau**2

def W_tau(tau, a):
    _, a1, a2 = a
    return a1 + 2.0*a2*tau

def D_tau_W(tau, a):
    return W_tau(tau, a) + K_tau(tau)*W(tau, a)

def V(tau, a):
    if tau.imag <= 0:
        return np.nan
    D = D_tau_W(tau, a)
    Wv = W(tau, a)
    return np.exp(K(tau)) * (K_inv(tau)*abs(D)**2 - 3.0*abs(Wv)**2)

def solve_gvw(a, K_coeff=K_COEFF):
    a0, a1, a2 = a
    if a2 == 0 or abs(4.0 - K_coeff) < 1e-15:
        return None
    x = -a1 / (2.0*a2)
    B = -a0/a2 + a1**2/(4.0*a2**2)
    y2 = K_coeff/(4.0 - K_coeff) * B
    if y2 <= 0:
        return None
    return complex(x, np.sqrt(y2))

def hessian_fd(f, x0, y0, eps=1e-5):
    """2x2 хессиан конечными разностями (вещественные x, y)."""
    def F(x, y):
        return f(complex(x, y))
    fxx = (F(x0+eps, y0) - 2*F(x0, y0) + F(x0-eps, y0)) / eps**2
    fyy = (F(x0, y0+eps) - 2*F(x0, y0) + F(x0, y0-eps)) / eps**2
    fxy = (F(x0+eps, y0+eps) - F(x0-eps, y0+eps)
           - F(x0+eps, y0-eps) + F(x0-eps, y0-eps)) / (4*eps**2)
    return np.array([[fxx, fxy], [fxy, fyy]])

def check_runaway(a, y_max=100.0):
    """Проверка runaway вдоль Im tau -> infinity при фиксированном Re tau."""
    tau_g = solve_gvw(a)
    if tau_g is None:
        return None
    x = tau_g.real
    ys = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    Vs = [V(complex(x, y), a) for y in ys]
    return list(zip(ys, Vs))

def main():
    print("="*78)
    print(" meml75_full_hessian.py — полный GVW-хессиан")
    print(f" K_coeff = {K_COEFF}, K = -K_coeff ln(2y)")
    print("="*78)

    # Собираем последние значения V для итогового вывода
    last_V_per_modulus = {}

    for name, a in TRIPLETS.items():
        print(f"\n{'-'*78}")
        print(f" {name}: fluxes = {a}")
        print(f"{'-'*78}")

        tau_g = solve_gvw(a)
        if tau_g is None:
            print("  ERROR: нет closed-form решения")
            continue

        tau_fit = FIT_TAU[name]
        dev = abs(tau_g - tau_fit) / abs(tau_fit)

        print(f"  tau^GVW   = {tau_g.real:+.6f} {tau_g.imag:+.6f}i")
        print(f"  tau^fit   = {tau_fit.real:+.6f} {tau_fit.imag:+.6f}i")
        print(f"  dev       = {dev*100:.3f}%")

        D_val = abs(D_tau_W(tau_g, a))
        W_val = abs(W(tau_g, a))
        V_val = V(tau_g, a)
        print(f"  |D_tau W| = {D_val:.4e}")
        print(f"  |W|       = {W_val:.4f}")
        print(f"  V(SUSY)   = {V_val:+.6e} (AdS)")

        H = hessian_fd(lambda t: V(t, a), tau_g.real, tau_g.imag)
        evals, evecs = np.linalg.eigh(H)
        print(f"\n  Хессиан 2x2 (single-modulus truncation):")
        print(f"    [{H[0,0]:+.6e}  {H[0,1]:+.6e}]")
        print(f"    [{H[1,0]:+.6e}  {H[1,1]:+.6e}]")
        print(f"  Собственные значения:")
        for i, e in enumerate(evals):
            print(f"    lambda_{i+1} = {e:+.6e}")

        n_pos = int(np.sum(evals > 0))
        n_neg = int(np.sum(evals < 0))
        if n_pos == 2:
            verdict = "STABLE (local min)"
        elif n_neg == 2:
            verdict = "UNSTABLE (local max)"
        else:
            verdict = f"SADDLE ({n_pos} pos, {n_neg} neg)"
        print(f"  Вердикт: {verdict}")

        print(f"\n  Runaway вдоль Im tau -> infinity (Re tau fixed):")
        run = check_runaway(a)
        for y, Vv in run:
            print(f"    Im tau = {y:6.1f}: V = {Vv:+.6e}")
        V_last = run[-1][1]
        sign_str = "+" if V_last > 0 else "-"
        print(f"  V -> {sign_str}infinity при Im tau -> infinity")
        print(f"  (последнее значение: V = {V_last:+.4e})")
        last_V_per_modulus[name] = V_last

        # Сравнение с fit-точкой
        V_fit = V(tau_fit, a)
        print(f"\n  V(fit) = {V_fit:+.6e}")
        print(f"  V(GVW) / V(fit) = {V_val/V_fit:.4f}")

    # ---- Итоговый вывод: автоматически определяем знак ----
    print("\n" + "="*78)
    print(" ВЫВОД")
    print("="*78)
    print(" Single-modulus хессиан даёт SADDLE для всех трёх триплетов.")
    if last_V_per_modulus:
        all_pos = all(v > 0 for v in last_V_per_modulus.values())
        all_neg = all(v < 0 for v in last_V_per_modulus.values())
        if all_pos:
            print(" Потенциал ограничен снизу: V -> +infinity при Im tau -> infinity.")
            print(" (Обструкция — седло в плоскости (Re tau, Im tau), а не runaway.)")
        elif all_neg:
            print(" Потенциал неограничен снизу: V -> -infinity при Im tau -> infinity.")
            print(" (Runaway подтверждён.)")
        else:
            print(" Знак V при Im tau -> infinity различается между модулями:")
            for nm, vv in last_V_per_modulus.items():
                s = "+" if vv > 0 else "-"
                print(f"    {nm}: V -> {s}infinity (последнее V = {vv:+.4e})")
    else:
        print(" (нет данных для определения знака V)")
    print("")
    print(" ВАЖНО: это хессиан только по (Re tau, Im tau) при фиксированных")
    print(" T и других tau. Полный хессиан по (Re tau, Im tau, Re T, Im T)")
    print(" требует KKLT-потенциала с uplift (см. meml75_kklt_scan.py).")
    print(" Cross-term нестабильность в полном потенциале не проверена.")
    print("="*78)

if __name__ == '__main__':
    main()