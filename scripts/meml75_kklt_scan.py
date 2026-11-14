#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_kklt_scan.py — KKLT-скан с полным 4x4 хессианом (исправлен).

Исправления относительно v1:
  - Только вещественные координаты: x = [Re tau, Im tau, Re T, Im T].
  - T + conj(T) = 2 Re T вычисляется явно.
  - Guard: Im tau > 0.01, Re T > 0.01 (строго положительные).
  - V возвращает np.real(...) и обрабатывает overflow/NaN.
  - Hessian по вещественным x, без комплексной арифметики.
  - exp(-a*T) с ограничением аргумента.

ИСПРАВЛЕНИЕ v2 (после аудита):
  - Итоговый вывод различает три случая:
      (a) dev < 1% и n_pos == 4: стабилизирует на фите;
      (b) n_pos == 4 и dev >= 1%: стабилизирует, но НЕ на фите
          (обструкция — смещение, а не седло);
      (c) n_pos < 4: не стабилизирует (седло/нестабильность).
    Ранее случай (b) ошибочно попадал в ветку "не стабилизирует",
    что противоречило вердикту Hessian "STABLE (все > 0)".

Запуск: python3 meml75_kklt_scan.py
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

TRIPLET = (-50, 1, 3)
TAU_FIT = -0.1577944085 + 2.3851782590j
N_UPLIFT = 2  # anti-D3


def unpack(x):
    return complex(x[0], x[1]), complex(x[2], x[3])


def safe_log(z, floor=1e-30):
    return np.log(max(z, floor))


def V(x, W0, A, a, D):
    """Полный KKLT-потенциал, только вещественный выход."""
    try:
        tau_re, tau_im, T_re, T_im = x
        if tau_im < 0.01 or T_re < 0.01:
            return 1e10

        tau = complex(tau_re, tau_im)
        T = complex(T_re, T_im)
        a0, a1, a2 = TRIPLET

        K = -3.0 * safe_log(2.0 * T_re) - safe_log(2.0 * tau_im)

        # W = W0 + A exp(-a T) + a0 + a1 tau + a2 tau^2
        exp_arg = -a * T
        if exp_arg.real < -50.0:
            Aexp = 0.0 + 0j
        else:
            Aexp = A * np.exp(exp_arg)
        W = W0 + Aexp + a0 + a1 * tau + a2 * tau**2

        W_tau = a1 + 2.0 * a2 * tau
        W_T = -a * Aexp
        K_tau = 1j / (2.0 * tau_im)
        K_T = -3.0 / (2.0 * T_re)
        K_inv_tau = 4.0 * tau_im**2
        K_inv_T = (2.0 * T_re)**2 / 3.0

        D_tau = W_tau + K_tau * W
        D_T = W_T + K_T * W

        V_F = np.exp(K) * (
            K_inv_tau * abs(D_tau)**2
            + K_inv_T * abs(D_T)**2
            - 3.0 * abs(W)**2
        )
        V_up = D / (2.0 * T_re)**N_UPLIFT
        val = np.real(V_F) + np.real(V_up)

        if not np.isfinite(val):
            return 1e10
        return float(val)
    except (OverflowError, ZeroDivisionError, ValueError):
        return 1e10


def hessian_4d(f, x0, eps=1e-4):
    """4x4 хессиан по вещественным координатам."""
    n = 4
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            xpp = x0.copy(); xpp[i] += eps; xpp[j] += eps
            xpm = x0.copy(); xpm[i] += eps; xpm[j] -= eps
            xmp = x0.copy(); xmp[i] -= eps; xmp[j] += eps
            xmm = x0.copy(); xmm[i] -= eps; xmm[j] -= eps
            H[i, j] = (f(xpp) - f(xpm) - f(xmp) + f(xmm)) / (4 * eps**2)
    return H


def minimize_tau_T(W0, A, a, D, x0=None):
    """Минимизация по 4 вещественным переменным."""
    if x0 is None:
        x0 = [TAU_FIT.real, TAU_FIT.imag, 1.0, 0.0]
    f = lambda x: V(x, W0, A, a, D)
    res = minimize(f, x0=x0, method='Nelder-Mead',
                   options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})
    return res


def main():
    print("=" * 78)
    print(" meml75_kklt_scan.py — исправленный KKLT-скан")
    print("=" * 78)
    print(f" TRIPLET = {TRIPLET}")
    print(f" tau_fit = {TAU_FIT}")
    print(f" N_uplift = {N_UPLIFT}")
    print()

    # DE-скан по параметрам
    bounds = [
        (0.1, 10.0),   # W0
        (0.1, 10.0),   # A
        (0.1, 2.0),    # a
        (1e-8, 1e-2),  # D
    ]

    def obj(p):
        W0, A, a, D = p
        res = minimize_tau_T(W0, A, a, D)
        return res.fun

    print(" DE-скан (maxiter=80, popsize=15)")
    res_de = differential_evolution(obj, bounds, maxiter=80, popsize=15,
                                     seed=42, tol=1e-4, polish=False)
    W0, A, a, D = res_de.x
    print(f"\n Лучшие параметры: W0={W0:.4f}, A={A:.4f}, "
          f"a={a:.4f}, D={D:.4e}")
    print(f" V_min = {res_de.fun:+.6e}")

    # Финальная минимизация
    res = minimize_tau_T(W0, A, a, D)
    tau_min, T_min = unpack(res.x)
    dev = abs(tau_min - TAU_FIT) / abs(TAU_FIT)
    print(f"\n Минимум:")
    print(f"   tau = {tau_min.real:+.6f} {tau_min.imag:+.6f}i")
    print(f"   T   = {T_min.real:+.6f} {T_min.imag:+.6f}i")
    print(f"   V   = {res.fun:+.6e}")
    print(f"   dev(tau) = {dev*100:.3f}%")

    # 4x4 хессиан
    f = lambda x: V(x, W0, A, a, D)
    H = hessian_4d(f, res.x)
    evals = np.linalg.eigvalsh(H)
    print(f"\n 4x4 хессиан (eigenvalues):")
    for i, e in enumerate(evals):
        print(f"   lambda_{i+1} = {e:+.6e}")
    n_pos = int(np.sum(evals > 0))
    n_neg = int(np.sum(evals < 0))
    if n_pos == 4:
        verdict = "STABLE (все > 0)"
    elif n_neg == 4:
        verdict = "UNSTABLE (все < 0)"
    else:
        verdict = f"SADDLE ({n_pos} pos, {n_neg} neg)"
    print(f" Вердикт: {verdict}")

    # Проверка runaway в полном потенциале
    print(f"\n Runaway вдоль Im tau -> infinity (Re tau, T fixed):")
    for y in [5, 10, 20, 50, 100]:
        x = [tau_min.real, y, T_min.real, T_min.imag]
        print(f"   Im tau = {y:6.1f}: V = {V(x, W0, A, a, D):+.6e}")

    print("\n" + "=" * 78)
    print(" ВЫВОД")
    print("=" * 78)
    print(f" KKLT-минимум: dev(tau) = {dev*100:.2f}%")
    if n_pos == 4 and dev < 0.01:
        print(" => KKLT стабилизирует tau на фит-значении. N_eff может сократиться.")
    elif n_pos == 4:
        print(f" => KKLT стабилизирует tau, но НЕ на фит-значении")
        print(f"    (dev = {dev*100:.2f}%). Обструкция — смещение, а не седло:")
        print(f"    4x4 Hessian положительно определён (STABLE),")
        print(f"    минимум устойчив, но tau = {tau_min.real:+.3f}{tau_min.imag:+.3f}i")
        print(f"    вместо tau^fit = {TAU_FIT.real:+.3f}{TAU_FIT.imag:+.3f}i.")
    else:
        print(" => KKLT не стабилизирует tau (седло/нестабильность).")
    print(f" N_eff: 13 → 10–11 (W0, A, a, D — новые параметры), не 7.")
    print("=" * 78)


if __name__ == '__main__':
    main()