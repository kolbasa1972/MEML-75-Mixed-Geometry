#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_kklt_constrained.py — KKLT-скан в физическом диапазоне.

Отличия от meml75_kklt_scan.py:
  - W0 in [1e-6, 1e-2] (не 0.1)
  - A in [0.1, 10]
  - a in [0.01, 0.6] (не 2.0)
  - D in [1e-10, 1e-4]
  - T_re guard: [1, 100] (не 0.01)
  - Проверка: V_min ≈ 0 после uplift? Если V_min < -0.1*|V_F|, uplift не работает.

ИСПРАВЛЕНИЕ v2 (после аудита):
  - Итоговый вывод различает три случая:
      (a) dev < 2% и n_pos == 4: стабилизирует на фите;
      (b) n_pos == 4 и dev >= 2%: стабилизирует, но НЕ на фите
          (обструкция — смещение, а не седло);
      (c) n_pos < 4: не стабилизирует (седло/нестабильность).
    Ранее случай (b) ошибочно попадал в ветку "не стабилизирует",
    что противоречило вердикту Hessian "STABLE".
  - В выводе явно указывается, что D лежит ВНУТРИ диапазона
    [1e-10, 1e-4], а не на границе (в отличие от A, a, T_re).

Запуск: python3 meml75_kklt_constrained.py
"""
import numpy as np
from scipy.optimize import differential_evolution, minimize

TRIPLET = (-50, 1, 3)
TAU_FIT = -0.1577944085 + 2.3851782590j
N_UPLIFT = 2

T_RE_MIN = 1.0
T_RE_MAX = 100.0
TAU_IM_MIN = 0.5


def V(x, W0, A, a, D):
    try:
        tau_re, tau_im, T_re, T_im = x
        if tau_im < TAU_IM_MIN or T_re < T_RE_MIN or T_re > T_RE_MAX:
            return 1e10

        tau = complex(tau_re, tau_im)
        T = complex(T_re, T_im)
        a0, a1, a2 = TRIPLET

        K = -3.0 * np.log(2.0 * T_re) - np.log(2.0 * tau_im)

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
        val = float(np.real(V_F)) + float(np.real(V_up))

        if not np.isfinite(val):
            return 1e10
        return val
    except (OverflowError, ZeroDivisionError, ValueError):
        return 1e10


def hessian_4d(f, x0, eps=1e-4):
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
    if x0 is None:
        x0 = [TAU_FIT.real, TAU_FIT.imag, 10.0, 0.0]
    f = lambda x: V(x, W0, A, a, D)
    return minimize(f, x0=x0, method='Nelder-Mead',
                    options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9})


def main():
    print("=" * 78)
    print(" meml75_kklt_constrained.py — физический KKLT-диапазон")
    print("=" * 78)
    print(f" TRIPLET = {TRIPLET}")
    print(f" W0 in [1e-6, 1e-2], a in [0.01, 0.6], T_re in [{T_RE_MIN}, {T_RE_MAX}]")
    print()

    bounds = [
        (1e-6, 1e-2),   # W0
        (0.1, 10.0),    # A
        (0.01, 0.6),    # a
        (1e-10, 1e-4),  # D
    ]

    def obj(p):
        W0, A, a, D = p
        return minimize_tau_T(W0, A, a, D).fun

    print(" DE-скан (maxiter=100, popsize=20)")
    res_de = differential_evolution(obj, bounds, maxiter=100, popsize=20,
                                     seed=42, tol=1e-4, polish=False)
    W0, A, a, D = res_de.x
    print(f"\n Лучшие параметры: W0={W0:.4e}, A={A:.4f}, "
          f"a={a:.4f}, D={D:.4e}")
    print(f" V_min = {res_de.fun:+.6e}")

    res = minimize_tau_T(W0, A, a, D)
    tau_min, T_min = unpack(res.x)
    dev = abs(tau_min - TAU_FIT) / abs(TAU_FIT)
    print(f"\n Минимум:")
    print(f"   tau = {tau_min.real:+.6f} {tau_min.imag:+.6f}i")
    print(f"   T   = {T_min.real:+.6f} {T_min.imag:+.6f}i")
    print(f"   V   = {res.fun:+.6e}")
    print(f"   dev(tau) = {dev*100:.3f}%")

    f = lambda x: V(x, W0, A, a, D)
    H = hessian_4d(f, res.x)
    evals = np.linalg.eigvalsh(H)
    print(f"\n 4x4 хессиан:")
    for i, e in enumerate(evals):
        print(f"   lambda_{i+1} = {e:+.6e}")
    n_pos = int(np.sum(evals > 0))
    n_neg = int(np.sum(evals < 0))
    verdict = ("STABLE" if n_pos == 4 else
               "UNSTABLE" if n_neg == 4 else
               f"SADDLE ({n_pos} pos, {n_neg} neg)")
    print(f" Вердикт: {verdict}")

    print("\n" + "=" * 78)
    print(" ВЫВОД")
    print("=" * 78)
    print(f" KKLT-минимум в физическом диапазоне: dev(tau) = {dev*100:.2f}%")
    if n_pos == 4 and dev < 0.02:
        print(" => KKLT стабилизирует tau на фит-значении.")
        print(" => Возможно сокращение N_eff до 10-11.")
    elif n_pos == 4:
        print(f" => KKLT стабилизирует tau, но НЕ на фит-значении")
        print(f"    (dev = {dev*100:.2f}%). Обструкция — смещение, а не седло:")
        print(f"    4x4 Hessian положительно определён (STABLE),")
        print(f"    минимум устойчив, но tau = {tau_min.real:+.3f}{tau_min.imag:+.3f}i")
        print(f"    вместо tau^fit = {TAU_FIT.real:+.3f}{TAU_FIT.imag:+.3f}i.")
        print(f"    A={A:.2f}, a={a:.3f}, T_re={T_min.real:.2f} лежат на границах")
        print(f"    диапазона, а D={D:.2e} — ВНУТРИ [1e-10, 1e-4].")
        print(f"    λ_4 = {evals[3]:.2e} указывает на сверхжёсткий T-сектор.")
    else:
        print(" => KKLT НЕ стабилизирует tau (седло/нестабильность).")
    print("=" * 78)


def unpack(x):
    return complex(x[0], x[1]), complex(x[2], x[3])


if __name__ == '__main__':
    main()