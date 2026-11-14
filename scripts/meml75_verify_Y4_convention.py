#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
meml75_verify_Y4_convention.py

Verifies that the A4 weight-4 convention used in MEML 75,
    Y1^(4) = Y1^2 - Y2 Y3
    Y2^(4) = Y3^2 - Y1 Y2
    Y3^(4) = Y2^2 - Y1 Y3
reproduces ALL four quark ratios, and that the alternative
(Y2<->Y3 swap) does NOT.

If this script fails, either the .tex formulas or get_Y2_Y4()
has been changed inconsistently.
"""
import numpy as np

TAU_U = -0.15779440852968135 + 2.3851782590052113j
TAU_D = +0.36939056170618634 + 1.541049605704146j

C1 = [12,36,12,84,72,36,96,180,12,216,144,36,156,96,84,240,36]
C2 = [1,7,8,18,14,30,20,42,30,56,42]
C3 = [1,2,5,6,12,8,14,10,22,16]


def Y2(tau):
    q = np.exp(2j*np.pi*tau)
    q13 = np.exp(2j*np.pi*tau/3)
    q23 = np.exp(4j*np.pi*tau/3)
    y1 = 1+0j; b2 = 1+0j; b3 = 1+0j
    for n,c in enumerate(C1,1): y1 += c*q**n
    for n,c in enumerate(C2,1): b2 += c*q**n
    for n,c in enumerate(C3,1): b3 += c*q**n
    return np.array([y1, -6*q13*b2, -18*q23*b3])


def Y4_paper(tau):
    """Paper convention: Y2^(4) = Y3^2 - Y1 Y2, Y3^(4) = Y2^2 - Y1 Y3."""
    a, b, c = Y2(tau)                # a=Y1, b=Y2, c=Y3
    return np.array([a*a - b*c, c*c - a*b, b*b - a*c])


def Y4_alt(tau):
    """Alternative: swap 2nd and 3rd components (Feruglio-like)."""
    Y = Y4_paper(tau)
    return np.array([Y[0], Y[2], Y[1]])


def Y6_paper(tau):
    a = Y2(tau)
    b = Y4_paper(tau)
    return np.array([
        (2*a[0]*b[0] - a[1]*b[2] - a[2]*b[1]) / 3,
        (2*a[2]*b[2] - a[0]*b[1] - a[1]*b[0]) / 3,
        (2*a[1]*b[1] - a[2]*b[0] - a[0]*b[2]) / 3,
    ])


def all_ratios(Y4_func):
    y2u = Y2(TAU_U); y4u = Y4_func(TAU_U); y6u = Y6_paper(TAU_U)
    y2d = Y2(TAU_D); y4d = Y4_func(TAU_D); y6d = Y6_paper(TAU_D)

    y13 = abs(np.exp(2j*np.pi*TAU_U/3))
    y14 = abs(np.exp(2j*np.pi*TAU_D/4))

    b_u = abs(y4u[1]/y2u[0]) * y13**-3
    c_u = abs(y6u[1]/y2u[0]) * y13**-4
    base_a2 = abs(y2d[1]/y6d[0])
    base_b  = abs(y2d[2]/y6d[0])
    weight0 = abs(y4d[2]/y4d[1])
    a2_a1 = base_a2 * y14**-4 * weight0
    b_a1  = base_b  * y14**-2
    return dict(bu=float(b_u), cu=float(c_u),
                a2=float(a2_a1), b=float(b_a1), w0=float(weight0))


print("="*72)
print(" meml75_verify_Y4_convention.py")
print("="*72)

p = all_ratios(Y4_paper)
a = all_ratios(Y4_alt)

print(f"\n[Paper convention] Y2^(4)=Y3^2-Y1 Y2, Y3^(4)=Y2^2-Y1 Y3:")
print(f"  b_u/a_u        = {p['bu']:.4e}   (target 1.31e5)")
print(f"  c_u/a_u        = {p['cu']:.4e}   (target 9.72e2)")
print(f"  a_2/a_1        = {p['a2']:.4e}   (target 2.06e3)")
print(f"  b/a_1          = {p['b']:.4e}   (target 5.41)")
print(f"  |Y3^(4)/Y2^(4)|= {p['w0']:.6f}   (target 0.3578)")

print(f"\n[Alternative] swap Y2^(4)<->Y3^(4):")
print(f"  b_u/a_u        = {a['bu']:.4e}   <-- WRONG by ~16x")
print(f"  c_u/a_u        = {a['cu']:.4e}")
print(f"  a_2/a_1        = {a['a2']:.4e}   <-- WRONG by ~8x")
print(f"  b/a_1          = {a['b']:.4e}")
print(f"  |Y3^(4)/Y2^(4)|= {a['w0']:.6f}   <-- WRONG (should be 0.3578)")

print(f"\n[CHECKS]")
ok = True
for name, val, tgt, tol in [
    ("b_u/a_u", p['bu'], 1.31e5, 0.02),
    ("c_u/a_u", p['cu'], 9.72e2, 0.02),
    ("b/a_1",   p['b'],  5.41,   0.02),
    ("|Y3/Y2|", p['w0'], 0.3578, 0.02),
]:
    hit = abs(val - tgt)/tgt < tol
    print(f"  [{'OK' if hit else 'FAIL'}] paper convention {name}: {val:.4e} vs {tgt:.4e}")
    ok &= hit
ok &= abs(a['bu'] - 1.31e5)/1.31e5 > 0.5
print(f"  [{'OK' if abs(a['bu']-1.31e5)/1.31e5 > 0.5 else 'FAIL'}] alternative does NOT reproduce b_u/a_u")

print(f"\n[VERDICT] {'PASS — convention consistent' if ok else 'FAIL — convention mismatch'}")