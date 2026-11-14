#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Figure 1: modular tau space (v24.5.5)

Generates fig_modular_tau_space.{pdf,png} showing:
  - fundamental domain of the modular group (tessellation)
  - phenomenological fit values (filled circles):
      tau_u^fit = -0.158 + 2.385i
      tau_d^fit =  0.369 + 1.541i
      tau_l^fit =  1.540 + 1.551i (unconstrained, open circle)
      tau_l^fit =  1.584 + 1.552i (constrained, filled circle)
  - GVW closed-form solutions in the standard Kaehler convention
    K_cs = -ln(2y) (open squares):
      tau_u^GVW = -0.167 + 2.359i (dev 1.16%, fluxes (-50, 1, 3))
      tau_d^GVW =  0.375 + 1.543i (dev 0.37%, fluxes (-28,-3, 4))
      tau_l^GVW =  1.550 + 1.550i (dev 0.47%, fluxes (-48,-31,10))
  - arrow tau -> tau - 2 mapping the constrained tau_l into the
    SL(2,Z) central domain

Output directory: <script_dir>/../figures if it exists,
otherwise <script_dir>/figures.
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# --- Resolve output directory ---
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_figures = os.path.join(script_dir, '..', 'figures')
local_figures = os.path.join(script_dir, 'figures')
if os.path.exists(parent_figures):
    out_dir = parent_figures
else:
    out_dir = local_figures
    os.makedirs(out_dir, exist_ok=True)
out_dir = os.path.abspath(out_dir)
print(f"Saving to: {out_dir}")

# --- Data (v24.5.1) ---
# Fit values (filled circles; unconstrained tau_l is an open circle)
fit_u = (-0.158, 2.385)
fit_d = ( 0.369, 1.541)
fit_l_c = (1.584, 1.552)  # constrained
fit_l_u = (1.540, 1.551)  # unconstrained

# GVW closed-form solutions (open squares), standard Kaehler K_cs = -ln(2y)
gvw_u = (-0.167, 2.359, '(-50,1,3)',    '1.16%')
gvw_d = ( 0.375, 1.543, '(-28,-3,4)',   '0.37%')
gvw_l = ( 1.550, 1.5495, '(-48,-31,10)', '0.47%')

# --- Figure ---
fig, ax = plt.subplots(figsize=(11, 8))

# 1. Tessellation (light gray tiles)
for n in range(-3, 5):
    ax.plot([n - 0.5, n - 0.5], [0, 3.0], color='lightgray', lw=0.8, zorder=1)
    theta = np.linspace(0, np.pi, 200)
    x_arc_tile = n + np.cos(theta)
    y_arc_tile = np.sin(theta)
    mask = y_arc_tile >= np.sqrt(3)/2 - 0.01
    ax.plot(x_arc_tile[mask], y_arc_tile[mask],
            color='lightgray', lw=0.8, zorder=1)

# 2. Central fundamental domain
x_arc = np.linspace(-0.5, 0.5, 100)
y_arc = np.sqrt(1 - x_arc**2)
ax.plot(x_arc, y_arc, 'k-', lw=2, zorder=2)
ax.plot([-0.5, -0.5], [0, np.sqrt(3)/2], 'k-', lw=2, zorder=2)
ax.plot([ 0.5,  0.5], [0, np.sqrt(3)/2], 'k-', lw=2, zorder=2)
ax.fill_between(x_arc, 0, y_arc, color='lightblue', alpha=0.3, zorder=2)

# 3. Symmetric points (x markers)
ax.plot(0.0,  1.0,             'x', color='black', ms=10, mew=2, zorder=4)
ax.plot(0.5,  np.sqrt(3)/2,    'x', color='black', ms=10, mew=2, zorder=4)
ax.plot(-0.5, np.sqrt(3)/2,    'x', color='black', ms=10, mew=2, zorder=4)

# 4. Fit values — filled circles (constrained tau_l) / open circle (unconstrained)
ax.plot(fit_u[0], fit_u[1], 'o', color='blue',  ms=14,
        mec='black', mew=1.5, zorder=10)
ax.plot(fit_d[0], fit_d[1], 'o', color='green', ms=14,
        mec='black', mew=1.5, zorder=10)
ax.plot(fit_l_c[0], fit_l_c[1], 'o', color='red', ms=14,
        mec='black', mew=1.5, zorder=10)
ax.plot(fit_l_u[0], fit_l_u[1], 'o', color='red', mfc='none',
        ms=14, mew=2, zorder=9)

# 5. GVW closed-form solutions — open squares
ax.plot(gvw_u[0], gvw_u[1], 's', color='blue',  mfc='none',
        ms=14, mew=2, zorder=8)
ax.plot(gvw_d[0], gvw_d[1], 's', color='green', mfc='none',
        ms=14, mew=2, zorder=8)
ax.plot(gvw_l[0], gvw_l[1], 's', color='red',   mfc='none',
        ms=14, mew=2, zorder=8)

# 6. Arrow tau -> tau - 2 for the constrained tau_l (FD image)
ax.annotate('',
            xy=(-0.416, 1.552), xytext=(1.584, 1.552),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5,
                            ls="--", connectionstyle="arc3,rad=-0.25",
                            alpha=0.6),
            zorder=5)
ax.plot(-0.416, 1.552, 'o', color='red', mfc='white',
        mec='red', ms=9, mew=1.5, zorder=6, alpha=0.6)

# 7. Legend
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue',
           markeredgecolor='black', markersize=12,
           label=r'$\tau_u$ fit: $-0.158+2.385i$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green',
           markeredgecolor='black', markersize=12,
           label=r'$\tau_d$ fit: $0.369+1.541i$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
           markeredgecolor='black', markersize=12,
           label=r'$\tau_l$ (constr.) fit: $1.584+1.552i$'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
           markeredgecolor='red', markersize=12, markeredgewidth=2,
           label=r'$\tau_l$ (unconstr.) fit: $1.540+1.551i$'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
           markeredgecolor='blue', markersize=12, markeredgewidth=2,
           label=r'$\tau_u$ GVW: $(-50,1,3)$, dev 1.16%'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
           markeredgecolor='green', markersize=12, markeredgewidth=2,
           label=r'$\tau_d$ GVW: $(-28,-3,4)$, dev 0.37%'),
    Line2D([0], [0], marker='s', color='w', markerfacecolor='none',
           markeredgecolor='red', markersize=12, markeredgewidth=2,
           label=r'$\tau_l$ GVW: $(-48,-31,10)$, dev 0.47%'),
    Line2D([0], [0], marker='x', color='black', markersize=10,
           markeredgewidth=2, linestyle='None',
           label='Symmetric points'),
]
ax.legend(handles=legend_elements, loc='upper right', fontsize=10,
          framealpha=0.95, edgecolor='black',
          title='Moduli (fit and GVW) / symmetric points',
          title_fontsize=11)

# 8. Formatting
ax.set_xlim(-1.5, 3.0)
ax.set_ylim(0.0, 3.0)
ax.set_xlabel(r'Re$(\tau)$', fontsize=14)
ax.set_ylabel(r'Im$(\tau)$', fontsize=14)
ax.grid(True, alpha=0.1)

# 9. Save
pdf_path = os.path.join(out_dir, 'fig_modular_tau_space.pdf')
png_path = os.path.join(out_dir, 'fig_modular_tau_space.png')
fig.savefig(pdf_path, bbox_inches='tight')
fig.savefig(png_path, dpi=200, bbox_inches='tight')
print(f"[OK] saved: {pdf_path}")
print(f"[OK] saved: {png_path}")
plt.close(fig)
print("--- DONE ---")