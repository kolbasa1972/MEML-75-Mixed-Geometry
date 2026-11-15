#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Figure 3: theta_23 vs delta_CP (v24.5.1)

Octant tension between NuFIT 5.0 (upper octant) and NuFIT 5.2 + SK atm
(lower octant), with the two MEML 75 branches
delta_CP = 22.9 deg and 157.1 deg (at theta_23 = 49.05 deg).

Numbers synced with MEML_75_v24.5.1.tex Section 8:
    theta_23 = 49.05 deg
    delta_CP = 22.9 deg  or  157.1 deg

Output directory: <script_dir>/../figures if it exists,
otherwise <script_dir>/figures.
"""
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

# --- Numbers from MEML 75 v24.5.1, Section 8 ---
THETA23 = 49.05
DELTA_1 = 22.9
DELTA_2 = 157.1

# --- Figure ---
fig, ax = plt.subplots(figsize=(8, 6))

# 1. Allowed theta_23 bands
ax.axvspan(48.0, 50.5, color='blue', alpha=0.2,
           label=r'NuFIT 5.0 (allowed, upper octant)')
ax.axvspan(41.0, 43.5, color='red', alpha=0.2,
           label=r'NuFIT 5.2 + SK atm (tension, lower octant)')

# 2. Octant boundary
ax.axvline(45, color='gray', linestyle='--', alpha=0.5)
ax.text(45.2, 0.68, 'Octant boundary',
        transform=ax.get_xaxis_transform(),
        color='gray', rotation=90, fontsize=10,
        ha='left', va='center')

# 3. MEML 75 branches
ax.axhline(DELTA_1, color='black', linestyle='-', lw=2, alpha=0.8,
           label=fr'MEML 75: $\delta_{{CP}} = {DELTA_1}^\circ$')
ax.axhline(DELTA_2, color='black', linestyle='-', lw=2, alpha=0.8,
           label=fr'MEML 75: $\delta_{{CP}} = {DELTA_2}^\circ$')

# 4. Prediction points (stars)
ax.scatter(THETA23, DELTA_1, color='black', marker='*', s=280, zorder=10)
ax.scatter(THETA23, DELTA_2, color='black', marker='*', s=280, zorder=10)

# 5. Text labels
ax.annotate(fr'$\theta_{{23}}={THETA23}^\circ$',
            xy=(THETA23, DELTA_1),
            xytext=(20, 25), textcoords='offset points',
            ha='left', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='white', edgecolor='none', alpha=0.9))
ax.annotate(fr'$\delta_{{CP}}={DELTA_2}^\circ$',
            xy=(THETA23, DELTA_2),
            xytext=(20, 20), textcoords='offset points',
            ha='left', va='bottom', fontsize=12,
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='white', edgecolor='none', alpha=0.9))

# 6. Formatting
ax.set_xlabel(r'$\theta_{23}$ [deg]', fontsize=14)
ax.set_ylabel(r'$\delta_{CP}$ [deg]', fontsize=14)
ax.set_xlim(38, 54)
ax.set_ylim(0, 360)
ax.set_yticks([0, 90, 180, 270, 360])
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
pdf_path = os.path.join(out_dir, 'fig_theta23_delta.pdf')
png_path = os.path.join(out_dir, 'fig_theta23_delta.png')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path, dpi=200, bbox_inches='tight')
print(f"[OK] saved: {pdf_path}")
print(f"[OK] saved: {png_path}")
plt.close(fig)
print("--- DONE ---")