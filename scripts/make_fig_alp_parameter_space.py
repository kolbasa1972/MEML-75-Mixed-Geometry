#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MEML 75 — Figure 2: ALP parameter space (v24.5.1)

Generates fig_alp_parameter_space.{pdf,png} showing:
  - m_a vs |g_a gamma gamma| in the range [10^-7, 10^-4] eV x [10^-17, 10^-12] GeV^-1
  - DFSZ and KSVZ lines, both normalised to pass through the
    MEML 75 prediction (m_a = 28.407 mu eV, g_a gamma gamma = +4.38e-15 GeV^-1)
  - current experimental bands: ADMX, HAYSTAC
  - projected future-haloscope band
  - MEML 75 prediction (red star)

Output directory: <script_dir>/../figures if it exists,
otherwise <script_dir>/figures.
"""
import os
import numpy as np
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

# --- Model prediction ---
m_a_model = 28.407e-6     # eV
g_model   = 4.38e-15      # GeV^-1

# --- Figure ---
fig, ax = plt.subplots(figsize=(10, 7))

# Mass scan for the coupling lines
m_a = np.logspace(-7, -4, 400)

# DFSZ / KSVZ lines, both anchored at the MEML 75 point
g_DFSZ = abs(g_model) * (m_a / m_a_model)
g_KSVZ = 1.126e-14   * (m_a / m_a_model)

# Experimental / projected bands (illustrative, constant-y spans)
ax.fill_between([1.9e-6, 4.2e-6], [2e-15, 2e-15], [2e-13, 2e-13],
                color='gray', alpha=0.3, label='ADMX (current)')
ax.fill_between([18e-6, 24e-6], [5e-16, 5e-16], [1e-13, 1e-13],
                color='orange', alpha=0.3, label='HAYSTAC (current)')
ax.fill_between([20e-6, 100e-6], [1e-16, 1e-16], [1e-14, 1e-14],
                color='purple', alpha=0.15, label='Future haloscopes')

# Coupling lines
ax.plot(m_a, g_DFSZ, 'b-', lw=2, label='DFSZ (through MEML 75)')
ax.plot(m_a, g_KSVZ, 'g-', lw=2, label='KSVZ (through MEML 75 m_a)')

# MEML 75 prediction
ax.scatter(m_a_model, g_model, color='red', marker='*', s=350,
           zorder=10, edgecolors='black', linewidths=1,
           label='MEML 75 prediction')
ax.annotate('MEML 75\n(28.4 $\\mu$eV, 4.38e-15 GeV$^{-1}$)',
            xy=(m_a_model, g_model), xytext=(15, -35),
            textcoords='offset points', fontsize=11,
            color='red', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white', edgecolor='red', alpha=0.9))

# --- Formatting ---
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel(r'ALP mass $m_a$ [eV]', fontsize=16)
ax.set_ylabel(r'Photon coupling $|g_{a\gamma\gamma}|$ [GeV$^{-1}$]',
              fontsize=16)
ax.set_xlim(1e-7, 1e-4)
ax.set_ylim(1e-17, 1e-12)
ax.grid(True, which='both', ls='--', alpha=0.5)
ax.legend(loc='upper left', fontsize=11, framealpha=0.95)

plt.tight_layout()
pdf_path = os.path.join(out_dir, 'fig_alp_parameter_space.pdf')
png_path = os.path.join(out_dir, 'fig_alp_parameter_space.png')
fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
fig.savefig(png_path, dpi=200, bbox_inches='tight')
print(f"[OK] saved: {pdf_path}")
print(f"[OK] saved: {png_path}")
plt.close(fig)
print("--- DONE ---")