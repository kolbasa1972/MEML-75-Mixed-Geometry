# MEML 75 (v24.5.5) — Collected Files

This directory contains the complete, reproducible bundle for the
manuscript **Model MEML 75 (v24.5.5)**.

## Directory Structure

- `paper/` — LaTeX source and compiled PDF.
  - `MEML 75 v24.5.5.tex` (main article, includes 3 figures)
  - `MEML 75 v24.5.5.pdf`
- `figures/` — PDF and PNG plots used in the paper.
  - `fig_modular_tau_space.pdf/png` (Figure 1, Section 2)
  - `fig_alp_parameter_space.pdf/png` (Figure 2, Section 6)
  - `fig_theta23_delta.pdf/png` (Figure 3, Section 8)
- `scripts/` — Python scripts for all numerical checks.
  - `make_fig_*.py`, `lot_theta23_delta.py` (generate the figures)
  - `meml75_logutil.py` (shared `safe_log` utility)
  - **Master verifier:** `meml75_v24.5.1_verification.py`
  - **Audit (JSON-based):** `meml75_v24.5_audit.py`
  - **Quark sector:** `meml75_honest_pure2.py`, `meml75_close_tails.py`, etc.
  - **Lepton sector:** `meml75_tau_l_global_scan_v2.py`, `meml75_constrained_global_scan.py`, etc.
  - **GVW & KKLT:** `meml75_flux_scan_cs.py`, `meml75_full_hessian.py`, `meml75_kklt_scan.py`, etc.
  - **ALP / WGC:** `meml75_wgc_check.py`
  - **Leptogenesis / curvaton:** `meml75_leptogenesis.py`, `meml75_curvaton_dilution.py`, etc.
  - **Negative results:** `meml75_derive_cd.py`, `meml75_derive_c1c2.py`, etc.
- `scripts/*_log.txt` — text logs produced by each script.
- `scripts/*.json` — machine-readable results.

> **Note on removed files.** The following files from earlier
> iterations are not part of this bundle and are not referenced by
> the manuscript:
> `meml75_audit_tables.py`, `meml75_check_scripts.py`, 
> and hand-written mirror JSONs (`derived_v24.5.1.json`, etc.).

## How to Reproduce

1. Install requirements: `pip install -r requirements.txt`
2. Generate figures: