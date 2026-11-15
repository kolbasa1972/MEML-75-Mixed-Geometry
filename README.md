# MEML 75 (v24.5.5) - Collected Files

Reproducible bundle for the manuscript **Model MEML 75 (v24.5.5)**.

## Directory Structure

- `paper/` - LaTeX source and compiled PDF.
  - `MEML 75 v24.5.5.tex`
  - `MEML 75 v24.5.5.pdf`
- `figures/` - PDF and PNG plots used in the paper.
  - `fig_modular_tau_space.pdf/png` (Figure 1, Section 2)
  - `fig_alp_parameter_space.pdf/png` (Figure 2, Section 6)
  - `fig_theta23_delta.pdf/png` (Figure 3, Section 8)
- `scripts/` - Python scripts for all numerical checks.
  - Figure generators: `make_fig_modular_tau_space.py`,
    `make_fig_alp_parameter_space.py`, `plot_theta23_delta.py`.
  - Shared utility: `meml75_logutil.py` (`safe_log`).
  - **Master verifier:** `meml75_v24.5.1_verification.py`
  - **Audit (JSON-based):** `meml75_v24.5_audit.py`
  - **Quark sector:** `meml75_honest_pure2.py`, `meml75_close_tails.py`,
    `meml75_verify_Y4_convention.py`, `meml75_reconcile_twisted_vs_weight0_v2.py`,
    `meml75_a2_weight_audit_v2.py`, `meml75_quark_localization.py`,
    `meml75_quark_fractional_powers.py`, `meml75_magnetized_overlap2.py`,
    `meml75_flux_overlap_test.py`, `meml75_global_kahler_scan.py`,
    `meml75_kahler_scan_sqrt.py`.
  - **Lepton sector:** `meml75_tau_l_global_scan_v2.py`,
    `meml75_constrained_global_scan.py`, `meml75_constraint_test_extended2.py`,
    `meml75_delta_cp.py`, `meml75_s4prime_exact_fit.py`,
    `meml75_lepton_untested.py`, `meml75_lepton_untested_v2.py`.
  - **GVW & KKLT:** `meml75_flux_scan_cs.py`, `meml75_full_hessian.py`,
    `meml75_gvw_tau_closed_form_v2.py`, `meml75_kklt_scan.py`,
    `meml75_kklt_constrained.py`.
  - **ALP / WGC:** `meml75_wgc_check.py`.
  - **Leptogenesis / curvaton:** `meml75_leptogenesis.py`,
    `meml75_curvaton_dilution.py`, `meml75_curvaton_nongaussianity.py`,
    `meml75_curvaton_shift_symmetry.py`.
  - **Negative results:** `meml75_derive_cd.py`,
    `meml75_derive_cd_strict.py`, `meml75_derive_c1c2.py`,
    `meml75_derive_c1c2_strict.py`, `meml75_derive_final_attempt.py`,
    `meml75_stability_partB.py`.
- `scripts/*_log.txt` - authoritative text logs from each script.
- `scripts/*.json` - machine-readable mirrors where present.
- `A5-Modular-Exclusion/` - companion negative-result material.
  - `MEML 75-5 Collaboration.tex`

> Files from earlier iterations that are not part of this bundle and
> are not referenced by the manuscript: `meml75_audit_tables.py`,
> `meml75_check_scripts.py`, and hand-written mirror JSONs
> (`derived_v24.5.1.json`, etc.).

## Requirements

- Python 3.9+
- `numpy`, `scipy`, `matplotlib`

```bash
pip install -r requirements.txt
```

If `requirements.txt` is absent:

```bash
pip install numpy scipy matplotlib
```

## How to Reproduce

Run from the repository root unless noted otherwise. Each script writes
`scripts/<script>_log.txt` (and `scripts/*.json` where applicable).

### Figures

```bash
cd scripts
python make_fig_modular_tau_space.py     # -> ../figures/fig_modular_tau_space.*
python make_fig_alp_parameter_space.py   # -> ../figures/fig_alp_parameter_space.*
python plot_theta23_delta.py             # -> ../figures/fig_theta23_delta.*
cd ..
```

### Master verifier and audit

```bash
cd scripts
python meml75_v24.5.1_verification.py    # -> meml75_v24.5.1_verification_log.txt
python meml75_v24.5_audit.py             # -> meml75_v24.5_audit_log.txt / .json
```

The verifier checks the analytical numbers quoted in the paper:
`alpha_G`, `epsilon`, `m_KK`, `M_6`, `f_a`, `m_a`, `g_agamma`, the WGC
conditions, and the lepton/neutrino fits. The audit reproduces the
parameter-count table and the `m_bb` range.

### Sector-by-sector checks

```bash
# Quark
python meml75_honest_pure2.py            # chi2 = 13.93 (paper) / 29.50 (PDG 2024)
python meml75_close_tails.py
python meml75_verify_Y4_convention.py

# Lepton
python meml75_tau_l_global_scan_v2.py    # chi2 = 0.089
python meml75_constrained_global_scan.py # chi2 = 0.309 (c2+c3=0)
python meml75_delta_cp.py                # delta_CP = 22.9 / 157.1 deg

# GVW & KKLT
python meml75_flux_scan_cs.py
python meml75_full_hessian.py            # Hessian at SUSY points: SADDLE
python meml75_kklt_scan.py
python meml75_kklt_constrained.py        # dev = 9.25%
python meml75_gvw_tau_closed_form_v2.py

# ALP & WGC
python meml75_wgc_check.py

# Leptogenesis & curvaton
python meml75_leptogenesis.py
python meml75_curvaton_dilution.py
python meml75_curvaton_nongaussianity.py
python meml75_curvaton_shift_symmetry.py

# Negative results
python meml75_derive_cd.py
python meml75_derive_cd_strict.py
python meml75_derive_c1c2.py
python meml75_derive_c1c2_strict.py
python meml75_derive_final_attempt.py
python meml75_stability_partB.py
```

## Quick Reference

| Paper claim | Script | Output |
|---|---|---|
| Figure 1 (moduli) | `make_fig_modular_tau_space.py` | `figures/fig_modular_tau_space.*` |
| Figure 2 (ALP) | `make_fig_alp_parameter_space.py` | `figures/fig_alp_parameter_space.*` |
| Figure 3 (theta23/delta) | `plot_theta23_delta.py` | `figures/fig_theta23_delta.*` |
| Analytical ALP chain | `meml75_v24.5.1_verification.py` | `*_verification_log.txt` |
| Parameter count, `m_bb` | `meml75_v24.5_audit.py` | `*_audit_log.txt` / `.json` |
| Quark `chi2 = 13.93 / 29.50` | `meml75_honest_pure2.py` | `*_log.txt` |
| Lepton `chi2 = 0.089` | `meml75_tau_l_global_scan_v2.py` | `*_log.txt` |
| Lepton `chi2 = 0.309` | `meml75_constrained_global_scan.py` | `*_log.txt` |
| `delta_CP = 22.9 / 157.1 deg` | `meml75_delta_cp.py` | `*_log.txt` |
| GVW flux triplets | `meml75_flux_scan_cs.py` | `*_log.txt` |
| Hessian = SADDLE | `meml75_full_hessian.py` | `*_log.txt` |
| KKLT `dev = 9.25%` | `meml75_kklt_constrained.py` | `*_log.txt` |
| WGC consistency | `meml75_wgc_check.py` | `*_log.txt` |
| Leptogenesis `Delta ~ 1e6` | `meml75_leptogenesis.py` | `*_log.txt` |
| Negative results | `meml75_derive_*.py`, `meml75_stability_partB.py` | various |

## Notes

- The `v24.5.1` tag in `meml75_v24.5.1_verification.py` and `v24.5` in
  `meml75_v24.5_audit.py` are historical filenames; their output is
  intended for v24.5.5.
- `*_log.txt` files are authoritative; `.json` files (where present)
  are convenience mirrors.

## Related Publications

- **Model MEML 75 (v24.5.5)** - main model; see `paper/`.
- **Excluding Minimal A5 Modular Flavor Symmetry** - negative result
  with additional documented negatives; see `A5-Modular-Exclusion/`,
  including `MEML 75-5 Collaboration.tex`.