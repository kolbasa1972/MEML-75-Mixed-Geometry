# MEML 75 (v24.5.5) - Collected Files
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24252252.svg)](https://doi.org/10.5281/zenodo.24252252) [![Version](https://img.shields.io/badge/version-v24.6.0-blue)](https://zenodo.org/record/24252252)

Reproducible bundle for the manuscript Model MEML 75 (v24.5.5).

## Directory Structure

- paper/
  - MEML 75 v24.5.5.tex
  - MEML 75 v24.5.5.pdf
- figures/
  - fig_modular_tau_space.pdf/png (Figure 1, Section 2)
  - fig_alp_parameter_space.pdf/png (Figure 2, Section 6)
  - fig_theta23_delta.pdf/png (Figure 3, Section 8)
- scripts/
  - make_fig_modular_tau_space.py
  - make_fig_alp_parameter_space.py
  - plot_theta23_delta.py
  - meml75_logutil.py
  - meml75_v24.5.1_verification.py (Master verifier)
  - meml75_v24.5_audit.py (Audit)
  - meml75_honest_pure2.py, meml75_close_tails.py, meml75_verify_Y4_convention.py, meml75_reconcile_twisted_vs_weight0_v2.py, meml75_a2_weight_audit_v2.py, meml75_quark_localization.py, meml75_quark_fractional_powers.py, meml75_magnetized_overlap2.py, meml75_flux_overlap_test.py, meml75_global_kahler_scan.py, meml75_kahler_scan_sqrt.py
  - meml75_tau_l_global_scan_v2.py, meml75_constrained_global_scan.py, meml75_constraint_test_extended2.py, meml75_delta_cp.py, meml75_s4prime_exact_fit.py, meml75_lepton_untested.py, meml75_lepton_untested_v2.py
  - meml75_flux_scan_cs.py, meml75_full_hessian.py, meml75_gvw_tau_closed_form_v2.py, meml75_kklt_scan.py, meml75_kklt_constrained.py
  - meml75_wgc_check.py
  - meml75_leptogenesis.py, meml75_curvaton_dilution.py, meml75_curvaton_nongaussianity.py, meml75_curvaton_shift_symmetry.py
  - meml75_derive_cd.py, meml75_derive_cd_strict.py, meml75_derive_c1c2.py, meml75_derive_c1c2_strict.py, meml75_derive_final_attempt.py, meml75_stability_partB.py
- scripts/logs/
  - *_log.txt (authoritative text logs)
  - *.json (machine-readable mirrors)
- A5-Modular-Exclusion/
  - MEML 75-5 Collaboration.tex

> Files from earlier iterations that are not part of this bundle: meml75_audit_tables.py, meml75_check_scripts.py, and hand-written mirror JSONs.

## Requirements

Python 3.9+
numpy, scipy, matplotlib

Install with:
pip install -r requirements.txt

## How to Reproduce

Run from the repository root unless noted otherwise.

### Figures
cd scripts
python make_fig_modular_tau_space.py
python make_fig_alp_parameter_space.py
python plot_theta23_delta.py
cd ..

### Master verifier and audit
cd scripts
python meml75_v24.5.1_verification.py
python meml75_v24.5_audit.py

### Sector-by-sector checks
Quark:
python meml75_honest_pure2.py
python meml75_close_tails.py
python meml75_verify_Y4_convention.py

Lepton:
python meml75_tau_l_global_scan_v2.py
python meml75_constrained_global_scan.py
python meml75_delta_cp.py

GVW and KKLT:
python meml75_flux_scan_cs.py
python meml75_full_hessian.py
python meml75_kklt_scan.py
python meml75_kklt_constrained.py
python meml75_gvw_tau_closed_form_v2.py

ALP and WGC:
python meml75_wgc_check.py

Leptogenesis and curvaton:
python meml75_leptogenesis.py
python meml75_curvaton_dilution.py
python meml75_curvaton_nongaussianity.py
python meml75_curvaton_shift_symmetry.py

Negative results:
python meml75_derive_cd.py
python meml75_derive_cd_strict.py
python meml75_derive_c1c2.py
python meml75_derive_c1c2_strict.py
python meml75_derive_final_attempt.py
python meml75_stability_partB.py

## Notes

- The v24.5.1 tag in meml75_v24.5.1_verification.py and v24.5 in meml75_v24.5_audit.py are historical filenames; their output is intended for v24.5.5.
- Log files are authoritative; JSON files are convenience mirrors.

## Related Publications

- Model MEML 75 (v24.5.5) - main model; see paper/.
- Excluding Minimal A5 Modular Flavor Symmetry - negative result with additional documented negatives; see A5-Modular-Exclusion/, including MEML 75-5 Collaboration.tex.
