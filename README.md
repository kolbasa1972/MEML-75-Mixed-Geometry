# MEML 75 — Mixed Geometry (v16.02)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23644888.svg)](https://doi.org/10.5281/zenodo.23644888)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**MEML 75** is a phenomenological model that parametrizes the Standard Model fermion sector through the geometry of mixed six-dimensional orbifolds. Quarks are localized on a T²/ℤ₃ torus (employing A₄ modular forms for the hierarchical mass matrix), while leptons are localized on a T²/ℤ₄ torus (employing S₄' modular forms for the mixing pattern). The dual-orbifold approach correlates quark masses, lepton mixing, axion-like dark matter, and leptogenesis within a single geometric framework.

## Status

- **v13.0** (quarks + ALP): submitted to Nuclear Physics B, NPB-D-26-02519.
- **v16.0** (full model): current release, including EDM, LFV, and a negative result on the geometric derivation of lepton Yukawas.

## Key Predictions & Testable Consequences

While the model fits 16 observables with 18 effective parameters (acknowledging overfitting with N_dof_fit = -2), it makes several sharp, falsifiable predictions:

- **Axion-like Dark Matter (ALP)**: The model predicts an ALP with f_a = 1.98 × 10¹¹ GeV, corresponding to a mass of **mₐ = 28.4 μeV** and a frequency of **νₐ = 6.87 GHz**. This is testable at future high-frequency haloscope upgrades (e.g., ADMX-EFR or HAYSTAC Phase III).
- **Neutrino Mass Sum**: Σmᵢ ≈ 0.068 eV.
- **Neutrinoless Double Beta Decay**: The effective Majorana mass is predicted to be m_ββ ≈ 8.37 meV (assuming vanishing Majorana phases; range 0–8.7 meV for arbitrary phases). This lies below the current KamLAND-Zen limit and is a prime target for next-generation experiments like nEXO and LEGEND-1000.
- **CP Violation**: The Dirac CP phase is predicted to be δ_CP ≈ 23° or 157°.
- **Atmospheric Octant**: The model selects the upper octant for θ₂₃ (≈ 49.04°). While in mild tension with recent SK atmospheric data, this is a definitive, testable consequence that will be settled by DUNE and Hyper-Kamiokande.

## Files

| File | Purpose |
|---|---|
| `MEML_75_v16.0.tex` | Manuscript source (LaTeX) |
| `MEML_75_v16.0.pdf` | Compiled manuscript |
| `meml75_v16.0_master.py` | Main verification script (sections A–F) |
| `meml75_constraint_test_extended.py` | Test of 7 residual symmetries in the lepton sector |
| `meml75_tau_l_global_scan_v2.py` | Unconstrained global scan for the lepton modulus |
| `meml75_constrained_global_scan.py` | Constrained global scan (c2 + c3 = 0) |
| `master_verification_log.txt` | Output of `master.py` (`ALL SECTIONS COMPLETE`) |
| `constraint_test_log.txt` | Output of `constraint_test_extended.py` (`1 PASS, 6 FAIL`) |
| `global_scan_log.txt` | Output of `tau_l_global_scan_v2.py` (`UNIQUE minimum`) |
| `constrained_scan_log.txt` | Output of `constrained_global_scan.py` (`c = (-6.59, -17.40, +17.40)`) |

## Reproducing the Results

Ensure you have the required dependencies (NumPy, SciPy) installed.

```bash
pip install -r requirements.txt
python meml75_v16.0_master.py
python meml75_constraint_test_extended.py
python meml75_tau_l_global_scan_v2.py
python meml75_constrained_global_scan.py
