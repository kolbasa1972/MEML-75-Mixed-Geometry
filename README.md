# MEML 75 — Mixed Geometry (v17.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23439547.svg)](https://doi.org/10.5281/zenodo.23439547)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

**MEML 75** is a phenomenological model that parametrizes the Standard Model fermion sector through the geometry of mixed six-dimensional orbifolds. Quarks are localized on a T²/ℤ₃ torus (employing A₄ modular forms), while leptons are localized on a T²/ℤ₄ torus (employing S₄' modular forms). 

In version 17.0, we demonstrate that the three complex moduli (\(\tau_u, \tau_d, \tau_l\)) are not free parameters, but are topologically stabilized by discrete GVW fluxes. This reduces the number of effective free parameters from 18 to **12**, yielding \(N_{\rm dof}^{\rm fit} = +4\).

## Key Predictions & Testable Consequences

While the model fits 16 observables, it makes several sharp, falsifiable predictions:

- **Axion-like Dark Matter (ALP)**: The model predicts an ALP with \(f_a = 1.98 \times 10^{11}\) GeV, corresponding to a mass of **\(m_a = 28.4 \ \mu\)eV** and a frequency of **\(\nu_a = 6.87\) GHz**. This is testable at future high-frequency haloscope upgrades (ADMX-EFR, HAYSTAC Phase III).
- **Neutrino Mass Sum**: \(\sum m_i \approx 0.068\) eV.
- **Neutrinoless Double Beta Decay**: \(m_{\beta\beta} \approx 8.37\) meV (range \(0-8.7\) meV for arbitrary Majorana phases). This lies below the current KamLAND-Zen limit and is a prime target for nEXO and LEGEND-1000.
- **CP Violation**: The Dirac CP phase is predicted to be \(\delta_{CP} \approx 23^\circ\) or \(157^\circ\).
- **Atmospheric Octant**: The model selects the upper octant for \(\theta_{23}\) (\(\approx 49.04^\circ\)), a definitive testable consequence for DUNE and Hyper-Kamiokande.

## Files

| File | Purpose |
|---|---|
| `MEML_Article/` | Folder containing the manuscript source (`MEML_75_v17.0.tex`), PDF, and `figures/`. |
| `meml75_v17.0_master.py` | Main verification script (sections A–F) |
| `constraint_test_extended2.py` | Test of 7 residual symmetries and stress test |
| `meml75_tau_l_global_scan_v2.py` | Unconstrained global scan for the lepton modulus |
| `meml75_constrained_global_scan.py` | Constrained global scan (\(c_2 + c_3 = 0\)) |
| `meml75_flux_scan_final.py` | GVW flux scan for topological stabilization |
| `meml75_j_invariant.py` | \(j\)-invariant check for algebraic points |
| `meml75_flux_conventions.py` | Kähler convention checks for fluxes |
| `master_verification_log.txt` | Output of `master.py` (`ALL SECTIONS COMPLETE`) |
| `constraint_test_log.txt` | Output of `constraint_test_extended2.py` (`1 PASS, 6 FAIL`) |
| `global_scan_log.txt` | Output of `tau_l_global_scan_v2.py` (`UNIQUE minimum`) |
| `constrained_scan_log.txt` | Output of `constrained_global_scan.py` (`c = (-6.59, -17.40, +17.40)`) |
| `flux_scan_log.txt` | Output of `flux_scan_final.py` (`|D_tau W| < 0.25`) |
| `j_invariant_log.txt` | Output of `j_invariant.py` (generic points) |
| `flux_conventions_log.txt` | Output of `flux_conventions.py` |

## Reproducing the Results

Ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt