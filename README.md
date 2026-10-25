[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23638606.svg)](https://doi.org/10.5281/zenodo.23638606)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![Verification](https://img.shields.io/badge/verification-ALL%20SECTIONS%20COMPLETE-brightgreen)]()

# MEML 75 — Mixed Geometry (v16.0)

Model of quark and lepton masses from $T^2/\mathbb{Z}_3 \times T^2/\mathbb{Z}_4$ geometry, with axion-like dark matter and leptogenesis.

## Status

- **v13.0** (quarks + ALP): submitted to Nuclear Physics B, NPB-D-26-02519.
- **v16.0** (full model): ready for submission to Physical Review D.

## Key Physics Results

- **18 effective parameters** (17 Lagrangian + ALP misalignment angle $\theta_i$) fitting **16 observables**.
- $N_{\rm dof}^{\rm fit} = -2$, $N_{\rm dof}^{\rm total} = +1$.
- **3 Conditional Predictions:** $\delta_{CP} \approx 23^\circ$ or $157^\circ$, $\sum m_i \approx 0.068$ eV, and $m_{\beta\beta} \approx 8.37$ meV (range $0$–$8.7$ meV, entirely below KamLAND-Zen).
- **ALP Dark Matter:** $f_a = 1.9801 \times 10^{11}$ GeV, $m_a = 28.407~\mu$eV, $\nu_a = 6.8688$ GHz. Testable at ADMX/HAYSTAC.
- **5 Honest Negative Results:** Explicitly detailed in Section 14.

## Files

| File | Purpose |
|---|---|
| `MEML_75_v16.0.tex` | Manuscript source (LaTeX) |
| `MEML_75_v16.0.pdf` | Compiled manuscript |
| `meml75_v16.0_master.py` | Main verification script (sections A–F) |
| `meml75_constraint_test_extended.py` | Test of 7 residual symmetries |
| `meml75_tau_l_global_scan_v2.py` | Unconstrained global scan |
| `meml75_constrained_global_scan.py` | Constrained global scan |
| `master_verification_log.txt` | Output of `master.py` (`ALL SECTIONS COMPLETE`) |
| `constraint_test_log.txt` | Output of `constraint_test_extended.py` (`1 PASS, 6 FAIL`) |
| `global_scan_log.txt` | Output of `tau_l_global_scan_v2.py` |

## Reproducing the results

Ensure you have `numpy` and `scipy` installed (see `requirements.txt`), then run:

```bash
python meml75_v16.0_master.py
python meml75_constraint_test_extended.py
python meml75_tau_l_global_scan_v2.py
python meml75_constrained_global_scan.py
