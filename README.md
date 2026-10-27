# Model MEML 75: Mixed Geometry (v19.0)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23644888.svg)](https://doi.org/10.5281/zenodo.23644888)

This repository contains the manuscript, Python verification scripts, and numerical logs for **Model MEML 75 (v19.0)**. 

The model parametrizes the fermion sector of the Standard Model through the geometry of mixed six-dimensional orbifolds (\(T^2/\mathbb{Z}_3 \times T^2/\mathbb{Z}_4\)), incorporating axion-like dark matter (ALP) and leptogenesis.

## 📂 Repository Structure
*   `MEML_75_v19.0.tex` / `MEML_75_v19.0.pdf` — The final manuscript source and compiled PDF.
*   `scripts/` — All 16 Python scripts used for numerical fits, global scans, and verification.
*   `logs/` — Output logs corresponding to each script (proving the reproducibility of all results discussed in the paper).
*   `figures/` — PDF figures used in the manuscript.

## 🔬 Main Results
*   **Parameter Balance:** The model has **11 effective fitted parameters** (10 Lagrangian parameters + the ALP misalignment angle \(\theta_i\)) fitting 16 observables, yielding \(N_{\rm dof}^{\rm fit} = +5\) (assuming the moduli are fixed by GVW fluxes).
*   **ALP Dark Matter:** \(f_a = 1.9801 \times 10^{11}\) GeV, \(m_a = 28.407 \mu\)eV (frequency \(6.87\) GHz). Testable at future haloscope experiments (ADMX-EFR, HAYSTAC Phase III).
*   **Leptogenesis:** \(\eta_B^{\rm pre} \approx 1.9 \times 10^{-3}\), resolved by entropy dilution \(\Delta \approx 3.13 \times 10^6\) (treated as a phenomenological parameter).
*   **Testable Predictions:** \(\sum m_i \approx 0.068\) eV, \(m_{\beta\beta} < 8.7\) meV, and \(\delta_{CP} \approx 23^\circ\) or \(157^\circ\). The model also selects the upper octant of \(\theta_{23}\).

## ⚙️ Reproducibility

To reproduce the main results and verify the assertions in the manuscript, install the required dependencies:

```bash
pip install -r requirements.txt