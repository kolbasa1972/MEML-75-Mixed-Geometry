# MEML 75 — Mixed Geometry (v16.0)

Model of quark and lepton masses from \(T^2/\mathbb{Z}_3 \times T^2/\mathbb{Z}_4\) geometry, with axion-like dark matter and leptogenesis.

## Status

- **v13.0** (quarks + ALP): submitted to Nuclear Physics B, NPB-D-26-02519.
- **v16.0** (full model): ready for submission to Physical Review D.

## Files

| File | Purpose |
|---|---|
| `MEML_75_v16.0.tex` | Manuscript source |
| `meml75_v16.0_master.py` | Main verification script (sections A–F) |
| `meml75_constraint_test_extended.py` | Test of 7 residual symmetries |
| `meml75_tau_l_global_scan_v2.py` | Unconstrained global scan |
| `meml75_constrained_global_scan.py` | Constrained global scan |
| `master_verification_log.txt` | Output of `master.py` (`ALL SECTIONS COMPLETE`) |
| `constraint_test_log.txt` | Output of `constraint_test_extended.py` (`1 PASS, 6 FAIL`) |
| `global_scan_log.txt` | Output of `tau_l_global_scan_v2.py` |

## Reproducing the results

```bash
pip install -r requirements.txt
python meml75_v16.0_master.py
python meml75_constraint_test_extended.py
python meml75_tau_l_global_scan_v2.py
python meml75_constrained_global_scan.py