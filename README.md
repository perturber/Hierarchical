# Hierarchical_BeyondvacGR
Decoupling local and global Beyond vacuum-GR effects in EMRIs with hierarchical modeling.

## Installation Guide

Dependencies:
- [`FastEMRIWaveforms(FEW)v2.0`](https://github.com/znasipak/FastEMRIWaveforms-Soton-Hackathon-2025/tree/Kerr_Equatorial_Eccentric). This branch is fully compatible with the default version, although [this](https://github.com/znasipak/FastEMRIWaveforms-Soton-Hackathon-2025/issues/55) issue needs fixing.
- [`StableEMRIFisher(SEF)`](https://github.com/perturber/StableEMRIFisher/tree/few2_package). This branch is fully compatible with the default version of the few2_package branch.
- [`LISAanalysistools(LAT)`](https://github.com/mikekatz04/LISAanalysistools/tree/main). LISA response util.
- [`lisa-on-gpu`](https://github.com/mikekatz04/lisa-on-gpu/tree/master). LISA response util.

To reproduce the results from the paper, download the datafiles [here](https://zenodo.org/records/15362412?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6IjY2N2RjZjkwLTdjYmYtNDEyMi05YjI2LWNiYTFjNTg0MzFiNSIsImRhdGEiOnt9LCJyYW5kb20iOiI0NzI1ODIxM2U1YWVlNjQ2ZTY0YjA3NjU1Njg1YTliMyJ9.FioeGIWlXePv3N0ozFbiWZOgCARcYeYx-J6y4Yy1DJ_xrVVVB5paCgbrXQBoyOj_Lpm7tl5zX-vjwelFDJkF5Q) and unzip contents. Execute notebooks in the *results* folder to replot everything. Additionally, the *execution* folder contains all files used to generate the populations and perform the analysis. Here's a brief description of the different files inside the *execution* folder:

- `hierarchical/Hierarchical_Class.py`: the main script file hosting the Hierarchical class. Given a cosmology/EMRI parametrization setup, this script calculates (i) the Fisher matrices at the true parameter point, (ii) the biased-inference points in the vacuum, local-only, and global-only hypotheses, and (iii) Bayes factors comparing the three hypothesis.
- `hierarchical/FisherValidation.py`: script file that can validate Fishers calculated at the true parameter point and the biased-inference point using KL-divergence. If KL is above a given threshold, the Fishers at the two points are assumed to be 'too different', breaking the linear-signal approximation. Validation can be performed from the Hierarchical class in Hierarchical_class.py.
- `hierarchical/utility.py`: various utility functions (cosmology, analysis) used by `Hierarchical_Class.py` and `FisherValidation.py`.
- `class_execution_Hierarchical.py`: execution file with fixed f and Gdot in the true population (single evaluation).
- `class-execution_varied_f.py`: execution file where the fractional EMRIs with a local effect, f, varies from 0 to 1. Gdot (global effect amplitude) is fixed to 1e-9.


