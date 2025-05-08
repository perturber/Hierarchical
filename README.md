# Hierarchical_BeyondvacGR
Decoupling local and global Beyond vacuum-GR effects in EMRIs with hierarchical modeling.

## Installation Guide

Dependencies:
- [`FastEMRIWaveforms(FEW)`](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) (modified version with two beyond-vacuum-GR effects). Access will be provided upon request.
- [`StableEMRIFisher(SEF)`](https://github.com/perturber/StableEMRIFisher) Modified version for the hierarchical analysis is provided in the repo.
- [`LISAanalysistools(LAT)`](https://github.com/mikekatz04/LISAanalysistools/tree/main). Specific version of this open-source tool is made available as a zip file.
- [`lisa-on-gpu`](https://github.com/mikekatz04/lisa-on-gpu/tree/master). Specific version of this open-source tool is made available as a zip file.

Download the zip file and unzip contents. Here's a brief description of the different files:

- `Hierarchical_Class.py`: the main script file hosting the Hierarchical class. Given a cosmology/EMRI parametrization setup, this script calculates (i) the Fisher matrices at the true parameter point, (ii) the biased-inference points in the vacuum, local-only, and global-only hypotheses, and (iii) Bayes factors comparing the three hypothesis.
- `FisherValidation.py`: script file that can validate Fishers calculated at the true parameter point and the biased-inference point using KL-divergence. If KL is above a given threshold, the Fishers at the two points are assumed to be 'too different', breaking the linear-signal approximation. Validation can be performed from the Hierarchical class in Hierarchical_class.py.
- `utility.py`: various utility functions (cosmology, analysis) used by `Hierarchical_Class.py` and `FisherValidation.py`.
- `class_execution_HPC.py`: execution file for a single full run with the true population being vacuum-GR consistent as an example.
- `class-execution_varied_f.py`: execution file where the fractional EMRIs with a local effect, f, varies from 0 to 1. Gdot (global effect amplitude) is fixed to 1e-9.
