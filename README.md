# Hierarchical_BeyondvacGR
Decoupling local and global Beyond vacuum-GR effects in EMRIs with hierarchical modeling.

## Installation Guide

Dependencies:
- [FastEMRIWaveforms](https://github.com/BlackHolePerturbationToolkit/FastEMRIWaveforms) (modified version with two beyond-vacuum-GR effects). Access will be provided upon request.

Download the zip file and unzip contents. Here's a brief description of the different files:

- Hierarchical_Class.py: the main script file hosting the Hierarchical class. Given a cosmology/EMRI parametrization setup, this script calculates (i) the Fisher matrices at the true parameter point, (ii) the biased-inference points in the vacuum, local-only, and global-only hypotheses, and (iii) Bayes factors comparing the three hypothesis.
- FisherValidation.py: script file that can validate Fishers calculated at the true parameter point and the biased-inference point using KL-divergence. If KL is above a given threshold, the Fishers at the two points are assumed to be 'too different', breaking the linear-signal approximation. Validation can be performed from the Hierarchical class in Hierarchical_class.py. 
- class_execution_HPC.py: execution file for a single full run with the true population being vacuum-GR consistent as an example.
- class-execution_varied_f.py: execution file where the fractional EMRIs with a local effect, f, is allowed to vary from 0 to 1. Gdot (global effect amplitude) is fixed to 1e-9.
