#imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cupy as cp
import pickle
import corner

from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import inner_product, generate_PSD, padding

from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform, Joint_RelativisticKerrCircularFlux
from few.summation.aakwave import AAKSummation
from fastlisaresponse import ResponseWrapper  # Response function 
from few.utils.constants import YRSID_SI, C_SI

from scipy.integrate import quad, nquad
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.stats import uniform
from scipy.special import factorial
from scipy.optimize import root

use_gpu = True
from scipy.stats import multivariate_normal

from Hierarchical_Class import Hierarchical


T_LISA = 1. #LISA observation duration
dt = 10.0 #sampling rate

insp_kwargs = { "err": 1e-13,
                "DENSE_STEPPING": 0,
                "max_init_len": int(1e8),
               "func":"Joint_Relativistic_Kerr_Circ_Flux",
               "use_rk4":False,
                }

sum_kwargs = {
    "use_gpu": use_gpu,  # GPU is availabel for this type of summation
    "pad_output": True, # True if expecting waveforms smaller than LISA observation window.
}

Waveform_model = GenerateEMRIWaveform(
            Joint_RelativisticKerrCircularFlux,
            inspiral_kwargs=insp_kwargs,
            sum_kwargs=sum_kwargs,
            use_gpu=use_gpu,
            return_list=False,
            frame="detector"
            )

orbit_file_esa = "/home/shubham/FEW_KerrEcc/Github_Repos/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
# orbit_file_esa = "/data/lsperi/lisa-on-gpu/orbit_files/equalarmlength-trailing-fit.h5"
orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

tdi_gen ="1st generation"# "2nd generation"#

order = 20  # interpolation order (should not change the result too much)
tdi_kwargs_esa = dict(
    orbit_kwargs=orbit_kwargs_esa, order=order, tdi=tdi_gen, tdi_chan="AE",
)  # could do "AET"

index_lambda = 8
index_beta = 7

# with longer signals we care less about this
t0 = 10000.0  # throw away on both ends when our orbital information is weird

EMRI_TDI = ResponseWrapper(
                        Waveform_model,
                        T_LISA,
                        t0=t0,
                        dt=dt,
                        index_lambda=index_lambda,
                        index_beta=index_beta,
                        flip_hx=True,  # set to True if waveform is h+ - ihx (FEW is)
                        use_gpu=use_gpu,
                        is_ecliptic_latitude=False,  # False if using polar angle (theta)
                        remove_garbage="zero",  # removes the beginning of the signal that has bad information
                        **tdi_kwargs_esa,
                        )

#cosmological parameters
cosmo_params={'Omega_m0':0.30,'Omega_Lambda0':0.70,'H0':70e3}

#Mstar normalization term for the EMRI MBH mass distribution
Mstar = 3e6

#True size of the population
Npop = int(30)

#detection SNR threshold
SNR_thresh = 20.0

#true values of population hyperparameters.
true_hyper={'K':5e-3,'alpha':0.,'beta':0., #vacuum hyperparameters
            'f':0.0,'mu_Al':1e-5,'mu_nl':8.0,'sigma_Al':1e-6,'sigma_nl':1.0, #local effect hyper
            'Gdot':0.0 #global effect hyper
           }

#range of source parameters
source_bounds={'M':[1e5,1e7],'z':[0.01,1.0], #vacuum parameters
               'Al':[0.0,1e-4],'nl':[0.0,10.0], #local effect parameters
               'Ag':[0.0,1e-8] #global effect parameters
              }

#range of population hyperparameters
hyper_bounds={'K':[1e-3,1e-2],'alpha':[-0.5,0.5],'beta':[-0.5,0.5], #vacuum hyperparameters
             'f':[0.0,1.0],'mu_Al':[1e-5,1e-5],'mu_nl':[8.0,8.0],'sigma_Al':[1e-6,1e-6],'sigma_nl':[1.0,1.0],#local effect hyper
             'Gdot':[0.0,1e-8] #global effect hyper
             }
             
filename = f'Hierarchical_Npop_{Npop}_f_{true_hyper['f']}_Gdot_{true_hyper['Gdot']}_K_{true_hyper['K']}_alpha_{true_hyper['alpha']}_beta_{true_hyper['beta']}' #folder with all the analysis data and plots
#filename = 'test_Gdot_1e-9' #folder with all the analysis data and plots
filename_Fishers = 'Fishers' #subfolder with all the Fisher matrices

#setting up kwargs to pass to StableEMRIFishers class
sef_kwargs = {'EMRI_waveform_gen':EMRI_TDI, #EMRI waveform model with TDI response
              'param_names': ['M','dist','A_l','n_l','A_g'], #params to be varied
              'der_order':4, #derivative order
              'Ndelta':12, #number of stable points
              'stats_for_nerds': False, #true if you wanna print debugging info
              'stability_plot': False, #true if you wanna plot stability surfaces
              'use_gpu':use_gpu,
              #'filename': filename_Fishers, #filename will be added inside the class definition
              #'suffix':i #suffix for ith EMRI source will be added inside class definition
              'interpolation_factor':1}

hier = Hierarchical(Npop=Npop,SNR_thresh=SNR_thresh,sef_kwargs=sef_kwargs,
                    filename=filename,filename_Fishers=filename_Fishers,
                    cosmo_params=cosmo_params,true_hyper=true_hyper,
                    source_bounds=source_bounds,hyper_bounds=hyper_bounds,Mstar=Mstar,
                    T_LISA=T_LISA,make_nice_plots=True,M_random=int(1e3))

hier()
