#imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import cupy as cp
import pickle
import corner
import time

from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import inner_product, generate_PSD, padding

from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform, Joint_RelativisticKerrCircularFlux
from few.summation.aakwave import AAKSummation
from few.utils.constants import YRSID_SI, C_SI

from fastlisaresponse import ResponseWrapper  # Response function 
from lisatools.detector import ESAOrbits #ESAOrbits correspond to esa-trailing-orbits.h5

from scipy.integrate import quad, nquad
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.stats import uniform
from scipy.special import factorial
from scipy.optimize import brentq, root

use_gpu = True
from scipy.stats import multivariate_normal
import warnings

from Hierarchical_Class import getdistGpc, check_prior

#waveform setup

T_LISA = 1. #LISA observation duration
dt = 10.0 #sampling rate

insp_kwargs = { "err": 1e-10,
                "DENSE_STEPPING": 0,
                "max_init_len": int(1e8),
               "func":"Joint_Relativistic_Kerr_Circ_Flux",
               "use_rk4":True,
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

#orbit_file_esa = "/home/shubham/FEW_KerrEcc/Github_Repos/lisa-on-gpu/orbit_files/esa-trailing-orbits.h5"
# orbit_file_esa = "/data/lsperi/lisa-on-gpu/orbit_files/equalarmlength-trailing-fit.h5"
#orbit_kwargs_esa = dict(orbit_file=orbit_file_esa)

tdi_gen ="1st generation"# "2nd generation"#

order = 20  # interpolation order (should not change the result too much)
tdi_kwargs_esa = dict(
    orbits=ESAOrbits(use_gpu=use_gpu), order=order, tdi=tdi_gen, tdi_chan="AE",
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
              
#initialize
Npop = int(1e3)

Omega_m0 = 0.30
Omega_Lambda0 = 0.70
H0 = 70e3

#true values of population hyperparameters.
true_hyper={'K':5e-3,'alpha':0.2,'beta':0.2, #vacuum hyperparameters
            'f':0.5,'mu_Al':1e-5,'mu_nl':8.0,'sigma_Al':1e-6,'sigma_nl':1.0, #local effect hyper
            'Gdot':1e-9 #global effect hyper
           }

#prior bounds on source parameters
source_bounds={'M':[1e5,1e7],'z':[0.01,1.0], #vacuum parameters
               'Al':[0.0,1e-4],'nl':[-20.0,20.0], #local effect parameters
               'Ag':[-1e-8,1e-8] #global effect parameters
              }

#prior bounds on population hyperparameters
hyper_bounds={'K':[1e-3,1e-2],'alpha':[-0.5,0.5],'beta':[-0.5,0.5], #vacuum hyperparameters
             'f':[0.0,1.0],'mu_Al':[true_hyper['mu_Al']*0.9,true_hyper['mu_Al']*1.1],'mu_nl':[true_hyper['mu_nl']*0.9,true_hyper['mu_nl']*1.1],
              'sigma_Al':[true_hyper['sigma_Al']*1e-1,true_hyper['sigma_Al']*1e2],
              'sigma_nl':[true_hyper['sigma_nl']*1e-1,true_hyper['sigma_nl']*1e2],#local effect hyper
             'Gdot':source_bounds['Ag'] #global effect hyper
             }

filename = f'Hierarchical_Npop_{Npop}_f_{true_hyper['f']}_Gdot_{true_hyper['Gdot']}_K_{true_hyper['K']}_alpha_{true_hyper['alpha']}_beta_{true_hyper['beta']}' #folder with all the analysis data and plots
#filename = 'test_Tplunge'
filename_Fishers = 'Fishers' #subfolder with all the Fisher matrices
filename_Fishers_loc = 'Fishers_loc' #subfolder with inferred FIMs in local hypothesis
filename_Fishers_glob = 'Fishers_glob' #subfolder with inferred FIMs in global hypothesis

filename_Fishers = os.path.join(filename,filename_Fishers)
filename_Fishers_loc = os.path.join(filename,filename_Fishers_loc)
filename_Fishers_glob = os.path.join(filename,filename_Fishers_glob)

#imports
detected_EMRIs = np.load(f'{filename}/detected_EMRIs.npy',allow_pickle=True)
print(detected_EMRIs[0])
Fishers_injected_all = []
for i in range(len(detected_EMRIs)):
    index = int(detected_EMRIs[i]["index"])
    Fishers_injected_all.append(np.load(f"{filename_Fishers}/Fisher_transformed_{index}.npy"))
    
#calculate Fishers at the biased point in the local hypothesis
if true_hyper['f'] > 0.0:
    #if fraction of local EMRIs > 0, calculate FIM at biased point in local hypothesis
    for i in tqdm(range(len(detected_EMRIs))):
        M = np.exp(detected_EMRIs[i]['local_params'][0])
        mu = detected_EMRIs[i]['true_params'][1]
        a = detected_EMRIs[i]['true_params'][2]
        p0 = detected_EMRIs[i]['true_params'][3]
        e0 = detected_EMRIs[i]['true_params'][4]
        Y0 = detected_EMRIs[i]['true_params'][5]
        dL = getdistGpc(detected_EMRIs[i]['local_params'][1],H0=H0,Omega_m0=Omega_m0,Omega_Lambda0=Omega_Lambda0)
        qS = detected_EMRIs[i]['true_params'][7]
        phiS = detected_EMRIs[i]['true_params'][8]
        qK = detected_EMRIs[i]['true_params'][9]
        phiK = detected_EMRIs[i]['true_params'][10]
        Phi_phi0 = detected_EMRIs[i]['true_params'][11]
        Phi_theta0 = detected_EMRIs[i]['true_params'][12]
        Phi_r0 = detected_EMRIs[i]['true_params'][13]
        Al = detected_EMRIs[i]['local_params'][2]
        nl = detected_EMRIs[i]['local_params'][3]
        Ag = detected_EMRIs[i]['local_params'][4] #will be zero in local hypothesis
        ng = 4.0

        if check_prior(Al,source_bounds['Al']) != 0.:
            continue #ignore inferred EMRIs beyond the source bounds.
        if check_prior(nl,source_bounds['nl']) != 0.:
            continue #ignore inferred EMRIs beyond the source bounds.

        T = T_LISA
        dt = dt

        emri_kwargs = {'T': T, 'dt': dt}

        param_list = [M,mu,a,p0,e0,Y0,
                      dL,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,
                      Al,nl,Ag,ng]

        sef_kwargs['filename'] = filename_Fishers_loc
        sef_kwargs['suffix'] = detected_EMRIs[i]['index']

        sef = StableEMRIFisher(*param_list, **emri_kwargs, **sef_kwargs)
        sef()
        
#calculate Fishers at the biased point in the global hypothesis
if true_hyper['Gdot'] > 0.0:
    #if Gdot > 0, calculate FIM at biased point in global hypothesis
    for i in tqdm(range(len(detected_EMRIs))):
        M = np.exp(detected_EMRIs[i]['global_params'][0])
        mu = detected_EMRIs[i]['true_params'][1]
        a = detected_EMRIs[i]['true_params'][2]
        p0 = detected_EMRIs[i]['true_params'][3]
        e0 = detected_EMRIs[i]['true_params'][4]
        Y0 = detected_EMRIs[i]['true_params'][5]
        dL = getdistGpc(detected_EMRIs[i]['global_params'][1],H0=H0,Omega_m0=Omega_m0,Omega_Lambda0=Omega_Lambda0)
        qS = detected_EMRIs[i]['true_params'][7]
        phiS = detected_EMRIs[i]['true_params'][8]
        qK = detected_EMRIs[i]['true_params'][9]
        phiK = detected_EMRIs[i]['true_params'][10]
        Phi_phi0 = detected_EMRIs[i]['true_params'][11]
        Phi_theta0 = detected_EMRIs[i]['true_params'][12]
        Phi_r0 = detected_EMRIs[i]['true_params'][13]
        Al = detected_EMRIs[i]['global_params'][2] #will be zero in global hypothesis
        nl = detected_EMRIs[i]['global_params'][3] #will be zero in global hypothesis
        Ag = detected_EMRIs[i]['global_params'][4]
        ng = 4.0

        if check_prior(Ag,source_bounds['Ag']) != 0.:
            continue #ignore inferred EMRIs beyond the source bounds.

        T = T_LISA
        dt = dt

        emri_kwargs = {'T': T, 'dt': dt}

        param_list = [M,mu,a,p0,e0,Y0,
                      dL,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,
                      Al,nl,Ag,ng]

        sef_kwargs['filename'] = filename_Fishers_glob
        sef_kwargs['suffix'] = detected_EMRIs[i]['index']

        sef = StableEMRIFisher(*param_list, **emri_kwargs, **sef_kwargs)
        sef()
