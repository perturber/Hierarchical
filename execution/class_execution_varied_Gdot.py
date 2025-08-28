#imports
import numpy as np
import os

from stableemrifisher.fisher import StableEMRIFisher

from few.waveform import GenerateEMRIWaveform

from fastlisaresponse import ResponseWrapper  # Response function 
from lisatools.detector import EqualArmlengthOrbits #ESAOrbits correspond to esa-trailing-orbits.h5
from lisatools.sensitivity import get_sensitivity, A1TDISens, E1TDISens

from hierarchical.Hierarchical_Class import Hierarchical
from hierarchical.JointWave import JointKerrWaveform

use_gpu = True

if not use_gpu:
    import few
    cfg_set = few.get_config_setter(reset=True)
    cfg_set.enable_backends("cpu")
    cfg_set.set_log_level("info")
else:
    pass #let the backend decide for itself.

### INIT PARAMETERS FOR SEF INIT ###
T_LISA = 1.0 #LISA observation duration
dt = 10.0 #sampling rate

#waveform class setup
waveform_class = JointKerrWaveform
waveform_class_kwargs = dict(inspiral_kwargs=dict(err=1e-11,),
                             mode_selector_kwargs=dict(mode_selection_threshold=1e-5))

#waveform generator setup
waveform_generator = GenerateEMRIWaveform
waveform_generator_kwargs = dict(sum_kwargs=dict(pad_output=True),
                                return_list=False)

#ResponseWrapper setup
tdi_gen ="1st generation"# "2nd generation"#
order = 20  # interpolation order (should not change the result too much)
tdi_kwargs_esa = dict(
    orbits=EqualArmlengthOrbits(use_gpu=use_gpu), order=order, tdi=tdi_gen, tdi_chan="AE",
)
index_lambda = 8
index_beta = 7
# with longer signals we care less about this
t0 = 10000.0  # throw away on both ends when our orbital information is weird

ResponseWrapper_kwargs = dict(
    #waveform_gen=waveform_generator,
    Tobs = T_LISA,
    dt = dt,
    index_lambda = index_lambda,
    index_beta = index_beta,
    t0 = t0,
    flip_hx = True,
    use_gpu=use_gpu,
    is_ecliptic_latitude=False,
    remove_garbage="zero",
    **tdi_kwargs_esa
)

#noise setup
channels = [A1TDISens, E1TDISens]
noise_model = get_sensitivity
noise_kwargs = [{"sens_fn": channel_i} for channel_i in channels]

sef = StableEMRIFisher(waveform_class=waveform_class, 
                       waveform_class_kwargs=waveform_class_kwargs,
                       waveform_generator=waveform_generator,
                       waveform_generator_kwargs=waveform_generator_kwargs,
                       ResponseWrapper=ResponseWrapper, ResponseWrapper_kwargs=ResponseWrapper_kwargs,
                       noise_model=noise_model, noise_kwargs=noise_kwargs, channels=channels,
                       stats_for_nerds = True, 
                       deriv_type='stable',
                       use_gpu = use_gpu,
                      )

### SEF INITIALIZATION DONE ###
                        
### Fixed parameters
#cosmological parameters
cosmo_params={'Omega_m0':0.30,
              'Omega_Lambda0':0.70,
              'H0':70e6  #H0 in m/s/Gpc
              }

#Mstar normalization term for the EMRI MBH mass distribution
Mstar = 3e6

#True size of the population (varied)
Npop = int(5e2)

#detection SNR threshold
SNR_thresh = 20.0

f = 0.5 #grid of fraction of EMRIs with a local effect

N_Gdots = 11
Gdot_range = np.linspace(1e-13, 1e-12, N_Gdots)

true_K = 5e-3
true_alpha = 0.0
true_beta = 0.0

parent_filename = f'Hierarchical_varied_Gdot_Gdotmax_{Gdot_range[-1]}_Npop_{Npop}_f_{f}_K_{true_K}_alpha_{true_alpha}_beta_{true_beta}'

#delta_range for additional parameters (because the default ranges may not be suitable)
Ndelta = 8

os.makedirs(parent_filename, exist_ok=True)

for i in range(len(Gdot_range)):

    true_Gdot = Gdot_range[i]
    
    delta_range = {
    "Al":np.geomspace(1e-7,1e-12,Ndelta),
    "nl":np.geomspace(1.0,1e-5,Ndelta),
    "Ag":np.geomspace(true_Gdot*1e-1, true_Gdot*1e-4, Ndelta),
    }
    
    #true values of population hyperparameters.
    true_hyper={'K':true_K,'alpha':true_alpha,'beta':true_beta, #vacuum hyperparameters
                'f':f,'mu_Al':1e-6,'mu_nl':8.0,'sigma_Al':1e-7,'sigma_nl':1.0, #local effect hyper
                'Gdot':true_Gdot #global effect hyper
            }

    #prior bounds on source parameters. The true population would also be generated from this!
    source_bounds={'lnM':[np.log(10**(5.5)),np.log(10**(6.5))],'z':[0.01,1.0], #vacuum parameters
                'Al':[0.0,1e-5],'nl':[-10.0,10.0], #local effect parameters
                'Ag':[-5*true_hyper['Gdot'],5*true_hyper['Gdot']] #global effect parameters
                }

    hypint = 0.1 #percentage interval around true value to be used as hyperparam bounds

    if true_hyper['alpha'] == 0:
        alpha_bounds = [-0.1,0.1]
    elif true_hyper['alpha'] > 0:
        alpha_bounds = [true_hyper['alpha']*(1 - hypint),true_hyper['alpha']*(1 + hypint)]
    else:
        alpha_bounds = [true_hyper['alpha']*(1 + hypint),true_hyper['alpha']*(1 - hypint)]

    if true_hyper['beta'] == 0:
        beta_bounds = [-0.1,0.1]
    elif true_hyper['beta'] > 0:
        beta_bounds = [true_hyper['beta']*(1 - hypint),true_hyper['beta']*(1 + hypint)]
    else:
        beta_bounds = [true_hyper['beta']*(1 + hypint),true_hyper['beta']*(1 - hypint)]

    #prior bounds on population hyperparameters
    hyper_bounds={'K':[true_hyper['K']*(1 - hypint),true_hyper['K']*(1 + hypint)],
                'alpha':alpha_bounds,
                'beta':beta_bounds, #vacuum hyperparameters
                'f':[0.0,1.0],
                'mu_Al':[true_hyper['mu_Al']*(1 - hypint),true_hyper['mu_Al']*(1 + hypint)],
                'mu_nl':[true_hyper['mu_nl']*(1 - hypint),true_hyper['mu_nl']*(1 + hypint)],
                'sigma_Al':[true_hyper['sigma_Al']*(1 - hypint),true_hyper['sigma_Al']*(1 + hypint)],
                'sigma_nl':[true_hyper['sigma_nl']*(1 - hypint),true_hyper['sigma_nl']*(1 + hypint)],#local effect hyper
                'Gdot':source_bounds['Ag'] #global effect hyper
                }

    filename = parent_filename + f'/Gdot_{true_Gdot}' #folder with all the analysis data and plots
    filename_Fishers = 'Fishers' #subfolder with all the Fisher matrices
    plots_filename = 'fancy_plots' #subfolder where all the plots will be saved
    
    #setting up kwargs to pass to StableEMRIFishers class
    sef_kwargs = {
            'param_names': ['m1','dist','Al','nl','Ag'], #params to be varied
            'T': T_LISA, #LISA observation duration
            'dt':dt, #sampling rate
            'der_order':2, #derivative order
            'Ndelta':Ndelta, #number of stable points
            'delta_range':delta_range,#custom delta range for additional parameters
            'stability_plot': False, #true if you wanna  plot stability surfaces
            'plunge_check':False, #no need to check for plunge --- away from plunge already ensured.
            }
    
    hier = Hierarchical(Npop=Npop,SNR_thresh=SNR_thresh,sef=sef,sef_kwargs=sef_kwargs,
                        filename=filename,filename_Fishers=filename_Fishers,
                        cosmo_params=cosmo_params,true_hyper=true_hyper,
                        source_bounds=source_bounds,hyper_bounds=hyper_bounds,Mstar=Mstar,
                        T_LISA=T_LISA,make_nice_plots=True,plots_filename=plots_filename,
                        M_random=int(2e3),
                        #Fisher_validation_kwargs=Fisher_validation_kwargs #not used in varying-gdot study
                        out_of_bound_nature='remove'
                        )
    
    hier()
