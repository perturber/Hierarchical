#imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
try:
    import cupy as cp
    use_gpu = True
except:
    use_gpu = False
import pickle
import time
import h5py

from stableemrifisher.fisher import StableEMRIFisher
from stableemrifisher.utils import inner_product, generate_PSD, padding

from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import GenerateEMRIWaveform
from few.summation.aakwave import AAKSummation
from few.utils.constants import YRSID_SI 
from few.utils.constants import SPEED_OF_LIGHT as C_SI

from fastlisaresponse import ResponseWrapper  # Response function 
from lisatools.detector import ESAOrbits #ESAOrbits correspond to esa-trailing-orbits.h5

from scipy.integrate import quad, nquad
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.stats import uniform
from scipy.special import factorial
from scipy.optimize import brentq, root

from scipy.stats import multivariate_normal
import warnings

from hierarchical.utility import getdistGpc, Jacobian, check_prior

if not use_gpu:
    cfg_set = few.get_config_setter(reset=True)
    cfg_set.enable_backends("cpu")
    cfg_set.set_log_level("info")
else:
    pass #let the backend decide for itself.

#calculate the KL-divergence
def Fisher_KL(Gamma1, Gamma2):
    """
    calculate the KL divergence between two Fisher matrices Gamma1 and Gamma2 of the same dimensions assuming they are
    calculated at the same parameter point.
    Since KL divergence is not symmetric, we asuume Gamma1 to be the 'truth' and Gamma2 as the 'approximation'.
    """
    Sigma1_det = 1/np.linalg.det(Gamma1)
    Sigma2_det = 1/np.linalg.det(Gamma2)

    dim = len(Gamma1)

    return 0.5*(np.log(Sigma2_det/Sigma1_det) - dim + np.trace(Gamma2@np.linalg.inv(Gamma1)))

class FisherValidation:

    """
    Calculate the KL divergence between Fisher Information matrices at the true parameter point v/s at the bias-corrected point in a given hypothesis.
    
    args:
    
        sef_kwargs (dict): arguments for initializing the StableEMRIFisher class.
        filename (string): folder name where the data is being stored. No default because impractical to not save results.
        filename_Fishers (string): a sub-folder for storing Fisher files (book-keeping). If None, Fishers directly stored in filename.
        
        filename_Fishers_loc (string): name of the subfolder where Fisher matrices in the local hypothesis are/will be stored.
        filename_Fishers_glob (string): name of the subfolder where Fisher matrices in the global hypothesis are/will be stored.
        
        true_hyper (dict): true values of all hyperparameters. Default are fiducial values consistent with a population of vacuum EMRIs.
        cosmo_params (dict): true values of 'Omega_m0' (matter density), 'Omega_Lambda0' (DE density), and 'H0' (Hubble constant in m/s/Gpc).

        source_bounds (dict): prior range on source parameters in all three hypotheses. Keys are param names and values are lists of lower and upper bounds. 
                              Must be provided for all parameters. We assume flat priors in this range.
        hyper_bounds (dict): prior range on population (hyper)params in all three hypotheses. Keys are param names and values are lists of lower and upper bounds. 
                             Must be provided for all hyperparams. We assume flat priors in this range.
                             
        T_LISA (float): time (in years) of LISA observation window. Default is 1.0.
        dt (float): LISA sampling frequency. Default is 1.0.
        
        validate (bool): Whether to calculate Fisher matrices at the biased parameter point. Default is True.
        plot_KL (bool): plot the KL divergences on all param indices. Default is False.
    """

    def __init__(self, sef_kwargs,
                 filename, filename_Fishers, filename_Fishers_loc, filename_Fishers_glob,
                 true_hyper, cosmo_params, source_bounds, hyper_bounds,
                 T_LISA, dt,
                 validate=True):

        self.filename = filename
        self.filename_Fishers = os.path.join(self.filename,filename_Fishers)
        self.filename_Fishers_loc = os.path.join(filename,filename_Fishers_loc)
        self.filename_Fishers_glob = os.path.join(filename,filename_Fishers_glob)
        
        self.sef_kwargs = sef_kwargs

        self.detected_EMRIs = np.load(f'{self.filename}/detected_EMRIs.npy', allow_pickle=True)

        #true cosmology
        self.Omega_m0 = cosmo_params['Omega_m0']
        self.Omega_Lambda0 = cosmo_params['Omega_Lambda0']
        self.H0 = cosmo_params['H0']

        self.true_hyper = true_hyper
        self.source_bounds = source_bounds
        self.hyper_bounds = hyper_bounds

        self.validate = validate

        self.T_LISA = T_LISA
        self.dt = dt
        
    def __call__(self):

        if self.validate:
            self.calculate_Fisher_at_bias()

        self.transform_Fisher_at_bias()

        self.calculate_KL()
        

    def calculate_Fisher_at_bias(self):

        detected_EMRIs = self.detected_EMRIs
        
        #calculate Fishers at the biased point in the local hypothesis
        if self.true_hyper['f'] > 0.0:
            #if fraction of local EMRIs > 0, calculate FIM at biased point in local hypothesis
            for i in tqdm(range(len(detected_EMRIs))):
                M = np.exp(detected_EMRIs[i]['local_params'][0])
                mu = detected_EMRIs[i]['true_params'][1]
                a = detected_EMRIs[i]['true_params'][2]
                p0 = detected_EMRIs[i]['true_params'][3]
                e0 = detected_EMRIs[i]['true_params'][4]
                Y0 = detected_EMRIs[i]['true_params'][5]
                dL = getdistGpc(detected_EMRIs[i]['local_params'][1],H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
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
        
                T = self.T_LISA
                dt = self.dt
        
                emri_kwargs = {'T': T, 'dt': dt}
        
                param_list = [M,mu,a,p0,e0,Y0,
                              dL,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,
                             ]
                             
                add_param_args = {"Al":Al, "nl":nl, "Ag":Ag, "ng":ng}
        
                self.sef_kwargs['filename'] = self.filename_Fishers_loc
                self.sef_kwargs['suffix'] = detected_EMRIs[i]['index']
        
                sef = StableEMRIFisher(*param_list, add_param_args=add_param_args, **emri_kwargs, **self.sef_kwargs)
                sef()
    
        #calculate Fishers at the biased point in the global hypothesis
        if self.true_hyper['Gdot'] > 0.0:
            #if Gdot > 0, calculate FIM at biased point in global hypothesis
            for i in tqdm(range(len(detected_EMRIs))):
                M = np.exp(detected_EMRIs[i]['global_params'][0])
                mu = detected_EMRIs[i]['true_params'][1]
                a = detected_EMRIs[i]['true_params'][2]
                p0 = detected_EMRIs[i]['true_params'][3]
                e0 = detected_EMRIs[i]['true_params'][4]
                Y0 = detected_EMRIs[i]['true_params'][5]
                dL = getdistGpc(detected_EMRIs[i]['global_params'][1],H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
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
        
                T = self.T_LISA
                dt = self.dt
        
                emri_kwargs = {'T': T, 'dt': dt}
        
                param_list = [M,mu,a,p0,e0,Y0,
                              dL,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,
                             ]
                             
                add_param_args = {"Al":Al, "nl":nl, "Ag":Ag, "ng":ng}
        
                self.sef_kwargs['filename'] = self.filename_Fishers_glob
                self.sef_kwargs['suffix'] = detected_EMRIs[i]['index']
        
                sef = StableEMRIFisher(*param_list, add_param_args=add_param_args, **emri_kwargs, **self.sef_kwargs)
                sef()

    def transform_Fisher_at_bias(self):

        detected_EMRIs = self.detected_EMRIs

        if self.true_hyper['f'] > 0.0:
            for i in range(len(detected_EMRIs)):
            
                Al = detected_EMRIs[i]['local_params'][2]
                nl = detected_EMRIs[i]['local_params'][3]
                Ag = detected_EMRIs[i]['local_params'][4] #will be zero in local hypothesis
                ng = 4.0
                    
                #transform Fishers_loc[index]
                M_i = np.exp(detected_EMRIs[i]['local_params'][0])
                dist_i = getdistGpc(detected_EMRIs[i]['local_params'][1],H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
                
                J = Jacobian(M_i, dist_i,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
                
                with h5py.File(f"{self.filename_Fishers_loc}/Fisher_{detected_EMRIs[i]['index']}.h5", "r") as f:
                    Fisher_i = f["Fisher"][:]
                
                Fisher_transformed = J.T@Fisher_i@J
                
                with h5py.File(f"{self.filename_Fishers_loc}/Fisher_{detected_EMRIs[i]['index']}.h5", "a") as f:
                    if not "Fisher_transformed" in f:
                        f.create_dataset("Fisher_transformed", data = Fisher_transformed)
        
        if self.true_hyper['Gdot'] > 0.0:
            for i in range(len(detected_EMRIs)):
                
                Al = detected_EMRIs[i]['global_params'][2] #will be zero in global hypothesis
                nl = detected_EMRIs[i]['global_params'][3] #will be zero in global hypothesis
                Ag = detected_EMRIs[i]['global_params'][4]
                ng = 4.0
                
                #transform Fishers_glob[index]
                M_i = np.exp(detected_EMRIs[i]['global_params'][0])
                dist_i = getdistGpc(detected_EMRIs[i]['global_params'][1],H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
                
                J = Jacobian(M_i, dist_i,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
                
                with h5py.File(f"{self.filename_Fishers_glob}/Fisher_{detected_EMRIs[i]['index']}.h5", "r") as f:
                    Fisher_i = f["Fisher"][:]
        
                Fisher_transformed = J.T@Fisher_i@J
                
                with h5py.File(f"{self.filename_Fishers_glob}/Fisher_{detected_EMRIs[i]['index']}.h5", "a") as f:
                    if not "Fisher_transformed" in f:
                        f.create_dataset("Fisher_transformed", data = Fisher_transformed)

    def calculate_KL(self):

        detected_EMRIs = self.detected_EMRIs
        
        if self.true_hyper['f'] > 0.0:
            KL_loc = []
            for i in range(len(detected_EMRIs)):
                Al = detected_EMRIs[i]['local_params'][2]
                nl = detected_EMRIs[i]['local_params'][3]
                Ag = detected_EMRIs[i]['local_params'][4] #will be zero in local hypothesis
                ng = 4.0
        
                with h5py.File(f"{self.filename_Fishers}/Fisher_{detected_EMRIs[i]['index']}.h5", "r") as f:
                    Gamma1 = f["Fisher_transformed"][:] #true Fisher at biased inference point
                
                with h5py.File(f"{self.filename_Fishers_loc}/Fisher_{detected_EMRIs[i]['index']}.h5", "r") as f:
                    Gamma2 = f["Fisher_transformed"][:] #Fisher at injection
        
                KL_loc.append(Fisher_KL(Gamma1,Gamma2))
        
            self.KL_loc = np.array(KL_loc)
            np.savetxt(f'{self.filename}/Fishers_loc_KL.txt',self.KL_loc)
        
        if self.true_hyper['Gdot'] > 0.0:
            KL_glob = []
            for i in range(len(detected_EMRIs)):
                Al = detected_EMRIs[i]['global_params'][2] #will be zero in global hypothesis
                nl = detected_EMRIs[i]['global_params'][3] #will be zero in global hypothesis
                Ag = detected_EMRIs[i]['global_params'][4]
                ng = 4.0
        
                with h5py.File(f"{self.filename_Fishers}/Fisher_{detected_EMRIs[i]['index']}.h5", "r") as f:
                    Gamma1 = f["Fisher_transformed"][:] #true Fisher at biased inference point
                    
                with h5py.File(f"{self.filename_Fishers_glob}/Fisher_{detected_EMRIs[i]['index']}.h5", "r") as f:
                    Gamma2 = f["Fisher_transformed"][:] #Fisher at injection
        
                KL_glob.append(Fisher_KL(Gamma1,Gamma2))
        
            self.KL_glob = np.array(KL_glob)
            np.savetxt(f'{self.filename}/Fishers_glob_KL.txt',self.KL_glob)

    def KL_divergence_plot(self,plots_folder=None):
    
        if plots_folder == None:
            plots_folder = f'{self.filename}/fancy_plots'

        print("Plotting KL divergences...")
        
        #plot the KL-divergence for each index
        if self.true_hyper['f'] > 0.0:
            plt.figure(figsize=(7,5))
            plt.plot(self.KL_loc)
            plt.axhline(np.median(self.KL_loc),color='k',linestyle='--',linewidth=4,label='median')
            plt.xlabel('source index',fontsize=16)
            plt.ylabel(r'$KL(\Gamma_{\rm inj},\Gamma_{\rm bias}|\rm{local})$',fontsize=16)
            plt.legend()
            plt.yscale('log')
            plt.savefig(f'{plots_folder}/Fishers_loc_KL.png',dpi=300,bbox_inches='tight')
            plt.close()
        
        if self.true_hyper['Gdot'] > 0.0:
            plt.figure(figsize=(7,5))
            plt.plot(self.KL_glob)
            plt.axhline(np.median(self.KL_glob),color='k',linestyle='--',linewidth=4,label='median')
            plt.xlabel('source index',fontsize=16)
            plt.ylabel(r'$KL(\Gamma_{\rm inj},\Gamma_{\rm bias}|\rm{global})$',fontsize=16)
            plt.legend()
            plt.yscale('log')
            plt.savefig(f'{plots_folder}//Fishers_glob_KL.png',dpi=300,bbox_inches='tight')
            plt.close()
