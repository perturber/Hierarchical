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

from utility import getdistGpc, Jacobian, check_prior

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

    def __init__(self, sef_kwargs,
                 filename, filename_Fishers, filename_Fishers_loc, filename_Fishers_glob,
                 true_hyper, cosmo_params, source_bounds, hyper_bounds,
                 T_LISA, dt,
                 validate=False):

        self.filename = filename
        self.filename_Fishers = os.path.join(self.filename,filename_Fishers)
        self.filename_Fishers_loc = os.path.join(filename,filename_Fishers_loc)
        self.filename_Fishers_glob = os.path.join(filename,filename_Fishers_glob)
        
        self.sef_kwargs = sef_kwargs

        self.detected_EMRIs = np.load(f'{self.filename}/detected_EMRIs.npy',allow_pickle=True)

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
        
                if check_prior(Al,self.source_bounds['Al']) != 0.:
                    continue #ignore inferred EMRIs beyond the source bounds.
                if check_prior(nl,self.source_bounds['nl']) != 0.:
                    continue #ignore inferred EMRIs beyond the source bounds.
        
                T = self.T_LISA
                dt = self.dt
        
                emri_kwargs = {'T': T, 'dt': dt}
        
                param_list = [M,mu,a,p0,e0,Y0,
                              dL,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,
                              Al,nl,Ag,ng]
        
                sef_kwargs['filename'] = self.filename_Fishers_loc
                sef_kwargs['suffix'] = detected_EMRIs[i]['index']
        
                sef = StableEMRIFisher(*param_list, **emri_kwargs, **sef_kwargs)
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
        
                if check_prior(Ag,self.source_bounds['Ag']) != 0.:
                    continue #ignore inferred EMRIs beyond the source bounds.
        
                T = self.T_LISA
                dt = self.dt
        
                emri_kwargs = {'T': T, 'dt': dt}
        
                param_list = [M,mu,a,p0,e0,Y0,
                              dL,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,
                              Al,nl,Ag,ng]
        
                self.sef_kwargs['filename'] = self.filename_Fishers_glob
                self.sef_kwargs['suffix'] = detected_EMRIs[i]['index']
        
                sef = StableEMRIFisher(*param_list, **emri_kwargs, **self.sef_kwargs)
                sef()

    def transform_Fisher_at_bias(self):

        detected_EMRIs = self.detected_EMRIs

        if self.true_hyper['f'] > 0.0:
            for i in range(len(detected_EMRIs)):
            
                Al = detected_EMRIs[i]['local_params'][2]
                nl = detected_EMRIs[i]['local_params'][3]
                Ag = detected_EMRIs[i]['local_params'][4] #will be zero in local hypothesis
                ng = 4.0
        
                if check_prior(Al,self.source_bounds['Al']) != 0.:
                    continue #ignore inferred EMRIs beyond the source bounds.
                if check_prior(nl,self.source_bounds['nl']) != 0.:
                    continue #ignore inferred EMRIs beyond the source bounds.
                    
                #transform Fishers_loc[index]
                M_i = np.exp(detected_EMRIs[i]['local_params'][0])
                dist_i = getdistGpc(detected_EMRIs[i]['local_params'][1],H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
                
                J = Jacobian(M_i, dist_i,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
                
                Fisher_i = np.load(f"{self.filename_Fishers_loc}/Fisher_{detected_EMRIs[i]['index']}.npy")
        
                Fisher_transformed = J.T@Fisher_i@J
                
                np.save(f"{self.filename_Fishers_loc}/Fisher_transformed_{detected_EMRIs[i]['index']}",Fisher_transformed)
        
        if self.true_hyper['Gdot'] > 0.0:
            for i in range(len(detected_EMRIs)):
                
                Al = detected_EMRIs[i]['global_params'][2] #will be zero in global hypothesis
                nl = detected_EMRIs[i]['global_params'][3] #will be zero in global hypothesis
                Ag = detected_EMRIs[i]['global_params'][4]
                ng = 4.0
        
                if check_prior(Ag,self.source_bounds['Ag']) != 0.:
                    continue #ignore inferred EMRIs beyond the source bounds.
                
                #transform Fishers_glob[index]
                M_i = np.exp(detected_EMRIs[i]['global_params'][0])
                dist_i = getdistGpc(detected_EMRIs[i]['global_params'][1],H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
                
                J = Jacobian(M_i, dist_i,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
        
                Fisher_i = np.load(f"{self.filename_Fishers_glob}/Fisher_{detected_EMRIs[i]['index']}.npy")
        
                Fisher_transformed = J.T@Fisher_i@J
                
                np.save(f"{self.filename_Fishers_glob}/Fisher_transformed_{detected_EMRIs[i]['index']}",Fisher_transformed)

    def calculate_KL(self):

        detected_EMRIs = self.detected_EMRIs
        
        if self.true_hyper['f'] > 0.0:
            KL_loc = []
            for i in range(len(detected_EMRIs)):
                Al = detected_EMRIs[i]['local_params'][2]
                nl = detected_EMRIs[i]['local_params'][3]
                Ag = detected_EMRIs[i]['local_params'][4] #will be zero in local hypothesis
                ng = 4.0
        
                if check_prior(Al,self.source_bounds['Al']) != 0.:
                    continue #ignore inferred EMRIs beyond the source bounds.
                if check_prior(nl,self.source_bounds['nl']) != 0.:
                    continue #ignore inferred EMRIs beyond the source bounds.
        
                Gamma1 = np.load(f"{self.filename_Fishers}/Fisher_transformed_{detected_EMRIs[i]['index']}.npy") #true Fisher at biased inference point
                Gamma2 = np.load(f"{self.filename_Fishers_loc}/Fisher_transformed_{detected_EMRIs[i]['index']}.npy") #Fisher at injection
        
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
            
                if check_prior(Ag,self.source_bounds['Ag']) != 0.:
                    continue #ignore inferred EMRIs beyond the source bounds.
        
                Gamma1 = np.load(f"{self.filename_Fishers}/Fisher_transformed_{detected_EMRIs[i]['index']}.npy") #true Fisher at biased inference point
                Gamma2 = np.load(f"{self.filename_Fishers_glob}/Fisher_transformed_{detected_EMRIs[i]['index']}.npy") #Fisher at injection
        
                KL_glob.append(Fisher_KL(Gamma1,Gamma2))
        
            self.KL_glob = np.array(KL_glob)
            np.savetxt(f'{self.filename}/Fishers_glob_KL.txt',self.KL_glob)

    def plot_KL(self):

        #plot the KL-divergence for each index
        if self.true_hyper['f'] > 0.0:
            plt.figure(figsize=(7,5))
            plt.plot(self.KL_loc)
            plt.axhline(np.median(self.KL_loc),color='k',linestyle='--',linewidth=4,label='median')
            plt.xlabel('source index',fontsize=16)
            plt.ylabel(r'$KL(\Gamma_{\rm inj},\Gamma_{\rm bias}|\rm{local})$',fontsize=16)
            plt.legend()
            plt.yscale('log')
            plt.savefig(f'{self.filename}/fancy_plots/Fishers_loc_KL.png',dpi=300,bbox_inches='tight')
            plt.close()
        
        if self.true_hyper['Gdot'] > 0.0:
            plt.figure(figsize=(7,5))
            plt.plot(self.KL_glob)
            plt.axhline(np.median(self.KL_glob),color='k',linestyle='--',linewidth=4,label='median')
            plt.xlabel('source index',fontsize=16)
            plt.ylabel(r'$KL(\Gamma_{\rm inj},\Gamma_{\rm bias}|\rm{global})$',fontsize=16)
            plt.legend()
            plt.yscale('log')
            plt.savefig(f'{self.filename}/fancy_plots/Fishers_glob_KL.png',dpi=300,bbox_inches='tight')
            plt.close()
