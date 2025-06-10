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
from few.trajectory.ode import KerrEccEqFlux

from few.waveform import GenerateEMRIWaveform
from few.summation.aakwave import AAKSummation
from few.utils.constants import YRSID_SI
from few.utils.constants import SPEED_OF_LIGHT as C_SI
from few.utils.geodesic import ELQ_to_pex, get_kerr_geo_constants_of_motion

from fastlisaresponse import ResponseWrapper  # Response function 
from lisatools.detector import ESAOrbits #ESAOrbits correspond to esa-trailing-orbits.h5

from scipy.integrate import quad, nquad
from scipy.interpolate import RegularGridInterpolator, CubicSpline
from scipy.stats import uniform
from scipy.special import factorial
from scipy.optimize import brentq, root

from scipy.stats import multivariate_normal
import warnings

from hierarchical.FisherValidation import FisherValidation
from hierarchical.utility import H, integrand_dc, dc, getdistGpc, dlminusdistz, getz, Jacobian, check_prior
from hierarchical.JointWave import JointKerrWaveform, JointRelKerrEccFlux


#lots of supporting utility functions

if not use_gpu:
    cfg_set = few.get_config_setter(reset=True)
    cfg_set.enable_backends("cpu")
    cfg_set.set_log_level("info")
else:
    pass #let the backend decide for itself.
    
#source parameter prior pdfs in all three hypotheses
#supporting functions for prior_vac
def Mz_func(M, z, K, alpha, beta, H0, Omega_m0,Omega_Lambda0, Mstar):
    return K*(M/Mstar)**alpha*(1+z)**beta*4*np.pi*dc(z,H0,Omega_m0,Omega_Lambda0)**2

def prior_vac(M, z, K, alpha, beta, H0, Omega_m0,Omega_Lambda0, Mstar):
    """
    given the vacuum hyperparams [K,alpha,beta],
    calculate the UNNORMALIZED probability distribution function of 
    obtaining the source params [M,z]
    """
    
    return Mz_func(M,z,K,alpha,beta,H0, Omega_m0,Omega_Lambda0, Mstar)

def prior_loc(vec_l, f, mu_l, sigma_l):
    """
    given the local effect hyperparams [f,[mu_Al, mu_nl],[sigma_Al,sigma_nl]],
    calculate the probability distribution function of 
    obtaining the source params vec_l = [A_l,n_l]
    """

    vec_l = np.array(vec_l)
    mu_l = np.array(mu_l)
    sigma_l = np.array(sigma_l)
    
    Gamma_l = np.diag(1/sigma_l**2)

    return f*np.linalg.det(Gamma_l)**(1/2)/(2*np.pi)*np.exp(-0.5*(vec_l-mu_l)@Gamma_l@(vec_l-mu_l))

def prior_glob(A_g, Gdot, atol=1e-14):
    """
    given the global effect hyperparam [Gdot],
    calculate the probability distribution function of 
    obtaining the source params vec_g = [A_g]
    """
    if np.isclose(A_g,Gdot,rtol=atol, atol=atol):
        return 1.
    else:
        return 0.
        
#generating source parameter samples
def M_z_samples(N,M_range,z_range,lambda_v,grid_size,H0,Omega_m0,Omega_Lambda0,Mstar,seed):
    """ function to generate N samples of the local effect parameters M, z
    from M in Mrange, z in zrange, given hyperparameters lambda_v and a grid of size grid_size
    """
    
    np.random.seed(seed)

    K_truth, alpha_truth, beta_truth = lambda_v

    M_grid = uniform.rvs(loc=M_range[0],scale=M_range[1]-M_range[0],size=grid_size) #generating samples first from a uniform grid
    z_grid = uniform.rvs(loc=z_range[0],scale=z_range[1]-z_range[0],size=grid_size)

    prior_Mz = []
    for i in range(grid_size):
        prior_Mz.append(prior_vac(M_grid[i],z_grid[i],K_truth,alpha_truth,beta_truth,H0, Omega_m0,Omega_Lambda0,Mstar))
    
    prior_Mz = np.array(prior_Mz)/np.sum(np.array(prior_Mz)) #normalizing
    
    #choosing N sources based on the probability distribution
    indices = range(grid_size)
    chosen = np.random.choice(indices,size=N,p=prior_Mz)
    M_samples = M_grid[chosen]
    z_samples = z_grid[chosen]    

    return M_samples, z_samples

def A_n_samples(N,lambda_l,seed):
    """ function to generate N samples of the local effect parameters A_l, n_l
    from A_l in Al_range, n_l in nl_range, given hyperparameters lambda_l and a grid of size grid_size
    """
    np.random.seed(seed)
    
    f, mu_l, sigma_l = lambda_l

    if f == 0.0:
        Al_samples = np.zeros(N)
        nl_samples = np.zeros(N)
        
    else:
        cov = [[sigma_l[0]**2, 0],[0,sigma_l[1]**2]]
        Al_samples, nl_samples = np.random.multivariate_normal(mean=mu_l,cov=cov,size=int(f*N)).T
        Al_samples = np.concatenate((Al_samples,np.zeros(N-int(f*N))))
        nl_samples = np.concatenate((nl_samples,np.zeros(N-int(f*N))))
        
    return Al_samples, nl_samples

def Ag_samples(N,lambda_g):
    """ function to generate N samples of the global effect parameter Ag
    given hyperparameter lambda_g
    """
    return np.ones(N)*lambda_g #just a Dirac Delta

def other_param_samples(N,M_samples,Tplunge_range,seed):
    np.random.seed(seed)

    Trange = Tplunge_range #time-to-plunge (from initiation) of the EMRI population. This is an important (but unfortunately FREE) parameter to control the number of observed EMRIs.
    
    Tstar = uniform.rvs(loc=Trange[0],scale=Trange[1]-Trange[0],size=N) #randomly choosing time of plunge for the Nth EMRI in the population
    
    log10qrange = [-5.5,-4.5] #range of log10 mass ratios
    qrange = 10**uniform.rvs(loc=log10qrange[0],scale=log10qrange[1]-log10qrange[0],size=N) #samples of mass ratios
    mustar = M_samples*qrange #samples of CO masses
    
    arange = [0.5,0.99] #MBH spin
    astar = uniform.rvs(loc=arange[0],scale=arange[1]-arange[0],size=N)
    
    qSrange = [0.0,np.pi] #polar sky location
    qSstar = uniform.rvs(loc=qSrange[0],scale=qSrange[1]-qSrange[0],size=N)
    qKstar = uniform.rvs(loc=qSrange[0],scale=qSrange[1]-qSrange[0],size=N) #polar spin orientation
    
    phiSrange = [0.0,2*np.pi] #azimuthal sky location
    phiSstar = uniform.rvs(loc=phiSrange[0],scale=phiSrange[1]-phiSrange[0],size=N)
    phiKstar = uniform.rvs(loc=phiSrange[0],scale=phiSrange[1]-phiSrange[0],size=N) #azimuthal spin orientation
    Phi0star = uniform.rvs(loc=phiSrange[0],scale=phiSrange[1]-phiSrange[0],size=N) #circ-ecc init EMRI phase
    
    return mustar, astar, qSstar, qKstar, phiSstar, phiKstar, Phi0star, Tstar
    
def p0_samples_func(N,Msamps,musamps,asamps,Alsamps,nlsamps,Agsamps,Tsamps,seed,filename):
    np.random.seed(seed)
    
    traj_xy = EMRIInspiral(func=JointRelKerrEccFlux)
    
    p0samps = []
    
    for i in tqdm(range(N)):
        #print(Tsamps[i],Msamps[i],musamps[i],asamps[i],Alsamps[i],nlsamps[i],Agsamps[i])
        
        p_plunge = get_p_at_t(traj_xy,
                              Tsamps[i],
                              [Msamps[i],
                               musamps[i],
                               asamps[i],
                               0.0, #e0
                               1.0, #Y0
                               Alsamps[i],
                               nlsamps[i],
                               Agsamps[i],
                               4.0, #ng
                              ],
                              )
        
        p0samps.append(p_plunge + 1.0)

    np.savetxt(f"{filename}/p0samps.txt",p0samps)
    
    return np.array(p0samps)
    
#bias calculation functions
def bias(psi_signal,phi_signal,multiplicative_factor):
    """ 
    Given a true signal with decomposed param set (psi, phi) and the multiplicative factor,
    calculate the biased param vector psi_bias.
    (See Eq. 11 in https://arxiv.org/abs/2312.13028)
    """
    # d: number of measured source params
    # Nphi: number of unmeasured params
    # Npsi: number of measured params

    phi_signal = np.atleast_1d(phi_signal)
    
    delta_phi = phi_signal - np.zeros(len(phi_signal)) #1D array - 1D array = 1D array of length Nphi
        
    #print(delta_phi)
    
    delta_psi = multiplicative_factor@delta_phi # Npsi x Nphi array times Nphi 1D array: 1D array of length Npsi
    
    #print(psi_ML)

    return psi_signal + delta_psi # Npsi 1D array + Npsi 1D array = Npsi 1D array

def bias(psi_signal,phi_signal,multiplicative_factor):
    """ 
    Given a true signal with decomposed param set (psi, phi) and the multiplicative factor,
    calculate the biased param vector psi_bias.
    (See Eq. 11 in https://arxiv.org/abs/2312.13028)
    """
    # d: number of measured source params
    # Nphi: number of unmeasured params
    # Npsi: number of measured params

    phi_signal = np.atleast_1d(phi_signal)
    
    delta_phi = phi_signal - np.zeros(len(phi_signal)) #1D array - 1D array = 1D array of length Nphi
        
    #print(delta_phi)
    
    delta_psi = multiplicative_factor@delta_phi # Npsi x Nphi array times Nphi 1D array: 1D array of length Npsi
    
    #print(psi_ML)

    return psi_signal + delta_psi # Npsi 1D array + Npsi 1D array = Npsi 1D array

#source integral prior_vac derivatives
######## supporting functions ###########################################

def Ddc(z,H0,Omega_m0,Omega_Lambda0):
    """calculate the first derivative of comoving distance with respect to z"""
    return integrand_dc(z,H0,Omega_m0,Omega_Lambda0)

def DDdc(z,H0,Om,Ol):

    """
    Computes the expression:
    
    -((3 * c * H0 * Om^(2/3) * (1 + z) * (Om * (1 + z)^3)^(3/2) *
    (Ol^(4/3) * Om^(1/3) * (1 + z) +
     3 * H0^(2/3) * Ol^(2/3) * Om^(2/3) * (1 + z)^2 +
     H0^(4/3) * Om * (1 + z)^3 -
     2 * H0^(1/3) * Ol * sqrt(Om * (1 + z)^3) -
     2 * H0 * Ol^(1/3) * Om^(1/3) * (1 + z) * sqrt(Om * (1 + z)^3)))
     / (2 * (Ol^(1/3) * Om^(1/3) * (1 + z) + H0^(1/3) * sqrt(Om * (1 + z)^3))^2 *
        (Ol^(2/3) * Om^(1/3) * (1 + z) +
         H0^(2/3) * Om^(2/3) * (1 + z)^2 -
         H0^(1/3) * Ol^(1/3) * sqrt(Om * (1 + z)^3))^4))

    Parameters:
        c, H0, Om, Ol, z : float
            Constants and variable in the expression
    """
    
    # Numerator terms
    term1 = Ol ** (4/3) * Om ** (1/3) * (1 + z)
    term2 = 3 * H0 ** (2/3) * Ol ** (2/3) * Om ** (2/3) * (1 + z) ** 2
    term3 = H0 ** (4/3) * Om * (1 + z) ** 3
    term4 = -2 * H0 ** (1/3) * Ol * np.sqrt(Om * (1 + z) ** 3)
    term5 = -2 * H0 * Ol ** (1/3) * Om ** (1/3) * (1 + z) * np.sqrt(Om * (1 + z) ** 3)
    
    numerator = (-3 * C_SI * H0 * Om ** (2/3) * (1 + z) * (Om * (1 + z) ** 3) ** (3/2) *
                 (term1 + term2 + term3 + term4 + term5))

    # Denominator terms
    denom1 = Ol ** (1/3) * Om ** (1/3) * (1 + z) + H0 ** (1/3) * np.sqrt(Om * (1 + z) ** 3)
    denom2 = (Ol ** (2/3) * Om ** (1/3) * (1 + z) +
              H0 ** (2/3) * Om ** (2/3) * (1 + z) ** 2 -
              H0 ** (1/3) * Ol ** (1/3) * np.sqrt(Om * (1 + z) ** 3))

    denominator = 2 * denom1 ** 2 * denom2 ** 4

    return numerator / denominator

######## pvac derivatives #########################################################
def DDM_prior_vac(M, z, K, alpha, beta, H0, Omega_m0,Omega_Lambda0, Mstar):
    C = K*(1/Mstar)**alpha*4*np.pi
    
    return C*alpha*(alpha-1)*M**(alpha-2)*(1+z)**beta*dc(z,H0,Omega_m0,Omega_Lambda0)**2

def DDz_prior_vac(M, z, K, alpha, beta,H0,Omega_m0,Omega_Lambda0, Mstar):
    C = K*(1/Mstar)**alpha*4*np.pi

    """
    Computes the expression given in Mathematica syntax:
    
    (-1 + beta) * beta * C * M^alpha * (1 + z)^(-2 + beta) * dc[z]^2 
    + 4 * beta * C * M^alpha * (1 + z)^(-1 + beta) * dc[z] * Derivative[1][dc][z] 
    + 2 * C * M^alpha * (1 + z)^beta * Derivative[1][dc][z]^2 
    + 2 * C * M^alpha * (1 + z)^beta * dc[z] * (dc''[z])
    
    Parameters:
        beta, C, M, alpha, z : float
            Constants and variable in the expression
        dc : float
            dc[z] (function value at z)
        Ddc : float
            Derivative of dc with respect to z (dc'[z])
        DDdc : float
            Second derivative of dc with respect to z (dc''[z])
    """

    term1 = (-1 + beta) * beta * C * M ** alpha * (1 + z) ** (-2 + beta) * dc(z,H0,Omega_m0,Omega_Lambda0)** 2
    term2 = 4 * beta * C * M ** alpha * (1 + z) ** (-1 + beta) * dc(z,H0,Omega_m0,Omega_Lambda0)* Ddc(z,H0,Omega_m0,Omega_Lambda0)
    term3 = 2 * C * M ** alpha * (1 + z) ** beta * Ddc(z,H0,Omega_m0,Omega_Lambda0) ** 2
    term4 = 2 * C * M ** alpha * (1 + z) ** beta * dc(z,H0,Omega_m0,Omega_Lambda0) * DDdc(z,H0,Omega_m0,Omega_Lambda0)
    
    return term1 + term2 + term3 + term4

def DMDz_prior_vac(M, z, K, alpha, beta,H0,Omega_m0,Omega_Lambda0, Mstar):
    C = K*(1/Mstar)**alpha*4*np.pi

    """
    Computes the expression:
    
    alpha * beta * C * M^(-1 + alpha) * (1 + z)^(-1 + beta) * dc[z]^2 
    + 2 * alpha * C * M^(-1 + alpha) * (1 + z)^beta * dc[z] * Derivative[1][dc][z]
    
    Parameters:
        alpha, beta, C, M, z : float
            Constants and variable in the expression
        dc : float
            dc[z] (function value at z)
        Ddc : float
            Derivative of dc with respect to z (dc'[z])
    """
    term1 = alpha * beta * C * M ** (-1 + alpha) * (1 + z) ** (-1 + beta) * dc(z,H0,Omega_m0,Omega_Lambda0) ** 2
    term2 = 2 * alpha * C * M ** (-1 + alpha) * (1 + z) ** beta * dc(z,H0,Omega_m0,Omega_Lambda0) * Ddc(z,H0,Omega_m0,Omega_Lambda0)
    
    return term1 + term2

#supporting function for Matrix operations
def get_minor(matrix, i, j):
    """Return the minor of the element at row i and column j."""
    minor = np.delete(matrix, i, axis=0)  # Remove the i-th row
    minor = np.delete(minor, j, axis=1)  # Remove the j-th column
    return minor

def cofactor_matrix(matrix):
    """Compute the cofactor matrix of a square matrix."""
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix must be square.")
    
    n = matrix.shape[0]
    cofactor = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            minor = get_minor(matrix, i, j)
            cofactor[i, j] = ((-1) ** (i + j)) * np.linalg.det(minor)
    
    return cofactor
    
#individual source integral terms in all three hypotheses

def Isource_vac(M, z, K, alpha, beta, Fisher, H0,Omega_m0,Omega_Lambda0,Mstar, indices = {'M':0,'z':1}):
    """ Source Integral approximation in the vacuum-GR hypothesis. 
    M, z are the inferred source parameters. K, alpha, beta are the hyperparameters.
    Fisher is the full Fisher matrix in the vac+loc+glob hypothesis at the true parameter point.
    indices is a dict of indices of the vacuum parameters [M, z] in the Fisher matrix.
    !! Transform Fisher from M, dl to M, z before calling this function !!
    """
        
    Fisher_vac_inds = np.ix_(list(indices.values()),list(indices.values()))
    Fisher_vac = Fisher[Fisher_vac_inds]
    
    Fisher_vac_inv = np.linalg.inv(Fisher_vac)

    return (prior_vac(M=M, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar) + (1/2*DDM_prior_vac(M=M, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*Fisher_vac_inv[indices['M'],indices['M']] +
                                                1/2*DDz_prior_vac(M=M, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*Fisher_vac_inv[indices['z'],indices['z']] +
                                                DMDz_prior_vac(M=M, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*Fisher_vac_inv[indices['M'],indices['z']]))


def Isource_glob(M, z, Ag, K, alpha, beta, Gdot, Fisher,H0,Omega_m0,Omega_Lambda0, Mstar, indices = {'M':0,'z':1,'Ag':-1}):
    """ Source Integral approximation in the Global effect hypothesis.
    M, z, Ag are the inferred source parameters. K, alpha, beta, Gdot are the hyperparameters.
    Fisher is the full Fisher matrix in the vac+loc+glob hypothesis at the true parameter point.
    indices is a dict of indices of the global parameters [M, z, Ag] in the Fisher matrix.
    !! Transform Fisher from M, dl to M, z before calling this function !!
    """

    #getting the Fisher for vac+global effect parameters
    dpsi = 3 #len(list(indices.keys())) #number of all parameters in the global effect hypothesis.
    
    Fisher_psipsi_inds = np.ix_(list(indices.values()),list(indices.values()))
    Fisher_psipsi = Fisher[Fisher_psipsi_inds]

    #inverse of Fisher_psipsi
    Fisher_psipsi_inv = np.linalg.inv(Fisher_psipsi)

    #cofactor matrix of Fisher_psipsi
    cofactor_psipsi = cofactor_matrix(Fisher_psipsi)
    #print('C_MAg: ',cofactor_psipsi[indices['M'],indices['Ag']])
    #print('C_zAg: ',cofactor_psipsi[indices['z'],indices['Ag']])
    
    #getting the Fisher for vac-only parameters
    indices_vac = {}
    for key in list(indices.keys()):
        if key in ['M','z']:
            indices_vac[key] = indices[key]

    dv = 2 #len(list(indices_vac.keys()))

    Fisher_vac_inds = np.ix_(list(indices_vac.values()),list(indices_vac.values()))
    Fisher_vac = Fisher[Fisher_vac_inds]

    Fisher_vac_inv = np.linalg.inv(Fisher_vac)

    #actually calculating the source integral
    Constant = ((np.linalg.det(Fisher_psipsi)/np.linalg.det(Fisher_vac))**(1/2))/((2*np.pi)**((dpsi-dv)/2)) #first term

    Expterm = np.exp(-1/2 * np.linalg.det(Fisher_psipsi)/np.linalg.det(Fisher_vac) * (Ag-Gdot)**2) #second term

    pvacterm = (prior_vac(M=M, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar) + ((1/np.linalg.det(Fisher_vac)**2)*\
                                                                            (1/2*DDM_prior_vac(M=M, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*(Fisher_psipsi[0,0]+cofactor_psipsi[indices['M'],indices['Ag']]**2*(Ag-Gdot)**2)
                                                                      + 1/2*DDz_prior_vac(M=M, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*(Fisher_psipsi[1,1]+cofactor_psipsi[indices['z'],indices['Ag']]**2*(Ag-Gdot)**2)
                                                                      + DMDz_prior_vac(M=M, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*(-Fisher_psipsi[0,1]+cofactor_psipsi[indices['z'],indices['Ag']]*cofactor_psipsi[indices['M'],indices['Ag']]*(Ag-Gdot)**2))))

    return_val = Constant*Expterm*pvacterm

    if return_val < 1e-50: #underfloat handling
        return 1e-50
    else:
        return return_val

def Isource_loc(M, z, vec_l, K, alpha, beta, f, mu_l, sigma_l, Fisher, H0,Omega_m0,Omega_Lambda0, Mstar, indices = {'M':0, 'z': 1, 'Al':2, 'nl':3}):
    """
    Source Integral approximation in the local effect hypothesis.
    M, z (np.float64) are the inferred vacuum parameters of the source. K, alpha, beta are the corresponding hyperparameters.
    vec_l (list/numpy 1d.array) = [Al, nl] is the list of inferred local effect parameters of the source.
    f (np.float64), mu_l = [mu_Al, mu_nl], sigma_l = [sigma_Al, sigma_nl] are the hyperparameters of the local effect.
    Fisher = Fisher_psipsi with coordinates [lnM, z, Al, nl] 4x4
    """

    sigma_l = np.array(sigma_l)
    mu_l = np.array(mu_l)
    vec_l = np.array(vec_l)
    
    Al, nl = vec_l
    
    #getting the Fisher for vac+local effect parameters
    dpsi = len(list(indices.keys()))
    
    indices_vac = {}
    for key in list(indices.keys()):
        if key in ['M','z']:
            indices_vac[key] = indices[key]

    dv = len(list(indices_vac.keys()))

    indices_loc = {}
    for key in list(indices.keys()):
        if key in ['Al','nl']:
            indices_loc[key] = indices[key]
            
    dl = len(list(indices_loc.keys()))
    
    Fisher_psipsi_inds = np.ix_(list(indices.values()),list(indices.values()))
    Fisher_psipsi = Fisher[Fisher_psipsi_inds] #Full Fisher in vac+local

    Fisher_vac_inds = np.ix_(list(indices_vac.values()),list(indices_vac.values()))
    Fisher_vac = Fisher[Fisher_vac_inds] #vacuum elements only

    Fisher_loc_inds = np.ix_(list(indices_loc.values()),list(indices_loc.values()))
    Fisher_loc = Fisher[Fisher_loc_inds]  #local effect elements only

    ### calculating the standardization factor in product of multivariate Gaussians
    #Fisher_tilde = Fisher_psipsi + [[0,0],[0,Fisher_l]]
    
    Fisher_l = np.diag(1/sigma_l**2)

    Fisher_tilde_additional = np.zeros_like(Fisher_psipsi)
    Fisher_tilde_additional[dl:,dl:] = Fisher_l

    Fisher_tilde = Fisher_psipsi + Fisher_tilde_additional #Fisher_tilde

    #psitilde
    psi_tilde_additional = np.zeros(dpsi)
    psi_tilde_additional[dl:] = Fisher_l@mu_l

    psi_vec = np.array([np.log(M),z,Al,nl])
                
    psi_tilde = np.linalg.inv(Fisher_tilde)@(Fisher_psipsi@psi_vec  + psi_tilde_additional) #psi_tilde
    
    lnM_tilde, z_tilde = psi_tilde[:dv] #v_tilde for I2 evaluation
    
    #standardization factor
    S = np.linalg.det(Fisher_loc+Fisher_l)**(1/2)/((2*np.pi)**(dpsi/2))*np.exp(-1/2*(vec_l - mu_l)@(Fisher_loc+Fisher_l)@(vec_l - mu_l))

    ### Calculating the source terms
    I1 = (1-f)*((np.linalg.det(Fisher_psipsi)/np.linalg.det(Fisher_vac))**(1/2))/((2*np.pi)**((dpsi-dv)/2))*Isource_vac(M=M, z=z, K=K, alpha=alpha, beta=beta, Fisher=Fisher,H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)

    #print(np.linalg.det(Fisher_psipsi)/np.linalg.det(Fisher_vac))
    
    I2 = S*f*Isource_vac(M=np.exp(lnM_tilde), z=z_tilde, K=K, alpha=alpha, beta=beta, Fisher=Fisher, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)

    #print(((np.linalg.det(Fisher_psipsi)/np.linalg.det(Fisher_vac))**(1/2))/((2*np.pi)**((dpsi-dv)/2)))
    
    return I1 + I2
        
#####################################################################################################
#####################################################################################################
### MAIN CLASS DEFINITION ###
#####################################################################################################
#####################################################################################################

class Hierarchical:

    """
        This class generates a population of extreme-mass-ratio inspirals (EMRIs) 
        for a fiducial set of hyperparameters on local and global perturbative effects on top of vacuum evolution 
        and calculates the approximate Bayes factor (Savage-Dickey ratio) comparing 
        three hypothesis: vacuum, vac+local effect only, vac+global effect only.

    Args:
        Npop (int): Number of EMRIs in the true population.
        SNR_thresh (float): Signal-to-noise ratio threshold to claim 'detected' EMRI set.
        sef_kwargs (dict): keyword arguments to provide to the StableEMRIFishers class. Must include:
                           'EMRI_waveform_gen', 'param_names'. All others optional.

        filename (string): folder name where the data is being stored. No default because impractical to not save results.
        filename_Fisher (string): a sub-folder for storing Fisher files (book-keeping). If None, Fishers directly stored in filename. Default is None. 

        true_hyper (dict): true values of all hyperparameters. Default are fiducial values consistent with a population of vacuum EMRIs.
        cosmo_params (dict): true values of 'Omega_m0' (matter density), 'Omega_Lambda0' (DE density), and 'H0' (Hubble constant in m/s/Gpc).

        source_bounds (dict): prior range on source parameters in all three hypotheses. Keys are param names and values are lists of lower and upper bounds. 
                              Must be provided for all parameters. We assume flat priors in this range.
        hyper_bounds (dict): prior range on population (hyper)params in all three hypotheses. Keys are param names and values are lists of lower and upper bounds. 
                             Must be provided for all hyperparams. We assume flat priors in this range.

        Tplunge_range (Union(list,NoneType)): lower and upper bounds on the time-to-plunge on EMRIs in the population. This will be used to initialize p0's for all EMRIs.
                              Default is None corresponding to Tplunge_range = [0.5,T_LISA + 1.0].
        
        T_LISA (float): time (in years) of LISA observation window. Default is 1.0.
        dt (float): LISA sampling frequency. Default is 1.0.
        Mstar (float) Constant in prior_vac. Default is 3e6. We choose it here following https://arxiv.org/pdf/1703.09722. Future implementations can vary this also.

        M_random (int): Number of random samples for Savage-Dickey ratio calculation. Default is int(1e4).
        Fisher_validation_kwargs (dict): Keyword arguments for FisherValidation class for Kulback-Leibler divergence calculation. 
                                         If not empty, must provide keys: ('KL_threshold', 'filename_Fisher_loc', 'filename_Fisher_glob', 'validate').
        make_nice_plots (bool): Make and save visualizations: scatterplots of source param distributions, inferred bias corner plots, source integrals as a function
                                function of hyperparameters, etc.
        plots_filename (string): custom filename for the plots file if make_nice_plots is True. If not provided, but make_nice_plots is True, plots are saved under the default name "fancy_plots". 
        
        random_seed (int or None): seed for random processes throughout the code. If NoneType, no seed is implemented. Default is 42.
    
    Returns:
        Bvac_loc (float): Savage-Dickey ratio preferring the vacuum 
        Bvac_glob (float): Savage-Dickey ratio preferring the vacuum over the global hypothesis.
        Bloc_glob (float): Savage-Dickey ratio preferring the local over the global hypothesis.
    """

    def __init__(self, Npop, SNR_thresh, sef_kwargs,
                       filename,filename_Fishers=None,
                       true_hyper={'K':5e-3,'alpha':0.0,'beta':0.0,
                                   'f':0.0,'mu_Al':1e-5,'mu_nl':8.0,'sigma_Al':1e-6,'sigma_nl':0.8,
                                   'Gdot':0.0},
                       cosmo_params={'Omega_m0':0.30,'Omega_Lambda0':0.70,'H0':70e3}, 
                       source_bounds={'M':[1e4,1e7],'z':[0.01,10.0],'Al':[0.0,1e-4],'nl':[0.0,10.0],'Ag':[0.0,1e-8]},
                       hyper_bounds={'K':[1e-3,1e-2],'alpha':[-0.5,0.5],'beta':[-0.5,0.5],
                                     'f':[0.0,1.0],'mu_Al':[1e-5,1e-5],'mu_nl':[8.0,8.0],'sigma_Al':[1e-6,1e-6],'sigma_nl':[0.8,0.8],
                                     'Gdot':[0.0,1e-8]},
                       Tplunge_range = None,
                       T_LISA = 1.0, dt = 10.0, Mstar = 3e6,
                       M_random = int(1e4),
                       Fisher_validation_kwargs = {},
                       out_of_bound_nature = 'edge',
                       make_nice_plots=False,
                       plots_filename='fancy_plots',
                       random_seed=42):

        if isinstance(Npop, int):
            self.Npop = Npop
        else:
            raise ValueError("Npop must be an integer > 0.")

        self.SNR_thresh = SNR_thresh

        self.filename = filename
        self.filename_Fishers = os.path.join(self.filename,filename_Fishers)
        self.sef_kwargs = sef_kwargs
        self.sef_kwargs['filename'] = self.filename_Fishers

        #true cosmology
        self.cosmo_params = cosmo_params
        
        self.Omega_m0 = cosmo_params['Omega_m0']
        self.Omega_Lambda0 = cosmo_params['Omega_Lambda0']
        self.H0 = cosmo_params['H0']

        #true population hyperparams.
        # K, alpha, beta are vacuum population hyperparameters
        # f, mu_Al, mu_nl, sigma_Al, sigma_nl are local-effect population hyperparameters
        # Gdot is the global-effect population hyperparameter.
        self.true_hyper = true_hyper
        
        self.K_truth = true_hyper['K']
        self.alpha_truth = true_hyper['alpha']
        self.beta_truth = true_hyper['beta']
        self.Mstar_truth = Mstar
        self.lambda_truth_vac = [self.K_truth,self.alpha_truth,self.beta_truth]
        
        self.f_truth = true_hyper['f']
        self.mu_Al_truth = true_hyper['mu_Al']
        self.mu_nl_truth = true_hyper['mu_nl']
        self.sigma_Al_truth = true_hyper['sigma_Al']
        self.sigma_nl_truth = true_hyper['sigma_nl']
        self.lambda_truth_loc = [self.f_truth,[self.mu_Al_truth,self.mu_nl_truth],[self.sigma_Al_truth,self.sigma_nl_truth]]
        
        self.Gdot_truth = true_hyper['Gdot']
        self.lambda_truth_glob = [self.Gdot_truth]

        #prior ranges on source parameters
        self.source_bounds = source_bounds
        
        self.M_range = source_bounds['M']
        self.z_range = source_bounds['z']
        self.Al_range = source_bounds['Al']
        self.nl_range = source_bounds['nl']
        self.Ag_range = source_bounds['Ag']

        #prior ranges on population (hyper)params
        self.hyper_bounds = hyper_bounds
        self.K_range = hyper_bounds['K']
        self.alpha_range = hyper_bounds['alpha']
        self.beta_range = hyper_bounds['beta']
        self.f_range = hyper_bounds['f']
        self.mu_Al_range = hyper_bounds['mu_Al']
        self.mu_nl_range = hyper_bounds['mu_nl']
        self.sigma_Al_range = hyper_bounds['sigma_Al']
        self.sigma_nl_range = hyper_bounds['sigma_nl']
        self.Gdot_range = hyper_bounds['Gdot']

        if Tplunge_range == None:
            self.Tplunge_range = [0.5,T_LISA + 1.0]
        else:
            self.Tplunge_range = Tplunge_range

        self.T_LISA = T_LISA
        self.dt = dt

        self.M_random = M_random

        self.Fisher_validation_kwargs = Fisher_validation_kwargs

        if out_of_bound_nature in ['edge', 'remove']:
            self.out_of_bound_nature = out_of_bound_nature
        else:
            warnings.warn("valid option for out_of_bound_nature: ['edge','remove']. Assuming default ('edge').")
            self.out_of_bound_nature = 'edge'
            
        self.make_nice_plots = make_nice_plots

        if self.make_nice_plots:
            self.plots_folder = os.path.join(self.filename, plots_filename)
                
            os.makedirs(self.plots_folder, exist_ok=True)

        self.seed = random_seed

    def __call__(self):

        ###########################################################################
        #generate a population according to prior distribution of model parameters
        #and the true values of population parameters
        ###########################################################################

        grid_size = int(1e4) #harcoded because does not matter as long as reasonably large.

        #generating vacuum parameter samples
        self.M_truth_samples, self.z_truth_samples = M_z_samples(N=self.Npop,
                                                                 M_range=self.M_range,z_range=self.z_range,
                                                                 lambda_v=self.lambda_truth_vac,grid_size=grid_size,
                                                                 H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0,Mstar=self.Mstar_truth,
                                                                 seed=self.seed)

        if self.make_nice_plots:
            plt.figure(figsize=(7,5))
            plt.scatter(np.log10(self.M_truth_samples),self.z_truth_samples,color='grey',alpha=0.5)
            plt.xlabel(r"$\log_{10}$(MBH masses M)",fontsize=16)
            plt.ylabel("redshifts z", fontsize=16)
            plt.title("True population",fontsize=16)
            plt.savefig(f'{self.plots_folder}/M_z_truth.png',dpi=300,bbox_inches='tight')
            plt.close()
        
        #generating local effect parameters samples
        self.Al_truth_samples, self.nl_truth_samples = A_n_samples(N=self.Npop,lambda_l=self.lambda_truth_loc,seed=self.seed)
        
        if self.make_nice_plots:
            plt.figure(figsize=(7,5))
            plt.scatter(self.Al_truth_samples,self.nl_truth_samples,color='grey',alpha=0.5)
            plt.xlabel(r"local-effect amp $A_l$",fontsize=16)
            plt.ylabel(r"local-effect slope $n_l$", fontsize=16)
            plt.title("True population",fontsize=16)
            plt.savefig(f'{self.plots_folder}/Al_nl_truth.png',dpi=300,bbox_inches='tight')
            plt.close()

        
        #generating global effect parameter samples 
        self.Ag_truth_samples = Ag_samples(N=self.Npop,lambda_g=self.lambda_truth_glob)

        if self.make_nice_plots:
            plt.figure(figsize=(7,5))
            plt.plot(self.Ag_truth_samples,color='grey')
            plt.ylabel(r"global-effect amp $A_g$", fontsize=16)
            plt.xlabel("Source index",fontsize=16)
            plt.title("True population",fontsize=16)
            plt.savefig(f'{self.plots_folder}/Ag_truth.png',dpi=300,bbox_inches='tight')
            plt.close()

        #generating all other model parameter samples
        (self.mu_truth_samples,
         self.a_truth_samples,
         self.qS_truth_samples,
         self.qK_truth_samples,
         self.phiS_truth_samples,
         self.phiK_truth_samples,
         self.Phi0_truth_samples,
         self.T_truth_samples) = other_param_samples(N=self.Npop,M_samples=self.M_truth_samples,Tplunge_range=self.Tplunge_range,seed=self.seed)
        
        try:
            self.p0_truth_samples = np.loadtxt(f"{self.filename}/p0samps.txt")
            print("p0 samples found")
        except FileNotFoundError:
            print("calculating p0 samples")
            self.p0_truth_samples = p0_samples_func(N=self.Npop,Msamps=self.M_truth_samples,
                                                    musamps=self.mu_truth_samples,
                                                    asamps=self.a_truth_samples,
                                                    Alsamps=self.Al_truth_samples,
                                                    nlsamps=self.nl_truth_samples,
                                                    Agsamps=self.Ag_truth_samples,
                                                    Tsamps=self.T_truth_samples,
                                                    seed=self.seed,
                                                    filename=self.filename)

        if self.make_nice_plots:
            plt.figure(figsize=(7,5))
            plt.scatter(np.log10(self.mu_truth_samples/self.M_truth_samples),self.p0_truth_samples,color='grey',alpha=0.5)
            plt.xlabel(r"$\log_{10}$(Mass ratio)",fontsize=16)
            plt.ylabel(r"$p_0$", fontsize=16)
            plt.title("True population",fontsize=16)
            plt.savefig(f'{self.plots_folder}/q_p0_truth.png',dpi=300,bbox_inches='tight')
            plt.close()

        #####################################################################
        #extracting the detected population using SNR threshold calculation
        #####################################################################
        self.calculate_detected()

        #print(self.detected_EMRIs)

        ####################################################################
        #transforming the Fishers from [M,dL,Al,nl,Ag] to [logM,z,Al,nl,Ag]
        ####################################################################

        Fisher_index = []
        varied_params = []
        for i in range(len(self.detected_EMRIs)):
            varied_params.append(np.array(np.array(self.detected_EMRIs[i]['transformed_params'])))
            Fisher_index.append(int(self.detected_EMRIs[i]['index']))
            
        varied_params = np.array(varied_params)
        Fisher_index = np.array(Fisher_index)

        for index, i in zip(Fisher_index,range(len(Fisher_index))):
        
            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                Gamma_i = f["Fisher"][:]
                    
            dist_i = self.detected_EMRIs[i]['true_params'][6] #true_params[6] = dist
            M_i = self.detected_EMRIs[i]['true_params'][0] #true_params[0] = M
            
            J = Jacobian(M_i, dist_i,self.H0,self.Omega_m0,self.Omega_Lambda0)
            
            Fisher_transformed = J.T@Gamma_i@J

            if (np.linalg.eigvals(Fisher_transformed) < 0.0).any():
                warnings.warn("positive-definiteness check failed for index: ", index)
                warnings.warn(f"removing source {index}...")
                self.detected_EMRIs = np.delete(self.detected_EMRIs, i)
                
            else:    
                print("positive-definiteness check passed for index: ", index, ". Saving Fisher...")
                with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "a") as f:
                    if not "Fisher_transformed" in f:
                        f.create_dataset("Fisher_transformed", data = Fisher_transformed)
                
        np.save(f"{self.filename}/detected_EMRIs",self.detected_EMRIs) #save updated list with check on positive-definiteness of Fisher_transformed

        ##################################################################
        #calculating the biased inferrence params in all three hypotheses
        ##################################################################
    
        self.inferred_params(hypothesis='vacuum')         
        self.inferred_params(hypothesis='local')         
        self.inferred_params(hypothesis='global')

        if self.make_nice_plots:
            self.corner_plot_biases()

        #######################################################
        #perform Fisher validation if KL_threshold is provided
        #######################################################

        if len(self.Fisher_validation_kwargs.keys()) > 0:
            print('Validating Fishers using KL-divergence...')
            
            self.KL_threshold = self.Fisher_validation_kwargs['KL_threshold']
            _, filename_Fishers = os.path.split(self.filename_Fishers)
            self.filename_Fishers_loc = self.Fisher_validation_kwargs['filename_Fishers_loc']
            self.filename_Fishers_glob = self.Fisher_validation_kwargs['filename_Fishers_glob']
            validate = self.Fisher_validation_kwargs['validate']

            fishervalidate = FisherValidation(self.sef_kwargs,
                     self.filename, filename_Fishers, self.filename_Fishers_loc, self.filename_Fishers_glob,
                     self.true_hyper, self.cosmo_params, self.source_bounds, self.hyper_bounds,
                     self.T_LISA, self.dt,
                     validate)
    
            fishervalidate()
            
            if self.make_nice_plots:
                fishervalidate.KL_divergence_plot(self.plots_folder)

        #############################################################
        #calculating the Savage-Dickey ratios in different hypotheses        
        #############################################################

        #savage-dickey preferring the vacuum hypothesis over local
        Bvac_loc = self.savage_dickey_vacloc()
        #print("Preference for vacuum over local: ", Bvac_loc)

        Bvac_glob = self.savage_dickey_vacglob()
        #print("Preference for vacuum over global: ", Bvac_glob)

        Bglob_loc = Bvac_loc/Bvac_glob
        #print("Preference for global over local: ", Bglob_loc)

        np.savetxt(f"{self.filename}/SD_ratios.txt",np.array([Bvac_loc,Bvac_glob,Bglob_loc]))

        return Bvac_loc, Bvac_glob, Bglob_loc

    def calculate_detected(self):
        """ calculate the SNRs of the sources in the population. For sources with SNR > thresh,
        calculate and save the FIMs and parameter values. """

        try:
            self.detected_EMRIs = np.load(f'{self.filename}/detected_EMRIs.npy', allow_pickle=True)
            all_SNRs = np.loadtxt(f'{self.filename}/all_SNRs.txt')
            
        except FileNotFoundError:
            print("Calculating FIMs for the detectable EMRI population.")
            
            self.detected_EMRIs = []
            all_SNRs = []
                    
            for i in tqdm(range(self.Npop)):
                M = self.M_truth_samples[i]
                mu = self.mu_truth_samples[i]
                a = self.a_truth_samples[i]
                e0 = 0.0
                Y0 = 1.0
                dL = getdistGpc(self.z_truth_samples[i],self.H0,self.Omega_m0,self.Omega_Lambda0) #Gpc
                
                qS = self.qS_truth_samples[i]
                phiS = self.phiS_truth_samples[i]
                qK = self.qK_truth_samples[i]
                phiK = self.phiK_truth_samples[i]
                Phi_phi0 = self.Phi0_truth_samples[i]
                Phi_theta0 = 0.0
                Phi_r0 = 0.0
                T = self.T_LISA #all sources plunge at or after T_LISA, so the observation window is T_LISA at max.
                dt = self.dt
    
                Al = self.Al_truth_samples[i]
                nl = self.nl_truth_samples[i]
    
                Ag = self.Ag_truth_samples[i]
                ng = 4.0
    
                p0 = self.p0_truth_samples[i]
    
                self.sef_kwargs['suffix'] = i
    
                param_list = [M,mu,a,p0,e0,Y0,
                              dL,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,
                              ] #SEF param args (vacuum-GR EMRI)

                add_param_args = {"Al":Al, "nl":nl, "Ag":Ag, "ng":ng} #dict of additional parameters
    
                transformed_params = [np.log(M),self.z_truth_samples[i],Al,nl,Ag]
                
                emri_kwargs = {'T': T, 'dt': dt}
    
                #print(param_list, self.T_truth_samples[i])
                
                sef = StableEMRIFisher(*param_list, add_param_args=add_param_args, **emri_kwargs, **self.sef_kwargs)
                all_SNRs.append(sef.SNRcalc_SEF())
    
                if all_SNRs[i] >= self.SNR_thresh:
                    self.detected_EMRIs.append({'index': i,'true_params': np.array(param_list),'SNR':all_SNRs[i], 
                                                'lambda_v':self.lambda_truth_vac, 'lambda_l':self.lambda_truth_loc, 'lambda_g':self.lambda_truth_glob,
                                               'transformed_params':np.array(transformed_params)})
                    try:
                        with h5py.File(f"{self.filename_Fishers}/Fisher_{i}.h5", "r") as f:
                            Gamma_i = f["Fisher"][:]
                    except FileNotFoundError:
                        sef() #calculate and save the FIM for the detected EMRI
    
            all_SNRs = np.array(all_SNRs)
            self.detected_EMRIs = np.array(self.detected_EMRIs)
            np.save(f"{self.filename}/detected_EMRIs",self.detected_EMRIs)
            np.savetxt(f"{self.filename}/all_SNRs.txt",np.array(all_SNRs))
    
        print(f"#detected EMRIs: {len(self.detected_EMRIs)}")

        if self.make_nice_plots:
            counts, bins, patches = plt.hist(all_SNRs, bins=50)
            for patch, bin_left in zip(patches, bins[:-1]):
                if bin_left >= self.SNR_thresh:
                    patch.set_facecolor('red')
                else:
                    patch.set_facecolor('grey')

            plt.axvline(self.SNR_thresh,color='k',linestyle='--',label='SNR threshold')
            plt.legend()
            plt.xlabel("SNRs",fontsize=16)
            plt.yscale("log")
            plt.savefig(f"{self.plots_folder}/SNR_dist.png",dpi=300,bbox_inches='tight')
            plt.close()

    def inferred_params(self,hypothesis='vacuum'):
        """ calculate and save the inferred biased params in the given hypothesis.
        choose between 'vacuum', 'local', or 'global' 
        """
        
        for i in range(len(self.detected_EMRIs)):
        
            # d: number of measured source params
            # Nphi: number of unmeasured params
            # Npsi: number of measured params
        
            index = int(self.detected_EMRIs[i]["index"])
            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                Gamma_i = f["Fisher_transformed"][:] #Fisher in transformed coords [lnM,z,Al,nl,Ag]
    
            if hypothesis == 'vacuum':
                #vacuum hypothesis
                indices_psi = [0,1]  #indices of measured params (lnM, z)
                indices_phi = [2,3,4] #indices of unmeasured params (Al, nl, Ag)
                
                i_psipsi = np.ix_(indices_psi,indices_psi)
                i_psiphi = np.ix_(indices_psi,indices_phi)
                i_phiphi = np.ix_(indices_phi,indices_phi) 
                
                Gamma_i_psipsi_inv = np.linalg.inv(Gamma_i[i_psipsi]) # Npsi x Npsi array
                Gamma_i_psiphi = Gamma_i[i_psiphi] # Npsi x Nphi array
            
                multiplicative_factor = Gamma_i_psipsi_inv@Gamma_i_psiphi # Npsi x Nphi array
                
                psi_i = np.array(self.detected_EMRIs[i]["transformed_params"])[indices_psi] #Npsi 1D array
                phi_i = np.array(self.detected_EMRIs[i]["transformed_params"])[indices_phi] #Nphi 1D array
    
                psi_i_inferred = bias(psi_i, phi_i, multiplicative_factor)
                psi_i_inferred = np.concatenate((psi_i_inferred,[0.0,0.0,0.0])) #size = Npsi + Nphi
    
                self.detected_EMRIs[i]["vacuum_params"] = np.array(psi_i_inferred) #save [lnM_bias,z_bias]
    
            if hypothesis == 'local':
                #local hypothesis
                indices_psi = [0,1,2,3]  #indices of measured params (lnM,z,Al,nl)
                indices_phi = [4] #indices of unmeasured params (Ag)
                
                i_psipsi = np.ix_(indices_psi,indices_psi)
                i_psiphi = np.ix_(indices_psi,indices_phi)
                i_phiphi = np.ix_(indices_phi,indices_phi) 
                
                Gamma_i_psipsi_inv = np.linalg.inv(Gamma_i[i_psipsi]) # Npsi x Npsi array
                Gamma_i_psiphi = Gamma_i[i_psiphi] # Npsi x Nphi array
            
                multiplicative_factor = Gamma_i_psipsi_inv@Gamma_i_psiphi # Npsi x Nphi array
                
                psi_i = np.array(self.detected_EMRIs[i]["transformed_params"])[indices_psi] #Npsi 1D array
                phi_i = np.array(self.detected_EMRIs[i]["transformed_params"])[indices_phi] #Nphi 1D array
    
                psi_i_inferred = bias(psi_i, phi_i, multiplicative_factor)
                """
                if psi_i_inferred[2] < 1e-14:
                    psi_i_inferred[2] = 1e-14 #Al cannot be negative.
                    psi_i_inferred[3] = 1e-14
                if psi_i_inferred[3] < 1e-14:
                    psi_i_inferred[3] = 1e-14 #nl cannot be negative.
                    psi_i_inferred[2] = 1e-14
                """
                psi_i_inferred = np.concatenate((psi_i_inferred,[0.0])) #size = Npsi + Nphi
    
                self.detected_EMRIs[i]["local_params"] = np.array(psi_i_inferred) #save [lnM_bias,z_bias,Al_bias, nl_bias]
    
            if hypothesis == 'global':
                #global hypothesis
                indices_psi = [0,1,4]  #indices of measured params (lnM,z,Ag)
                indices_phi = [2,3] #indices of unmeasured params (Al,nl)
                
                i_psipsi = np.ix_(indices_psi,indices_psi)
                i_psiphi = np.ix_(indices_psi,indices_phi)
                i_phiphi = np.ix_(indices_phi,indices_phi) 
    
                Gamma_i_psipsi_inv = np.linalg.inv(Gamma_i[i_psipsi]) # Npsi x Npsi array
                Gamma_i_psiphi = Gamma_i[i_psiphi] # Npsi x Nphi array
    
                multiplicative_factor = Gamma_i_psipsi_inv@Gamma_i_psiphi # Npsi x Nphi array
    
                psi_i = np.array(self.detected_EMRIs[i]["transformed_params"])[indices_psi] #Npsi 1D array
                phi_i = np.array(self.detected_EMRIs[i]["transformed_params"])[indices_phi] #Nphi 1D array
    
                psi_i_inferred = bias(psi_i, phi_i, multiplicative_factor)
                """
                if psi_i_inferred[-1] < 1e-14:
                    psi_i_inferred[-1] = 1e-14 #global effect cannot be negative.
                """
                psi_i_inferred = np.concatenate((np.concatenate((psi_i_inferred[:2],[0.0,0.0])),[psi_i_inferred[-1]]))
                
                self.detected_EMRIs[i]["global_params"] = np.array(psi_i_inferred) #save [lnM_bias,z_bias,Ag_bias]

        self.Nobs = len(self.detected_EMRIs) #number of detected EMRIs.
        np.save(f'{self.filename}/detected_EMRIs',self.detected_EMRIs)

    def source_integral_vac(self,K,alpha,beta):
        
        """Calculate the source integral in the vacuum hypothesis.
        bounds_vac is a dict of bounds on M and z. Bounds can be given for any subset of the parameters."""
                
        #calculate source integral
        Ivac_all = []

        Nobs = self.Nobs
        count = 0.0 #number of out of bound EMRIs
        
        bounds_vac = {'logM':np.log(self.source_bounds['M']),'z':self.source_bounds['z']}

        for i in range(len(self.detected_EMRIs)):
            
            out_of_bounds = False
            index = int(self.detected_EMRIs[i]["index"])
            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                Fisher = f["Fisher_transformed"][:] #Fisher in transformed coords [lnM,z,Al,nl,Ag]
            
            vacparams = self.detected_EMRIs[i]["vacuum_params"] # logMvac, zvac, Alvac, nlvac, Agvac
        
            for param,j in zip(bounds_vac.keys(),range(len(bounds_vac.keys()))):
                if check_prior(vacparams[j],bounds_vac[param]) == 1: #if the source parameters hits the upper limit
                    out_of_bounds = True
                    warnings.warn(f"source {index} is out of prior bounds on {param} (upper bound hit). \n\
                            Parameter value: {vacparams[j]}. Bound: {bounds_vac[param]}.")
                    varparams[j] = bounds_vac[param][1] #varparam takes the upper limit value
                elif check_prior(vacparams[j],bounds_vac[param]) == -1: #if the source parameter hits the lower limit
                    out_of_bounds = True
                    warnings.warn(f"source {index} is out of prior bounds on {param} (lower bound hit). \n\
                            Parameter value: {vacparams[j]}. Bound: {bounds_vac[param]}.")
                    vacparams[j] = bounds_vac[param][0] #vacparam takes the lower limit value

            if out_of_bounds:
                count+=1

            Ivac_i = Isource_vac(M=np.exp(vacparams[0]),z=vacparams[1], 
                                K=K, alpha=alpha, beta=beta, #variable hyperparameters
                                Fisher=Fisher,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0,Mstar=self.Mstar_truth)

            if out_of_bounds and self.out_of_bound_nature == 'remove':
                Ivac_all.append(1.0)
                Nobs -= 1

            else:
                Ivac_all.append(Ivac_i)
    
        warnings.warn(f"EMRIs out-of-bounds: {int(count)} out of total {int(len(Fishers_all))}")
    
        return factorial(Nobs-1)*np.prod(np.array(Ivac_all))

    def source_integral_loc(self,K,alpha,beta,f,mu_Al,mu_nl,sigma_Al,sigma_nl,Fishers_all,indices_all,locparams_all):
        
        """Calculate the source integral in the local hypothesis.
        bounds_loc is a dict of bounds on M, z, Al, nl. Bounds can be given for any subset of the parameters."""
        
        Nobs = self.Nobs
        count = 0.0 #number of out of bound EMRIs
        
        #calculate source integral
        Iloc_all = []

        bounds_loc = {'logM':np.log(self.source_bounds['M']),'z':self.source_bounds['z'],
                      'Al':self.source_bounds['Al'],'nl':self.source_bounds['nl']} #prior range
    
        for i, index in zip(range(len(Fishers_all)),indices_all):
            out_of_bounds = False
            Fisher = Fishers_all[i] #Fisher in transformed coords [lnM,z,Al,nl,Ag]
            locparams = locparams_all[i]
            
            for param,j in zip(bounds_loc.keys(),range(len(bounds_loc.keys()))):
                if check_prior(locparams[j],bounds_loc[param]) == 1: #if the source parameters hits the upper limit
                    out_of_bounds = True
                    warnings.warn(f"source {index} is out of prior bounds on {param} (upper bound hit). \n\
                            Parameter value: {locparams[j]}. Bound: {bounds_loc[param]}.")
                    locparams[j] = bounds_loc[param][1] #locparam takes the upper limit value
                elif check_prior(locparams[j],bounds_loc[param]) == -1: #if the source parameter hits the lower limit
                    out_of_bounds = True
                    warnings.warn(f"source {index} is out of prior bounds on {param} (lower bound hit). \n\
                            Parameter value: {locparams[j]}. Bound: {bounds_loc[param]}.")
                    locparams[j] = bounds_loc[param][0] #locparam takes the lower limit value

            if out_of_bounds:
                count+=1
    
            Iloc_i = Isource_loc(M=np.exp(locparams[0]),z=locparams[1], vec_l=[locparams[2],locparams[3]], 
                                K=K, alpha=alpha, beta=beta, 
                                f=f, mu_l=[mu_Al,mu_nl], sigma_l=[sigma_Al,sigma_nl], 
                                Fisher=Fisher,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0,
                                Mstar=self.Mstar_truth)
    
            if out_of_bounds and self.out_of_bound_nature == 'remove':
                Iloc_all.append(1.0)
                Nobs -= 1
                
            elif np.isnan(Iloc_i):
                Iloc_all.append(1.0)
                Nobs -= 1
                
            else:
                Iloc_all.append(Iloc_i)

        lnposterior = np.sum(np.log(np.array(Iloc_all))) #avoid overflow by calculating log posterior
        if count > 0.0:
            warnings.warn(f"EMRIs out-of-bounds: {int(count)} out of total {int(len(Fishers_all))}")
        
        return lnposterior

    def source_integral_glob(self,K,alpha,beta,Gdot,Fishers_all,indices_all,globparams_all):
        
        """Calculate the source integral in the global hypothesis.
        bounds_loc is a dict of bounds on M, z, Al, nl. Bounds can be given for any subset of the parameters."""
            
        Nobs = self.Nobs
        count = 0.0 #number of out of bound EMRIs
        
        #calculate source integral
        Iglob_all = []

        bounds_glob = {'logM':np.log(self.source_bounds['M']),'z':self.source_bounds['z'],
                      'Ag':self.source_bounds['Ag']} #prior range
    
        for i, index in zip(range(len(Fishers_all)),indices_all):
            out_of_bounds = False
            Fisher = Fishers_all[i] #Fisher in transformed coords [lnM,z,Al,nl,Ag]
    
            globparams = globparams_all[i] # logMglob, zglob, Alglob, nlglob, Agglob
            
            for param,j in zip(bounds_glob.keys(),range(len(bounds_glob.keys()))):
                if check_prior(globparams[j],bounds_glob[param]) == 1: #if the source parameters hits the upper limit
                    out_of_bounds = True
                    warnings.warn(f"source {index} is out of prior bounds on {param} (upper bound hit). \n\
                            Parameter value: {globparams[j]}. Bound: {bounds_glob[param]}.")
                    globparams[j] = bounds_glob[param][1] #varparam takes the upper limit value
                elif check_prior(globparams[j],bounds_glob[param]) == -1: #if the source parameter hits the lower limit
                    out_of_bounds = True
                    warnings.warn(f"source {index} is out of prior bounds on {param} (lower bound hit). \n\
                            Parameter value: {globparams[j]}. Bound: {bounds_glob[param]}.")
                    globparams[j] = bounds_glob[param][0] #vacparam takes the lower limit value

            if out_of_bounds:
                count+=1
    
            Iglob_i = Isource_glob(M=np.exp(globparams[0]),z=globparams[1],Ag=globparams[-1],
                                  K=K, alpha=alpha, beta=beta, 
                                  Gdot=Gdot,Mstar=self.Mstar_truth,
                                  Fisher=Fisher,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
            
            if out_of_bounds and self.out_of_bound_nature == 'remove':
                Iglob_all.append(1.0)
                Nobs -= 1

            elif np.isnan(Iglob_i):
                Iglob_all.append(1.0)
                Nobs -= 1
                
            else:
                Iglob_all.append(Iglob_i)

        lnposterior = np.sum(np.log(np.array(Iglob_all))) #avoid overflow by calculating log posterior
        if count > 0.0:
            warnings.warn(f"EMRIs out-of-bounds: {int(count)} out of total {int(len(Fishers_all))}")

        return lnposterior

    def savage_dickey_vacloc(self):
        #no seed ideally required for this calculation
        t = 1e6 * time.time() # current time in microseconds
        np.random.seed(int(t) % 2**32)

        samples = np.random.uniform(
                                    low=[self.K_range[0], self.alpha_range[0], self.beta_range[0], self.f_range[0], 
                                         self.mu_Al_range[0], self.mu_nl_range[0], self.sigma_Al_range[0], self.sigma_nl_range[0]],
                                    high=[self.K_range[1], self.alpha_range[1], self.beta_range[1], self.f_range[1], 
                                          self.mu_Al_range[1], self.mu_nl_range[1], self.sigma_Al_range[1], self.sigma_nl_range[1]],
                                    size=(self.M_random, 8)
                                    )
        
        K_samples, alpha_samples, beta_samples, f_samples, mu_Al_samples, mu_nl_samples, sigma_Al_samples, sigma_nl_samples = samples.T

        #make sure f_samples have at least 10% draws at the null value for SD calculation
        f_samples = f_samples[:int(0.9*self.M_random)]
        f_samples = np.concatenate((f_samples,np.zeros(self.M_random-len(f_samples))))

        Fishers_all = []
        indices_all = []
        locparams_all = []      
        for i in range(len(self.detected_EMRIs)):
            index = int(self.detected_EMRIs[i]["index"])
            indices_all.append(index)
            
            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                Fish_trans = f["Fisher_transformed"][:]
                
            Fishers_all.append(Fish_trans)
            
            locparams_all.append(self.detected_EMRIs[i]["local_params"])

        indices_all = np.array(indices_all)
        Fishers_all = np.array(Fishers_all)
        locparams_all = np.array(locparams_all)
        
        #only choose Fishers which satisfy the KL-divergence threshold, if available.
        if len(self.Fisher_validation_kwargs.keys()) > 0:
            if self.f_truth > 0:
                Fishers_loc_KL = np.loadtxt(f'{self.filename}/Fishers_loc_KL.txt')
                Fishers_all_KL = []
                indices_all_KL = []
                locparams_all_KL = []
                j = 0
                for i in range(len(self.detected_EMRIs)):
                    Al = self.detected_EMRIs[i]['local_params'][2]
                    nl = self.detected_EMRIs[i]['local_params'][3]
                    Ag = self.detected_EMRIs[i]['local_params'][4] #will be zero in local hypothesis
                    ng = 4.0
            
                    if Fishers_loc_KL[j] < self.KL_threshold: #KL-divergence of jth source should be less than the threshold.
                        Fishers_all_KL.append(Fishers_all[j])
                        indices_all_KL.append(indices_all[j])
                        locparams_all_KL.append(locparams_all[j])

                    j += 1 #hacky afterthought to cycle through Fishers_loc_KL

                Fishers_all = np.array(Fishers_all_KL) #update Fishers_all
                indices_all = np.array(indices_all_KL) #update indices_all

                if len(Fishers_all) != len(self.detected_EMRIs):
                    warnings.warn(f"After KL-divergence validation, only {len(Fishers_all)} sources remain.")
        
        lnprodIsource = []
        removed_indices = []
    
        for j in tqdm(range(self.M_random)):
            lnprodIsource_j = self.source_integral_loc(K=K_samples[j],alpha=alpha_samples[j],beta=beta_samples[j],
                                                        f=f_samples[j],mu_Al=mu_Al_samples[j],mu_nl=mu_nl_samples[j],
                                                        sigma_Al=sigma_Al_samples[j],sigma_nl=sigma_nl_samples[j],
                                                        Fishers_all=Fishers_all, indices_all=indices_all,locparams_all=locparams_all)
            
            lnprodIsource.append(lnprodIsource_j)
    
        lnprodIsource = np.array(lnprodIsource) - np.max(lnprodIsource)
        prodIsource = np.exp(lnprodIsource)

        for i in range(len(prodIsource)):
            if prodIsource[i] < 1e-300: #control underflow
                prodIsource[i] = 1e-300
        
        prodIsource = prodIsource/np.sum(prodIsource)
        
        #f=0 mask
        num_bins = 40
        mask = np.abs(f_samples - 0.0) < (max(f_samples)-min(f_samples))/num_bins
        
        while sum(mask) < 10: #make sure at least ten sample point in the null hypothesis.
            warnings.warn("No samples consistent with the null hypothesis. Reducing bin size. The Bayes factor may be incorrect. Increase M_samples!")
            num_bins -= 5
            mask = np.abs(f_samples - 0.0) < (max(f_samples)-min(f_samples))/num_bins
            
        prior_f0 = sum(mask)/len(prodIsource) #prior number of points within the bin for f = 0 
        posterior_f0 = np.sum(prodIsource[mask])
    
        print("prior_f0: ", prior_f0)
        print("posterior_f0: ", posterior_f0)

        if self.make_nice_plots:
            plt.figure(figsize=(7,5))
            plt.scatter(f_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(f_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.f_truth,color='k',linestyle='--',label='truth')
            plt.xlabel("fraction of local-effect EMRIs (f)", fontsize=16)
            plt.ylabel("posterior pdf p(f|data)",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_f.png",dpi=300,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(7,5))
            plt.scatter(mu_Al_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(mu_Al_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.mu_Al_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"mean disk-effect amplitude of local-effect EMRIs ($\mu_{Al}$)", fontsize=16)
            plt.ylabel(r"posterior pdf p($\mu_{Al}$|data)",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_muAl.png",dpi=300,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(7,5))
            plt.scatter(mu_nl_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(mu_nl_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.mu_nl_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"mean disk-effect slope of local-effect EMRIs ($\mu_{nl}$)", fontsize=16)
            plt.ylabel(r"posterior pdf p($\mu_{nl}$|data)",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_munl.png",dpi=300,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(7,5))
            plt.scatter(sigma_Al_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(sigma_Al_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.sigma_Al_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"std of disk-effect amplitude of local-effect EMRIs ($\sigma_{Al}$)", fontsize=16)
            plt.ylabel(r"posterior pdf p($\sigma_{Al}$|data)",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_sigmaAl.png",dpi=300,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(7,5))
            plt.scatter(sigma_nl_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(sigma_nl_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.sigma_nl_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"std of disk-effect slope of local-effect EMRIs ($\sigma_{nl}$)", fontsize=16)
            plt.ylabel(r"posterior pdf p($\sigma_{nl}$|data)",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_sigmanl.png",dpi=300,bbox_inches='tight')
            plt.close()
        
        return posterior_f0/prior_f0
    
    def savage_dickey_vacglob(self):
        #no seed ideally required for this calculation
        t = 1e6 * time.time() # current time in microseconds
        np.random.seed(int(t) % 2**32)

        samples = np.random.uniform(
                                    low=[self.K_range[0], self.alpha_range[0], self.beta_range[0], self.Gdot_range[0]],
                                    high=[self.K_range[1], self.alpha_range[1], self.beta_range[1], self.Gdot_range[1]],
                                    size=(self.M_random, 4)
                                    )
        
        K_samples, alpha_samples, beta_samples, Gdot_samples = samples.T

        #make sure Gdot_samples have at least 10% draws at the null value for SD calculation
        Gdot_samples = Gdot_samples[:int(0.9*self.M_random)]
        Gdot_samples = np.concatenate((Gdot_samples,np.zeros(self.M_random-len(Gdot_samples))))

        indices_all = []
        Fishers_all = []
        globparams_all = []
        for i in range(len(self.detected_EMRIs)):
            index = int(self.detected_EMRIs[i]["index"])
            indices_all.append(index)
            
            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                Fish_trans = f["Fisher_transformed"][:]
                
            Fishers_all.append(Fish_trans)
            globparams_all.append(self.detected_EMRIs[i]["global_params"])

        indices_all = np.array(indices_all)
        Fishers_all = np.array(Fishers_all)
        globparams_all = np.array(globparams_all)

        #only choose Fishers which satisfy the KL-divergence threshold, if available.
        if len(self.Fisher_validation_kwargs.keys()) > 0:
            if self.Gdot_truth > 0:
                Fishers_glob_KL = np.loadtxt(f'{self.filename}/Fishers_glob_KL.txt')
                Fishers_all_KL = []
                indices_all_KL = []
                globparams_all_KL = []
                j = 0
                for i in range(len(self.detected_EMRIs)):
                    Al = self.detected_EMRIs[i]['global_params'][2] #will be zero in global hypothesis
                    nl = self.detected_EMRIs[i]['global_params'][3] #will be zero in global hypothesis
                    Ag = self.detected_EMRIs[i]['global_params'][4] 
                    ng = 4.0
            
                    if Fishers_glob_KL[j] < self.KL_threshold: #KL-divergence of jth source should be less than the threshold.
                        Fishers_all_KL.append(Fishers_all[j])
                        indices_all_KL.append(indices_all[j])
                        globparams_all_KL.append(globparams_all[j])

                    j += 1 #hacky afterthought to cycle through Fishers_glob_KL

                Fishers_all = np.array(Fishers_all_KL) #update Fishers_all
                indices_all = np.array(indices_all_KL) #update indices_all

                if len(Fishers_all) != len(self.detected_EMRIs):
                    warnings.warn(f"After KL-divergence validation, only {len(Fishers_all)} sources remain.")
    
        lnprodIsource = []
        #removed_indices = []
        for j in tqdm(range(self.M_random)):
                
            lnprodIsource_j = self.source_integral_glob(K=K_samples[j], alpha=alpha_samples[j], beta=beta_samples[j],
                                                  Gdot=Gdot_samples[j],Fishers_all=Fishers_all,indices_all=indices_all,globparams_all=globparams_all)

            #print(K_samples[j], alpha_samples[j], beta_samples[j], Gdot_samples[j], prodIsource_j)
    
            lnprodIsource.append(lnprodIsource_j)
    
        lnprodIsource = np.array(lnprodIsource) - np.max(lnprodIsource)
        prodIsource = np.exp(lnprodIsource)

        for i in range(len(prodIsource)):
            if prodIsource[i] < 1e-300: #control underflow
                prodIsource[i] = 1e-300
                
        prodIsource = prodIsource/np.sum(prodIsource)
            
        #Gdot=0 mask
        num_bins = 40
        mask = np.abs(Gdot_samples - 0.0) < (max(Gdot_samples)-min(Gdot_samples))/num_bins

        while sum(mask) < 10: #make sure at least ten sample point in the null hypothesis.
            warnings.warn("No samples consistent with the null hypothesis. Reducing bin size. The Bayes factor may be incorrect. Increase M_samples!")
            num_bins -= 5
            mask = np.abs(Gdot_samples - 0.0) < (max(Gdot_samples)-min(Gdot_samples))/num_bins
        
        prior_Gdot0 = sum(mask)/self.M_random #prior number of points within the bin for Gdot = 0 
        posterior_Gdot0 = np.sum(prodIsource[mask])
    
        print("prior_Gdot0: ", prior_Gdot0)
        print("posterior_Gdot0: ", posterior_Gdot0)

        if self.make_nice_plots:
            plt.figure(figsize=(7,5))
            plt.scatter(Gdot_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(Gdot_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.Gdot_truth,color='k',linestyle='--',label='truth')
            plt.xlabel("value of global-effect (Gdot)",fontsize=16)
            plt.ylabel("posterior pdf p(Gdot|data)",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_glob.png",dpi=300,bbox_inches='tight')
            plt.close()
        
        return posterior_Gdot0/prior_Gdot0

    def corner_plot_biases(self):
        Fisher_index = []
        varied_params = []
        vacuum_params = []
        local_params = []
        global_params = []
        SNRs = []
        
        for i in range(len(self.detected_EMRIs)):
            varied_params.append(np.array(np.array(self.detected_EMRIs[i]['transformed_params'])))
            vacuum_params.append(self.detected_EMRIs[i]['vacuum_params'])
            local_params.append(self.detected_EMRIs[i]['local_params'])
            global_params.append(self.detected_EMRIs[i]['global_params'])
            Fisher_index.append(int(self.detected_EMRIs[i]['index']))
            SNRs.append(self.detected_EMRIs[i]['SNR'])
            
        varied_params = np.array(varied_params)
        vacuum_params = np.array(vacuum_params)
        local_params = np.array(local_params)
        global_params = np.array(global_params)
        Fisher_index = np.array(Fisher_index)
        SNRs = np.array(SNRs)
        
        params = ['$\\log{M}$','$z$','$A_l$','$n_l$','$A_g$']
        param_lims = [np.log(self.M_range),self.z_range,self.Al_range,self.nl_range,self.Ag_range]
        fig, axs = plt.subplots(len(params),len(params),figsize=(40,40))

        plt.subplots_adjust(hspace=0, wspace=0)
        
        for i in range(len(params)):
            for j in range(len(params)):
                if j < i:
                    axs[i,j].scatter(varied_params[:,j],varied_params[:,i],s=SNRs,label='true',alpha=0.5)
                    axs[i,j].scatter(vacuum_params[:,j],vacuum_params[:,i],s=SNRs,label='vac',alpha=0.5)
                    axs[i,j].scatter(local_params[:,j],local_params[:,i],s=SNRs,label='loc',alpha=0.5)
                    axs[i,j].scatter(global_params[:,j],global_params[:,i],s=SNRs,label='global',alpha=0.5)

                    axs[i, j].grid(linestyle='--')
                    
                    if i == len(params)-1:
                        axs[i,j].set_xlabel(params[j],fontsize=46)
                    else:
                        axs[i, j].set_xticklabels([])
                    if j == 0:
                        axs[i,j].set_ylabel(params[i],fontsize=46)
                    else:
                        axs[i, j].set_yticklabels([])
                    
                    axs[i,j].set_xlim(param_lims[j])
                    axs[i,j].set_ylim(param_lims[i])
                    
                else:
                    axs[i,j].remove()
        
        handles, labels = axs[1, 0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=60, loc='upper right', bbox_to_anchor=(0.7, 0.7))  # Place legend outside

        plt.savefig(f"{self.plots_folder}/inferred_vs_truth.png",dpi=300,bbox_inches='tight')
        plt.close()
