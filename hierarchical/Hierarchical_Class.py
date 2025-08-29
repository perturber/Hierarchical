#imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
try:
    import cupy as cp
except:
    pass #SEF would shout anyway.
import time
import h5py

from few.utils.utility import get_p_at_t
from few.trajectory.inspiral import EMRIInspiral

from few.utils.constants import SPEED_OF_LIGHT as C_SI

from scipy.stats import uniform
import warnings

from hierarchical.FisherValidation import FisherValidation
from hierarchical.utility import integrand_dc, dc, getdist, Jacobian, check_prior
from hierarchical.JointWave import JointRelKerrEccFlux

########################################
#lots of supporting utility functions
########################################

#source parameter prior pdfs in all three hypotheses

def prior_vac(lnM, z, K, alpha, beta, H0, Omega_m0,Omega_Lambda0, Mstar):
    """
    given the vacuum hyperparams [K,alpha,beta],
    calculate the UNNORMALIZED probability distribution function of 
    obtaining the source params [M,z]
    """
    M = np.exp(lnM)

    C = K * ((1/Mstar)**alpha) * 4 * np.pi

    return C * (M**alpha) * ((1+z)**beta) * (dc(z,H0,Omega_m0,Omega_Lambda0)**2)

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
def lnM_z_samples(N,lnM_range,z_range,lambda_v,grid_size,H0,Omega_m0,Omega_Lambda0,Mstar,seed):
    """ function to generate N samples of the local effect parameters M, z
    from lnM in lnMrange, z in zrange, given hyperparameters lambda_v and a grid of size grid_size
    """
    
    np.random.seed(seed)

    K_truth, alpha_truth, beta_truth = lambda_v

    lnM_grid = uniform.rvs(loc=lnM_range[0],scale=lnM_range[1]-lnM_range[0],size=grid_size) #generating samples first from a uniform grid
    z_grid = uniform.rvs(loc=z_range[0],scale=z_range[1]-z_range[0],size=grid_size)

    prior_lnMz = []
    for i in range(grid_size):
        prior_lnMz.append(prior_vac(lnM_grid[i],z_grid[i],K_truth,alpha_truth,beta_truth,H0,Omega_m0,Omega_Lambda0,Mstar))
    
    prior_lnMz = np.array(prior_lnMz)/np.sum(np.array(prior_lnMz)) #normalizing
    
    #choosing N sources based on the probability distribution
    indices = range(grid_size)
    chosen = np.random.choice(indices,size=N,p=prior_lnMz)
    lnM_samples = lnM_grid[chosen]
    z_samples = z_grid[chosen]    

    return lnM_samples, z_samples

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
        p_plunge = get_p_at_t(traj_xy,
                              Tsamps[i],
                              [Msamps[i],
                               musamps[i],
                               asamps[i],
                               0.0, #e0
                               1.0, #xI0
                               Alsamps[i],
                               nlsamps[i],
                               Agsamps[i],
                               4.0, #ng
                              ],
                              )
        
        print("T, M, mu, a, Al, nl, Ag: ", Tsamps[i],Msamps[i],musamps[i],asamps[i],Alsamps[i],nlsamps[i],Agsamps[i])
        print("corresponding p0_plunge: ", p_plunge)

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
        
    delta_psi = multiplicative_factor @ delta_phi # Npsi x Nphi array times Nphi 1D array: 1D array of length Npsi
    
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
def DM_prior_vac(M, z, K, alpha, beta, H0, Omega_m0, Omega_Lambda0, Mstar):
    C = K * ((1/Mstar)**alpha) * 4 * np.pi
    
    return C * alpha * (M ** (alpha - 1)) * ((1 + z) ** beta) * (dc(z, H0, Omega_m0, Omega_Lambda0) ** 2)

def DDlnM_prior_vac(lnM, z, K, alpha, beta, H0, Omega_m0,Omega_Lambda0, Mstar):
    M = np.exp(lnM)
    C = K*(1/Mstar)**alpha*4*np.pi
    
    return ((C * alpha * (alpha-1) * M ** (alpha - 2) * (1 + z) ** beta * dc(z, H0, Omega_m0, Omega_Lambda0) ** 2)* M ** 2 + 
            DM_prior_vac(M, z, K, alpha, beta, H0, Omega_m0, Omega_Lambda0, Mstar) * M)

def DDz_prior_vac(lnM, z, K, alpha, beta,H0,Omega_m0,Omega_Lambda0, Mstar):
    M = np.exp(lnM)
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

def DlnMDz_prior_vac(lnM, z, K, alpha, beta,H0,Omega_m0,Omega_Lambda0, Mstar):
    M = np.exp(lnM)
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
    
    return M * (term1 + term2) #multiplied by M for chain rule since this is actually d/dlnMdz
    
#individual source integral terms in all three hypotheses

def Isource_vac(lnM, z, K, alpha, beta, Fisher, H0,Omega_m0,Omega_Lambda0,Mstar, indices = {'lnM':0,'z':1}):
    """ Source Integral approximation in the vacuum-GR hypothesis. 
    lnM, z are the inferred source parameters. K, alpha, beta are the hyperparameters.
    Fisher is the full Fisher matrix in the vac+loc+glob hypothesis at the true parameter point (after transformation M, dist -> lnM, z).
    indices is a dict of indices of the vacuum parameters [lnM, z] in the Fisher matrix.
    !! Transform Fisher from M, dl to lnM, z before calling this function !!
    """
        
    Fisher_vac_inds = np.ix_(list(indices.values()),list(indices.values()))
    Fisher_vac = Fisher[Fisher_vac_inds]
    
    Fisher_vac_inv = np.linalg.inv(Fisher_vac)

    return (prior_vac(lnM=lnM, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar) + 
            (1/2*DDlnM_prior_vac(lnM=lnM, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*Fisher_vac_inv[indices['lnM'],indices['lnM']] +
            1/2*DDz_prior_vac(lnM=lnM, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*Fisher_vac_inv[indices['z'],indices['z']] +
            DlnMDz_prior_vac(lnM=lnM, z=z, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*Fisher_vac_inv[indices['lnM'],indices['z']]))

def Isource_glob(lnM, z, Ag, K, alpha, beta, Gdot, Fisher,H0,Omega_m0,Omega_Lambda0, Mstar, indices = {'lnM':0,'z':1,'Ag':-1}):
    """ Source Integral approximation in the Global effect hypothesis.
    lnM, z, Ag are the inferred source parameters. K, alpha, beta, Gdot are the hyperparameters.
    Fisher is the full Fisher matrix in the vac+loc+glob hypothesis at the true parameter point (after transformation M, dist -> lnM, z).
    indices is a dict of indices of the global parameters [M, z, Ag] in the Fisher matrix.
    !! Transform Fisher from lnM, dl to lnM, z before calling this function !!
    """

    vec_v = np.array([lnM,z])
    vec_g = np.array([Ag])

    #getting the Fisher for vac+global effect parameters
    dpsi = len(list(indices.keys())) #number of all parameters in the global effect hypothesis.
    
    Fisher_psipsi_inds = np.ix_(list(indices.values()),list(indices.values()))
    Fisher_psipsi = Fisher[Fisher_psipsi_inds]

    #getting the different Fisher blocks
    indices_vac = {}
    for key in list(indices.keys()):
        if key in ['lnM','z']:
            indices_vac[key] = indices[key]

    indices_glob = {}
    for key in list(indices.keys()):
        if key in ['Ag']:
            indices_glob[key] = indices[key]

    Fisher_vac_inds = np.ix_(list(indices_vac.values()),list(indices_vac.values()))
    Fisher_vac = Fisher[Fisher_vac_inds] #vacuum elements only
    Fisher_vac_inv = np.linalg.inv(Fisher_vac)

    Fisher_glob_inds = np.ix_(list(indices_glob.values()),list(indices_glob.values()))
    Fisher_glob = Fisher[Fisher_glob_inds]  #global effect elements only

    Fisher_vacglob_inds = np.ix_(list(indices_vac.values()),list(indices_glob.values()))
    Fisher_vacglob = Fisher[Fisher_vacglob_inds]

    Fisher_globvac = Fisher_vacglob.T

    dv = len(list(indices_vac.keys()))

    v_dagger = vec_v + (Fisher_vac_inv @ Fisher_vacglob) @ (vec_g - Gdot) #biased point after marginalizing over Ag

    lnM_dagger, z_dagger = v_dagger 

    #actually calculating the source integral
    I0 = ((np.linalg.det(Fisher_psipsi)/np.linalg.det(Fisher_vac))**(1/2)/((2*np.pi)**((dpsi-dv)/2)) *  
          np.exp(-1/2 * (Fisher_glob - ((Fisher_globvac @ Fisher_vac_inv) @ Fisher_vacglob))[0][0] * (vec_g - Gdot)**2)
     ) #first term

    def conditional_expectation(first_index, second_index):
        #calculate the conditional expectation on v^k * v^m moment of the vacuum vector for I1
        return Fisher_vac_inv[first_index, second_index]

    return_val = (I0 * 
                  (prior_vac(lnM=lnM_dagger, z=z_dagger, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar) +  
                    1/2*DDlnM_prior_vac(lnM=lnM_dagger, z=z_dagger, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*conditional_expectation(indices['lnM'],indices['lnM'])
                    + 1/2*DDz_prior_vac(lnM=lnM_dagger, z=z_dagger, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*conditional_expectation(indices['z'],indices['z'])
                    + DlnMDz_prior_vac(lnM=lnM_dagger, z=z_dagger, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*conditional_expectation(indices['lnM'],indices['z'])
                    )
    )[0]

    return return_val

def Isource_loc(lnM, z, vec_l, K, alpha, beta, f, mu_l, sigma_l, Fisher, H0,Omega_m0,Omega_Lambda0, Mstar, indices = {'lnM':0, 'z': 1, 'Al':2, 'nl':3}):
    """
    Source Integral approximation in the local effect hypothesis.
    M, z (np.float64) are the inferred vacuum parameters of the source. K, alpha, beta are the corresponding hyperparameters.
    vec_l (list/numpy 1d.array) = [Al, nl] is the list of inferred local effect parameters of the source.
    f (np.float64), mu_l = [mu_Al, mu_nl], sigma_l = [sigma_Al, sigma_nl] are the hyperparameters of the local effect.
    Fisher = Fisher_psipsi with coordinates [lnM, z, Al, nl] 4x4
    """

    sigma_l = np.array(sigma_l)
    mu_l = np.array(mu_l)
    vec_l = np.array(vec_l) #lhat

    Al, nl = vec_l
    vec_v = np.array([lnM,z])

    #getting the Fisher for vac+local effect parameters
    dpsi = len(list(indices.keys()))
    
    indices_vac = {}
    for key in list(indices.keys()):
        if key in ['lnM','z']:
            indices_vac[key] = indices[key]

    dv = len(list(indices_vac.keys()))

    indices_loc = {}
    for key in list(indices.keys()):
        if key in ['Al','nl']:
            indices_loc[key] = indices[key]
            
    dl = len(list(indices_loc.keys()))
    
    Fisher_psipsi_inds = np.ix_(list(indices.values()),list(indices.values()))
    Fisher_psipsi = Fisher[Fisher_psipsi_inds] #Full Fisher in vac+local
    Fisher_psipsi_inv = np.linalg.inv(Fisher_psipsi)

    Fisher_vac_inds = np.ix_(list(indices_vac.values()),list(indices_vac.values()))
    Fisher_vac = Fisher[Fisher_vac_inds] #vacuum elements only
    Fisher_vac_inv = np.linalg.inv(Fisher_vac)

    Fisher_loc_inds = np.ix_(list(indices_loc.values()),list(indices_loc.values()))
    Fisher_loc = Fisher[Fisher_loc_inds]  #local effect elements only

    Fisher_vacloc_inds = np.ix_(list(indices_vac.values()),list(indices_loc.values()))
    Fisher_vacloc = Fisher[Fisher_vacloc_inds]

    Fisher_locvac = Fisher_vacloc.T

    ### Calculating the first source term

    v_dagger = vec_v + (Fisher_vac_inv @ Fisher_vacloc) @ vec_l #biased point after marginalizing over Al, nl
    
    lnM_dagger, z_dagger = v_dagger 

    I0_1 = (((np.linalg.det(Fisher_psipsi)/np.linalg.det(Fisher_vac))**(1/2))/((2*np.pi)**((dpsi-dv)/2)) * 
            np.exp(-1/2 * vec_l.T @ (Fisher_loc - (Fisher_locvac @ Fisher_vac_inv) @ Fisher_vacloc) @ vec_l)) #marginalization term
    
    def conditional_expectation(first_index, second_index):
        #calculate the conditional expectation on v^k * v^m moment of the vacuum vector for I1
        return Fisher_vac_inv[first_index, second_index]
    
    I1 = ((1-f) * 
          I0_1 * 
          (
          prior_vac(lnM=lnM_dagger, z=z_dagger, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar) + 
          (1/2*DDlnM_prior_vac(lnM=lnM_dagger, z=z_dagger,  K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*conditional_expectation(indices['lnM'],indices['lnM']) +
            1/2*DDz_prior_vac(lnM=lnM_dagger, z=z_dagger,  K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*conditional_expectation(indices['z'],indices['z']) +
            DlnMDz_prior_vac(lnM=lnM_dagger, z=z_dagger,  K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*conditional_expectation(indices['lnM'],indices['z']))
            )
          )

    ### calculating the second source term

    #projection matrix: vec_l = np.dot(P,psi)
    P = np.zeros((dl,dpsi))

    for i, key in enumerate(list(indices_loc.keys())):
        j = indices_loc[key]  # column index in full vector
        P[i, j] = 1.0
    
    Fisher_l = np.diag(1/sigma_l**2)

    vec_psi = np.array([lnM,z,Al,nl])            
    
    #standardization factor
    S_covar = P @ Fisher_psipsi_inv @ P.T + np.linalg.inv(Fisher_l)
    S_gamma = np.linalg.inv(S_covar)
    S = ((np.linalg.det(S_gamma)**(1/2))/((2*np.pi)**(dl/2)) * 
         np.exp(-0.5 * (mu_l - vec_l) @ S_gamma @ (mu_l - vec_l)))

    #tilde quantities
    Fisher_tilde = Fisher_psipsi + P.T @ Fisher_l @ P
    psi_tilde = np.linalg.inv(Fisher_tilde) @ (Fisher_psipsi @ vec_psi + P.T @ Fisher_l @ mu_l)

    lnM_tilde, z_tilde = psi_tilde[:2]

    I2 = (f * 
          S * 
          (prior_vac(lnM=lnM_tilde, z=z_tilde, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar) + 
           (1/2*DDlnM_prior_vac(lnM=lnM_tilde, z=z_tilde, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*np.linalg.inv(Fisher_tilde)[indices['lnM'],indices['lnM']] +
            1/2*DDz_prior_vac(lnM=lnM_tilde, z=z_tilde, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*np.linalg.inv(Fisher_tilde)[indices['z'],indices['z']] +
            DlnMDz_prior_vac(lnM=lnM_tilde, z=z_tilde, K=K, alpha=alpha, beta=beta, H0=H0, Omega_m0=Omega_m0, Omega_Lambda0=Omega_Lambda0,Mstar=Mstar)*np.linalg.inv(Fisher_tilde)[indices['lnM'],indices['z']])
            )
        )
    
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
        sef (class): initialized instance of the StableEMRIFishers class object.
        sef_kwargs (dict): keyword arguments to provide to the StableEMRIFishers class.

        filename (string): folder name where the data is being stored. No default because impractical to not save results.
        filename_Fisher (string): a sub-folder for storing Fisher files (book-keeping). If None, Fishers directly stored in filename. Default is None. 

        true_hyper (dict): true values of all hyperparameters. Default are fiducial values consistent with a population of vacuum EMRIs.
        cosmo_params (dict): true values of 'Omega_m0' (matter density), 'Omega_Lambda0' (DE density), and 'H0' (Hubble constant in m/s/Gpc).

        source_bounds (dict): prior range on source parameters in all three hypotheses. Keys are param names and values are lists of lower and upper bounds. 
                              Must be provided for all parameters. We assume flat priors in this range.
        out_of_bound_nature (str): If MLE estimates outside the source_bounds, what to do with them? 'remove': remove them, 'edge': the MLE takes the edge value from source bounds, 'None': Keep everything.
        hyper_bounds (dict): prior range on population (hyper)params in all three hypotheses. Keys are param names and values are lists of lower and upper bounds. 
                             Must be provided for all hyperparams. We assume flat priors in this range.

        Tplunge_range (Union(list,NoneType)): lower and upper bounds on the time-to-plunge on EMRIs in the population. This will be used to initialize p0's for all EMRIs.
                              Default is None corresponding to Tplunge_range = [0.5,T_LISA + 1.0].
        
        T_LISA (float): time (in years) of LISA observation window. Default is 1.0.
        dt (float): LISA sampling frequency. Default is 10.
        Mstar (float) Constant in prior_vac. Default is 3e6. We choose it here following https://arxiv.org/pdf/1703.09722.

        M_random (int): Number of random samples for Savage-Dickey ratio calculation. Default is int(2e3).
        Fisher_validation_kwargs (dict): Keyword arguments for FisherValidation class for Kullback-Leibler divergence calculation. 
                                         If not empty, must provide keys: ('KL_threshold', 'filename_Fisher_loc', 'filename_Fisher_glob', 'validate').
        make_nice_plots (bool): Make and save visualizations: scatterplots of source param distributions, inferred bias corner plots, source integrals as a function
                                function of hyperparameters, etc.
        plots_filename (string): custom filename for the plots file if make_nice_plots is True. If not provided, but make_nice_plots is True, plots are saved under the default name "fancy_plots". 
        
        random_seed (int or None): seed for random source and hyperparameter samples. If NoneType, no seed is implemented. Default is 42.
    
    Returns:
        Bvac_loc (float): Savage-Dickey ratio preferring the vacuum over the local hypothesis. 
        Bvac_glob (float): Savage-Dickey ratio preferring the vacuum over the global hypothesis.
        Bglob_loc (float): Savage-Dickey ratio preferring the global over the local hypothesis.
    """

    def __init__(self, Npop, SNR_thresh, sef, sef_kwargs,
                       filename,filename_Fishers=None,
                       true_hyper={'K':5e-3,'alpha':0.0,'beta':0.0,
                                   'f':0.0,'mu_Al':1e-5,'mu_nl':8.0,'sigma_Al':1e-6,'sigma_nl':0.8,
                                   'Gdot':0.0},
                       cosmo_params={'Omega_m0':0.30,'Omega_Lambda0':0.70,'H0':70e3}, 
                       source_bounds={'lnM':[np.log(1e4),np.log(1e7)],'z':[0.01,10.0],'Al':[0.0,1e-4],'nl':[0.0,10.0],'Ag':[0.0,1e-8]},
                       out_of_bound_nature = None,
                       hyper_bounds={'K':[1e-3,1e-2],'alpha':[-0.5,0.5],'beta':[-0.5,0.5],
                                     'f':[0.0,1.0],'mu_Al':[1e-5,1e-5],'mu_nl':[8.0,8.0],'sigma_Al':[1e-6,1e-6],'sigma_nl':[0.8,0.8],
                                     'Gdot':[0.0,1e-8]},
                       Tplunge_range = None,
                       T_LISA = 1.0, dt = 10.0, Mstar = 3e6,
                       M_random = int(2e3),
                       Fisher_validation_kwargs = {},
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
        self.sef = sef
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
        
        self.lnM_range = source_bounds['lnM']
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

        if (out_of_bound_nature == None) or (out_of_bound_nature in ['edge', 'remove']):
            self.out_of_bound_nature = out_of_bound_nature
        else:
            warnings.warn("valid option for out_of_bound_nature: ['edge','remove', None]. Assuming default (None).")
            self.out_of_bound_nature = None
            
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
        self.lnM_truth_samples, self.z_truth_samples = lnM_z_samples(N=self.Npop,
                                                                 lnM_range=self.lnM_range,z_range=self.z_range,
                                                                 lambda_v=self.lambda_truth_vac,grid_size=grid_size,
                                                                 H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0,Mstar=self.Mstar_truth,
                                                                 seed=self.seed)
        
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
         self.T_truth_samples) = other_param_samples(N=self.Npop,M_samples=np.exp(self.lnM_truth_samples),Tplunge_range=self.Tplunge_range,seed=self.seed)
        
        try:
            self.p0_truth_samples = np.loadtxt(f"{self.filename}/p0samps.txt")
            print("p0 samples found")
        except FileNotFoundError:
            print("calculating p0 samples")
            self.p0_truth_samples = p0_samples_func(N=self.Npop,Msamps=np.exp(self.lnM_truth_samples),
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
            plt.scatter(np.log10(self.mu_truth_samples/np.exp(self.lnM_truth_samples)),self.p0_truth_samples,color='grey',alpha=0.5)
            plt.xlabel(r"$\log_{10}$(Mass ratio)",fontsize=16)
            plt.ylabel(r"$p_0$", fontsize=16)
            plt.title("True population",fontsize=16)
            plt.savefig(f'{self.plots_folder}/q_p0_truth.png',dpi=300,bbox_inches='tight')
            plt.close()

        #####################################################################
        #extracting the detected population using SNR threshold calculation
        #####################################################################
        self.calculate_detected() 

        ####################################################################
        #transforming the Fishers from [M,dL,Al,nl,Ag] to [lnM,z,Al,nl,Ag]
        ####################################################################

        Fisher_index = []
        varied_params = []
        for i in range(len(self.detected_EMRIs)):
            varied_params.append(np.array(np.array(self.detected_EMRIs[i]['transformed_params']))) #lnM, z, Al, nl, Ag
            Fisher_index.append(int(self.detected_EMRIs[i]['index']))
            
        varied_params = np.array(varied_params)
        Fisher_index = np.array(Fisher_index)

        for index, i in zip(Fisher_index,range(len(Fisher_index))):
        
            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                Gamma_i = f["Fisher"][:]
                
            dist_i = self.detected_EMRIs[i]['true_params'][6] #true_params[6] = dist
            M_i = self.detected_EMRIs[i]['true_params'][0] #true_params[0] = M
            
            J = Jacobian(M_i, dist_i, self.H0, self.Omega_m0, self.Omega_Lambda0) #converts from M, dist, ... -> lnM, z
            
            Fisher_transformed = J.T@Gamma_i@J #Fisher transformed now [lnM, z, ...]

            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "a") as f:
                if "Fisher_transformed" in f:
                    del f["Fisher_transformed"] #overwrite Fisher_transformed
                f.create_dataset("Fisher_transformed", data = Fisher_transformed)

        ##################################################################
        #calculating the biased inferrence params in all three hypotheses
        ##################################################################
    
        self.inferred_params()

        if self.make_nice_plots:
            self.corner_plot_biases()

        #######################################################
        #perform Fisher validation if KL_threshold is provided
        #######################################################

        if len(self.Fisher_validation_kwargs.keys()) > 0:
            print('Validating Fishers using KL-divergence...')
            
            self.KL_threshold = self.Fisher_validation_kwargs['KL_threshold']
            _, filename_Fishers = os.path.split(self.filename_Fishers)
            filename_Fishers_loc = self.Fisher_validation_kwargs['filename_Fishers_loc']
            filename_Fishers_glob = self.Fisher_validation_kwargs['filename_Fishers_glob']
            validate = self.Fisher_validation_kwargs['validate']

            fishervalidate = FisherValidation(sef = self.sef, sef_kwargs = self.sef_kwargs,
                     filename = self.filename, filename_Fishers = filename_Fishers, filename_Fishers_loc = filename_Fishers_loc, 
                     filename_Fishers_glob = filename_Fishers_glob,
                     true_hyper = self.true_hyper, cosmo_params = self.cosmo_params, source_bounds = self.source_bounds, hyper_bounds = self.hyper_bounds,
                     T_LISA = self.T_LISA, dt = self.dt,
                     validate = validate)
    
            fishervalidate()
            
            if self.make_nice_plots:
                fishervalidate.KL_divergence_plot(self.plots_folder)

        #############################################################
        #calculating the Savage-Dickey ratios in different hypotheses        
        #############################################################

        #savage-dickey preferring the vacuum hypothesis over local
        Bvac_loc = self.savage_dickey_vacloc()

        Bvac_glob = self.savage_dickey_vacglob()

        Bglob_loc = Bvac_loc/Bvac_glob

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
                sef_kwargs = self.sef_kwargs.copy()
                M = np.exp(self.lnM_truth_samples[i])
                mu = self.mu_truth_samples[i]
                a = self.a_truth_samples[i]
                e0 = 0.0
                xI0 = 1.0
                dL = getdist(self.z_truth_samples[i],self.H0,self.Omega_m0,self.Omega_Lambda0) #Gpc
                
                qS = self.qS_truth_samples[i]
                phiS = self.phiS_truth_samples[i]
                qK = self.qK_truth_samples[i]
                phiK = self.phiK_truth_samples[i]
                Phi_phi0 = self.Phi0_truth_samples[i]
                Phi_theta0 = 0.0
                Phi_r0 = 0.0
                T = self.T_LISA
                dt = self.dt
    
                Al = self.Al_truth_samples[i]
                nl = self.nl_truth_samples[i]
    
                Ag = self.Ag_truth_samples[i]
                ng = 4.0
    
                p0 = self.p0_truth_samples[i]
    
                sef_kwargs['suffix'] = i
    
                param_list = [M,mu,a,p0,e0,xI0,
                              dL,qS,phiS,qK,phiK,Phi_phi0,Phi_theta0,Phi_r0,
                              ] #SEF param args (vacuum-GR EMRI)

                add_param_args = {"Al":Al, "nl":nl, "Ag":Ag, "ng":ng} #dict of additional parameters
                sef_kwargs['add_param_args'] = add_param_args

                transformed_params = [np.log(M),self.z_truth_samples[i],Al,nl,Ag] #lnM, z, Al, nl, Ag
                
                emri_kwargs = {"T": T, "dt": dt}
                sef_kwargs["T"] = T
                sef_kwargs["dt"] = dt
                
                param_list_SNR = param_list.copy()+[Al,nl,Ag,ng] #param_list for SNR calculation in order that you'd supply to the waveform generator
                source_optimal_SNR = self.sef.SNRcalc_SEF(*param_list_SNR, **emri_kwargs, use_gpu=self.sef.use_gpu) #calculate optimal SNR
                all_SNRs.append(source_optimal_SNR)

                print(all_SNRs[-1])
                if all_SNRs[i] >= self.SNR_thresh:
                    self.detected_EMRIs.append({'index': i,
                                                'true_params': np.array(param_list_SNR), #copy the beyond-vacuum-GR params at the end as well.
                                                'SNR':all_SNRs[i], 
                                                'lambda_v':self.lambda_truth_vac, 
                                                'lambda_l':self.lambda_truth_loc, 
                                                'lambda_g':self.lambda_truth_glob,
                                               'transformed_params':np.array(transformed_params)})
                    #calculate and save the FIM for this source
                    try:
                        with h5py.File(f"{self.filename_Fishers}/Fisher_{i}.h5", "r") as f:
                            _ = f["Fisher"][:]
                    except FileNotFoundError:
                        self.sef(*param_list, **sef_kwargs) #calculate and save the FIM for the detected EMRI
    
            all_SNRs = np.array(all_SNRs)
            self.detected_EMRIs = np.array(self.detected_EMRIs)
            np.save(f"{self.filename}/detected_EMRIs",self.detected_EMRIs)
            np.savetxt(f"{self.filename}/all_SNRs.txt",np.array(all_SNRs))
    
        print(f"#detected EMRIs: {len(self.detected_EMRIs)}")

        if self.make_nice_plots:
            
            indices_detected = []

            for i in range(len(self.detected_EMRIs)):
                indices_detected.append(self.detected_EMRIs[i]['index'])
            
            indices_detected = np.array(indices_detected)

            plt.figure(figsize=(7,5))
            plt.scatter(self.lnM_truth_samples,self.z_truth_samples,color='grey',alpha=0.5,label='truth')
            plt.scatter(self.lnM_truth_samples[indices_detected], self.z_truth_samples[indices_detected], color='orange', edgecolor='k', marker='*', s=100, label='detected')
            plt.xlabel(r"$\log(m_1)$",fontsize=16)
            plt.ylabel(r"$z$", fontsize=16)
            plt.legend(fontsize=14)
            plt.savefig(f'{self.plots_folder}/M_z_truth.png',dpi=300,bbox_inches='tight')
            plt.savefig(f'{self.plots_folder}/M_z_truth.pdf',dpi=300,bbox_inches='tight') #for the paper
            plt.close()

            fig, axs = plt.subplots(1,2,figsize=(10,5), sharey=True)
            axs[0].hist(self.lnM_truth_samples, bins=20, histtype='stepfilled', edgecolor='k', color='grey', alpha = 0.5, label='truth')
            axs[0].hist(self.lnM_truth_samples[indices_detected], bins=20, histtype='stepfilled', edgecolor='k', color='orange', label='detected')
            axs[0].set_xlabel(r"$\log(M))$",fontsize=16)
            axs[0].tick_params(axis='both', which='major', labelsize=14)

            axs[1].hist(self.z_truth_samples, bins=20, histtype='stepfilled', edgecolor='k', color='grey', alpha = 0.5, label='truth')
            axs[1].hist(self.z_truth_samples[indices_detected], bins=20, histtype='stepfilled', edgecolor='k', color='orange', label='detected')
            axs[1].set_xlabel(r"$z$", fontsize=16)
            axs[1].tick_params(axis='both', which='major', labelsize=14)
            
            plt.savefig(f'{self.plots_folder}/M_z_truth_vs_detected.png',dpi=300,bbox_inches='tight')
            plt.savefig(f'{self.plots_folder}/M_z_truth_vs_detected.pdf',dpi=300,bbox_inches='tight') #for the paper.
            plt.close()

        if self.make_nice_plots:
            counts, bins, patches = plt.hist(all_SNRs, bins=50)
            for patch, bin_left in zip(patches, bins[:-1]):
                if bin_left >= self.SNR_thresh:
                    patch.set_facecolor('orange')
                else:
                    patch.set_facecolor('grey')

            plt.axvline(self.SNR_thresh,color='k',linestyle='--',label='SNR threshold')
            plt.legend()
            plt.xlabel("SNRs",fontsize=16)
            plt.yscale("log")
            plt.savefig(f"{self.plots_folder}/SNR_dist.png",dpi=300,bbox_inches='tight')
            plt.savefig(f"{self.plots_folder}/SNR_dist.pdf",dpi=300,bbox_inches='tight') #for the paper
            plt.close()

    def inferred_params(self):
        """ calculate and save the inferred biased params in the given hypothesis.
        """
        
        for i in range(len(self.detected_EMRIs)):
        
            # d: number of measured source params
            # Nphi: number of unmeasured params
            # Npsi: number of measured params

            detected_EMRIs = self.detected_EMRIs.copy()
            index = int(self.detected_EMRIs[i]["index"])
            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                Gamma_i = f["Fisher_transformed"][:] #Fisher in transformed coords [lnM,z,Al,nl,Ag]
    
            ### vacuum hypothesis ###
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

            detected_EMRIs[i]["vacuum_params"] = np.array(psi_i_inferred) #save [lnM_bias,z_bias]
    
            ### local hypothesis ###
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
            psi_i_inferred = np.concatenate((psi_i_inferred,[0.0])) #size = Npsi + Nphi

            detected_EMRIs[i]["local_params"] = np.array(psi_i_inferred) #save [M_bias,z_bias,Al_bias, nl_bias]
    
            ### global hypothesis ###
            indices_psi = [0,1,4]  #indices of measured params (M,z,Ag)
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
            psi_i_inferred = np.concatenate((np.concatenate((psi_i_inferred[:2],[0.0,0.0])),[psi_i_inferred[-1]]))
            detected_EMRIs[i]["global_params"] = np.array(psi_i_inferred) #save [lnM_bias,z_bias,Ag_bias]
            
            self.detected_EMRIs[i] = detected_EMRIs[i] #update

        self.Nobs = len(self.detected_EMRIs) #number of detected EMRIs.
        np.save(f'{self.filename}/detected_EMRIs',self.detected_EMRIs) #save updated

    def source_integral_vac(self,K,alpha,beta):
        
         """Calculate the source integral in the vacuum hypothesis.
         bounds_vac is a dict of bounds on M and z. Bounds can be given for any subset of the parameters.
         !!! This code hasn't been checked for a while. Proceed with caution."""
                
         #calculate source integral
         Ivac_all = []

         count = 0.0 #number of out of bound EMRIs
        
         bounds_vac = {'lnM':self.source_bounds['M'],'z':self.source_bounds['z']}

         for i in range(len(self.detected_EMRIs)):
            
             out_of_bounds = False
             index = int(self.detected_EMRIs[i]["index"])
             with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                 Fisher = f["Fisher_transformed"][:] #Fisher in transformed coords [M,z,Al,nl,Ag]

             vacparams = self.detected_EMRIs[i]["vacuum_params"] # Mvac, zvac, Alvac, nlvac, Agvac

             if not self.out_of_bound_nature is None:
                 for param,j in zip(bounds_vac.keys(),range(len(bounds_vac.keys()))):
                     if check_prior(vacparams[j],bounds_vac[param]) == 1: #if the source parameters hits the upper limit
                         out_of_bounds = True
                         warnings.warn(f"source {index} is out of prior bounds on {param} (upper bound hit). \n\
                                 Parameter value: {vacparams[j]}. Bound: {bounds_vac[param]}.")
                         if self.out_of_bound_nature == 'edge':
                             vacparams[j] = bounds_vac[param][1] #varparam takes the upper limit value
                     elif check_prior(vacparams[j],bounds_vac[param]) == -1: #if the source parameter hits the lower limit
                         out_of_bounds = True
                         warnings.warn(f"source {index} is out of prior bounds on {param} (lower bound hit). \n\
                                 Parameter value: {vacparams[j]}. Bound: {bounds_vac[param]}.")
                         if self.out_of_bound_nature == 'edge':
                             vacparams[j] = bounds_vac[param][0] #vacparam takes the lower limit value

             if out_of_bounds:
                 count+=1

             Ivac_i = Isource_vac(M=vacparams[0],z=vacparams[1], 
                                 K=K, alpha=alpha, beta=beta, #variable hyperparameters
                                 Fisher=Fisher,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0,Mstar=self.Mstar_truth)

             if out_of_bounds & self.out_of_bound_nature == 'remove':
                 Ivac_all.append(1.0)

             else:
                 Ivac_all.append(Ivac_i)
    
         warnings.warn(f"EMRIs out-of-bounds: {int(count)} out of total {int(len(Fishers_all))}")
    
         return np.prod(np.array(Ivac_all))

    def source_integral_loc(self,K,alpha,beta,f,mu_Al,mu_nl,sigma_Al,sigma_nl,Fishers_all,indices_all,locparams_all):
        
        """Calculate the source integral in the local hypothesis.
        bounds_loc is a dict of bounds on M, z, Al, nl. Bounds can be given for any subset of the parameters."""
        
        count = 0.0 #number of out of bound EMRIs
        
        #calculate source integral
        Iloc_all = []

        bounds_loc = {'lnM':self.source_bounds['lnM'],'z':self.source_bounds['z'],
                      'Al':self.source_bounds['Al'],'nl':self.source_bounds['nl']} #prior range
    
        for i, index in zip(range(len(Fishers_all)),indices_all):
            out_of_bounds = False
            Fisher = Fishers_all[i] #Fisher in transformed coords [lnM,z,Al,nl,Ag]

            locparams = locparams_all[i].copy()
            
            out_of_bound_nature = self.out_of_bound_nature

            #check prior bounds on all model parameters.
            #out_of_bound_nature for nl is always "edge" because of the way it contributes in the model.
            if not out_of_bound_nature is None:
                for param,j in zip(bounds_loc.keys(),range(len(bounds_loc.keys()))):
                    if param == 'nl':
                        out_of_bound_nature = 'edge'

                    if check_prior(locparams[j],bounds_loc[param]) == 1: #if the source parameters hits the upper limit
                        out_of_bounds = True
                        warnings.warn(f"source {index} is out of prior bounds on {param} (upper bound hit). \n\
                                Parameter value: {locparams[j]}. Bound: {bounds_loc[param]}.")
                        if out_of_bound_nature == 'edge':
                            locparams[j] = bounds_loc[param][1] #locparam takes the upper limit value
                    elif check_prior(locparams[j],bounds_loc[param]) == -1: #if the source parameter hits the lower limit
                        out_of_bounds = True
                        warnings.warn(f"source {index} is out of prior bounds on {param} (lower bound hit). \n\
                                Parameter value: {locparams[j]}. Bound: {bounds_loc[param]}.")
                        if out_of_bound_nature == 'edge':
                            locparams[j] = bounds_loc[param][0] #locparam takes the lower limit value

            if out_of_bounds:
                count+=1
                
            if (out_of_bounds) & (out_of_bound_nature == 'remove'):
                Iloc_all.append(1.0)
                    
            else: 
                Iloc_i = Isource_loc(lnM=locparams[0],z=locparams[1], vec_l=[locparams[2],locparams[3]], 
                                    K=K, alpha=alpha, beta=beta, 
                                    f=f, mu_l=[mu_Al,mu_nl], sigma_l=[sigma_Al,sigma_nl], 
                                    Fisher=Fisher,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0,
                                    Mstar=self.Mstar_truth)
        
                if np.isnan(Iloc_i):
                    Iloc_all.append(1.0)

                elif Iloc_i <= 1e-200: #assuming fiducial baseline 'noisy' posterior value of 1e-200
                    Iloc_all.append(1e-200)
                    
                else:
                    Iloc_all.append(Iloc_i)

        lnposterior = np.sum(np.log(np.array(Iloc_all))) #avoid overflow by calculating log posterior
        
        return lnposterior

    def source_integral_glob(self,K,alpha,beta,Gdot,Fishers_all,indices_all,globparams_all):
        
        """Calculate the source integral in the global hypothesis.
        bounds_loc is a dict of bounds on M, z, Al, nl. Bounds can be given for any subset of the parameters."""
            
        count = 0.0 #number of out of bound EMRIs
        
        #calculate source integral
        Iglob_all = []

        bounds_glob = {'lnM':self.source_bounds['lnM'],'z':self.source_bounds['z'],
                      'Ag':self.source_bounds['Ag']} #prior range
    
        for i, index in zip(range(len(Fishers_all)),indices_all):
            out_of_bounds = False

            Fisher = Fishers_all[i] #Fisher in transformed coords [lnM,z,Al,nl,Ag]

            globparams = globparams_all[i].copy() #ln Mglob, zglob, Alglob, nlglob, Agglob
            
            if not self.out_of_bound_nature is None:
                for param, j in zip(bounds_glob.keys(),range(len(bounds_glob.keys()))):
                    if check_prior(globparams[j],bounds_glob[param]) == 1: #if the source parameters hits the upper limit
                        out_of_bounds = True
                        warnings.warn(f"source {index} is out of prior bounds on {param} (upper bound hit). \n\
                                Parameter value: {globparams[j]}. Bound: {bounds_glob[param]}.")
                        if self.out_of_bound_nature == 'edge':
                            globparams[j] = bounds_glob[param][1] #varparam takes the upper limit value
                    elif check_prior(globparams[j],bounds_glob[param]) == -1: #if the source parameter hits the lower limit
                        out_of_bounds = True
                        warnings.warn(f"source {index} is out of prior bounds on {param} (lower bound hit). \n\
                                Parameter value: {globparams[j]}. Bound: {bounds_glob[param]}.")
                        if self.out_of_bound_nature == 'edge':
                            globparams[j] = bounds_glob[param][0] #vacparam takes the lower limit value

                if out_of_bounds:
                    count+=1
                    
                if (out_of_bounds) & (self.out_of_bound_nature == 'remove'):
                    Iglob_all.append(0.0)
                    continue
        
            Iglob_i = Isource_glob(lnM=globparams[0],z=globparams[1],Ag=globparams[-1],
                                K=K, alpha=alpha, beta=beta, 
                                Gdot=Gdot,Mstar=self.Mstar_truth,
                                Fisher=Fisher,H0=self.H0,Omega_m0=self.Omega_m0,Omega_Lambda0=self.Omega_Lambda0)
            
            if Iglob_i <= (1e-200): #assuming fiducial baseline 'noisy' posterior value of 1e-200
                Iglob_all.append((1e-200))
                
            else:
                Iglob_all.append(Iglob_i)

        lnposterior = np.sum(np.log(np.array(Iglob_all))) #avoid overflow by calculating log posterior

        return lnposterior

    def savage_dickey_vacloc(self):
        #use a random seed.
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

        Fishers_all = []
        indices_all = []
        locparams_all = []      
        for i in range(len(self.detected_EMRIs)):
            index = int(self.detected_EMRIs[i]["index"])
            indices_all.append(index)
            
            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                Fish_trans = f["Fisher_transformed"][:] #[lnM, z, Al, nl, Ag]
                
            Fishers_all.append(Fish_trans)
            
            locparams_all.append(self.detected_EMRIs[i]["local_params"])

        indices_all = np.array(indices_all)
        Fishers_all = np.array(Fishers_all)
        locparams_all = np.array(locparams_all)
        
        #only choose Fishers which satisfy the KL-divergence threshold and are positive semi-definite.
        indices_valid_KL = []
        
        if (len(self.Fisher_validation_kwargs.keys()) > 0) & (self.f_truth > 0):
            Fishers_loc_KL = np.loadtxt(f'{self.filename}/Fishers_loc_KL.txt')
            for i in range(len(self.detected_EMRIs)):
                if (Fishers_loc_KL[i] <= self.KL_threshold): #KL-divergence of jth source should be less than the threshold. out_of_bounds sources will have KL = -1 and will also be ignored here.
                    indices_valid_KL.append(i)
                else:
                    warnings.warn(f"source {indices_all[i]} failed KL test.")

        else:
            indices_valid_KL = list(range(len(self.detected_EMRIs)))

        Fishers_all = Fishers_all[indices_valid_KL]
        indices_all = indices_all[indices_valid_KL]
        locparams_all = locparams_all[indices_valid_KL]

        indices_valid_Fishers = []
        for i in range(len(Fishers_all)):
            if (np.linalg.eigvals(Fishers_all[i]) > 0.0).all():
                indices_valid_Fishers.append(i)
            else:
                warnings.warn(f"source {indices_all[i]} is not positive-definite.")

        Fishers_all = Fishers_all[indices_valid_Fishers]
        indices_all = indices_all[indices_valid_Fishers]
        locparams_all = locparams_all[indices_valid_Fishers]

        if len(Fishers_all) != len(self.detected_EMRIs):
            warnings.warn(f"omitted {len(self.detected_EMRIs) - len(Fishers_all)} sources in the observed population of size {self.Nobs}.")

        lnprodIsource = []
        
        for j in tqdm(range(self.M_random)):
        
            lnprodIsource_j = self.source_integral_loc(K=K_samples[j],alpha=alpha_samples[j],beta=beta_samples[j],
                                                        f=f_samples[j],mu_Al=mu_Al_samples[j],mu_nl=mu_nl_samples[j],
                                                        sigma_Al=sigma_Al_samples[j],sigma_nl=sigma_nl_samples[j],
                                                        Fishers_all=Fishers_all, indices_all=indices_all,locparams_all=locparams_all)
         
            lnprodIsource.append(lnprodIsource_j)
    
        lnprodIsource = np.array(lnprodIsource) - np.max(lnprodIsource)
        prodIsource = np.exp(lnprodIsource)

        for i in range(len(prodIsource)):
            if prodIsource[i] <= 1e-200: #control underflow
                prodIsource[i] = 1e-200
        
        prodIsource = prodIsource/np.sum(prodIsource)
        
        #f=0 mask
        num_bins = 40
        mask = np.abs(f_samples - 0.0) < (max(f_samples)-min(f_samples))/num_bins
        
        prior_f0 = sum(mask)/len(prodIsource) #prior number of points within the bin for f = 0 !!! Only works for uniform prior !!!
        posterior_f0 = np.sum(prodIsource[mask])

        with h5py.File(f"{self.filename}/samples_f.h5", "w") as f:
            f.create_dataset("posteriors_f_samples", data = prodIsource)
            f.create_dataset("f_samples", data = f_samples)
            f.create_dataset("mu_Al_samples", data = mu_Al_samples)
            f.create_dataset("mu_nl_samples", data = mu_nl_samples)
            f.create_dataset("sigma_Al_samples", data = sigma_Al_samples)
            f.create_dataset("sigma_nl_samples", data = sigma_nl_samples)
            f.create_dataset("K_samples", data = K_samples)
            f.create_dataset("alpha_samples", data = alpha_samples)
            f.create_dataset("beta_samples", data = beta_samples)
            f.create_dataset("null_mask", data = mask)
    
        print("prior_f0: ", prior_f0)
        print("posterior_f0: ", posterior_f0)

        #expectation of f:
        expectation_f = np.sum(prodIsource * f_samples)
        print("expectation of f: ", expectation_f)

        np.savetxt(f"{self.filename}/expectation_f.txt", np.array([expectation_f]))

        if self.make_nice_plots:
            fig, axs = plt.subplots(1, 3, figsize=(15,7), sharey=True)
            
            axs[0].scatter(K_samples, prodIsource, color='k', alpha=0.5)
            axs[0].scatter(K_samples[mask],prodIsource[mask],color='orange',alpha=1.0)
            axs[0].axvline(self.K_truth,color='k',linestyle='--')
            axs[0].set_xlabel(r"$K$", fontsize=16)
            
            axs[1].scatter(alpha_samples, prodIsource, color='k', alpha=0.5)
            axs[1].scatter(alpha_samples[mask],prodIsource[mask],color='orange',alpha=1.0)
            axs[1].axvline(self.alpha_truth,color='k',linestyle='--')
            axs[1].set_xlabel(r"$\alpha$", fontsize=16)
            
            axs[2].scatter(beta_samples, prodIsource, color='k', alpha=0.5)
            axs[2].scatter(beta_samples[mask],prodIsource[mask],color='orange',alpha=1.0)
            axs[2].axvline(self.beta_truth,color='k',linestyle='--')
            axs[2].set_xlabel(r"$\beta$", fontsize=16)
            
            axs[0].set_ylabel(r"$p(\vec{\lambda}_v|\{\vec{d}\}_i)$",fontsize=16)
            axs[0].set_yscale('log')
            axs[1].set_yscale('log')
            axs[2].set_yscale('log')
            
            plt.savefig(f"{self.plots_folder}/posterior_vacparams_f.png",dpi=300,bbox_inches='tight')
            plt.close()


            plt.figure(figsize=(7,5))
            plt.scatter(f_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(f_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.f_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"$f$", fontsize=16)
            plt.ylabel(r"$p(f|\mathcal{D})$",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_f.png",dpi=300,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(7,5))
            plt.scatter(mu_Al_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(mu_Al_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.mu_Al_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"$\mu_{Al}$", fontsize=16)
            plt.ylabel(r"$p(\mu_{Al}|\mathcal{D})$",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_muAl.png",dpi=300,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(7,5))
            plt.scatter(mu_nl_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(mu_nl_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.mu_nl_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"$\mu_{nl}$", fontsize=16)
            plt.ylabel(r"$p(\mu_{nl}|\mathcal{D})$",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_munl.png",dpi=300,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(7,5))
            plt.scatter(sigma_Al_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(sigma_Al_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.sigma_Al_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"$\sigma_{Al}$", fontsize=16)
            plt.ylabel(r"$p(\sigma_{Al}|\mathcal{D})$",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_sigmaAl.png",dpi=300,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(7,5))
            plt.scatter(sigma_nl_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(sigma_nl_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.sigma_nl_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"$\sigma_{nl}$", fontsize=16)
            plt.ylabel(r"$p(\sigma_{nl}|\mathcal{D})$",fontsize=16)
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_loc_sigmanl.png",dpi=300,bbox_inches='tight')
            plt.close()
        
        return posterior_f0 / prior_f0
    
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

        indices_all = []
        Fishers_all = []
        globparams_all = []
        for i in range(len(self.detected_EMRIs)):
            index = int(self.detected_EMRIs[i]["index"])
            indices_all.append(index)
            
            with h5py.File(f"{self.filename_Fishers}/Fisher_{index}.h5", "r") as f:
                Fish_trans = f["Fisher_transformed"][:] #[lnM, z, Al, nl, Ag]
                
            Fishers_all.append(Fish_trans)
            globparams_all.append(self.detected_EMRIs[i]["global_params"])

        indices_all = np.array(indices_all)
        Fishers_all = np.array(Fishers_all)
        globparams_all = np.array(globparams_all)

        #only choose Fishers which satisfy the KL-divergence threshold and are positive semi-definite.
        indices_valid_KL = []
        
        if (len(self.Fisher_validation_kwargs.keys()) > 0) & (self.Gdot_truth != 0):
            Fishers_glob_KL = np.loadtxt(f'{self.filename}/Fishers_glob_KL.txt')
            for i in range(len(self.detected_EMRIs)):
                if (Fishers_glob_KL[i] <= self.KL_threshold): #KL-divergence of jth source should be less than the threshold. out_of_bounds sources will have KL = -1 and will also be ignored here.
                    indices_valid_KL.append(i)
                else:
                    warnings.warn(f"source {indices_all[i]} failed KL test.")

        else:
            indices_valid_KL = list(range(len(self.detected_EMRIs)))

        Fishers_all = Fishers_all[indices_valid_KL]
        indices_all = indices_all[indices_valid_KL]
        globparams_all = globparams_all[indices_valid_KL]

        indices_valid_Fishers = []
        for i in range(len(Fishers_all)):
            if (np.linalg.eigvals(Fishers_all[i]) > 0.0).all():
                indices_valid_Fishers.append(i)
            else:
                warnings.warn(f"source{indices_all[i]} is not positive-definite.")
        
        Fishers_all = Fishers_all[indices_valid_Fishers]
        indices_all = indices_all[indices_valid_Fishers]
        globparams_all = globparams_all[indices_valid_Fishers]

        if len(Fishers_all) != len(self.detected_EMRIs):
            warnings.warn(f"omitted {len(self.detected_EMRIs) - len(Fishers_all)} sources in the observed population of size {self.Nobs}.")
    
        lnprodIsource = []
        
        for j in tqdm(range(self.M_random)):
            
            lnprodIsource_j = self.source_integral_glob(K=K_samples[j], alpha=alpha_samples[j], beta=beta_samples[j],
                                                  Gdot=Gdot_samples[j],Fishers_all=Fishers_all,indices_all=indices_all,globparams_all=globparams_all)
        
            lnprodIsource.append(lnprodIsource_j)

        lnprodIsource = np.array(lnprodIsource) - np.max(lnprodIsource)
        prodIsource = np.exp(lnprodIsource)

        for i in range(len(prodIsource)):
            if prodIsource[i] <= 1e-200: #control underflow
                prodIsource[i] = 1e-200
                
        prodIsource = prodIsource/np.sum(prodIsource)
            
        #Gdot=0 mask
        num_bins = 40
        mask = np.abs(Gdot_samples - 0.0) < (max(Gdot_samples)-min(Gdot_samples))/num_bins
        
        prior_Gdot0 = sum(mask)/self.M_random #prior number of points within the bin for Gdot = 0 !!! Only works for uniform prior !!!
        posterior_Gdot0 = np.sum(prodIsource[mask])

        with h5py.File(f"{self.filename}/samples_Gdot.h5", "w") as f:
            f.create_dataset("posteriors_Gdot_samples", data = prodIsource)
            f.create_dataset("Gdot_samples", data = Gdot_samples)
            f.create_dataset("K_samples", data = K_samples)
            f.create_dataset("alpha_samples", data = alpha_samples)
            f.create_dataset("beta_samples", data = beta_samples)
            f.create_dataset("null_mask", data = mask)
    
        print("prior_Gdot0: ", prior_Gdot0)
        print("posterior_Gdot0: ", posterior_Gdot0)

        #expectation of Gdot:
        expectation_Gdot = np.sum(prodIsource * Gdot_samples)
        print("expectation of Gdot: ", expectation_Gdot)

        np.savetxt(f"{self.filename}/expectation_Gdot.txt", np.array([expectation_Gdot]))

        if self.make_nice_plots:
            fig, axs = plt.subplots(1, 3, figsize=(15,7), sharey=True)
            
            axs[0].scatter(K_samples, prodIsource, color='k', alpha=0.5)
            axs[0].scatter(K_samples[mask],prodIsource[mask],color='orange',alpha=1.0)
            axs[0].axvline(self.K_truth,color='k',linestyle='--')
            axs[0].set_xlabel(r"$K$", fontsize=16)
            
            axs[1].scatter(alpha_samples, prodIsource, color='k', alpha=0.5)
            axs[1].scatter(alpha_samples[mask],prodIsource[mask],color='orange',alpha=1.0)
            axs[1].axvline(self.alpha_truth,color='k',linestyle='--')
            axs[1].set_xlabel(r"$\alpha$", fontsize=16)
            
            axs[2].scatter(beta_samples, prodIsource, color='k', alpha=0.5)
            axs[2].scatter(beta_samples[mask],prodIsource[mask],color='orange',alpha=1.0)
            axs[2].axvline(self.beta_truth,color='k',linestyle='--')
            axs[2].set_xlabel(r"$\beta$", fontsize=16)
            
            axs[0].set_ylabel(r"$p(\vec{\lambda}_v|\{\vec{d}\}_i)$",fontsize=16)
            axs[0].set_yscale('log')
            axs[1].set_yscale('log')
            axs[2].set_yscale('log')
            
            plt.savefig(f"{self.plots_folder}/posterior_vacparams_Gdot.png",dpi=300,bbox_inches='tight')
            plt.close()

            plt.figure(figsize=(7,5))
            plt.scatter(Gdot_samples, prodIsource,color='grey',alpha=0.5,label='all posterior samples')
            plt.scatter(Gdot_samples[mask],prodIsource[mask],color='red',alpha=0.5,label='posterior consistent with null')
            plt.axvline(self.Gdot_truth,color='k',linestyle='--',label='truth')
            plt.xlabel(r"$\bar{A}_g$",fontsize=16) #Gdot -> barA_g name changed for paper plots
            plt.ylabel(r"$p(\bar{A}_g|\mathcal{D})$",fontsize=16) #Gdot -> barA_g name changed for paper plots
            plt.yscale('log')
            plt.legend()
            plt.savefig(f"{self.plots_folder}/posterior_vac_glob.png",dpi=300,bbox_inches='tight')
            plt.close()
        
        return posterior_Gdot0 / prior_Gdot0

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
        
        params = [r'$\log M$','$z$','$A_l$','$n_l$','$A_g$']
        param_lims = [self.lnM_range,self.z_range,self.Al_range,self.nl_range,self.Ag_range]
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
