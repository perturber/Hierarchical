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

def H(z,H0,Omega_m0,Omega_Lambda0):
    """
    calculate the Hubble parameter at redshift z in m/s/Gpc
    """
    return H0 * np.sqrt(Omega_m0*(1+z)**3 + Omega_Lambda0)

def integrand_dc(z,H0,Omega_m0,Omega_Lambda0):
    return C_SI/H(z,H0,Omega_m0,Omega_Lambda0)

def dc(z,H0,Omega_m0,Omega_Lambda0):
    """
    returns the comoving distance in Gpc for a given redshift z
    """
    return quad(integrand_dc,0,z,args=(H0,Omega_m0,Omega_Lambda0),epsabs=1e-1,epsrel=1e-1)[0]/1000

def getdistGpc(z,H0,Omega_m0,Omega_Lambda0):
    """
    returns the luminosity distance in Gpc for a given redshift z
    """
    return (1+z)*dc(z,H0,Omega_m0,Omega_Lambda0)

def dlminusdistz(z, dl, H0,Omega_m0,Omega_Lambda0):
    return dl - getdistGpc(z,H0,Omega_m0,Omega_Lambda0)

def getz(dl,H0,Omega_m0,Omega_Lambda0):
    """
    returns the redshift for a given luminosity distance in Gpc
    """    
    return (root(dlminusdistz,x0=0.1,args=(dl,H0,Omega_m0,Omega_Lambda0)).x)[0]
    
def Jacobian(M,dist,H0,Omega_m0,Omega_Lambda0):
    """ 
    Jacobian for Fisher parameter transformation from [M,dist,Al,nl,Ag] to [lnM,z,Al,nl,Ag]
    Returns a 5x5 diagonal np.ndarray.
    """
    
    #Jacobian = partial old/partial new
    
    delta = dist*1e-5
    del_z_del_dist = ((getz(dist+delta,H0,Omega_m0,Omega_Lambda0)-getz(dist-delta,H0,Omega_m0,Omega_Lambda0))/(2*delta))
    diag = np.diag((M,(del_z_del_dist)**-1,1.0,1.0,1.0))
    
    return diag
    
#supporting function
def check_prior(param,bound):
    """ return True if param within bound (including edges), False otherwise """

    if (param >= bound[0]) & (param <= bound[1]):
        return 0 #within bounds
    elif (param < bound[0]):
        return -1 #lower bound hit
    else:
        return 1 #upper bound hit
