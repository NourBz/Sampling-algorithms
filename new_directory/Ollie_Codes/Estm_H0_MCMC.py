import numpy as np
import matplotlib.pyplot as plt
from MCMC_func import MCMC_run
from astropy.cosmology import FlatLambdaCDM

H0_true = 70.5   # True value of hubble constant 
Omega_m = 0.274
Omega_Lambda = 1 - Omega_m
c = 299792458.0 / 1e3
z_s = 1.0 

kwargs = {'c':3e5, 'H0_true':H0_true, 'Omega_m':Omega_m, 
            'Omega_Lambda':Omega_Lambda, 'z_s':z_s}
# Assign a cosmology
cosmo = FlatLambdaCDM(H0=H0_true, Om0=Omega_m)

# Assume known redshift of source  

# Calculate luminosity distance of source assuming a cosmology
D_Mpc = cosmo.luminosity_distance(z_s).value      # Measure in Mpc
# Give rough estimate of precision in parameters
delta_D_Mpc = (1/30) * 1e3

# Input data
data_dl  = np.array([D_Mpc])
delta_dl = np.array([delta_D_Mpc])
data_z   = np.array([z_s])

# Start initial H0 
param_start =[H0_true]

# Run MCMC algorithm
H0_chain, lp = MCMC_run(data_dl, delta_dl, param_start, **kwargs)

# Output results
ma = np.mean(H0_chain)
sda = np.std(H0_chain)

print("mean of H0 =", ma, '_', "standard deviation of H0 =", sda)
# Plot result
plt.hist(H0_chain, bins = 45, color = 'turquoise', edgecolor = 'darkcyan', alpha = 1, label = 'Posterior')
plt.xlabel(r'$H_{0}$',fontsize = 16)
plt.ylabel(r'$P(H_{0}|d)$',fontsize = 16)
plt.title(r'Constraint on Hubble Constant',fontsize = 20)
plt.axvline(x = H0_true, label = 'True value', c = 'red', linestyle = 'dashed')
plt.legend()
plt.tight_layout()
plt.show() 

