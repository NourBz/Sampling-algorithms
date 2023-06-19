import numpy as np
import matplotlib.pyplot as plt
from MCMC_func import MCMC_run
from astropy.cosmology import FlatLambdaCDM

H0_true = 70.5
cosmo = FlatLambdaCDM(H0=H0_true, Om0=0.274, Ob0 = 0.046)

# Check the result: 
z_s = 0.8

D_Mpc = cosmo.luminosity_distance(z_s).value      # Measure in Mpc
delta_D_Mpc = (1/10) * 1e3

data_dl  = np.array([0, D_Mpc])
delta_dl = np.array([1e-10, delta_D_Mpc])
data_z   = np.array([0,z_s])

param_start =[60]
Ntotal = 100000 # Total number of iterations
burnin = 30000  # Set burn-in. This is the amount of samples we will discard whilst looking 
                # for the true parameters
H0_var_prop = 0.8743905288970055
printerval = 50000

N_obs = len(data_z)

H0_chain, lp = MCMC_run(data_dl, data_z, delta_dl, param_start)

plt.hist(H0_chain, bins = 45, color = 'turquoise', edgecolor = 'darkcyan', alpha = 1, label = 'Posterior')
plt.xlabel(r'$H_{0}$',fontsize = 16)
plt.ylabel(r'$P(H_{0}|d)$',fontsize = 16)
plt.title(r'Constraint on Hubble Constant',fontsize = 20)
plt.axvline(x = H0_true, label = 'True value', c = 'red', linestyle = 'dashed')
plt.legend()
plt.tight_layout()
plt.show() 
ma = np.mean(H0_chain)
sda = np.std(H0_chain)

print("mean of H0 =", ma, '_', "standard deviation of H0 =", sda)

