import numpy as np
import matplotlib.pyplot as plt
from MCMC_func import MCMC_run
from astropy.cosmology import FlatLambdaCDM

H0_true = 70.5   # True value of hubble constant 
Omega_m_true = 0.274
Omega_Lambda = 1 - Omega_m_true
c = 299792458.0 / 1e3
z_s = np.array([0.5, 1.0])

kwargs = {'c':3e5, 'H0_true':H0_true, 'Omega_m_true':Omega_m_true, 
            'Omega_Lambda':Omega_Lambda, 'z_s':z_s}
# Assign a cosmology
cosmo = FlatLambdaCDM(H0=H0_true, Om0=Omega_m_true)

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
param_start =[H0_true, Omega_m_true]

# Run MCMC algorithm
H0_chain, Omega_m_chain, lp = MCMC_run(data_dl, delta_dl, param_start, **kwargs)

# Output results
ma = np.mean(H0_chain)
sda = np.std(H0_chain)
# Hello, how are things getting on? 

mb = np.mean(Omega_m_chain)
sdb = np.std(Omega_m_chain)

print("mean of H0 =", ma, '_', "standard deviation of H0 =", sda)
print("mean of Omega_m =", mb, '_', "standard deviation of Omega_m =", sdb)

# corner-plot result
samples = np.column_stack([H0_chain,Omega_m_chain])

params = [r"$H_0$", r"$\Omega_m$"]

figure = corner.corner(samples, bins = 30, color = 'purple', labels = params, 
                plot_datapoints = False, smooth1d = True, smooth = True, quantiles=[0.16, 0.5, 0.84],
                show_titles = True, label_kwargs = {"fontsize":12}, title_fmt='.2f',title_kwargs={"fontsize": 12})

axes = np.array(figure.axes).reshape((2, 2))

m = [H0_true, Omega_m_true]

for yi in range(2):
    for xi in range(yi):
        ax = axes[yi, xi]
        ax.axvline(m[xi], color="g")
        ax.axhline(m[yi], color="g")
        ax.plot(m[xi], m[yi], "sg")
