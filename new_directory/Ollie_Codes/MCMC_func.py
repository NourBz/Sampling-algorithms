import numpy as np
from scipy import integrate
from tqdm import tqdm

def llike(data_dl, data_z, H0_prop, delta_dl, N_obs):

    c = 3e5
    omega = 0.274

    func = 1/(np.sqrt(omega*((data_z+1)**3)+1-omega))
    
    integ = integrate.cumtrapz(func, data_z, initial = 0) 
    
    init_point = 0.5*0*(func[0] + 1)
    first_term = -(N_obs/2) * np.log(2 * np.pi) - np.log(np.prod(delta_dl))
    second_term = -np.sum((1/(2 * delta_dl**2)) * (data_dl - ((c/H0_prop)*((1+data_z)*(integ + init_point))))**2)
    

    return(first_term + second_term)

def lprior_uniform(param, param_low_val, param_high_val):
    """
    Set uniform priors on parameters with select ranges.
    """
    if param < param_low_val or param > param_high_val:
        return -np.inf
    else:
        return 0

def lpost(data_dl, data_z, H0_prop, delta_dl, N_obs, param1, param1_low_range = 30, param1_high_range = 100):
    '''
    Compute log posterior - require log likelihood and log prior.
    '''
    return(lprior_uniform(param1, param1_low_range, param1_high_range) 
           + llike(data_dl, data_z, H0_prop, delta_dl, N_obs))

def accept_reject(lp_prop, lp_prev):
    '''
    Compute log acceptance probability (minimum of 0 and log acceptance rate)
    Decide whether to accept (1) or reject (0)
    '''
    u = np.random.uniform(size = 1)  # U[0, 1]
    logalpha = np.minimum(0, lp_prop - lp_prev)  # log acceptance probability
    if np.log(u) < logalpha:
        return(1)  # Accept
    else:
        return(0)  # Reject

def MCMC_run(data_dl, data_z, delta_dl, param_start,  printerval = 50000, H0_var_prop = 0.8,  Ntotal = 300000, burnin = 10000):
    '''
    Metropolis MCMC sampler
    '''

    # Set starting values

    H0_chain = [param_start[0]]
    N_obs = len(data_z)
                                           
    # Initial value for log posterior
    lp = []
    lp.append(lpost(data_dl, data_z, H0_chain[0], delta_dl, N_obs, H0_chain[0])) # Append first value of log posterior 
    
    lp_store = lp[0]  # Create log posterior storage to be overwritten

    #####                                                  
    # Run MCMC
    #####
    accept_reject_count = [1]

    for i in tqdm(range(1, Ntotal)):
        
        if i % printerval == 0: # Print accept/reject ratio.
            # tqdm.write("Iteration ", i, "accept_reject =",sum(accept_reject_count)/len(accept_reject_count))
            accept_reject_ratio = sum(accept_reject_count)/len(accept_reject_count)
            tqdm.write("Iteration {0}, accept_reject = {1}".format(i,accept_reject_ratio))
            # print("Iteration ", i, "accept_reject =",sum(accept_reject_count)/len(accept_reject_count))
            
        lp_prev = lp_store  # Call previous stored log posterior
        
        # Propose new points according to a normal proposal distribution of fixed variance 

        H0_prop = np.random.normal(H0_chain[i - 1], np.sqrt(H0_var_prop))
    
        # Compute log posterior
        lp_prop = lpost(data_dl, data_z, H0_prop, delta_dl, N_obs, H0_prop)
        
        if accept_reject(lp_prop, lp_prev) == 1:  # Accept
            H0_chain.append(H0_prop)    # accept H0_{prop} as new sample
            accept_reject_count.append(1)
            lp_store = lp_prop  # Overwrite lp_store
            
        else:  # Reject, if this is the case we use previously accepted values
            H0_chain.append(H0_chain[i - 1])
            accept_reject_count.append(0)

        lp.append(lp_store)
    
    # Recast as np.arrays
    H0_chain = np.array(H0_chain)
    H0_chain = H0_chain[burnin:]

    
    return H0_chain, lp  # Return chains and log posterior.    