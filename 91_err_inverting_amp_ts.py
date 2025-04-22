#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:41:47 2025

@author: vcabiativapico
"""


import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate  import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, differential_evolution
from stochopy.optimize  import cpso,pso,cmaes
import pyswarms as ps


def find_first_index_greater_than(lst, target):
    for index, value in enumerate(lst):
        if value > target:
            return index
    return -1

def create_gaussian_array(idx_fb,sigma,length,amp_factor):
    # Generate an array of x values around the peak
    x = np.arange(length)
    
    # Calculate the Gaussian function values for each x
    gaussian_array = np.exp(-(x - idx_fb)**2 / (2 * sigma**2))* amp_factor + 1
    return gaussian_array


def create_time_shift_array(idx_fb,length,ts_array,smooth_fact):
    
    tau = np.zeros(length)
    
    tau[idx_fb:idx_fb+len(ts_array)] = ts_array
    
    for i in range(idx_fb+len(ts_array),len(tau)): 
        tau[i] = tau[idx_fb+len(ts_array)-1]
    
    tau_smooth = gaussian_filter(tau,smooth_fact)
    
    return tau_smooth



#%%


title = 301
year = 30
part = '_p2_v1'
name = str(year)+part


nt = 1801
dt = 1.41e-3
ft = -100.11e-3
nz = 151
fz = 0.0
dz = 12.0/1000.
nx = 601
fx = 0.0
dx = 12.0/1000.
no = 251
do = dx
fo = -(no-1)/2*do
ao = fo + np.arange(no)*do
at = ft + np.arange(nt)*dt
az = fz + np.arange(nz)*dz
ax = fx + np.arange(nx)*dx


tr1 = '../output/90_kimberlina_mod_v3_high/full_sum/f_y0/t1_obs_000'+str(title).zfill(3)+'.dat'
tr2 = '../output/92_kimberlina_corr_amp/full_sum/f_'+name+'/t1_obs_000'+str(title).zfill(3)+'.dat'


org = -gt.readbin(tr1, no, nt).transpose()[:,125]
ano = -gt.readbin(tr2, no, nt).transpose()[:,125]

diff = org - ano



'''Create amplitude factor curve'''

trace_wv_length = -ft/dt
sigma = trace_wv_length / 2.355  # Standard deviation (width of the Gaussian)
length = len(at)  # Length of the array
amp_factor = 2
idx =  find_first_index_greater_than(diff, np.max(diff) * 0.05)-200
a = create_gaussian_array(idx,sigma,length,amp_factor) 



'''Create time-shift curve'''

ts_max = 0.005
ts_array = np.linspace(0,ts_max,100)
smooth_fact = 10
tau_smooth = create_time_shift_array(idx,len(org),ts_array,smooth_fact)

''' Create interpolation function to call new time-shift index'''
f_new = CubicSpline(at, org, extrapolate=True)


def mod_trace_in_two(param):
    "Forward modelling"
    dm = [] 
    a_vals = param[:len(at)] 
    tau_vals = param[len(at):] 
    for i, t in enumerate(at):
        t2 = t - tau_vals[i]
        if t2 < t:
            dmod = f_new(t2) * a_vals[i]
        else: 
            dmod = org[i]
        dm.append(dmod)
    dm = np.array(dm) 
    return dm




''' Build the monitor trace dm'''
param_m = np.concatenate([a,tau_smooth])
dm = mod_trace_in_two(param_m)

plt.figure(figsize=(4,8))
plt.plot(org, at, '-')
plt.plot(dm, at, '-') 
plt.ylim(0.5,2.0)
plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()

#%%

def objective_with_args(param_pred):
    '''Objective function'''
    # print('value =', param_pred)
    d_pred = mod_trace_in_two(param_pred)
    error = (d_pred  - dm)**2
    error = np.sum(error)
    return error



a_init = create_gaussian_array(idx,20,length,amp_factor) 
tau_init = np.zeros_like(dm) + 0.007

param_init = np.concatenate([a_init,tau_init])

# Define constraints and bounds
a_lower, a_upper = 0.1, 5.0  # Example range for a(t)
tau_lower, tau_upper = 0, 0.01  # Example range for tau(t)

bounds = [(a_lower, a_upper)] * len(at) + [(tau_lower, tau_upper)] * len(at)


method = 'Nelder-Mead'
method = 'L-BFGS-B'

result = minimize(objective_with_args, param_init, bounds= bounds, method=method)

# result = cmaes(objective_with_args,bounds, x0=param_init, maxiter= 100,popsize=len(param_init))
# result = pso(objective_with_args,bounds, maxiter= 100,popsize=len(param_init))


print(result)


a_opt = result.x[:len(at)]
tau_opt = result.x[len(at):]



plt.figure(figsize=(4,8))
plt.title('Amplitude')
plt.plot(a_opt,at,label='Final')
plt.plot(a_init,at,'--',label='INIT')
plt.plot(a,at,label='True')
plt.legend()
plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()

plt.figure(figsize=(4,8))
plt.title('TS')
plt.plot(tau_smooth,at,label='True ')
plt.plot(tau_init,at,label='INIT')
plt.plot(tau_opt,at,label='Final')
plt.legend()
plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()


#%%
bounds = [(a_lower, a_upper)] * len(at) + [(tau_lower, tau_upper)] * len(at)


      

def mod_trace_in_one(a_vals, tau_vals):
    dm = [] 
    for i, t in enumerate(at):
        t2 = t - tau_vals[i]
        if t2 < t:
            dmod = f_new(t2) * a_vals[i]
        else: 
            dmod = org[i]
        dm.append(dmod)
    dm = np.array(dm) 
    return dm

dm = mod_trace_in_one(a, tau_smooth).T

def objective_with_args_in_one(a_pred,tau_pred):
    '''Objective function'''
    
    d_pred = mod_trace_in_two(a_pred,tau_pred)
    error = (d_pred  - dm)**2
    error = np.sum(error)
    return error

# param2 = np.array([a_init, tau_init])




#%%
# Set-up hyperparameters
options = {'c1': 0.5, 'c2': 0.3, 'w':0.9}

# Call instance of PSO
optimizer = ps.single.GlobalBestPSO(n_particles=nt*5, dimensions=nt*2, options=options)

# Perform optimization
cost, pos = optimizer.optimize(objective_with_args, iters=1000)

     

plt.plot(pos,at)