#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:32:19 2025

@author: vcabiativapico
"""

import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate  import CubicSpline
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, differential_evolution
from stochopy.optimize  import cpso,pso
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




trace_wv_length = -ft/dt
sigma = trace_wv_length / 2.355  # Standard deviation (width of the Gaussian)
length = len(at)  # Length of the array
amp_factor = 2


idx =  find_first_index_greater_than(diff, np.max(diff) * 0.05)-200

a = create_gaussian_array(idx,sigma,length,amp_factor) 





ts_max = 0.005
ts_array = np.linspace(0,ts_max,100)
smooth_fact = 10

tau_smooth = create_time_shift_array(idx,len(org),ts_array,smooth_fact)






f_new = CubicSpline(at, org, extrapolate=True)

def mod_trace(trace, a, tau,at ): 
    dshift = []
    for i, t in enumerate(at):
        t2 = t - tau[i]
        if t2 < t:
            j = f_new(t2) * a[i]
        else: 
            j = trace[i]
        dshift.append(j)
    dm = np.array(dshift) 
    return dm

dm = mod_trace(org,a,tau_smooth,at)


plt.figure(figsize=(4,8))
plt.plot(org, at, '-')
plt.plot(dm, at, '-') 
plt.ylim(0.5,2.0)
plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()


a_init = create_gaussian_array(idx,20,length,amp_factor) 
tau_init = np.zeros_like(dm) + 0.007

plt.figure(figsize=(4,8))
plt.plot(tau_smooth,at,label=['True TS'])
plt.plot(tau_init,at,label=['TS INIT'])
plt.ylim(0.5,2.0)
plt.gca().invert_yaxis()

plt.figure(figsize=(4,8))
plt.plot(a,at,label=['True amp'])
plt.plot(a_init,at,label=['amp INIT'])
plt.ylim(0.5,2.0)
plt.gca().invert_yaxis()

params = np.concatenate([a_init,tau_init])



def objective(x):
    print(x)
    error = []
    a_vals = x[:len(at)] 
    tau_vals = x[len(at):] 
    for i, t in enumerate(at):
        t2 = t - tau_vals[i]
        if t2 < t:
            d_pred = f_new(t2) * a_vals[i]
        else: 
            d_pred = org[i]
        error.append((d_pred  - dm[i])**2)
    error = np.sum(error)
    return error
    

# Define constraints and bounds
a_lower, a_upper = 0.1, 5.0  # Example range for a(t)
tau_lower, tau_upper = 0, 0.01  # Example range for tau(t)

bounds = [(a_lower, a_upper)] * len(at) + [(tau_lower, tau_upper)] * len(at)


method = 'Nelder-Mead'
method = 'L-BFGS-B'

# result = minimize(objective, params, bounds= bounds, method=method)

print(result)


a_opt = result.x[:len(at)]
tau_opt = result.x[len(at):]



plt.figure(figsize=(4,8))
plt.plot(a_opt,at,label='Final')
plt.plot(a_init,at,label='INIT')
plt.plot(a,at,label='True')
plt.legend()
plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()

plt.figure(figsize=(4,8))
plt.plot(tau_smooth,at,label='True ')
plt.plot(tau_init,at,label='INIT')
plt.plot(tau_opt,at,label='Final')
plt.legend()
plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()


#%%

#%%

a_init = create_gaussian_array(idx,20,length,amp_factor) 
tau_init = np.zeros_like(dm) + 0.007


a_lower, a_upper = 0.1, 5.0  # Example range for a(t)
tau_lower, tau_upper = 0, 0.01  # Example range for tau(t)
bounds = [(a_lower, a_upper)] * len(at) + [(tau_lower, tau_upper)] * len(at)



params = np.concatenate([a_init,tau_init])

params2 = np.zeros((2,len(params)))
params2[0] = params
params2_list = params2.tolist()

def pso_objective(x):
    error = []

    a_vals = x[:len(at)] 
    tau_vals = x[len(at):] 
    # print(len(x))
    for i, t in enumerate(at):
        t2 = t - tau_vals[i]
        if t2 < t:
            d_pred = f_new(t2) * a_vals[i]
        else: 
            d_pred = org[i]
        error.append((d_pred  - dm[i])**2)
    error = np.sum(error)
    return error


result2 = cpso(pso_objective,bounds,x0=params2, maxiter= 100)


a_opt = result2.x[:len(at)]
tau_opt = result2.x[len(at):]



plt.figure(figsize=(4,8))
plt.plot(a_opt,at,label='Final')
plt.plot(a_init,at,label='INIT')
plt.plot(a,at,label='True')
plt.legend()
plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()

plt.figure(figsize=(4,8))
plt.plot(tau_smooth,at,label='True ')
plt.plot(tau_init,at,label='INIT')
plt.plot(tau_opt,at,label='Final')
plt.legend()
plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()
