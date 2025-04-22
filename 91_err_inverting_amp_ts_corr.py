#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  3 15:32:19 2025

@author: vcabiativapico
"""

import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate  import CubicSpline, interp1d
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, differential_evolution

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

plt.figure(figsize=(4,8))
plt.plot(tau_smooth,at)
plt.ylim(0.5,2.0)
plt.gca().invert_yaxis()

plt.figure(figsize=(4,8))
plt.plot(a,at)
plt.ylim(0.5,2.0)
plt.gca().invert_yaxis()

db = np.copy(org)
dshift = []

test = []

f_new = CubicSpline(at, org, extrapolate=True)

def mod_trace(trace, a, tau,at ): 
    
    for i in range(len(trace)):
            t  = at[i]
            t2 = t - tau[i]
            if t2 < t:
                test.append(t)
                j = f_new(t2)
            else: 
                j = trace[i]
            dshift.append(j)
    dm = np.array(dshift) * a
    return dm

dm = mod_trace(org,a,tau_smooth,at)





def objective(params,t,dm): 
    a_t, tau_t = params
    t2 = t - tau_t
    if t2 < t:
        j = f_new(t2)
    else: 
        j = org[i]
    
    pred_m = j * a_t
    return np.sum((dm - pred_m) ** 2) 





plt.figure(figsize=(4,8))
plt.plot(org, at, '-')
plt.plot(dm, at, '-') 
plt.ylim(0.5,2.0)
plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()


# Initial guesses for a(t) and tau(t)
initial_guess = [1, 0.005]  # Assume initial values for a and tau

# Bounds to ensure physical constraints (e.g., tau should be positive)
bounds = [(0, None), (0, 0.005)]  # a ≥ 0, 0 ≤ tau ≤ max time

# Solve for each t
a_results = []
tau_results = []

for i, t in enumerate(at):
    m_t = dm[i]

    # Minimize the objective function
    result = minimize(objective, initial_guess, args=(t, m_t), bounds=bounds, method='L-BFGS-B')
    # result = differential_evolution(objective, bounds, args=(t, m_t), strategy='best1bin', tol=1e-6)
    
    a_opt, tau_opt = result.x

    a_results.append(a_opt)
    tau_results.append(tau_opt)

a_results = np.array(a_results)
tau_results = np.array(tau_results)

plt.figure(figsize=(4,8))
plt.plot(a_results,at)
plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()

plt.figure(figsize=(4,8))
plt.plot(tau_results,at)
plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()