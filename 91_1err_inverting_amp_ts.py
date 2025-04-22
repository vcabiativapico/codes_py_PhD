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
from scipy.optimize import minimize

def find_first_index_greater_than(lst, target):
    for index, value in enumerate(lst):
        if value > target:
            return index
    return -1



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

# Interpolate at according to org

f = CubicSpline(at,org)

times_nt = 5
at_int = np.linspace(at[0], at[-1], num=nt*times_nt)
org_int = f(at_int)  


idx_fb =  find_first_index_greater_than(diff, np.max(diff) * 0.05) * times_nt


trace_wv_length = -ft*2/dt




def create_gaussian_array(idx_fb,sigma,length,amp_factor):
    # Generate an array of x values around the peak
    x = np.arange(length)
    
    # Calculate the Gaussian function values for each x
    gaussian_array = np.exp(-(x - idx_fb)**2 / (2 * sigma**2))* amp_factor + 1
    return gaussian_array


sigma = trace_wv_length / 2.355  # Standard deviation (width of the Gaussian)
length = len(at_int)  # Length of the array
amp_factor = 2
a = create_gaussian_array(idx_fb,sigma,length,amp_factor) 


def shift_with_roll(shift,data,start_idx):
    shifted_data = data.copy()  # Make a copy to avoid modifying the original array
    
    # Shift elements after the idx_fb
    shifted_data[start_idx:] = np.roll(shifted_data[start_idx:], shift)
    
    # Optionally, set the first few elements of the shifted part to NaN (or another value) to indicate the shift
    shifted_data[start_idx:start_idx + shift] = np.nan  # Fill with NaN or use 0 if preferred
    
    # Now interpolate the NaN values using numpy
    # Find indices where NaN values exist
    nan_indices = np.isnan(shifted_data)
    x = np.arange(len(shifted_data))

    # Perform cubic spline interpolation using CubicSpline
    # We need to interpolate using only the non-NaN values
    valid_x = x[~nan_indices]
    valid_y = shifted_data[~nan_indices]
    
    # Create the cubic spline interpolator
    cs = CubicSpline(valid_x, valid_y, bc_type='natural')  # 'natural' boundary conditions
    
    # Interpolate the NaN values using the cubic spline
    shifted_data[nan_indices] = cs(x[nan_indices])
    return shifted_data


shift = int(trace_wv_length/2)

def modif_trace(org_int,a,shift,start_idx_ts):
    
    org_mod_a = org_int * a
    org_shifted = shift_with_roll(shift,org_mod_a,start_idx_ts)
    return org_shifted
    
final_mod_trace = modif_trace(a,org_int,shift,idx_fb-100)


plt.figure(figsize=(4,8))
# plt.plot(org_mod_a,at_int)
plt.plot(final_mod_trace,at_int)
plt.plot(org_int,at_int)
plt.ylim(0.5,2.0)
plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()



# plt.figure(figsize=(4,8))
# plt.plot(tau_smooth,at_int,'-') 
# plt.gca().invert_yaxis()

# plt.figure(figsize=(4,8))
# plt.plot(a,at_int,'-') 
# plt.gca().invert_yaxis()








#%%


def create_time_shift_array(idx_fb,length,ts_array):
    
    tau = np.zeros(length)
    
    tau[idx_fb:idx_fb+len(ts_array)] = ts_array
    
    for i in range(idx_fb+len(ts_array),len(tau)): 
        tau[i] = tau[idx_fb+len(ts_array)-1]
    
    tau_smooth = gaussian_filter(tau,100)
    
    return tau_smooth



ts_array = [0.0, 0.0, 0.05, 0.010, 0.010]
tau_smooth = create_time_shift_array(idx_fb-100,length,ts_array)

plt.figure(figsize=(4,8))
plt.plot(tau_smooth,at_int)
plt.ylim(0.5,2.0)
plt.gca().invert_yaxis()

plt.figure(figsize=(4,8))
plt.plot(a,at_int)
plt.ylim(0.5,2.0)
plt.gca().invert_yaxis()


def modif_trace(org_int,at_int,a,tau):
    org_mod_a = org_int * a
    at_mod_ts = at_int + tau
    return org_mod_a, at_mod_ts

org_mod, at_mod = modif_trace(org_int,at_int,a,tau_smooth)

# param = np.concatenate((a,tau_smooth))

# x0 = param
# res = minimize(modif_trace, x0, method='Nelder-Mead', tol=1e-6)
# res.x

plt.figure(figsize=(4,8))
plt.plot(org_int, at_int, '-')
plt.plot(org_mod, at_mod, '-') 
plt.ylim(0.5,2.0)
plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()


#%%


f = CubicSpline(at,org)
# at_int = np.linspace(at[0], at[-1], num=nt*times_nt)
# org_int = f(at_int)  

ts_array = [0.0, 0.0, 0.05, 0.010, 0.010]

idx =  find_first_index_greater_than(diff, np.max(diff) * 0.05) 

tau_smooth = create_time_shift_array(idx,len(org),ts_array)

db = np.copy(org)
dm = []

test = []

for i in range(nt):
    t  = i * dt
    t2 = t - tau_smooth[i]
    if t2 < t:
        test.append(t)
        j = f(t)
    else: 
        j = org[i]
    dm.append(j)

plt.figure(figsize=(4,8))
plt.plot(db, at, '-')
plt.plot(dm, at, '-') 
# plt.ylim(0.5,2.0)
plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()

