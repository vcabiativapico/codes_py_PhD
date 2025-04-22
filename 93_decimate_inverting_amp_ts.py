#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:41:47 2025

@author: vcabiativapico
"""


import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate import CubicSpline, interp1d,PchipInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, differential_evolution,curve_fit
from stochopy.optimize import cpso, pso, cmaes, cpso
import pyswarms as ps
from spotfunk.res import procs

def find_first_index_greater_than(lst, target):
    for index, value in enumerate(lst):
        if value > target:
            return index
    return -1


def create_gaussian_array(idx_fb, sigma, length, amp_factor):
    # Generate an array of x values around the peak
    x = np.arange(length)

    # Calculate the Gaussian function values for each x
    gaussian_array = np.exp(-(x - idx_fb)**2 / (2 * sigma**2)) * amp_factor + 1
    return gaussian_array


def create_time_shift_array(idx_fb, length, ts_array, smooth_fact):

    tau = np.zeros(length)

    tau[idx_fb:idx_fb+len(ts_array)] = ts_array

    for i in range(idx_fb+len(ts_array), len(tau)):
        tau[i] = tau[idx_fb+len(ts_array)-1]

    tau_smooth = gaussian_filter(tau, smooth_fact)

    return tau_smooth

# %%

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


tr1 = '../output/90_kimberlina_mod_v3_high/full_sum/f_y0/t1_obs_000' + \
    str(title).zfill(3)+'.dat'
tr2 = '../output/92_kimberlina_corr_amp/full_sum/f_' + \
    name+'/t1_obs_000'+str(title).zfill(3)+'.dat'


org = -gt.readbin(tr1, no, nt).transpose()[:, 125]
ano = -gt.readbin(tr2, no, nt).transpose()[:, 125]

diff = org - ano

org = 2*np.cos(2*np.pi*7.5*at) + 1*np.cos(2*np.pi*10*at) \
    + 2*np.cos(np.pi*12.5*at) + 2*np.cos(np.pi*15*at)


# noise = np.random.normal(0,0.3,nt)
# org = 2*np.cos(1.1*np.pi*7.5*at) + 2*np.cos(1.5*np.pi*10*at) \
#     + 2*np.cos(2*np.pi*12.5*at) + 2*np.cos(2*np.pi*15*at)



org = org / np.max(org)


'''Create amplitude factor curve'''

# trace_wv_length = -ft/dt
# sigma = trace_wv_length / 2.355  # Standard deviation (width of the Gaussian)
# length = len(at)  # Length of the array
# amp_factor = 3
# idx = find_first_index_greater_than(diff, np.max(diff) * 0.05)-400
# a = create_gaussian_array(idx, sigma, length, amp_factor)


idx_for_chg = 3
nb_points = 8
a_cut = np.ones(nb_points)
a_cut[idx_for_chg] = 1.25


'''Create time-shift curve'''

ts_max = 0.005
# ts_array = np.linspace(0, ts_max, 100)
# smooth_fact = 10
# # tau_smooth = create_time_shift_array(idx, len(org), ts_array, smooth_fact)



tau_cut =  np.zeros(nb_points)
tau_cut[idx_for_chg:] = ts_max

''' Define 8 node points'''


int_idx = nt//nb_points +1
at_cut = at[::int_idx]




''' Create interpolation function to call new time-shift index'''


def forward_sample(param):
    "Forward modelling"
    dm = []
    a_vals = param[:len(at_cut)]
    tau_vals = param[len(at_cut):]
    
    f_a = PchipInterpolator(at_cut, a_vals)
    f_tau = PchipInterpolator(at_cut, tau_vals)
    f_new = CubicSpline(at, org, extrapolate=True)

    for i, t in enumerate(at):
        t2 = t - f_tau(t)
        if t2 < t:
            dmod = f_new(t2) * f_a(t)
        else:
            dmod = org[i]
        dm.append(dmod)
    dm = np.array(dm)
    return dm, f_a, f_tau

param = np.concatenate((a_cut,tau_cut))
dm_samp,f_a,f_tau = forward_sample(param)


plt.figure(figsize=(4, 8))
plt.plot(a_cut,at_cut,'o')
plt.plot(f_a(at), at, '-')
plt.title('amplitude vector')
plt.gca().invert_yaxis()

plt.figure(figsize=(4, 8))
plt.plot(tau_cut,at_cut,'o')
plt.plot(f_tau(at), at, '-')
plt.title('time-shift vector')
plt.gca().invert_yaxis()

plt.figure(figsize=(4, 8))
plt.plot(org, at, '-',color='tab:blue', label='Baseline')
plt.plot(dm_samp, at, '-',color='tab:orange',label='Monitor')
# plt.ylim(0.6,1.7)
plt.legend(loc='upper right')
plt.xlim(-2, 2)
plt.gca().invert_yaxis()




# %%


def objective(param_pred):
    '''Objective function'''
    # print('value =', param_pred)
    global aj
    d_pred = forward_sample(param_pred)[0]
    error = (d_pred  - dm_samp)**2
    error = np.sum(error)
    aj.append(error)
    return error

aj = []

a_init = np.zeros_like(at_cut) + 1
tau_init = np.zeros_like(at_cut)

param_init = np.concatenate([a_init, tau_init])

# Define constraints and bounds
a_lower, a_upper = 0.1, 5.0  # range for a(t)
tau_lower, tau_upper = 0, 0.01  # range for tau(t)

bounds = [(a_lower, a_upper)] * len(at_cut) + \
    [(tau_lower, tau_upper)] * len(at_cut)


method = 'Nelder-Mead'
method = 'L-BFGS-B'

result = minimize(objective, param_init,
                  bounds=bounds, method=method)

# result = cmaes(objective,bounds, x0=param_init, maxiter= 500,popsize=len(param_init))
# result = pso(objective,bounds, maxiter= 500,popsize=len(param_init))

# result = cpso(objective,bounds, maxiter= 2000,popsize=len(param_init))

print(result)


a_opt = result.x[:len(at_cut)]
tau_opt = result.x[len(at_cut):]





#%%




plt.rcParams['font.size'] = 17


# fig = plt.figure(figsize=(4, 8))
# plt.title('Amplitude')
# plt.plot(a_opt, at_cut,'o-', color= 'tab:blue',label='Inv')
# plt.plot(a_init, at_cut, '--', color= 'tab:orange',label='Init')
# plt.plot(a_cut, at_cut, 'o-', color= 'tab:green',label='True')
# plt.legend()
# # plt.ylim(0.5,2.0)
# # plt.xlim(-0.1,0.1)
# plt.gca().invert_yaxis()
# flout = '../png/94_amp_ts_inversion/amp_points.png'
# fig.tight_layout()
# print("Export to file:", flout)
# fig.savefig(flout, bbox_inches='tight')


# fig = plt.figure(figsize=(4, 8))
# plt.title('TS')
# plt.plot(tau_cut, at_cut, 'o-',  color= 'tab:green',label='True')
# plt.plot(tau_init, at_cut, '--', color= 'tab:orange',label='Init')
# plt.plot(tau_opt, at_cut, 'o-', color= 'tab:blue',label='Inv')
# plt.legend()
# # plt.ylim(0.5,2.0)
# # plt.xlim(-0.1,0.1)
# plt.gca().invert_yaxis()
# flout = '../png/94_amp_ts_inversion/ts_points.png'
# fig.tight_layout()
# print("Export to file:", flout)
# fig.savefig(flout, bbox_inches='tight')


f_a_opt = PchipInterpolator(at_cut, a_opt, extrapolate=True)
f_tau_opt = PchipInterpolator(at_cut, tau_opt, extrapolate=True)


fig = plt.figure(figsize=(4, 8))
plt.title('TS')
plt.plot(f_tau_opt(at), at, '-', color= 'tab:blue',label='Inv')
plt.plot(f_tau(at), at, '-', color= 'tab:green',label='True')
plt.plot(tau_init, at_cut, '--', color= 'tab:orange',label='Init')
plt.plot(tau_cut, at_cut, 'o',color= 'tab:green')
plt.plot(tau_opt, at_cut, 'o', color= 'tab:blue')
plt.legend()
# plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()
flout = '../png/94_amp_ts_inversion/ts_interpolate.png'
fig.tight_layout()
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')


fig = plt.figure(figsize=(4, 8))
plt.title('amplitude')
plt.plot(f_a_opt(at), at, '-', color= 'tab:blue',label='Inv')
plt.plot(f_a(at), at, '-', color= 'tab:green',label='True')
plt.plot(a_init, at_cut, '--', color= 'tab:orange',label='Init')
plt.plot(a_cut, at_cut,'o', color= 'tab:green')
plt.plot(a_opt, at_cut,'o', color= 'tab:blue')
plt.legend()
# plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()
flout = '../png/94_amp_ts_inversion/amp_interpolate.png'
fig.tight_layout()
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')




dm_samp_init,f_a_init,f_tau_init = forward_sample(param_init)


param_final =  np.concatenate([a_opt, tau_opt])

dm_samp_final,f_a_final,f_tau_final = forward_sample(param_final)


fig = plt.figure(figsize=(4, 8))
plt.plot(org, at, '-',label='Baseline')
plt.plot(dm_samp, at, '-',label='ancienen')
plt.plot(dm_samp_final, at, '--',label='inversion')
plt.legend()
plt.ylim(0.6,1.4)
# plt.ylim(1.5, 1.8)
# plt.xlim(-0.02, 0.02)
plt.gca().invert_yaxis()
flout = '../png/94_amp_ts_inversion/inverted_trace.png'
fig.tight_layout()
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')



fig = plt.figure(figsize=(4, 8))
plt.title('Difference')
plt.plot(org-dm_samp, at, '-',label='base - modelled')
plt.plot(dm_samp-dm_samp_final, at, '-',label='modelled - inverted')
fig.tight_layout()
plt.legend()
plt.gca().invert_yaxis()
fig.tight_layout()
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')


fig = plt.figure()
plt.title('Cost function')
plt.plot(aj)
plt.ylim(0,aj[0])
plt.ylabel('error')
plt.xlabel('iter')
flout = '../png/94_amp_ts_inversion/cost_function.png'
fig.tight_layout()
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')


#%%


sld_ts = procs.sliding_TS(dm_samp,org,oplen=350,si=dt,taper=80)




diff_syn = org -dm_samp
idx = find_first_index_greater_than(diff_syn, np.max(diff_syn) * 0.05)

cc_ts = []
win_width_all = []
win_start_all = []

for i in range(120,500,20):
    win_width = i
    
    for j in range(0,250,10):
        win1 = at[idx+j]
        win2 = win1 + win_width/1000
        cc_ts.append(procs.max_cross_corr(dm_samp,org,
                                          win1=win1*1000,win2=win2*1000,
                                          thresh=None,si=dt,taper=25)/1000)
        
        if i == 120:
            win_start_all.append(win1)
        if j == 0:
            win_width_all.append(win2-win1)


cc_ts_all = np.reshape(cc_ts,(len(win_width_all),len(win_start_all)))



# idx_start = 10

    
# plt.figure(figsize=(12, 8))
# plt.imshow(cc_ts_all,  extent=[win_width_all[0], win_width_all[-1], win_start_all[-1], win_start_all[0]])
# plt.ylabel('Window start')
# plt.xlabel('Window width')
# plt.colorbar()

# for i in range(0,20,4):
#     plt.scatter(win_width_all[i],win_start_all[idx_start],c= 'k')
    
    
  
# for i in range(0,20,4):   
#     fig = plt.figure(figsize=(4, 8))  
#     plt.title(win_width_all[i])
#     plt.plot(org, at, '-',label='Baseline')
#     plt.plot(dm_samp, at, '-',label='ancienne')
#     # plt.plot(dm_samp_final, at, '--',label='inversion')
#     plt.axhline(win_start_all[idx_start],color='tab:red')
#     plt.axhline(win_start_all[idx_start]+win_width_all[i],color='tab:red')
#     plt.ylim(0.5,1.8)
#     plt.legend()
#     plt.gca().invert_yaxis()
#     fig.tight_layout()


max_cc_ts = np.max(cc_ts_all)
idx_max_cc_ts = np.where(max_cc_ts == cc_ts_all)

win_width_max = win_width_all[idx_max_cc_ts[0][0]]
win_start_max = win_start_all[idx_max_cc_ts[1][0]]


fig = plt.figure(figsize=(4, 8))
plt.title('TS all')
plt.plot(sld_ts/1000,at,color= 'tab:red',label='SLD_ts')
plt.axvline(max_cc_ts,color= 'tab:purple',label='max_cc')
plt.axhline(win_start_max,linestyle= '--',color='tab:brown')
plt.axhline(win_start_max+win_width_max,linestyle= '--',color='tab:brown')
plt.plot(f_tau_opt(at), at, '-', color= 'tab:blue',label='Inv')
plt.plot(f_tau(at), at, '-', color= 'tab:green',label='True')
# plt.plot(tau_init, at_cut, '--', color= 'tab:orange',label='Init')
plt.plot(tau_cut, at_cut, 'o',color= 'tab:green')
plt.plot(tau_opt, at_cut, 'o', color= 'tab:blue')
plt.legend()
plt.gca().invert_yaxis()
flout = '../png/94_amp_ts_inversion/ts_all.png'
fig.tight_layout()
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')

