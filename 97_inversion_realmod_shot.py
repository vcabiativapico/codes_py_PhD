#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 10:41:47 2025

@author: vcabiativapico
"""


import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from matplotlib.ticker import FormatStrFormatter
from scipy.interpolate import CubicSpline, interp1d,PchipInterpolator
from scipy.ndimage import gaussian_filter
from scipy.optimize import minimize, differential_evolution,curve_fit
from stochopy.optimize import cpso, pso, cmaes, cpso
import pyswarms as ps
from spotfunk.res import procs
import pandas as pd
from tqdm import tqdm
from wiggle.wiggle import wiggle

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


def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier


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
            if inv_type==0:
                dmod = f_new(t2) * f_a(t)
            elif inv_type == 1 :
                dmod = f_new(t2) + f_a(t)
        else:
            dmod = org[i]
        dm.append(dmod)
    dm = np.array(dm)
    return dm, f_a, f_tau


def objective(param_pred):
    '''Objective function'''
    # print('value =', param_pred)
    global aj
    d_pred = forward_sample(param_pred)[0]
    error = (d_pred  - dm_samp)**2
    error = np.sum(error)
    aj.append(error)
    return error


no = 251
do = 12.0/1000.
fo = -(no-1)/2*do
ao = fo + np.arange(no)*do

# inv_type = 0 # multiply a
inv_type = 1 # sum a



full_org     = []
full_dm_samp = []
full_a_opt   = []
full_tau_opt = []
full_sld_ts  = []
full_max_cross_corr = []
full_picked_ts = []


title = 166
off = 0.57
idx_off = int(off * 1000 // 12 + 125)

ran_off = np.arange(0,no,4)


for idx_off in ran_off:    
    print(idx_off)
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
    
    
    noise_g = np.random.normal(0,0.002,nt)
    
    noise = procs.freq_filtering(noise_g,10,15,80,100,butter=False,slope1=12,slope2=18,bandstop=False,si=dt)
    noise = 0
    
    tr1 = '../output/94_kimberlina_v4/full_sum/medium/f_0/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/94_kimberlina_v4/full_sum/medium/f_'+name+'/t1_obs_000'+str(title).zfill(3)+'.dat'
     
    
 
        
    org = gt.readbin(tr1, no, nt).transpose()[:, idx_off] + noise
    ano = gt.readbin(tr2, no, nt).transpose()[:, idx_off] + noise
    
    
    
    org = org / np.max(org)
    ano = ano / np.max(ano)
    
    diff = org - ano
    
    
    # cut_idx_in = 550
    
    perc = 0.01
    first_arr_idx = find_first_index_greater_than(org, np.max(org)*perc)+250
    cut_idx = first_arr_idx
    cut_idx_in = np.array(cut_idx)*dt +ft
    
    print(cut_idx)
    
    
    org[:cut_idx]= 0
    ano[:cut_idx] = 0
    
    # cut_idx = 0
    # org = org[cut_idx:]
    # ano = ano[cut_idx:]
    # at = at[cut_idx:]
    
    idx_max_diff = np.argmax(diff[:1701])
    
    '''Picking the event according to amplitude'''
    
    perc = 0.01
    first_idx = find_first_index_greater_than(diff, np.max(diff)*perc)-150
    fb_idx = np.argmax(org[first_idx:])+first_idx
    fb_t = np.array(fb_idx)*dt +ft
    
    win1_add = -0.03
    win2_add = win1_add+0.2
    win1_array = (fb_t + win1_add) * 1000
    win2_array = (fb_t + win2_add) * 1000
    
    if at[idx_max_diff] > 1:
        win1 = win1_array
        win2 = win2_array
        if win1 > at[-1]*1000-100: 
            win1 = at[-1]*1000-100
            
        if win2 > at[-1]*1000: 
            win2 = at[-1]*1000
        
        max_cross_corr = procs.max_cross_corr(org,ano,win1=win1,win2=win2,thresh=None,si=dt,taper=25)
    else: 
        max_cross_corr=0
    
    pk_base = procs.extremum_func(org,win1=win1,win2=win2,maxi=True,si=dt)
    pk_monitor = procs.extremum_func(ano,win1=win1,win2=win2,maxi=True,si=dt)
    
    ts_picked = pk_base[0] - pk_monitor[0]
    
    
    
    ''' Create interpolation function to call new time-shift index'''
    
    dm_samp = np.copy(ano)
    
    aj = []
    
    
    ''' Define # node points'''
    
    nb_points = 16
    a_cut = np.ones(nb_points)
    tau_cut =  np.zeros(nb_points)
    
    int_idx = len(at)//nb_points +1
    at_cut = at[::int_idx]
    
    if inv_type==0:
        a_init = np.zeros_like(at_cut) + 1
    elif inv_type == 1 :
        a_init = np.zeros_like(at_cut) 
    
    
    
    tau_init = np.zeros_like(at_cut)
    param_init = np.concatenate([a_init, tau_init])
    
    ''' Define constraints and bounds'''
    
    if inv_type==0:
        a_lower, a_upper = 0.75, 1.25  # range for a(t)
        tau_lower, tau_upper = 0, 0.1  # range for tau(t)
    elif inv_type == 1 :
        a_lower, a_upper = -1, 1  # range for a(t)
        tau_lower, tau_upper = 0, 0.15  # range for tau(t)
    
    
    
    
    bounds = [(a_lower, a_upper)] * len(at_cut) + \
        [(tau_lower, tau_upper)] * len(at_cut)
    
    
    method = 'Nelder-Mead'
    # method = 'L-BFGS-B'
    
    # result = minimize(objective, param_init,
    #                   bounds=bounds, method=method)
    
    # # result = cmaes(objective,bounds, x0=param_init, maxiter= 500,popsize=len(param_init))
    # # result = pso(objective,bounds, maxiter= 500,popsize=len(param_init),seed=len(param_init)*2)
    
    # # result = cpso(objective,bounds, maxiter= 2000,popsize=len(param_init))
    
    # print(result)
    
    
    # a_opt = result.x[:len(at_cut)]
    # tau_opt = result.x[len(at_cut):]
    
    
  
    fig = plt.figure(figsize=(4, 8))
    plt.title('Traces')
    plt.plot(org, at, '-',label='Baseline')
    plt.plot(dm_samp, at, '-',label='Monitor')
    # plt.ylim(0.6,1.7)
    plt.xlabel('Amplitude')
    plt.ylabel('Time (s)')
    # plt.ylim(1.4,2.3)
    plt.xlim(-1, 1)
    plt.legend(loc='upper right')
    fig.tight_layout()
    plt.xlim(-np.max(dm_samp), np.max(dm_samp))
    plt.gca().invert_yaxis()




    
    # plt.rcParams['font.size'] = 17
    
    # f_a_opt = PchipInterpolator(at_cut, a_opt, extrapolate=True)
    # f_tau_opt = PchipInterpolator(at_cut, tau_opt, extrapolate=True)
    
    # fig = plt.figure(figsize=(4, 8))
    # plt.title('TS')
    # plt.plot(f_tau_opt(at), at, '-', color= 'tab:blue',label='Inv')
    # # plt.plot(f_tau(at), at, '-', color= 'tab:green',label='True')
    # plt.plot(tau_init, at_cut, '--', color= 'tab:orange',label='Init')
    # # plt.plot(tau_cut, at_cut, 'o',color= 'tab:green')
    # plt.plot(tau_opt, at_cut, 'o', color= 'tab:blue')
    # plt.legend()
    # plt.xlabel('time-shift')
    # plt.ylabel('Time (s)')
    # # plt.ylim(0.9,2.0)
    # # plt.ylim(0.5,2.0)
    # # plt.xlim(-0.1,0.1)
    # plt.gca().invert_yaxis()
    # flout = '../png/97_inversion_realmod_shot/ts_interpolate'+str(title)+'_off_'+str(idx_off)+'.png'
    # fig.tight_layout()
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    
    
    # fig = plt.figure(figsize=(4, 8))
    # plt.title('amplitude')
    # plt.plot(f_a_opt(at), at, '-', color= 'tab:blue',label='Inv')
    # # plt.plot(f_a(at), at, '-', color= 'tab:green',label='True')
    # plt.plot(a_init, at_cut, '--', color= 'tab:orange',label='Init')
    # # plt.plot(a_cut, at_cut,'o', color= 'tab:green')
    # plt.plot(a_opt, at_cut,'o', color= 'tab:blue')
    # plt.legend()
    # plt.xlabel('Amplitude')
    # plt.ylabel('Time (s)')
    # # plt.ylim(0.9,2.0)
    # # plt.ylim(0.5,2.0)
    # # plt.xlim(-0.1,0.1)
    # plt.gca().invert_yaxis()
    # flout = '../png/97_inversion_realmod_shot/amp_interpolate'+str(title)+'_off_'+str(idx_off)+'.png'
    # fig.tight_layout()
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    
    # dm_samp_init,f_a_init,f_tau_init = forward_sample(param_init)
    
    # param_final =  np.concatenate([a_opt, tau_opt])
    
    # dm_samp_final,f_a_final,f_tau_final = forward_sample(param_final)
    
    # dm_max = np.max(dm_samp_final)
    # dm_min = -dm_max
    
    # fig = plt.figure(figsize=(4, 8))
    # plt.title('Traces')
    # plt.plot(org, at, label='Baseline')
    # plt.plot(dm_samp, at, label='Monitor')
    # plt.plot(dm_samp_final, at,'--', label='inversion')
    # plt.legend()
    # plt.legend(loc='upper right')
    # # plt.ylim(0.9,2.0)
    # plt.ylim(1.5, 2.3)
    # plt.xlim(dm_min, dm_max)
    # plt.xlabel('amplitude')
    # plt.ylabel('Time (s)')
    # plt.gca().invert_yaxis()
    # flout = '../png/97_inversion_realmod_shot/inverted_trace'+str(title)+'_off_'+str(idx_off)+'.png'
    # fig.tight_layout()
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    
    
    
    # fig = plt.figure(figsize=(4, 8))
    # plt.title('Difference')
    # plt.plot(org-dm_samp, at, '-',label='base - modelled')
    # plt.plot(dm_samp-dm_samp_final, at, '-',label='modelled - inverted')
    # fig.tight_layout()
    # plt.legend()
    # plt.xlabel('Difference')
    # plt.ylabel('Time (s)')
    # plt.xlim(np.min(org-dm_samp), -np.min(org-dm_samp))
    # plt.gca().invert_yaxis()
    # flout = '../png/97_inversion_realmod_shot/difference_trace'+str(title)+'_off_'+str(idx_off)+'.png'
    # fig.tight_layout()
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    
    
    
    # fig = plt.figure()
    # plt.title('Cost function')
    # plt.plot(aj)
    # plt.ylim(0,aj[0])
    # plt.ylabel('error')
    # plt.xlabel('iter')
    # flout = '../png/97_inversion_realmod_shot/cost_function'+str(title)+str(idx_off)+'.png'
    # fig.tight_layout()
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')

    
    
    
    
 
    
    sld_ts = procs.sliding_TS(dm_samp,org,oplen=300,si=dt,taper=100)
    
    
    
    
    plt.figure(figsize=(4, 8))
    plt.plot(org, at, '-',label='Baseline')
    plt.plot(dm_samp, at, '-',label='Monitor')
    plt.scatter(pk_base[1],pk_base[0]/1000+ft)
    plt.scatter(pk_monitor[1],pk_monitor[0]/1000+ft)
    plt.axhline(win1/1000)
    plt.axhline(win2/1000)
    # plt.ylim(0.6,1.7)
    plt.ylim(1.0,2.3)
    fig.tight_layout()
    plt.xlabel('Difference')
    plt.ylabel('Time (s)')
    plt.legend(loc='upper right')
    plt.xlim(np.min(dm_samp), np.max(dm_samp))
    plt.gca().invert_yaxis()
    flout = '../png/97_inversion_realmod_shot/traces_all'+str(title)+'_off_'+str(idx_off)+'.png'
    fig.tight_layout()
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')

    
    
    xmax = np.max(sld_ts/1000)+np.max(sld_ts/1000)*0.2
    xmin =- np.max(sld_ts/1000)+np.max(sld_ts/1000)*0.2
    
    # fig = plt.figure(figsize=(4, 8))
    # plt.title('TS all')
    # plt.plot(sld_ts/1000,at,color= 'tab:red',label='SLD_ts')
    # plt.plot(f_tau_opt(at), at, '-', color= 'tab:blue',label='Inv')
    # plt.plot(tau_opt, at_cut, 'o', color= 'tab:blue')
    # plt.axvline(-ts_picked/1000,color= 'tab:purple',label='picked_ts')
    # plt.axvline(-max_cross_corr/1000,color= 'tab:green',label='cc_ts')
    # # plt.plot(f_tau(at), at, '-', color= 'tab:green',label='True')
    # # plt.plot(tau_init, at_cut, '--', color= 'tab:orange',label='Init')
    # # plt.plot(tau_cut, at_cut, 'o',color= 'tab:green')
    # # plt.xlim(-0.001,0.007)
    # # plt.ylim(0.9,2.0)
    # plt.xlim(xmin,xmax)
    # plt.xlabel('time-shift')
    # plt.ylabel('Time (s)')
    # plt.legend()
    # plt.gca().invert_yaxis()
    # flout = '../png/97_inversion_realmod_shot/ts_all'+str(title)+'_off_'+str(idx_off)+'.png'
    # fig.tight_layout()
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')


    full_org.append(org)
    full_dm_samp.append(dm_samp)
    # full_a_opt.append(a_opt)
    # full_tau_opt.append(tau_opt)
    full_sld_ts.append(sld_ts)
    full_max_cross_corr.append(-max_cross_corr)
    full_picked_ts.append(-ts_picked)
    

    

file = '../png/97_inversion_realmod_shot/results_bound_001.pkl'
    
df = pd.DataFrame({'full_org':full_org,
                   'full_dm_samp':full_dm_samp,
                   # 'full_a_opt':full_a_opt,
                   # 'full_tau_opt':full_tau_opt,
                   'full_sld_ts':full_sld_ts,
                   'full_max_cross_corr':full_max_cross_corr,
                  'full_picked_ts':full_picked_ts})
df.to_pickle(file)
    

    
    #%%

# test = pd.read_pickle(file)
# list_test= test['full_tau_opt'].tolist()


# plt.figure()
# plt.imshow(np.transpose(list_test),aspect='auto',
#            vmin=np.min(list_test),vmax=-np.min(list_test),
#            extent=[ao[0],ao[-1],at[-1],at[0]],cmap= 'seismic')
# plt.colorbar()

    
plt.figure()
plt.imshow(np.transpose(full_org),aspect='auto',
           vmin=np.min(full_org),vmax=-np.min(full_org),
           extent=[ao[0],ao[-1],at[-1],at[0]],cmap= 'seismic')
plt.colorbar()


plt.figure()
plt.imshow(np.transpose(full_org)-np.transpose(full_dm_samp),aspect='auto',
           vmin=np.min(full_org)/2,vmax=-np.min(full_org)/2,
           extent=[ao[0],ao[-1],at[-1],at[0]],cmap= 'seismic')
plt.colorbar()

plt.figure()
plt.imshow(np.transpose(full_tau_opt),
           vmin=np.min(full_tau_opt),vmax=np.max(full_tau_opt),
           aspect='auto',extent=[ao[0],ao[-1],at[-1],at[0]])
plt.colorbar()    




plt.figure()
plt.imshow(np.transpose(full_sld_ts)/1000,
           vmin=np.min(full_tau_opt),vmax=np.max(full_tau_opt),
           aspect='auto',
           extent=[ao[0],ao[-1],at[-1],at[0]])
plt.colorbar()    




full_sld_ts_sm = []
max_array_ts_sld_sm = []

for array in full_sld_ts:
    full_sld_ts_sm.append(gaussian_filter(array,15))
    max_array_ts_sld_sm.append(np.max(array)/1000)
    
    
plt.figure()
plt.imshow(np.transpose(full_sld_ts_sm),
           # vmin=0,vmax=np.max(full_tau_opt),
           vmin=0,vmax=8,
           aspect='auto',
           extent=[ao[0],ao[-1],at[-1],at[0]])
plt.colorbar()   




max_array_ts_inv = []
max_array_ts_sld = []
max_array_ts_cc = []


for array in full_tau_opt: 
    max_array_ts_inv.append(np.max(array[14:]))

for array in full_sld_ts:
    max_array_ts_sld.append(np.max(array)/1000)

for array in full_max_cross_corr:
    max_array_ts_cc.append(np.max(array)/1000)

plt.figure(figsize=(12,8))
plt.title('ts')
plt.plot(ao,np.array(full_picked_ts)/1000,label='picked')
plt.plot(ao,max_array_ts_sld,label='sld')
plt.plot(ao,max_array_ts_sld_sm, label='sld_smooth')
plt.plot(ao,max_array_ts_inv,label='inv')
plt.plot(ao,max_array_ts_cc,label='cc')
plt.legend()
plt.xlabel('offset')
plt.ylabel('time-shit')


# axi = np.zeros(np.size(full_tau_opt))
# fig, (axi) = plt.subplots(nrows=1, ncols=np.shape(full_tau_opt)[0],
#                           sharey=True,
#                           figsize=(20, 8),
#                           facecolor="white")

# for i in range(len(ran_off)):
#     xmax = np.max(full_tau_opt)
#     xmin = -xmax
    
#     axi[i].plot(full_sld_ts[i],at, 'r')
#     axi[i].plot(full_tau_opt[i],at_cut, 'b')
#     axi[i].set_xlim(xmin, xmax)
#     axi[i].set_ylim(at[-1], at[0])
#     # axi.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
#     plt.rcParams['font.size'] = 14
#     # plt.colorbar()
#     fig.tight_layout()
#     axi[0].set_ylabel('Time (s)')
   
#     fig.text(0.48, -0.01, "Amplitude")
#     fig.text(0.5, 1, '$\delta p$')
#     flout = '../png/97_inversion_realmod_shot/many_shot.png'
#     print("Export to file:", flout)
#     fig.savefig(flout, bbox_inches='tight')



# axi = np.zeros(np.size(full_tau_opt))
# fig, (axi) = plt.subplots(nrows=1, ncols=np.shape(full_tau_opt)[0],
#                           sharey=True,
#                           figsize=(20, 8),
#                           facecolor="white")

# for i in range(len(ran_off)):
#     xmax = np.max(full_max_cross_corr)/1000
#     xmin = -xmax
    
#     axi[i].plot(full_tau_opt[i],at_cut, 'r')
    
#     axi[i].set_xlim(xmin, xmax)
#     axi[i].set_ylim(at[-1], at[0])
#     # axi.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
#     plt.rcParams['font.size'] = 14
#     # plt.colorbar()
#     fig.tight_layout()
#     axi[0].set_ylabel('Time (s)')
   
#     fig.text(0.48, -0.01, "Amplitude")
#     fig.text(0.5, 1, '$\delta p$')
#     flout = '../png/97_inversion_realmod_shot/many_shot_2.png'
#     print("Export to file:", flout)
#     fig.savefig(flout, bbox_inches='tight')


