#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 15:59:37 2025

@author: vcabiativapico
"""


import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from PIL import Image 
from scipy.ndimage import gaussian_filter
from scipy import signal
from spotfunk.res import procs
import pandas as pd
from matplotlib.gridspec import GridSpec
import functions_ts_inversion as ts
from scipy.interpolate import CubicSpline, interp1d,PchipInterpolator
from scipy.optimize import minimize

labelsize = 16
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

def plot_model(inp):
    plt.rcParams['font.size'] = 20
    hmin = np.min(inp)
    hmax = -hmin
    # hmax = np.max(inp)
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', cmap='seismic')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig1, format='%1.1f',label='m/s')
    # fig.tight_layout()
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    return inp, fig

def convert_slowness_to_vel(inp):
    inp = inp.reshape(nz*nx)
    inp_corr_amp = [0]*(nz*nx) 
    for i,x in enumerate(inp):
        inp_corr_amp[i] = 1/np.sqrt(inp[i])
    inp_corr_amp = np.reshape(inp_corr_amp,(nz,nx))
    return inp_corr_amp


def resize_model(new_nz,new_nx,model):
    '''Modifies the model to the desired dimensions'''
    images = Image.fromarray(model)
    resized_images = images.resize((new_nx, new_nz), Image.LANCZOS)
    resized_array = np.array(resized_images)
    print(resized_array.shape)
    return resized_array


def plot_test_model(inp,hmin=-9999,hmax=9999):
    if hmax==9999:
        hmin = np.min(inp)
        hmax = np.max(inp)
    fig  = plt.figure(figsize=(14, 7), facecolor="white")
    av   = plt.subplot(1, 1, 1)
    hfig = av.imshow(inp, 
                     vmin=hmin, vmax=hmax, aspect='auto',
                      extent=[ax[0], ax[-1], az[-1], az[0]],
                     cmap='viridis')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig, format='%1.2f',label='m/s')
    fig.tight_layout()
    return fig

# def model_window(win1=0,win2=200,mode='horizontal'):
#     if mode=='horizontal':
#         window = np.zeros_like(org)
#         win_size = win2-win1
#         win_tuk = signal.windows.tukey(win_size,alpha=0.6)
#         print(win_tuk.shape)
#         for i in range(nz):  
#             window[i,win1:win2] = win_tuk
#         print(window[i,win1:win2].shape)
#     elif mode == 'vertical':
#         window = np.zeros_like(org)
#         win_size = win2-win1
#         win_tuk = signal.windows.tukey(win_size,alpha=0.6)
#         for i in range(nx):  
#             window[win1:win2,i] = win_tuk
#     return window

   
def crop_fill(inp1,inp2,z_limit):
    inp_overthrust_rs = resize_model(z_limit, 600, inp1[:135])
    # inp_overthrust_rs = resize_model(z_limit, 600, inp1)
    # plot_test_model(inp_overthrust_rs)
    inp_kim2d_crop = np.copy(inp2)
    inp_kim2d_crop[:z_limit] = 0
    inp_mix = np.copy(inp_kim2d_crop)
    inp_mix[:z_limit] = inp_overthrust_rs
    return inp_mix

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier





def ts_inversion(param_init,at_cut_final,at,inv_type=0,method = 'Nelder-Mead'):
  
    
    ''' Define constraint bounds'''
        
    if inv_type==0:
        a_lower, a_upper = -4, 4  # range for a(t)
    elif inv_type == 1  or inv_type == 2:
        a_lower, a_upper = -1, 1  # range for a(t)
     
    tau_lower, tau_upper = 0, 0.15 # range for tau(t)
    
    
    if inv_type==3:
        bounds = [(tau_lower, tau_upper)] * len(at_cut_final)
    else:
        bounds = [(a_lower, a_upper)] * len(at_cut_final) + \
            [(tau_lower, tau_upper)] * len(at_cut_final)

    '''Run minimization'''
    # method = 'Nelder-Mead'
    # method = 'L-BFGS-B'
    
    result = minimize(objective, param_init,
                      bounds=bounds, method=method)
    
    # result = cmaes(objective,bounds, x0=param_init, maxiter= 500,popsize=len(param_init))
    # result = pso(objective,bounds, maxiter= 500,popsize=len(param_init),seed=len(param_init)*2)
    
    # result = cpso(objective,bounds, maxiter= 2000,popsize=len(param_init))
    
    print(result)
    
    
    a_opt = result.x[:len(at_cut_final)]
    tau_opt = result.x[len(at_cut_final):]
    
    if inv_type == 3:
        tau_opt = result.x
        a_opt = 0
    return a_opt,tau_opt


def find_first_index_greater_than(lst, target):
    for index, value in enumerate(lst):
        if value > target:
            return index
    return -1


def node_points(first_idx1,regular = True, nb_points = 16,nb_points1= 5, nb_points2=11):
    '''
    Node points can be regular or irregular
    In case of regular= True only one nb_points needs to be given
    In case of regular= False 
    
    '''
    if regular == True:
      
        int_idx = len(at)//nb_points +1
        at_cut_final = at[::int_idx]      
     
    else:     
        int_idx1 = len(at[:first_idx1])//nb_points1 +1
        at_cut1 = at[:first_idx1:int_idx1]
        
        int_idx2 = len(at[first_idx1:])//nb_points2 +1
        at_cut2 = at[first_idx1::int_idx2]
        
        at_cut_final = np.append(at_cut1,at_cut2)
    return at_cut_final

def initial_values(at_cut_final, inv_type = 0):
    ''' Create initial profiles'''
    if inv_type==0:
        a_init = np.zeros_like(at_cut_final) + 1
    elif inv_type == 1 or inv_type == 2:
        a_init = np.zeros_like(at_cut_final)  
    else:
        a_init = 0
    
    tau_init = np.zeros_like(at_cut_final)
    
    '''Merge initial profiles into a 1D vector param_init'''
    
    if inv_type ==3:
        param_init = tau_init
    else:
        param_init = np.concatenate([a_init, tau_init])
       
    return param_init, a_init, tau_init
    
#%%

path = 'c:/users/victorcabiativapico/SpotLight/SpotLighters - SpotLight/R&D/DOSSIER_PERSO_SpotLighters_RD/SpotVictor/Data_synthetics'


name = 'p2_v1'

# for i in range(30,5,-5):
for i in range(20,5,-5):
    test = gt.readbin(path+ '/94_kimberlina_v4/full_sum/medium/sum_kim_model_y0.dat', 151, 601)
    test10 = gt.readbin(path+ '/94_kimberlina_v4/full_sum/medium/sum_kim_model_y'+str(i)+'_'+name+'.dat', 151, 601)
    # test15 = gt.readbin('../input/90_kimberlina_mod_v3_high/full_sum/sum_kim_model_y15_p2_v1.dat', 151, 601)
    # test20 = gt.readbin('../input/90_kimberlina_mod_v3_high/full_sum/sum_kim_model_y20_p2_v1.dat', 151, 601)
  
    
    diff_test10 = test - test10
    
    # diff_test20 = test - test20
    if i == 20:
        hmin = np.min(diff_test10)
        hmax = np.max(diff_test10)
    
    fig  = plt.figure(figsize=(14, 7), facecolor="white")
    av   = plt.subplot(1, 1, 1)
    hfig = av.imshow(diff_test10, 
                      vmin=hmin, vmax=hmax, aspect='auto',
                       extent=[ax[0], ax[-1], az[-1], az[0]],
                      cmap='viridis')
    plt.title(str(i))   
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig, format='%1.2f',label='m/s')
    fig.tight_layout()
    # flout = '../png/90_kimberlina_mod_v3_high/y'+str(i)+'_'+name+'_diff.png' 
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    
    # print(np.max(diff_test10))
    
    idx_diff = np.where(diff_test10 > 0.05)
    
    sum_ts = []
    for i in range(np.min(idx_diff[1]),np.max(idx_diff[1]),4):
        idx_151 = np.where(idx_diff[1] == i)
        
        velocity_org = test[idx_diff[0][idx_151],idx_diff[1][idx_151]]
        velocity_ano = test10[idx_diff[0][idx_151],idx_diff[1][idx_151]]           
         
        thickness = dz
        
        
        ts1 = 2 * thickness / velocity_org - 2 * thickness / velocity_ano
        
        
        sum_ts.append(np.sum(ts1))
        
    print(np.min(idx_diff[1]),np.max(idx_diff[1]))
    
    

#%%

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

year = 20
part = '_p2_v1'
name = str(year)+part


a_opt = []
tau_opt = []

shot = np.arange(np.min(idx_diff[1]),np.max(idx_diff[1]),4)

# shot= [166]
# title = 190

picked_ts = []
cc_ts = []
sld_ts = []

for title in shot: 
    
    
    tr1 = path+'/94_kimberlina_v4/full_sum/medium/f_0/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = path+'/94_kimberlina_v4/full_sum/medium/f_'+name+'/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    off = 0
    idx_off =  int(off * 1000 // 12 + 125)
    
    inp1 = gt.readbin(tr1, no, nt).transpose()
    inp2 = gt.readbin(tr2, no, nt).transpose()
    
    diff = inp1-inp2
    
    idx_fb = np.argmax(inp1[1100:,idx_off])+1100 
    fb_t = idx_fb *dt+ft
    
    win1_add = -0.03
    win2_add = win1_add+0.2
    win1 = (fb_t + win1_add) * 1000
    win2 = (fb_t + win2_add) * 1000
    
    current_sld_ts = procs.sliding_TS(inp1[:,idx_off],inp2[:,idx_off],oplen=200,taper=25)
    sld_ts.append(current_sld_ts)
    
    # current_cc_TS = procs.max_cross_corr(inp1[:,idx_off],inp2[:,idx_off],win1=win1,win2=win2,thresh=None,si=dt,taper=25)
    # cc_ts.append(current_cc_TS)
    
    # pk_base = procs.extremum_func(inp1[:,idx_off],win1=win1,win2=win2,maxi=True,si=dt)
    # pk_monitor = procs.extremum_func(inp2[:,idx_off],win1=win1,win2=win2,maxi=True,si=dt)

    # picked_ts.append( pk_base[0] - pk_monitor[0])
    
        
    
    # aj = []
    
    # nb_points1 = 5
    # nb_points2 = 11
    
    # perc = 0.01
    
    # org = inp1
    # dm_samp = inp2
    # inv_type=0
    # first_idx1 = find_first_index_greater_than(diff[:,idx_off], np.max(diff[:,idx_off])*perc)-150
    
    # at_cut_final = node_points(first_idx1,regular = False,nb_points1= nb_points1, nb_points2=nb_points2)
    
    
    # param_init, a_init, tau_init = initial_values(at_cut_final, inv_type)
    
        
    # def forward_sample(param):
    #     "Forward modelling"
    #     dm = []
        
    #     if inv_type==3:
    #         tau_vals = np.copy(param)
    #         f_tau = PchipInterpolator(at_cut_final, tau_vals)
    #         f_new = CubicSpline(at, org, extrapolate=True)
    #         f_a = 0
    #     else:   
        
    #         a_vals = param[:len(at_cut_final)]
    #         tau_vals = param[len(at_cut_final):]
            
    #         f_a = PchipInterpolator(at_cut_final, a_vals)
    #         f_tau = PchipInterpolator(at_cut_final, tau_vals)
    #         f_new = CubicSpline(at, org, extrapolate=True)
        
    #     for i, t in enumerate(at):
    #         t2 = t - f_tau(t)
    #         if t2 < t:
    #             if inv_type==0:
    #                 dmod = f_new(t2) * f_a(t)
    #             elif inv_type == 1 :
    #                 dmod = f_new(t2) + f_a(t)
    #             elif inv_type == 2 :
    #                 t_mov = f_new(t2)
    #                 dmod = t_mov + f_a(t_mov)
    #             elif inv_type ==3:
    #                 dmod = f_new(t2)
    #         else:
    #             dmod = org[i]
    #         dm.append(dmod)
    #     dm = np.array(dm)
    #     return dm, f_a, f_tau
    
    # def objective(param_pred):
    #     '''Objective function'''
    #     # print('value =', param_pred)
    #     global aj
    #     d_pred = forward_sample(param_pred)[0]
    #     error = (d_pred  - dm_samp)**2
    #     error = np.sum(error)
    #     aj.append(error)
    #     return error

            
    # a_tau  = ts_inversion(param_init,at_cut_final,at,inv_type=0,method = 'Nelder-Mead')
    # a_opt.append(a_tau[0])
    # tau_opt.append(a_tau[1])
    

    # plt.rcParams['font.size'] = 22
    # fig = plt.figure(figsize=(5, 10))
    # gs = GridSpec(1, 1, figure=fig)
    # ax1 = fig.add_subplot(gs[:, 0])
    # ax1.set_title('Trace \n src = '+str(title*12) + ' m \noff = '+str(int(off*1000))+' m\nTS = '+str(truncate_float(current_cc_TS,2))+' ms')
    # ax1.axhline(win1/1000)
    # ax1.axhline(win2/1000)
    # ax1.plot(inp1[:,idx_off], at[:], label='org',linewidth=2)
    # ax1.plot(inp2[:,idx_off], at[:], label='ano',linewidth=2)
    # ax1.legend()
    # ax1.set_xlim(-0.03, 0.03)
    # ax1.set_ylim(1,2.5)
    # ax1.set_ylabel('Time (s)')
    # ax1.set_xlabel('Amplitude')
    # ax1.grid()
    # plt.gca().invert_yaxis()
    # fig.tight_layout()

#%%
max_ts_inverted = []
for i in tau_opt: 
    max_ts_inverted.append(np.max(i))
max_ts_inverted = np.array(max_ts_inverted)

plt.figure(figsize=(8,6))
plt.plot(tau_opt[0],at_cut_final,'.-')
plt.gca().invert_yaxis()

    
plt.figure(figsize=(8,6))
plt.plot(np.array(shot)*dx,picked_ts,'.-',label='picked ts')
plt.plot(np.array(shot)*dx,cc_ts,'.-',label='cross-corr ts')
plt.plot(np.array(shot)*dx,np.array(sum_ts)*1000,'.-',label='theo ts')
plt.plot(np.array(shot)*dx,np.array(-max_ts_inverted)*1000,'.-',label='max_inv ts')

plt.legend()
plt.xlabel('Position (x)')
plt.ylabel('time-shift')
