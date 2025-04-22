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
no =251
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

#%%



name = 'p2_v1'

# for i in range(30,5,-5):
for i in range(30,5,-25):
    test = gt.readbin('../input/94_kimberlina_v4/full_sum/medium/sum_kim_model_y0.dat', 151, 601)
    test10 = gt.readbin('../input/94_kimberlina_v4/full_sum/medium/sum_kim_model_y'+str(i)+'_'+name+'.dat', 151, 601)
    # test15 = gt.readbin('../input/90_kimberlina_mod_v3_high/full_sum/sum_kim_model_y15_p2_v1.dat', 151, 601)
    # test20 = gt.readbin('../input/90_kimberlina_mod_v3_high/full_sum/sum_kim_model_y20_p2_v1.dat', 151, 601)
    
    
    diff_test10 = test - test10
    
    # diff_test20 = test - test20
    if i == 30:
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
    for i in range(np.min(idx_diff[1]),np.max(idx_diff[1])):
        idx_151 = np.where(idx_diff[1] == i)
        
        velocity_org = test[idx_diff[0][idx_151],idx_diff[1][idx_151]]
        velocity_ano = test10[idx_diff[0][idx_151],idx_diff[1][idx_151]]           
         
        thickness = dz
        
        
        ts = 2 * thickness / velocity_org - 2 * thickness / velocity_ano
        
        
        sum_ts.append(np.sum(ts))
        
    print(np.min(idx_diff[1]),np.max(idx_diff[1]))


year = 30
part = '_p2_v1'
name = str(year)+part




shot = np.arange(np.min(idx_diff[1]),np.max(idx_diff[1]))
# title = 190


cc_ts = []
for title in shot: 
    
    
    tr1 = '../output/94_kimberlina_v4/full_sum/medium/f_0/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/94_kimberlina_v4/full_sum/medium/f_'+name+'/t1_obs_000'+str(title).zfill(3)+'.dat'
     
    
    
    
    off = 0
    idx_off =  int(off * 1000 // 12 + 125)
    
    inp1 = -gt.readbin(tr1, no, nt).transpose()
    inp2 = -gt.readbin(tr2, no, nt).transpose()
    
    
    idx_fb = np.argmin(inp1[1100:,idx_off])+1100 
    fb_t = idx_fb *dt+ft
    
    win1_add = -0.03
    win2_add = win1_add+0.2
    win1 = (fb_t + win1_add) * 1000
    win2 = (fb_t + win2_add) * 1000
    
    current_cc_TS = procs.max_cross_corr(inp1[:,idx_off],inp2[:,idx_off],win1=win1,win2=win2,thresh=None,si=dt,taper=25)
    cc_ts.append(current_cc_TS)
    
    plt.rcParams['font.size'] = 22
    fig = plt.figure(figsize=(5, 10))
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.set_title('Trace \n src = '+str(title*12) + ' m \noff = '+str(int(off*1000))+' m\nTS = '+str(truncate_float(current_cc_TS,2))+' ms')
    ax1.axhline(win1/1000)
    ax1.axhline(win2/1000)
    ax1.plot(inp1[:,idx_off], at[:], label='org',linewidth=2)
    ax1.plot(inp2[:,idx_off], at[:], label='ano',linewidth=2)
    ax1.legend()
    ax1.set_xlim(-0.03, 0.03)
    ax1.set_ylim(1,2.5)
    ax1.set_ylabel('Time (s)')
    ax1.set_xlabel('Amplitude')
    ax1.grid()
    plt.gca().invert_yaxis()
    fig.tight_layout()


    
plt.figure(figsize=(8,6))
plt.plot(shot*dx,cc_ts,'.',label='cross-corr ts')
plt.plot(shot*dx,np.array(sum_ts)*1000,'.',label='theo ts')
plt.legend()
plt.xlabel('Position (x)')
plt.ylabel('time-shift')