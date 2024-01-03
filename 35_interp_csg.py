#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:31:13 2023

@author: vcabiativapico
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter
from scipy import interpolate
# from scipy.interpolate import splrep, BSpline, interpn, RegularGridInterpolator
from scipy.signal import hilbert
import csv
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from wiggle.wiggle import wiggle
import tqdm
if __name__ == "__main__":
  
  
## Building simple vel and rho models to test modeling
    labelsize = 16
    nt = 1501
    dt = 1.41e-3
    ft = -100.11e-3
    nz = 151
    fz = 0.0
    dz = 12.0/1000.
    nx = 601
    fx = 0.0
    dx = 12.0/1000.
    no =251
   # no        = 2002
    do = dx
    fo = -(no-1)/2*do
    ao = fo + np.arange(no)*do
    at = ft + np.arange(nt)*dt
    az = fz + np.arange(nz)*dz
    ax = fx + np.arange(nx)*dx
## Add y dimension    
    fy = -500 
    ny = 21
    dy = 50
    ay = fy + np.arange(ny)*dy
    
    
def plot_shot_gathers(hmin, hmax, inp, flout):
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                     vmin=hmin, vmax=hmax, aspect='auto',
                     cmap='seismic')
    plt.colorbar(hfig, format='%2.2f')
    plt.rcParams['font.size'] = 16
    plt.xlabel('Offset (km)')
    plt.ylabel('Time (s)')
    fig.tight_layout()
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')
    return fig
    
    
def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr

## Read the results from demigration
def read_results(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
        # attr = [x for x in attr if str(x) != 'nan']
    return attr
 
#%%   
""" READ AND PLOT SOURCE RECEIVER AND TRAVELTIME """

# path1  = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/Demigration_Victor/015_marm_flat_v2_p100_zero.csv'
# path1 ='/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/Demigration_Victor/015_marm_flat_v2_test_1e_4.csv'
path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/baptiste_corr_v181223_y0.01/015_marm_flat_v2_test_p0001_v2_test_bp_az_0_4992.csv'

src_x = np.array(read_results(path1,1))
src_y = np.array(read_results(path1,2))
src_z = np.array(read_results(path1,3))    
rec_x = np.array(read_results(path1,4))  
rec_y = np.array(read_results(path1,5))    
rec_z = np.array(read_results(path1,6))
spot_x = np.array(read_results(path1,7)) 
spot_y = np.array(read_results(path1,8))
spot_z= np.array(read_results(path1,9))
off_x  = np.array(read_results(path1,16))
tt_inv = np.array(read_results(path1,17))

indices = []
for i in range(0,101):
    if str(src_x[i]) != 'nan':
        indices.append(i)

""" PLOT TRAVELTIMES OVER THE SHOTS """

# ## Calculate the index of the shot
shot = np.round(src_x/12)
# shot = np.round(src_x/6)
shot = np.array((np.rint(shot)).astype(int))


csg_trace= np.zeros((nt,len(indices)))
csg_trace_INT = np.zeros((nt,len(indices)))
shot_int = np.zeros((nt,476-126))

dec_off_x = -off_x/1000
no = 251
dec_off_x[0]


shot_idx = np.arange(126,476+1)
dec_title = shot_idx*12

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


sd_val = []
sd_idx = []
for k in range(len(src_x)):
    sd_val.append(find_nearest(dec_title,src_x[k])[0])
    sd_idx.append(find_nearest(dec_title,src_x[k])[1])


fd_val = []
fd_idx = []
for k in range(len(dec_off_x)):
    fd_val.append(find_nearest(ao, dec_off_x[k])[0])
    fd_idx.append(find_nearest(ao, dec_off_x[k])[1])
    
    



## Pour les traces interpolées
# idx_off = np.round(off_x/6)
# tr = idx_off[:]+251
# tr = np.array((np.rint(tr)).astype(int))
d_dox = dec_off_x[0] - dec_off_x[1]
sd_dox = src_x[0] - src_x[1]


inp_hilb3 = np.zeros((351,1501, 251),dtype = 'complex_')

# for i in range(351):
for i in range(351):
    txt = str(i+126)
    title = txt.zfill(3)
    # title = shot[indices[i]]  
## Read the shots that converged
    tr3 = '../output/30_marm_flat/t1_obs_000'+str(title)+'.dat'
    inp3 = -gt.readbin(tr3, no, nt).transpose()
    # inp_hilb3 = np.zeros_like(inp3,dtype = 'complex_')  
    for j in range(no):
        inp_hilb3[i][:,j] = hilbert(inp3[:,j]) 
inp_hilb3 = inp_hilb3.imag

# for i in range(0,351,50):
#     hmin = -0.15
#     hmax = 0.15
#     fig = plt.figure(figsize=(10, 8), facecolor="white")
#     av = plt.subplot(1, 1, 1)
#     hfig = av.imshow(inp_hilb3[i], extent=[0, len(indices), at[-1], at[0]],
#                       vmin=hmin, vmax=hmax, aspect='auto',
#                       cmap='seismic')     
#     # wiggle(inp_hilb3[i],tt=at,xx=ao   )

    

## INTERPOLATION OF THE RECEIVERS
rec_to_int = np.zeros((4,1501,4))
tr_INT     = np.zeros((4,1501,5)) 
for i in range(len(indices)):    
    # Index creation, we take extract four values around the desired offset to interpolate
    ind_tr_int = np.arange(fd_idx[i]-2,fd_idx[i]+2)
    
    ind_shot_int = np.arange(sd_idx[i]-2,sd_idx[i]+2)
    shot_to_int = inp_hilb3[ind_shot_int][:,ind_tr_int]
    # Call the traces wit index
    for j in range(4):
        rec_to_int[j][:,:] = inp_hilb3[ind_shot_int[j]][:,ind_tr_int]
        f = interpolate.RegularGridInterpolator((at,ao[ind_tr_int]), rec_to_int[j],method='linear',bounds_error=False, fill_value=None) 
        at_new = np.linspace(at[0], at[-1], 1501)
        ao_new = np.linspace(dec_off_x[i]-d_dox*2,dec_off_x[i]+d_dox*2, 5)
        AT, AO = np.meshgrid(at_new, ao_new, indexing='ij')
        tr_INT[j][:,:] = f((AT,AO))
        rec_int = tr_INT[:,:,2]
            # shot_int[:,i] = tr_INT[:,2]
        # plot_rec_int = np.asarray(csg_trace)
        # plot_rec_int[:,i] = rec_int[3]
        
    f = interpolate.RegularGridInterpolator((at,(ind_shot_int+126)*12), rec_int.T,method='linear',bounds_error=False, fill_value=None) 
    at_new = np.linspace(at[0], at[-1], 1501)
    src_new = np.linspace(src_x[i]-sd_dox*2, src_x[i]+sd_dox*2, 5)
    AT, SRC = np.meshgrid(at_new, src_new, indexing='ij')
    src_INT = f((AT,SRC))
    csg_trace[:,i] = src_INT[:,2] 

    
    
# ## INTERPOLATION OF THE SHOTS
# for i in range(len(indices)):   
#     ind_shot_int = np.arange(sd_idx[i]-2,sd_idx[i]+2)
#     shot_to_int = inp_hilb3[ind_shot_int][:,ind_tr_int]
    
#     f = interpolate.RegularGridInterpolator((at,ind_shot_int*12), tr_to_int,method='linear',bounds_error=False, fill_value=None) 
#     at_new = np.linspace(at[0], at[-1], 1501)
#     src_new = np.linspace(src_x[i]-sd_dox*2, src_x[i]+sd_dox*2, 5)
#     AT, SRC = np.meshgrid(at_new, src_new, indexing='ij')
#     src_INT = f((AT,SRC))
#     csg_trace[:,i] = src_INT[:,2] 
   


## PLOT THE RAYTRACING TRAVELTIMES OVERLAYING COMMON SPOT GATHER TRACES
hmin = -0.15
hmax = 0.15
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
# hfig = av.imshow(plot_rec_int, extent=[0, len(indices), at[-1], at[0]],
#                  vmin=hmin, vmax=hmax, aspect='auto',
#                  cmap='seismic')
hfig = av.imshow(csg_trace, extent=[0, len(indices), at[-1], at[0]],
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='seismic')    



## PLOT A WIGGLE OVERLAY
fig = plt.figure(figsize=(12, 10), facecolor="white")
## First and main plot 
av = plt.subplot2grid((5, 1), (0, 0),rowspan=4)

plt.plot(np.arange(len(indices)),tt_inv[indices]/1000,'-or',markersize=3)
wiggle(csg_trace,tt=at,xx=np.arange(len(indices)))
## Define the tick axis
av.xaxis.set_ticks(np.arange(len(indices))) 
av.xaxis.set_ticklabels(np.rint(off_x[indices]).astype(int))
xticks = plt.gca().xaxis.get_major_ticks()
for i in range(len(xticks)):
    if i % 10 != 0:
        xticks[i].set_visible(False)
## Labels
plt.legend(['raytracing','modeling'])
plt.rcParams['font.size'] = 16
plt.title('Common spot gather Raytracing and Modelling')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')       
    # if i == 0:
    #     ind_tr_int = tr1[np.arange(indices[i],indices[i+4])]
        
    # elif i == 1:
    #     ind_tr_int = tr1[np.arange(indices[i-1],indices[i+3])]
        
    # elif i == len(indices)-1:
    #     ind_tr_int = tr1[np.arange(indices[i-3],indices[i]+1)]
        
    # elif i == len(indices)-2:
    #     ind_tr_int = tr1[np.arange(indices[i-2],indices[i]+2)]
    
    # else:
    #     ind_tr_int = tr1[np.arange(indices[i-2],indices[i+2])]  
    # print(ind_tr_int,tr1[i])
    
    
    
#     ## Interpolation
#     f = interpolate.RegularGridInterpolator((at,ao[ind_tr_int]), tr_for_int,method='linear',bounds_error=False, fill_value=None) 
#     at_new = np.linspace(at[0], at[-1], 1501)
#     ao_new = np.linspace(ao[ind_tr_int[-1]],[ind_tr_int[0]], 27)
#     AT, AO = np.meshgrid(at_new, ao_new, indexing='ij')
#     tr_INT = f((AT,AO))
#     print(tr_INT,tr1[i])
# ## Calculate the index of the offset in the shot according to the off_x
#     k =   
#     csg_trace[:,i] = tr_INT[:,k]    
#     csg_trace[:,i] = tr_for_int[:,tr1[indices[i]]] 
    # csg_trace_INT[:,i] = INT_inp_hilb3[:,tr[indices[i]]] 
    