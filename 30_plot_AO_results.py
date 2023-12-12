#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 11:43:15 2023

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
    
def read_pick(path,srow):
    rec_x = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            rec_x.append(float(row[srow]))
    return rec_x

## Read the results from demigration
def read_results(path,srow):
    rec_x = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            rec_x.append(float(row[srow]))
        rec_x = [x for x in rec_x if str(x) != 'nan']
    return rec_x


## Read the results from demigration
# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/014_flat_marm_sm_demig_tol6.csv'

# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/011_marm_sm_binv_PP21_P021_hz02.csv'
# path2 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/011_marm_sm_badj_PP21_P021_hz02.csv'

path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/Demigration_Victor/015_marm_flat_v2.csv'


src_x = np.array(read_results(path1,1))
rec_x = np.array(read_results(path1,4))
off_x = np.array(read_results(path1,16))

tt_inv = np.array(read_results(path1,17))
# tt_adj = np.array(read_results(path2,17))



shot = np.round(src_x/12)
# shot = np.round(shot/4)*4
# shot = np.array(shot)+1

shot = np.array((np.rint(shot)).astype(int))


idx_off = np.round(off_x/12)
tr = idx_off[:]+125
tr = np.array((np.rint(tr)).astype(int))


idx_off_int = np.round(off_x/6)
tr_int = idx_off[:]+125*2
tr_int = np.array((np.rint(tr)).astype(int))


# for i in range(tr.size):
#     if tr[i] == -9223372036854775808:
#         tr[i] = 0
#         shot[i] = 0

# indx = np.where(shot==0)


tt_inv = np.array(tt_inv[:])-32
# tt_adj = np.array(tt_adj[:])-32

# idx = 0
# shot_idx = shot[idx]
# rec_x = (shot*dx+ao[tr])*1000
# flout  = './png/22_extend_anomaly/born_trace_'+str(title)+'.png'
# plot_trace(xmax_tr,inp2,inp2,flout,tr)

# tr_adj = '../output/26_mig_4_interfaces/badj_rc_norm/t1_obs_000'+str(shot)+'.dat'
# tr_inv = '../output/26_mig_4_interfaces/binv_rc_norm/t1_obs_000'+str(shot)+'.dat'

tr_adj = np.zeros_like(shot)
tr_inv = np.zeros_like(shot)
tr_mod = np.zeros_like(shot)
csg_tr = np.zeros((nt,shot.size))
csg_tr_int = np.zeros((nt,shot.size))
time_csg = np.zeros(shot.size)

for i in range(shot.size):
    if shot[i]>0:
        # tr_adj = '../output/27_marm/badj/t1_obs_000'+str(shot[i])+'.dat'
        tr_inv = '../output/27_marm/binv/t1_obs_000'+str(shot[i])+'.dat'
        tr_mod = '../output/27_marm/mod_marm_inv/t1_obs_000'+str(shot[i])+'.dat'
        # tr1 = '../output/27_marm/diff_marm_corr/t1_obs_000'+str(shot[i])+'.dat'
        tr1 = '../output/27_marm/flat_marm2/t1_obs_000'+str(shot[i])+'.dat'
        # inp_adj = -gt.readbin(tr_adj, no, nt).transpose()
        inp_inv = -gt.readbin(tr_inv, no, nt).transpose()
        inp_mod = -gt.readbin(tr_mod, no, nt).transpose()
        
        
        inp_inv = -gt.readbin(tr1, no, nt).transpose()
        
        
        # inp_hilb_adj = np.zeros_like(inp_adj,dtype = 'complex_')   
        inp_hilb_inv = np.zeros_like(inp_inv,dtype = 'complex_')  
        inp_hilb_mod = np.zeros_like(inp_mod,dtype = 'complex_')  
    
        for j in range(no):
            # inp_hilb_adj[:,j] = hilbert(inp_adj[:,j])    
            inp_hilb_inv[:,j] = hilbert(inp_inv[:,j])    
            inp_hilb_mod[:,j] = hilbert(inp_mod[:,j])  
            
        # imag_hilb_adj = inp_hilb_adj.imag
        imag_hilb_inv = inp_hilb_inv.imag
        imag_hilb_mod = inp_hilb_mod.imag    
        ### Difference 
        imag_hilb_diff = imag_hilb_mod - imag_hilb_inv
        imag_hilb_diff = imag_hilb_inv
        csg_tr[:,i] = imag_hilb_diff[:, tr[i]] 
        ## Interpolation
        f = interpolate.RegularGridInterpolator((at,ao), imag_hilb_diff,method='linear',bounds_error=False, fill_value=None) 
        at_new = np.linspace(at[0], at[-1], 1501)
        ao_new = np.linspace(ao[0], ao[-1], 502)
        AT, AO = np.meshgrid(at_new, ao_new, indexing='ij')
        INT_imag_hilb_diff = f((AT,AO))
        ## Calling traces according to demigration
        csg_tr_int[:,i] = INT_imag_hilb_diff[:, tr[i]]  
        
        time_csg[i] =  tt_inv[i]
        

### Common spot gather

hmin= np.min(csg_tr_int)*100
hmax = -hmin
ax = np.arange(time_csg.size)*10
# tt_inv[0] = 1000
fig = plt.figure(figsize=(10, 8), facecolor = "white")
plt.suptitle(f"Offset", fontsize=16)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()  
hmin, hmax = -0.05,0.05
hfig = ax1.imshow(csg_tr, vmin=hmin, vmax=hmax, 
                 extent=[ax[0], ax[-1], at[-1], at[0]],
                 aspect='auto',cmap='seismic')
# ax1.colorbar(hfig)
ax1.scatter(ax,time_csg/1000,marker='o')
ax1.plot(ax,shot/1000*10-3.5)
ax1.plot(ax,rec_x/1000-3.95)



### Interpolation

# x, y = np.array([-2, 0, 4]), np.array([-2, 0, 2, 5])
# x = np.linspace(-2, 5, 21)
# y = np.linspace(-2, 5, 11)
# def ff(x, y):
#     return x**2 + y**2


# xg, yg = np.meshgrid(x, y, indexing='ij')
# data = ff(xg, yg)
# interp = RegularGridInterpolator((x, y), data,
#                                  bounds_error=False, fill_value=None)

# fig1 = plt.figure(figsize=(10,10))
# axi1 = fig1.add_subplot()
# axi1.imshow(data,extent=[x[0],x[-1],y[0],y[-1]])
# fig = plt.figure()
# axi = fig.add_subplot(projection='3d')
# axi1.scatter(xg.ravel(), yg.ravel(), data.ravel(),
#            s=60, c='k', label='data')

# xx = np.linspace(-2, 5, 32)
# yy = np.linspace(-2, 5, 32)
# X, Y = np.meshgrid(xx, yy, indexing='ij')

# fig2 = plt.figure(figsize=(10,10))
# axi2 = fig2.add_subplot()
# axi2.imshow(interp((X, Y)),extent=[x[0],x[-1],y[0],y[-1]])
# axi.plot_wireframe(X, Y, interp((X, Y)), rstride=3, cstride=3,
#                   alpha=0.4, color='m', label='linear interp')
# axi.plot_wireframe(X, Y, ff(X, Y), rstride=3, cstride=3,
#                   alpha=0.4, label='ground truth')
# # plt.legend()
# plt.show()

## Application of real interpolation
# f = interpolate.RegularGridInterpolator((at,ao), imag_hilb_diff,method='linear',bounds_error=False, fill_value=None) 
# # f = interpn((at,ao), imag_hilb_diff,([0,0]),method="linear",bounds_error=False, fill_value=None) 



# at_new = np.linspace(at[0], at[-1], 3002)
# ao_new = np.linspace(ao[0], ao[-1], 502)
# AT, AO = np.meshgrid(at_new, ao_new, indexing='ij')
# INT_imag_hilb_diff = f((AT,AO))


# fig = plt.figure(figsize=(10, 8), facecolor="white")
# av = plt.subplot(1, 1, 1)
# hmin, hmax = -0.01,0.01
# hfig = av.imshow(imag_hilb_diff, vmin=hmin, vmax=hmax, 
#                  extent=[ao[0], ao[-1], at[-1], at[0]],
#                  aspect='auto',cmap='seismic')

# fig = plt.figure(figsize=(10, 8), facecolor="white")
# av = plt.subplot(1, 1, 1)
# hmin, hmax = -0.01,0.01
# hfig = av.imshow(INT_imag_hilb_diff, vmin=hmin, vmax=hmax, 
#                   extent=[ao[0], ao[-1], at[-1], at[0]],
#                   aspect='auto',cmap='seismic')


#%% PLOTS

def plot_gather_scatter(inp,t_time,lgd):
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    av = plt.subplot(1, 1, 1)
    # print(np.max(inp))
    c_point = ['yellow','b','r','k','greenyellow']
    if lgd == 'badj' :
        hmin, hmax = -0.5,0.5
        hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                         vmin=hmin, vmax=hmax, aspect='auto',
                         cmap='seismic')
        flout = '../png/27_marm/badj/100_compare_RT_OBS_'+str(shot_idx)+'.png'
        plt.title('Observed shot with the ADJOINT demigrated tt data \n')
        plt.colorbar(hfig,format='%1.e')
    else:
        hmin, hmax = -0.5,0.5
        hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                         vmin=hmin, vmax=hmax, aspect='auto',
                         cmap='seismic')
        flout = '../png/27_marm/binv/100_compare_RT_OBS_'+str(shot_idx)+'.png'
        plt.title('Observed shot with the INVERSE demigrated tt data \n')
        plt.colorbar(hfig,format='%1.e')
        av.scatter(off_x[idx]/1000,tt_inv[idx]/1000,marker='o')
    # for i in range(tr.size):
    #     plt.axvline(x=ao[tr[i]], color='k', ls='--',alpha=0.8)
    #     plt.scatter(ao[tr[i]],t_time[i]/1000, color=c_point[3], ls='--')
        
   
    fig.tight_layout()
    
    plt.rcParams['font.size'] = 16
    plt.xlabel('Offset (km)')
    plt.ylabel('Time (s)')
    fig.tight_layout()
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')
    

def plot_trace_scatter(inp,t_time,lgd,flout):
    axi = np.zeros(np.size(tr))
    fig, (axi) = plt.subplots(nrows=1, ncols=np.size(tr),
                              sharey=True,
                              figsize=(14, 8),
                              facecolor="white")

    c_point = ['yellow','b','r','k','greenyellow']
    # ratio = np.asarray(tr, dtype='f')
    for i in range(np.size(tr)):
        # xmin = np.min(inp1[:,tr[i]]) + np.min(inp1[:,tr[i]])/1.5
        # xmin = 1.0
        xmax = np.max(inp[:, tr[i]])
        xmin = -xmax
        
        # inp[:, tr[i]] = (inp[:, tr[i]]/np.max(inp[:, tr[i]]))
        axi[i].plot(inp[:, tr[i]], at, 'b')
    

        axi[i].set_xlim(xmin, xmax)
        axi[i].set_ylim(2, ft)
        axi[i].xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
        axi[i].set_xlabel('Offset: '+str(f'{ao[tr[i]]:.2f}'))
        # print('iter: '+str(i),'time: '+str(t_time[i]))
        axi[i].axhline(t_time[i]/1000, color=c_point[3], ls='--')
        fig.tight_layout()
        axi[i].grid()
    axi[0].set_ylabel('Time (s)')
    axi[0].legend([lgd,'RayTT'], loc='upper left', shadow=True)

    # axi[0].legend(['Baseline','Monitor'],loc='upper left',shadow=True)
    fig.text(0.48, -0.01, "Amplitude")
    fig.text(0.48, 1, 'Traces from observed data vs tt '+str(lgd))
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')




### MINIMUM PHASE ORIGINAL
lgd_inv = 'binv'    
# lgd_adj = 'badj'


shot_idx = 360
idx = 0
plot_gather_scatter(inp_inv,tt_inv,lgd_inv)
# plot_gather_scatter(inp_adj,tt_adj,lgd_adj)

   

flout1 = '../png/26_mig_4_interfaces/traces_inv'
plot_trace_scatter(inp_inv,tt_inv,lgd_inv,flout1) 
# flout2 = '../png/26_mig_4_interfaces/traces_adj'
# plot_trace_scatter(inp_adj,tt_adj,lgd_adj,flout2)

### ZERO PHASE CORRECTION

plot_gather_scatter(-imag_hilb_inv,tt_inv,lgd_inv)
# plot_gather_scatter(imag_hilb_adj,tt_adj,lgd_adj)

# plot_gather_scatter(imag_hilb_diff,tt_inv,lgd_inv)
# flout1 = '../png/26_mig_4_interfaces/traces_inv_hilb'
# plot_trace_scatter(imag_hilb_inv,tt_inv,lgd_inv,flout1) 
# flout2 = '../png/26_mig_4_interfaces/traces_adj_hilb'
# plot_trace_scatter(imag_hilb_adj,tt_adj,lgd_adj,flout2)
   

# flout3 = '../png/27_marm/diff_marm_inv/traces_phase_inv'
# plot_trace_scatter(imag_hilb_inv,tt_inv,lgd_inv,flout3) 
# flout4 = '../png/27_marm/diff_marm_inv/traces_phase_adj'
# plot_trace_scatter(imag_hilb_adj,tt_adj,lgd_adj,flout4)

    
# Calculate the difference
   
# tt_delta = np.array(tt_inv)-np.array(tt_adj)
tt_delta= np.array(tt_inv)


print('mean difference :',np.mean(tt_delta))
plt.figure(figsize=(16,8))
c_point = ['yellow','b','r','k','greenyellow']
for i in range(tt_delta.size):
    plt.plot(ao[tr[i]],-tt_delta[i],'-o',color=c_point[3])
plt.grid()
plt.title('Difference between raytracing with hz in the badj or binv image')
plt.xlabel('Offset (km)')
plt.ylabel('Time difference (ms)')

