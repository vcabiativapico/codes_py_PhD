#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 14:53:07 2023

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
path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/039_mig_marm_flat/badj/020_badj_marm_sm_flat_601.csv'

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
     


# ## Calculate the index time travel 
tt_inv = np.array(tt_inv[:])

""" READ AND PLOT SHOTS """

idx = 100

idx = indices[idx]

title = shot[idx]

no = 251
## READ SHOTS
# tr3 = '../output/27_marm/flat_marm/t1_obs_000'+str(title)+'.dat'
tr3 = '../output/30_marm_flat/t1_obs_000'+str(title)+'.dat'
inp3 = -gt.readbin(tr3, no, nt).transpose()
inp_hilb3 = np.zeros_like(inp3,dtype = 'complex_')  
for i in range(no):
    inp_hilb3[:,i] = hilbert(inp3[:,i]) 
inp_hilb3 = inp_hilb3.imag



## PLOT SHOTS   
hmin, hmax = -0.05,0.05
flout_gather = '../png/30_marm_flat/obs_'+str(title)+'.png'
fig = plot_shot_gathers(hmin, hmax, inp_hilb3, flout_gather)
fig = plt.plot(off_x[idx]/1000,tt_inv[idx]/1000,'ok')


"""PLOT COMMON SPOT GATHER (CSG)"""

csg_trace= np.zeros((nt,len(indices)))
csg_trace_INT = np.zeros((nt,len(indices)))


idx_off1 = np.round(off_x/12)
tr1 = 125-idx_off1[:]
tr1 = np.array((np.rint(tr1)).astype(int))
no = 251


## Pour les traces interpolées
idx_off = np.round(off_x/6)
tr = 251-idx_off[:]
tr = np.array((np.rint(tr)).astype(int))


for i in range(len(indices)):
    title = shot[indices[i]]
    
## Read the shots that converged
    tr3 = '../output/30_marm_flat/t1_obs_000'+str(title)+'.dat'
    inp3 = -gt.readbin(tr3, no, nt).transpose()
    inp_hilb3 = np.zeros_like(inp3,dtype = 'complex_')  
    for j in range(no):
        inp_hilb3[:,j] = hilbert(inp3[:,j]) 
    inp_hilb3 = inp_hilb3.imag
    
    ## Interpolation
    f = interpolate.RegularGridInterpolator((at,ao), inp_hilb3,method='linear',bounds_error=False, fill_value=None) 
    at_new = np.linspace(at[0], at[-1], 1501)
    ao_new = np.linspace(ao[0], ao[-1], 502)
    AT, AO = np.meshgrid(at_new, ao_new, indexing='ij')
    INT_inp_hilb3 = f((AT,AO))
    
## Calculate the index of the offset in the shot according to the off_x  
    # csg_trace[:,i] = inp_hilb3[:,tr1[indices[i]]] 
    csg_trace_INT[:,i] = INT_inp_hilb3[:,tr[indices[i]]] 
    
    
## PLOT THE RAYTRACING TRAVELTIMES OVERLAYING COMMON SPOT GATHER TRACES
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(csg_trace_INT, extent=[0, len(indices), at[-1], at[0]],
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='seismic')    
plt.plot(np.arange(len(indices)),tt_inv[indices]/1000,'k') ## RAYTRACING TRAVELTIMES
# plt.plot(off_x/1000,tt_inv/1000,'.k') ## RAYTRACING TRAVELTIMES
plt.colorbar(hfig, format='%2.2f')
plt.rcParams['font.size'] = 16
plt.xlabel('Trace nb')
plt.ylabel('Time (s)')
fig.tight_layout()
flout = '../png/27_marm/csg_'+str(title)+'_p10.png'
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')


## PLOT A WIGGLE OVERLAY
fig = plt.figure(figsize=(12, 10), facecolor="white")
## First and main plot 
av = plt.subplot2grid((5, 1), (0, 0),rowspan=4)

plt.plot(np.arange(len(indices)),tt_inv[indices]/1000,'-or',markersize=3)
wiggle(csg_trace_INT,tt=at,xx=np.arange(len(indices)))
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

## Secondary plot 
av1= plt.subplot2grid((5, 1), (4, 0),rowspan=1)
plt.plot(np.arange(len(indices)),off_x[indices],'.-')
av1.xaxis.set_ticks(np.arange(len(indices))) 
av1.xaxis.set_ticklabels(np.array(indices))
# av1.xaxis.set_ticklabels(np.rint(off_x[indices]).astype(int))
xticks = plt.gca().xaxis.get_major_ticks()
for i in range(len(xticks)):
    if i % 20 != 0:
        xticks[i].set_visible(False)
plt.ylabel('Offset (m)')
plt.xlabel('Raytrace nb')  
fig.tight_layout()
flout = '../png/27_marm/wiggle_csg_'+str(title)+'_p10.png'
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')


# flout = '../input/27_marm/csg_raytracing_modeling.dat'
# gt.writebin(csg_trace_INT,flout)
##PLOT ONLY THE TRACES WITH ITS CORRESPONDING TRAVELTIME
# for i in range(len(indices)):
#     fig = plt.figure(figsize=(5, 10), facecolor="white")
#     av = plt.subplot(1, 1, 1)
#     plt.plot(csg_trace[:,i],at)
#     plt.plot(tt_inv[indices[i]]/1000,'ok')
#     plt.title('Ray '+str(indices[i])+'\n src_x = '+str(int(src_x[indices[i]]))+' rec_x = '+str( int(rec_x[indices[i]])))
#     plt.xlim(-0.2,0.2)
#     plt.gca().invert_yaxis()

plt.figure(figsize=(12, 10), facecolor="white")
zero_ph_pick2 = np.array(read_results('../input/27_marm/29_pick_csg.csv', 0))
zero_ph_corr = zero_ph_pick2-100.11
plt.plot(off_x,zero_ph_corr)
plt.plot(off_x,tt_inv[indices],'-or',markersize=3)
plt.ylim(2000,0)
plt.xlabel('offset (m)')
plt.ylabel('time (ms)')

decalage = zero_ph_corr - tt_inv[indices]
plt.figure(figsize=(12, 10), facecolor="white")
plt.plot(off_x,decalage,'.')
plt.xlabel('offset (m)')
plt.ylabel('decalage (ms)')
#%%


""" PLOT THE POSITIONS AND RECEIVERS OBTAINED """
colors = src_x[::2]
fig = plt.figure(figsize= (10,7))
plt.scatter(src_x[::2],src_y[::2],c=colors,marker='*',cmap='jet')
plt.scatter(rec_x[::2],rec_y[::2],c=colors,marker='v',cmap='jet')
plt.xlabel('x')
plt.ylabel('y')


flout = '../png/27_marm/flat_marm/corr_az_pert/carte_p10.png'
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')


""" PLOT MIGRATED IMAGE AND RAYS """

## Read migrated image

file = '../output/27_marm/flat_marm/inv_betap_x_s.dat'
file = '../input/27_marm/marm2_sm15.dat'
# file = '../output/27_marm/flat_marm/dbetap_exact.dat'
Vit_model2 = gt.readbin(file,nz,nx).T*1000


## Read rays from raytracing
path_ray = [0]*102
plt.figure(figsize= (8,6))


indices = []
for i in range(0,101,10):
    if str(src_x[i]) != 'nan': 
        indices.append(i)
        
        path_ray[i] = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug141223_raytracing/rays/opti_"+str(i)+"_ray.csv"
        ray_x = np.array(read_results(path_ray[i], 0))
        ray_z = np.array(read_results(path_ray[i], 2))

        
        x_disc = np.arange(nx)*12.00
        z_disc = np.arange(nz)*12.00
        
        fig1 = plt.figure(figsize=(18, 8), facecolor="white")
        av1 = plt.subplot(1, 1, 1)
        hmin = np.min(Vit_model2)
        # hmax = -hmin
        hmax = np.max(Vit_model2)
        hfig = av1.imshow(Vit_model2.T[:,270:560],vmin = hmin,
                    vmax = hmax,aspect = 1, 
                    extent=(x_disc[270],x_disc[560],z_disc[-1],z_disc[0]),cmap='jet')
        # hfig = av1.imshow(Vit_model2.T,vmin = hmin,
        #            vmax = hmax,aspect = 1, 
        #            extent=(3400,6600,z_disc[-1],z_disc[0]),cmap='jet')
        plt.colorbar(hfig)
        av1.axhline(1190.7)
        av1.plot(src_x[i],24,'*')
        av1.plot(rec_x[i],24,'v')
        av1.plot(spot_x,-spot_z,'.k')
        av1.scatter(ray_x,-ray_z, c="r", s=0.1)
        av1.set_title('ray number '+str(i)+'; offset = ' +str(int(off_x[i])))
        flout1 = "../png/27_marm/flat_marm/corr_az_pert/ray_sm_img_"+str(i)+"_f_p0007.png"
        print("Export to file:", flout1)
        fig1.savefig(flout1, bbox_inches='tight')
        
        
        
        # fig2 = plt.figure(figsize=(16, 8), facecolor="white")
        # av2 = plt.subplot(1, 1, 1)
        # av2.plot(src_x[i],-24,'*')
        # av2.plot(rec_x[i],-24,'v')
        # av2.plot(spot_x,spot_z,'ok',label='spot')
        # av2.plot(ray_x,ray_z,'.',label='ray',markersize=1)
        # av2.set_title('ray number '+str(i))
        # av2.legend()
        # av2.set_xlim(3750,5000)
        # flout2 = "../png/27_marm/flat_marm/p100/ray_plot_"+str(i)+"_p100.png"
        # print("Export to file:", flout2)
        # fig2.savefig(flout2, bbox_inches='tight')
        
        fig3 =plt.figure(figsize=(10, 10), facecolor="white")
        vp = np.array(read_results(path_ray[i], 6))
        ray_z = np.array(read_results(path_ray[i], 2))
        plt.plot(vp,ray_z,'.')
        plt.xlabel('Vp (m/s)')
        plt.ylabel('Z (m)')
        flout3 = "../png/27_marm/flat_marm/corr_az_pert/ray_vp_z_"+str(i)+"_corr.png"
        print("Export to file:", flout1)
        fig3.savefig(flout3, bbox_inches='tight') 
        
# path_ray_y = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/baptiste_corr_v181223/rays/opti_50_ray.csv"
# ray_y = np.array(read_results(path_ray_y, 1))

## Definition of a new ray path
# plt.figure()
# path_ray = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/opti_100_ray.csv"
# vp = np.array(read_results(path_ray, 6))
# ray_z = np.array(read_results(path_ray, 2))
# plt.plot(vp,ray_z,'.')
# plt.ylim(0,-1000)
