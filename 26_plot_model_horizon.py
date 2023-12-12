#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 27 15:19:47 2023

@author: vcabiativapico
"""
import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from numba import jit
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import time
import csv
import sys
import gc
from spotfunk.res.input import segy_reader


from spotfunk.res import bspline

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


def read_results(path,srow):
    rec_x = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            rec_x.append(float(row[srow]))
    return rec_x

def read_bspline(path,srow):
    rec_x = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            rec_x.append(float(row[srow]))
    return rec_x

def readbin(flnam,nz,nx):
    # Read binary file (32 bits)
    with open(flnam,"rb") as fl:
        im = np.fromfile(fl, dtype=np.float32)
    im = im.reshape(nz,nx,order='F')
    return im

def plot_mig_hz(mig_img, horizon,shot_x):
    c_point = ['yellow','b','r','k','greenyellow']
    x_spline = np.arange(601)*12.00
    plt.figure(figsize=(16,8))
    plt.imshow(mig_img.T,vmin = -np.max(np.abs(mig_img)),
                vmax = np.max(np.abs(mig_img)),aspect = 2, 
                extent=(x_spline[0],x_spline[-1],z_disc[-1],z_disc[0]),cmap='seismic')
    
    plt.plot(x_spline,horizon,c='black',linewidth=3)  
    
    plt.colorbar(format='%1.e')
 
    for i in range(np.size(rec_x)):
        plt.scatter(rec_x[i],30,s=80,color=c_point[i], marker='v',alpha=0.5)
        plt.axvline(shot_x[i],color=c_point[i],ls='--',alpha=0.75)
    plt.xlabel('Distance (m)')
    plt.ylabel('Depth (m)')
    plt.rcParams['font.size'] = 18
    plt.tight_layout()
nx = 601
nz = 151
x_disc = np.arange(601)*12.00
z_disc = np.arange(151)*12.00
x_spline = np.arange(601)*12.00
# z_spline = np.arange(len(d_interp))*5




#%%
path_adj = '../../../../Demigration_SpotLight_Septembre2023/output/008_demig_sm_SR_az_12_all5_iz_ADJ_PP21.csv'
path_inv = '../../../../Demigration_SpotLight_Septembre2023/output/008_demig_sm_SR_az_12_all5_iz_INV_PP21.csv'

# path_adj = '../../../../Demigration_SpotLight_Septembre2023/output/012_demig_AO_INV_PP21.csv'
# path_inv = '../../../../Demigration_SpotLight_Septembre2023/output/012_demig_AO_ADJ_PP21.csv'


spot_x_adj = read_results(path_adj,7)
spot_y_adj = read_results(path_adj,8)
spot_z_adj = read_results(path_adj,9)


spot_x_adj = np.array(spot_x_adj)
spot_y_adj = np.array(spot_y_adj)
spot_z_adj = -np.array(spot_z_adj)

spot_x_inv = read_results(path_inv,7)
spot_y_inv = read_results(path_inv,8)
spot_z_inv = read_results(path_inv,9)



spot_x_inv = np.array(spot_x_inv)
spot_y_inv = np.array(spot_y_inv)
spot_z_inv = -np.array(spot_z_inv)
    
path = '../../../../Demigration_SpotLight_Septembre2023/output/008_demig_sm_SR_az_12_all5.csv'

rec_x = read_results(path,4)

# rec_x= read_results(path_inv,4) 
shot_x = read_results(path,1)

hz_adj_path_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/horizon_badj_smooth_rc_norm_12.csv'
hz_adj_path_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/horizon_badj_rc_norm_12.csv'
horizon_adj_sm = read_bspline(hz_adj_path_sm,0)   
horizon_adj_org = read_bspline(hz_adj_path_org,0)   


hz_inv_path_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/horizon_binv_smooth_rc_norm_12.csv'
hz_inv_path_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/horizon_binv_rc_norm_12.csv'
horizon_inv_sm = read_bspline(hz_inv_path_sm,0)   
horizon_inv_org = read_bspline(hz_inv_path_org,0) 

fl_badj = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/Model_Vit_discret/badj_inv_betap_x_s.dat'
fl_binv = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/Model_Vit_discret/inv_betap_x_s.dat'

mig_badj = readbin(fl_badj,nz,nx).T
mig_binv = readbin(fl_binv,nz,nx).T


## Plot overlay model + horizons SMOOTHED + spots
plt.figure(figsize=(13,7))
c_point = ['yellow','b','r','k','greenyellow']
plot_mig_hz(mig_badj,horizon_adj_sm,shot_x)
for i in range(len(c_point)):
    plt.plot(spot_x_adj[i],spot_z_adj[i],marker='o',color=c_point[i])
flout = '../png/26_mig_4_interfaces/mig_image_adj_sm_spotxz.png'
plt.savefig(flout, bbox_inches='tight')
print("Export to file:", flout)

plot_mig_hz(mig_binv,horizon_inv_sm,shot_x)
for i in range(len(c_point)):
    plt.plot(spot_x_inv[i],spot_z_inv[i],marker='o',color=c_point[i])
flout = '../png/26_mig_4_interfaces/mig_image_inv_sm_spotxz.png'
plt.savefig(flout, bbox_inches='tight')
print("Export to file:", flout)


## Plot overlay model + horizons as ORIGINALLY PICKED + spots
plt.figure(figsize=(13,7))
plot_mig_hz(mig_badj,horizon_adj_org,shot_x)
for i in range(len(c_point)):
    plt.plot(spot_x_adj[i],spot_z_adj[i],marker='o',color=c_point[i])
flout = '../png/26_mig_4_interfaces/mig_image_adj_org_spotxz.png'
plt.savefig(flout, bbox_inches='tight')
print("Export to file:", flout)


plot_mig_hz(mig_binv,horizon_inv_org,shot_x)
for i in range(len(c_point)):
    plt.plot(spot_x_inv[i],spot_z_inv[i],marker='o',color=c_point[i])
flout = '../png/26_mig_4_interfaces/mig_image_inv_org_spotxz.png'
plt.savefig(flout, bbox_inches='tight')
print("Export to file:", flout)


## Plot horizons smoothed
plt.figure(figsize=(13,7))
plt.plot(ax,horizon_inv_sm,'.b')
plt.plot(ax,horizon_adj_sm,'.r')
plt.tight_layout()
plt.legend(['Quantitative','Standard'])
plt.rcParams['font.size'] = 18
plt.ylim(1200,1500)
plt.xlabel('Distance (km)')
plt.ylabel('Depth (m)')
plt.gca().invert_yaxis()
flout = '../png/26_mig_4_interfaces/compare_smooth_hz_inv_vs_adj.png'
plt.savefig(flout, bbox_inches='tight')
print("Export to file:", flout)



## Plot horizons as picked
plt.figure(figsize=(13,7))
plt.plot(ax,horizon_inv_org,'.b')
plt.plot(ax,horizon_adj_org,'.r')
plt.tight_layout()
plt.legend(['Quantitative','Standard'])
plt.ylim(1200,1500)
plt.xlabel('Distance (km)')
plt.ylabel('Depth (m)')
plt.gca().invert_yaxis()
flout = '../png/26_mig_4_interfaces/compare_org_hz_inv_vs_adj.png'
plt.savefig(flout, bbox_inches='tight')
print("Export to file:", flout)



hz_delta_sm = np.arange(601)
hz_delta_org = np.arange(601)
hz_sm_org = np.arange(601)

for i in range(np.size(horizon_inv_org)):
    hz_delta_org[i] = horizon_inv_org[i]-horizon_adj_org[i]
    hz_delta_sm[i] = horizon_inv_sm[i]-horizon_adj_sm[i]
    hz_sm_org[i] = horizon_inv_org[i]-horizon_inv_sm[i]
    
plt.figure(figsize=(13,7))
plt.plot(ax,hz_delta_org,'.k')
plt.title('Difference horizon original')
plt.xlabel('Distance (km)')
plt.ylabel('Difference (m)')
plt.ylim(-10,10)
flout = '../png/26_mig_4_interfaces/diff_hz_org_inv_vs_adj.png'
plt.savefig(flout, bbox_inches='tight')
print("Export to file:", flout)



plt.figure(figsize=(13,7))
plt.plot(ax,hz_delta_sm,'.k')
plt.title('Difference horizon smooth')
plt.xlabel('Distance (km)')
plt.ylabel('Difference (m)')
plt.ylim(-10,10)
flout = '../png/26_mig_4_interfaces/diff_hz_sm_inv_vs_adj.png'
plt.savefig(flout, bbox_inches='tight')
print("Export to file:", flout)

plt.figure(figsize=(13,7))
plt.plot(ax,hz_sm_org,'.k')
plt.title('Difference horizon original')
plt.ylim(-10,10)

#%% MARMOUSI2 

# Read horizons
hz_adj_marm_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_sm3_marm_adj_02.csv'
hz_adj_marm_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_marm_adj_02.csv'
hz_adj_marm_sm = read_bspline(hz_adj_marm_sm,0)   
hz_adj_marm_org = read_bspline(hz_adj_marm_org,0)   


hz_inv_marm_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_sm3_marm_inv_02.csv'
hz_inv_marm_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_marm_inv_02.csv'


hz_inv_marm_sm =  np.array(read_bspline(hz_inv_marm_sm,0))   
hz_inv_marm_org =  np.array(read_bspline(hz_inv_marm_org,0)) 

# Read migrated images

fl_marm_badj = '../output/27_marm/badj/inv_betap_x_s.dat'
fl_marm_binv = '../output/27_marm/binv/inv_betap_x_s.dat'
marm_mig_badj = np.array(readbin(fl_marm_badj,nz,nx).T)
marm_mig_binv = np.array(readbin(fl_marm_binv,nz,nx).T)

# Read ray-tracing spot location results
path_adj = '../../../../Demigration_SpotLight_Septembre2023/output/010_marm_sm_badj_PP21_P005_hz02.csv'
path_inv = '../../../../Demigration_SpotLight_Septembre2023/output/010_marm_sm_binv_PP21_P005_hz02.csv'

path_adj = '../../../../Demigration_SpotLight_Septembre2023/output/011_marm_sm_badj_PP21_P021_hz02.csv'
path_inv = '../../../../Demigration_SpotLight_Septembre2023/output/011_marm_sm_binv_PP21_P021_hz02.csv'


spot_x_adj = read_results(path_adj,7)
spot_y_adj = read_results(path_adj,8)
spot_z_adj = read_results(path_adj,9)

spot_x_adj = np.array(spot_x_adj)
spot_y_adj = np.array(spot_y_adj)
spot_z_adj = -np.array(spot_z_adj)

spot_x_inv = read_results(path_inv,7)
spot_y_inv = read_results(path_inv,8)
spot_z_inv = read_results(path_inv,9)

spot_x_inv = np.array(spot_x_inv)
spot_y_inv = np.array(spot_y_inv)
spot_z_inv = -np.array(spot_z_inv)

# Read rec_x
rec_x = read_results(path_adj,4)
# rec_x = read_results(path,4)

shot_x = read_results(path_adj,1)

## Difference of horizons
hz_delta_sm = hz_inv_marm_sm - hz_adj_marm_sm
# Plot both horizons and difference
plt.figure(figsize=(13,7))
plt.plot(x_spline,hz_inv_marm_sm)
plt.plot(x_spline,hz_adj_marm_sm)
plt.legend(['inv','adj'])
plt.title('horizons smooth inv vs adj ')
plt.gca().invert_yaxis()

plt.figure(figsize=(13,7))
plt.plot(x_spline,hz_delta_sm)
plt.title('horizons its differnce')






# Plot image, spot and horizon
plt.figure(figsize=(13,7))
c_point = ['yellow','b','r','k','greenyellow']
# plot_mig_hz(marm_mig_binv,hz_inv_marm_org)


plot_mig_hz(marm_mig_binv,hz_inv_marm_sm,shot_x)
for i in range(len(spot_x_inv)):
    plt.plot(spot_x_inv[i],spot_z_inv[i],'.',
             color=c_point[i],markersize=12)

# plt.xlim(2800,5500)
plt.ylim(1800,0)




# plot_mig_hz(marm_mig_badj,hz_adj_marm_org)
# plt.plot(spot_x_adj,spot_z_adj,'o',color='greenyellow')

plot_mig_hz(marm_mig_badj,hz_adj_marm_sm,shot_x)
for i in range(len(spot_x_inv)):
    plt.plot(spot_x_adj[i],spot_z_adj[i],'.',
             color=c_point[i],markersize=12)
# plt.xlim(2800,5500)
plt.ylim(1800,0)
  

