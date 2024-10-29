#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 16:07:02 2024

@author: vcabiativapico
"""


import os
import numpy as np
from math import log, sqrt, log10, pi, cos, sin, atan
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert

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


def plot_mig(inp, flout):
    # hmax = np.max(np.abs(inp2))
    # hmin = -hmax

    # print(hmin,hmax)
    plt.rcParams['font.size'] = 20
    # hmax = 5.0
    # hmin = 1.5
    hmin = np.min(inp)
    hmax = -hmin
    # hmax = np.max(inp)

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', cmap='seismic')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    # av.set_xlim([2.5,4.5])
    # av.set_ylim([0.8,0.4])
    # plt.axvline(x=ax[tr], color='k',ls='--')
    # plt.axhline(0.606, color='w')
    plt.colorbar(hfig1, format='%1.1f',label='m/s')
    # plt.colorbar(hfig1, format='%1.1f',label='m/s')
    # fig.tight_layout()

    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    return inp, fig


def plot_mig_min_max(inp, hmin,hmax):
    # hmax = np.max(np.abs(inp2))
    # hmin = -hmax

    # print(hmin,hmax)
    plt.rcParams['font.size'] = 20
    # hmax = 5.0
    # hmin = 1.5
    # hmin = np.min(inp)
    # hmax = -hmin
    # hmax = np.max(inp)

    fig = plt.figure(figsize=(14, 7), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', cmap='viridis')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    # av.set_xlim([2.5,4.5])
    # av.set_ylim([0.8,0.4])
    # plt.axvline(x=ax[tr], color='k',ls='--')
    # plt.axhline(0.606, color='w')
    plt.colorbar(hfig1, format='%1.1f',label='m/s')
    # plt.colorbar(hfig1, format='%1.1f',label='m/s')
    # fig.tight_layout()

    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    return inp, fig

#%%
''' Read migration results for the inverse case'''


fl = '../input/org_full/marm2_full.dat'
inp = gt.readbin(fl,nz,nx)
plot_mig(inp,flout)

inp_gaus10= gaussian_filter(inp, 10)
plot_mig(inp_gaus10,flout)
flnam = '../input/org_full/marm2_gauss_10.dat'
gt.writebin(inp_gaus10, flnam)


fl1 = '../input/org_full/marm2_smooth.dat'
flout = '../png/inv_betap_x_s.png'
inp1 = gt.readbin(fl1,nz,nx)
plot_mig(inp1,flout)

fl3 = '../output/71_thick_marm_ano_born_mig/org_inv/inv_betap_x_s.dat'
inp_org = gt.readbin(fl3,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_org,flout)



fl4 = '../output/71_thick_marm_ano_born_mig/ano_inv/inv_betap_x_s.dat'
inp_ano= gt.readbin(fl4,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano,flout)


''' To extract the shape of the anomaly only from the migrated image'''
inp_diff_inv = inp_org-inp_ano
plot_mig(inp_diff_inv,flout)

# inp_new_betap_inv = inp1 + inp_diff_inv
# plot_mig(inp_new_betap_inv,flout)


# flnam2 = '../input/71_thick_marm_ano_born_mig/new_betap_inv.dat'
# gt.writebin(inp_new_betap_inv, flnam2)





inp_org_norm = inp_org/np.max(inp_org)
plot_mig(inp_org_norm,flout)

inp_ano_norm = inp_ano/np.max(inp_ano)
plot_mig(inp_ano_norm,flout)



inp_mig_plus_bg_org = inp1 + inp_org_norm +0.5
plot_mig(inp_mig_plus_bg_org,flout)

inp_mig_plus_bg_ano = inp1 + inp_ano_norm+0.5
plot_mig(inp_mig_plus_bg_ano,flout)

diff_inp = inp_mig_plus_bg_org-inp_mig_plus_bg_ano
plot_mig(diff_inp, flout)

# flnam2 = '../input/71_thick_marm_ano_born_mig/inp_mig_plus_bg_org.dat'
# gt.writebin(inp_mig_plus_bg_org, flnam2)

# flnam2 = '../input/71_thick_marm_ano_born_mig/inp_mig_plus_bg_ano.dat'
# gt.writebin(inp_mig_plus_bg_ano, flnam2)

#%%
'''Sum perturbations then convert to velocity'''


fl3 = '../output/75_marm_sum_pert/org/inv_betap_x_s.dat'
inp_org = gt.readbin(fl3,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_org,flout)

fl4 = '../output/75_marm_sum_pert/ano/inv_betap_x_s.dat'
inp_ano= gt.readbin(fl4,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano,flout)

fl_sm = '../input/org_full/marm2_smooth.dat'
flout = '../png/inv_betap_x_s.png'
inp_sm = gt.readbin(fl_sm,nz,nx)
plot_mig_min_max(inp_sm, 1.5,3)


inp_m0_sm = 1/inp_sm**2

plot_mig_min_max(inp_m0_sm,0.1,0.4)



inp_pert_sm_org = inp_m0_sm + inp_org
plot_mig( inp_pert_sm_org,flout)
plot_mig_min_max(inp_pert_sm_org,-1,1)

inp_pert_sm_ano = inp_m0_sm + inp_ano
plot_mig_min_max(inp_pert_sm_ano,-1,1)


def convert_slowness_to_vel(inp):
    inp = inp.reshape(nz*nx)
    inp_corr_amp = [0]*(nz*nx)
    
    for i,x in enumerate(inp):
        if x > 0:
            inp_corr_amp[i] = 1/np.sqrt(inp[i])
        elif x ==0:
            inp_corr_amp[i] = 0
        else:
            inp_corr_amp[i] = -1/np.sqrt(-inp[i])
    
    inp_corr_amp = np.reshape(inp_corr_amp,(nz,nx))
    return inp_corr_amp




inp_sm_recover = convert_slowness_to_vel(inp_m0_sm)

inp_corr_amp_org = convert_slowness_to_vel(inp_pert_sm_org)
inp_corr_amp_ano = convert_slowness_to_vel(inp_pert_sm_ano)


plot_mig_min_max(inp_corr_amp_org ,np.min(inp_corr_amp_org), np.max(inp_corr_amp_org))
plot_mig_min_max(inp_corr_amp_ano ,1.5, 3.5)
plot_mig_min_max(inp_sm_recover   ,1.5, 3.5)

# flnm_ano=  '../input/75_marm_sum_pert/full_org.dat'
# gt.writebin(inp_corr_amp_org, flnm_ano)

# flnm_ano=  '../input/75_marm_sum_pert/full_ano.dat'
# gt.writebin(inp_corr_amp_ano, flnm_ano)


#%%

'''Creating the input model for the full modelling'''

fl3 =  '../output/72_thick_marm_ano_born_mig_flat/org/inv_betap_x_s.dat'
inp3 = gt.readbin(fl3,nz,nx)Figure 2024-10-29 152204
flout = '../png/inv_betap_x_s.png'
inp3 = inp3/np.max(inp3)
plot_mig(inp3,flout)


fl4 = '../output/72_thick_marm_ano_born_mig_flat/ano/inv_betap_x_s.dat'
inp4 = gt.readbin(fl4,nz,nx)
flout = '../png/inv_betap_x_s.png'
inp4 = inp4/np.max(inp4)
plot_mig(inp4,flout)



plot_mig(inp3-inp4,flout)

fl_sm = '../input/72_thick_marm_ano_born_mig_flat/marm_sm_ext_flat.dat'
inp_sm = gt.readbin(fl_sm,nz,nx)  

new_flat_org = inp_sm + inp3 + 0.5
plot_mig(new_flat_org,flout)  
flnm_org=  '../input/72_thick_marm_ano_born_mig_flat/new_flat_org.dat'
# gt.writebin(new_flat_org, flnm_org)


new_flat_ano = inp_sm + inp4 + 0.5
plot_mig(new_flat_ano,flout)    
flnm_ano=  '../input/72_thick_marm_ano_born_mig_flat/new_flat_ano.dat'
# gt.writebin(new_flat_ano, flnm_ano)

#%%
def plot_sim_wf(bg, inp1):
    plt.rcParams['font.size'] = 22
    hmax = np.max(inp1)
    print('hmax: ', hmax)
    hmax = np.max(inp1)
    hmin = -hmax
    for i in range(450, 700, 25):
        fig = plt.figure(figsize=(13, 6), facecolor="white")
        av = plt.subplot(1, 1, 1)
        hfig = av.imshow(inp1[:, i, :]*2, extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                          vmin=hmin, vmax=hmax, aspect='auto', alpha=1,
                          cmap='jet')
        hfig1 = av.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                          aspect='auto', alpha=0.3,
                          cmap='gray')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
        # arrow1 = patches.Arrow(3.6, 0.2, -0.15, 0.18, width=0.1,color='black')
        # arrow2 = patches.Arrow(4.45, 0.6, -0.15, 0.2, width=0.1,color='white')
        # av.add_patch(arrow1)
        # av.add_patch(arrow2)
        # av.plot(ray_x,ray_z)
        # av.scatter(2.640,0.012,marker='*')
        av.set_title('t = '+str(i*dt*1000)+' s')
        plt.colorbar(hfig)
        fig.tight_layout()
        flout2 = '../png/71_thick_marm_ano_born_mig/sim_org_fwi_shot_'+str(shot_mod_idx)+'_t_'+str(i)+'.png'
        print("Export to file:", flout2)
        fig.savefig(flout2, bbox_inches='tight')
  
    
shot_mod_idx = 251

fl1   = '../output/71_thick_marm_ano_born_mig/simulations/inv/p2d_fwi_000'+str(shot_mod_idx)+'.dat'
fl2   = '../output/71_thick_marm_ano_born_mig/simulations/adj/p2d_fwi_000'+str(shot_mod_idx)+'.dat'
fl3 = '../input/71_thick_marm_ano_born_mig/inp_mig_plus_bg_org.dat'

fl4 = '../output/71_thick_marm_ano_born_mig/simulations/smooth/p2d_fwi_000'+str(shot_mod_idx)+'.dat'
fl5 = '../output/71_thick_marm_ano_born_mig/simulations/new_pert_mig_sm_org/p2d_fwi_000'+str(shot_mod_idx)+'.dat'
fl6 = '../output/71_thick_marm_ano_born_mig/simulations/new_pert_mig_sm_ano/p2d_fwi_000'+str(shot_mod_idx)+'.dat'


inp1 = gt.readbin(fl3, nz, nx)

# inp1 = np.copy(diff_inp)

nt    = 1801

nxl   = 291
h_nxl = int((nxl-1)/2)

org  =  -gt.readbin(fl1, nz, nxl*nt) 
ano  = -gt.readbin(fl2, nz, nxl*nt)   #
smooth = -gt.readbin(fl4, nz, nxl*nt) 
new_pert_mig_sm_org = -gt.readbin(fl5, nz, nxl*nt) 
new_pert_mig_sm_ano = -gt.readbin(fl6, nz, nxl*nt) 


# nxl = bites/4/nz/nt = bites/4/151/1501
# position central (301-1)*dx = 3600
# 291 = 1+2*145
# point à gauche = 3600-145*dx
# point à droite = 3600+145*dx
value = 300 - shot_mod_idx

# value = 300-221

left_p  = 300 - value - h_nxl  # left point
right_p = 300 - value + h_nxl  # right point

org = np.reshape(org, (nz, nt, nxl))
ano = np.reshape(ano, (nz, nt, nxl))
smooth = np.reshape(smooth, (nz, nt, nxl))
new_pert_mig_sm_org = np.reshape(new_pert_mig_sm_org, (nz, nt, nxl))
new_pert_mig_sm_ano = np.reshape(new_pert_mig_sm_ano, (nz, nt, nxl))

diff = new_pert_mig_sm_org-new_pert_mig_sm_ano

plot_sim_wf(inp1, diff)

# plot_sim_wf(inp1, smooth-org)
