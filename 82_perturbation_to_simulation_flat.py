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
    hmax = np.max(inp)

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
    fig.tight_layout()

    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')
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

fl_inp = '../input/simple/simple_2000.dat'
flou_inp = '../png/inv_betap_x_s.png'
inp = gt.readbin(fl_inp,nz,nx)
plot_mig(inp,flou_inp)

fl_org = '../output/74_test_flat/new_vop8/org/inv_betap_x_s.dat'
flou_org = '../png/inv_betap_x_s.png'
inp_org = gt.readbin(fl_org,nz,nx)
plot_mig(inp_org,flou_org)

fl_ano = '../output/74_test_flat/new_vop8/ano/inv_betap_x_s.dat'
flout = '../png/inv_betap_x_s.png'
inp_ano = gt.readbin(fl_ano,nz,nx)
plot_mig(inp_ano,flout)



inp_m0_sm = 1/inp**2
plot_mig_min_max(inp_m0_sm,0,0.3)




# inp_org_norm = inp_org/np.max(inp_org) 
# plot_mig(inp_org_norm,flout)

# inp_ano_norm = inp_ano/np.max(inp_ano)
# plot_mig(inp_ano_norm,flout)

# flout = '../png/inv_betap_x_s.png'


inp_mig_plus_bg_org = inp_m0_sm + inp_org
plot_mig_min_max(inp_mig_plus_bg_org,0,0.5)

inp_mig_plus_bg_ano = inp_m0_sm + inp_ano
plot_mig_min_max(inp_mig_plus_bg_ano,0,0.5)


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

inp_corr_amp_org = convert_slowness_to_vel(inp_mig_plus_bg_org)
inp_corr_amp_ano = convert_slowness_to_vel(inp_mig_plus_bg_ano)



plot_mig_min_max(inp_sm_recover,1.800,2.200)

plot_mig_min_max(inp_corr_amp_org,1.95,2.05)
plot_mig_min_max(inp_corr_amp_ano,1.95,2.05)


# diff = inp_mig_plus_bg_org - inp_mig_plus_bg_ano
# plot_mig(diff,flout)


flnam1 = '../input/74_test_flat/new_betap_test_org.dat'
gt.writebin(inp_corr_amp_org, flnam1)

flnam2 = '../input/74_test_flat/new_betap_test_ano.dat'
gt.writebin(inp_corr_amp_ano, flnam2)

# flnam2 = '../input/73_new_flat_sm/rho_mig_plus_bg_ano.dat'
# gt.writebin(inp_mig_plus_bg_ano, flnam2)
