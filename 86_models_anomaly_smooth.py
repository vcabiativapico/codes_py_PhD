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
import csv
import matplotlib.patches as patches


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

def read_results(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            attr.append(int(row[srow]))
        # attr = [x for x in attr if str(x) != 'nan']
    return np.array(attr)


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
    plt.rcParams['font.size'] = 25
    # hmax = 5.0
    # hmin = 1.5
    # hmin = np.min(inp)
    # hmax = -hmin
    # hmax = np.max(inp)
    ax1 = 190
    ax2 = 370
    az1 = 45
    az2 = 128
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', cmap='viridis')
    hfig1 = av.imshow(inp[az1:az2,ax1:ax2], extent=[ax[ax1], ax[ax2], az[az2], az[az1]],
                      vmin=hmin, vmax=hmax, aspect='auto', cmap='viridis')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    rec1 = patches.Rectangle((ax1*dx,az2*dz), (ax2-ax1)*dx, (az1-az2)*dx, linewidth =3,edgecolor='red',facecolor= 'none')
    arrow1 = patches.Arrow(3.6, 0.2, -0.15, 0.18, width=0.1,color='black')
    av.add_patch(rec1)

    # av.set_xlim([2.5,4.5])
    # av.set_ylim([0.8,0.4])
    # plt.axvline(x=ax[tr], color='k',ls='--')
    # plt.axhline(0.606, color='w')
    plt.colorbar(hfig1, format='%1.2f',label='km/s')
    # plt.colorbar(hfig1, format='%1.1f',label='m/s')
    # fig.tight_layout()

    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    return inp, fig

#%%
'''Sum perturbations then convert to velocity'''

fl_marm_org = '../input/org_full/marm2_full.dat'
inp_marm_org = gt.readbin(fl_marm_org,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_marm_org,flout)

flnm_ano_old = '../input/68_thick_marm_ano/marm_thick_org.dat'
inp_ano_old = gt.readbin(flnm_ano_old,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano_old,flout)






fl3 = '../output/78_marm_sm8_thick_sum_pert/org/inv_betap_x_s.dat'
inp_org = gt.readbin(fl3,nz,nx)
flout = '../png/inv_betap_x_s.png'
# plot_mig(inp_org,flout)

fl4 = '../output/78_marm_sm8_thick_sum_pert/ano/inv_betap_x_s.dat'
inp_ano= gt.readbin(fl4,nz,nx)
flout = '../png/inv_betap_x_s.png'
# plot_mig(inp_ano,flout)

fl_sm = '../input/68_thick_marm_ano/marm_thick_org_sm8.dat'
flout = '../png/inv_betap_x_s.png'
inp_sm = gt.readbin(fl_sm,nz,nx)
# plot_mig_min_max(inp_sm, 1.5,3)





inp_m0_sm = 1/inp_sm**2

# plot_mig_min_max(inp_m0_sm,0.1,0.4)



inp_pert_sm_org = inp_m0_sm + inp_org
inp_pert_sm_ano = inp_m0_sm + inp_ano


def convert_slowness_to_vel(inp):
    inp = inp.reshape(nz*nx)
    inp_corr_amp = [0]*(nz*nx) 
    for i,x in enumerate(inp):
        inp_corr_amp[i] = 1/np.sqrt(inp[i])
    inp_corr_amp = np.reshape(inp_corr_amp,(nz,nx))
    return inp_corr_amp




inp_sm_recover = convert_slowness_to_vel(inp_m0_sm)

inp_corr_amp_org = convert_slowness_to_vel(inp_pert_sm_org)
inp_corr_amp_ano = convert_slowness_to_vel(inp_pert_sm_ano)


# plot_mig_min_max(inp_corr_amp_org ,2.0, 3.5)




# plot_mig_min_max(inp_corr_amp_ano ,1.5, 3.5)
# plot_mig_min_max(inp_sm_recover   ,1.5, 3.5)

flnm_org=  '../input/78_marm_sm8_thick_sum_pert/full_org.dat'
# gt.writebin(inp_corr_amp_org, flnm_org)

flnm_ano=  '../input/78_marm_sm8_thick_sum_pert/full_ano.dat'
# gt.writebin(inp_corr_amp_ano, flnm_ano)



flnam = '../input/68_thick_marm_ano/new_idx.dat'
flnam2 = '../input/68_thick_marm_ano/new_idx2.dat'
flnam3 = '../input/68_thick_marm_ano/idx_mod.dat'

new_idx = [read_results(flnam,0), read_results(flnam,1)]
new_idx2 = [read_results(flnam2,0), read_results(flnam2,1)]
idx_mod = [read_results(flnam3,0), read_results(flnam3,1)]


inp_corr_amp_ano_mod = np.copy(inp_corr_amp_ano)

mean1 = np.mean(inp_corr_amp_ano_mod[new_idx])
mean2 = np.mean(inp_corr_amp_ano_mod[new_idx2])
mean  = np.mean([mean1,mean2])

extra_val = 1.05

inp_corr_amp_ano_mod[new_idx] = mean * extra_val
inp_corr_amp_ano_mod[new_idx2] = mean * extra_val
inp_corr_amp_ano_mod[idx_mod] = mean * extra_val


flnm_ano_mod =  '../input/78_marm_sm8_thick_sum_pert/full_ano_mod.dat'
# gt.writebin(inp_corr_amp_ano_mod, flnm_ano_mod)

inp_ano_poly = inp_corr_amp_ano_mod - inp_corr_amp_org
inp_ano_poly_gauss = gaussian_filter(inp_ano_poly, 5)
flout = '../png/inv_betap_x_s.png'
plot_mig_min_max(inp_ano_poly_gauss,np.min(inp_ano_poly_gauss), np.max(inp_ano_poly_gauss))



inp_poly_ano_sm = inp_corr_amp_org + inp_ano_poly_gauss
flout = '../png/inv_betap_x_s.png'
plot_mig_min_max(inp_poly_ano_sm,2.0, 3.5)
flnm_poly_ano_sm =  '../input/80_smooth_ano_sum_pert/full_ano_mod_5p.dat'
# gt.writebin(inp_poly_ano_sm, flnm_poly_ano_sm)


plot_mig_min_max(inp_corr_amp_org,2.0, 3.5)
plot_mig_min_max(inp_corr_amp_ano,2.0, 3.5)

# flnm_corr_org =  '../input/80_smooth_ano_sum_pert/full_org_mod_corr.dat'
# gt.writebin(inp_corr_amp_org, flnm_corr_org)

# flnm_corr_ano =  '../input/80_smooth_ano_sum_pert/full_ano_mod_corr.dat'
# gt.writebin(inp_corr_amp_ano, flnm_corr_ano)


plot_mig_min_max(inp_ano_poly,np.min(inp_ano_poly), np.max(inp_ano_poly))



'''New anomaly in layers'''

inp_ano_poly_div = np.copy(inp_ano_poly_gauss)
inp_ano_poly_div[86:93] = 0
# inp_ano_poly_div[88:91] = 0
# inp_ano_poly_div[96:99] = 0
inp_ano_poly_div = gaussian_filter(inp_ano_poly_div, 2)

plot_mig_min_max(inp_ano_poly_div,np.min(inp_ano_poly_div), np.max(inp_ano_poly_div))

inp_corr_amp_ano_layers = inp_corr_amp_org + inp_ano_poly_div


plot_mig_min_max(inp_corr_amp_ano_layers,2.0, 3.5)



flnm_corr_org =  '../input/83_smooth_ano_layers/full_org_mod_corr.dat'
gt.writebin(inp_corr_amp_org, flnm_corr_org)

flnm_corr_ano =  '../input/83_smooth_ano_layers/full_ano_mod_corr_layers.dat'
gt.writebin(inp_corr_amp_ano_layers, flnm_corr_ano)



#%%

'''Sum perturbations then convert to velocity'''


fl3 = '../output/77_flat_fw_focus/org/inv_betap_x_s.dat'
inp_org = gt.readbin(fl3,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_org,flout)

fl4 = '../output/77_flat_fw_focus/ano/inv_betap_x_s.dat'
inp_ano= gt.readbin(fl4,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano,flout)

fl_sm = '../input/31_const_flat_tap/2_0_sm_constant.dat'
flout = '../png/inv_betap_x_s.png'
inp_sm = gt.readbin(fl_sm,nz,nx)
plot_mig_min_max(inp_sm, 1.5,3)


inp_m0_sm = 1/inp_sm**2

plot_mig_min_max(inp_m0_sm,0.1,0.4)



inp_pert_sm_org = inp_m0_sm + inp_org
inp_pert_sm_ano = inp_m0_sm + inp_ano


def convert_slowness_to_vel(inp):
    inp = inp.reshape(nz*nx)
    inp_corr_amp = [0]*(nz*nx) 
    for i,x in enumerate(inp):
        inp_corr_amp[i] = 1/np.sqrt(inp[i])
    inp_corr_amp = np.reshape(inp_corr_amp,(nz,nx))
    return inp_corr_amp


inp_sm_recover = convert_slowness_to_vel(inp_m0_sm)

inp_corr_amp_org = convert_slowness_to_vel(inp_pert_sm_org)
inp_corr_amp_ano = convert_slowness_to_vel(inp_pert_sm_ano)


plot_mig_min_max(inp_corr_amp_org ,1.9, 2.1)
plot_mig_min_max(inp_corr_amp_ano ,1.9, 2.1)
plot_mig_min_max(inp_sm_recover   ,1.9, 2.1)

flnm_org=  '../input/77_flat_fw_focus/full_org.dat'
# gt.writebin(inp_corr_amp_org, flnm_org)

flnm_ano=  '../input/77_flat_fw_focus/full_ano.dat'
# gt.writebin(inp_corr_amp_ano, flnm_ano)



