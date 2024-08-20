#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:37:55 2024

@author: vcabiativapico
"""

import os
import numpy as np
from math import log, sqrt, log10, pi, cos, sin, atan
import csv
import matplotlib.pyplot as plt
from matplotlib import use
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel
import pickle as pk
from scipy import interpolate


if __name__ == "__main__":
    
    # Global parameters
    labelsize = 16
    nt        = 1001
    dt        = 2.08e-3
    ft        = -100.11e-3
    nz        = 151
    fz        = 0.0
    dz        = 12.49/1000.
    nx        = 601
    fx        = 0.0
    dx        = 12.49/1000.
    no        = 251
    do        = dx
    fo        = -(no-1)/2*do
    ao        = fo + np.arange(no)*do
    at        = ft + np.arange(nt)*dt
    az        = fz + np.arange(nz)*dz
    ax        = fx + np.arange(nx)*dx

    hmin,hmax = 1.5,4.0
    fl1       = '../input/org/marm2_sel.dat'
    inp_org   = gt.readbin(fl1,nz,nx)
    inp1      = gt.readbin(fl1,nz,nx)
    perc      = 0.9
    
        
    def plot_model(inp,hmin,hmax):
        plt.rcParams['font.size'] = 16
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto'\
                         )
        plt.colorbar(hfig)
        plt.xlabel('Distance (Km)')
        plt.ylabel('Profondeur (Km)')
        fig.tight_layout()
        return fig
        
    def export_model(inp,fig,imout,flout):
        fig.savefig(imout, bbox_inches='tight')
        gt.writebin(inp,flout)  
        
#%% MODIFY MARMOUSI2 FOR DEMIGRATION
fl1       = '../input/60_sigsbee2a/sigsbee2a_stratigraphy.dat'
# fl1       = '../input/45_marm_ano_v3/fwi_ano_45.dat'

fl2       = '../input/60_sigsbee2a/sigsbee2a_migration_velocity_sm.dat'


inp_org   = gt.readbin(fl1,nz,nx)
inp_sm    = gt.readbin(fl2,nz,nx)

x1 = 178
x2 = 243
z1 = 79
z2 = 84

# x1 = 295
# x2 = 317
# z1 = 75
# z2 = 100

inp_cut    = inp_org[z1:z2,x1:x2]     # Zone B - Cut the model in the area of the anomaly
old_vel    = np.max(inp_cut)                # Find the value where the anomaly wants to be changed    

hmin = 1.5
hmax = 4.5

plot_model(inp_org,hmin,hmax)
  

def modif_layer(inp1,r1,r2,nv): 
    area      = np.zeros(inp1.shape)
    for i in range(z1,z2): 
        for j in range(x1,x2):
            if inp1[i,j] > r1 and inp1[i,j] < r2 : 
                area[i,j] = 1
            else: 
                area[i,j] = 0
            
            
    index1     = np.where(area == 1)
    # new_vel    = nv
    inp1_before = inp1[index1]
    new_vel   = inp1[index1]*1.14
   
    inp1[index1] = new_vel
    
    # print(new_vel - inp1_before)
    # print(np.mean(new_vel - inp1_before))
    return inp1,index1

%matplotlib qt5
    
# inp_mod, ind_mod = modif_layer(inp_org, 2.55, 2.649, 4.5)

inp_mod, ind_mod = modif_layer(inp_org, 2.16, 2.31, 5)

# fl1       = '../input/org_full/marm2_full.dat'
# inp_org   = gt.readbin(fl1,nz,nx)
inp_diff = (inp_mod - inp_org)
inp_diff = (inp_mod - inp_org)*1.14


hmax = np.max(inp_cut)
hmin = np.min(inp_cut)
plot_model(inp_cut,hmin,hmax)



hmax = np.max(inp_mod)
hmin = np.min(inp_mod)
fig1 = plot_model(inp_mod,hmin,hmax)



# hmax = np.max(inp_diff)
# hmin = np.min(inp_diff)
# fig1 = plot_model(inp_diff,hmin,hmax)
# imout1 = '../png/45_marm_ano_v3/fwi_diff_ano.png'
# flout1 = '../input/45_marm_ano_v3/fwi_diff_ano.dat'
# # export_model(inp_diff,fig1,imout1,flout1)


# inp_diff_sm = inp_diff+inp_sm

# hmax = np.max(inp_diff_sm)
# hmin = np.min(inp_diff_sm)
# fig1 = plot_model(inp_diff_sm,hmin,hmax)
# imout1 = '../png/45_marm_ano_v3/fwi_diff_sm_ano.png'
# flout1 = '../input/45_marm_ano_v3/fwi_diff_sm_ano.dat'
# # export_model(inp_diff_sm,fig1,imout1,flout1)

# adbetap_exact = np.copy(inp_diff)

# for i in range(nz):
#     for j in range(nx): 
#         if inp_diff[i,j] == 0: 
#             adbetap_exact[i,j] = 0
#         else:
#             adbetap_exact[i,j] = 1/inp_diff_sm[i,j]**2 - 1/inp_sm[i,j]**2

# hmin = np.min(adbetap_exact)
# hmax = np.max(adbetap_exact)
# fig1 = plot_model(adbetap_exact,hmin,hmax)
# imout1 = '../png/45_marm_ano_v3/adbetap_diff_sm_ano.png'
# flout1 = '../input/45_marm_ano_v3/adbetap_diff_sm_ano.dat'
# export_model(adbetap_exact,fig1,imout1,flout1)