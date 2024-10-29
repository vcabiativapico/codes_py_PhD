#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 14:09:33 2024

@author: vcabiativapico
"""

import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate import interpolate
import csv
from wiggle.wiggle import wiggle
from spotfunk.res import procs, visualisation, input
import segyio
from PIL import Image 
from scipy.ndimage import gaussian_filter

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


# target images size
def resize_model(new_nz,new_nx,model):
    '''Modifies the model to the desired dimensions'''
    images = Image.fromarray(model)
    resized_images = images.resize((new_nx, new_nz), Image.LANCZOS)
    resized_array = np.array(resized_images)
    print(resized_array.shape)
    return resized_array

fl_ano = '../input/org_full/marm2_full.dat'
inp_ano = gt.readbin(fl_ano,nz,nx)
inp_ano = inp_ano[:,:255]

def plot_model(inp):
    hmin = np.min(inp)
    hmax = np.max(inp)
    fig = plt.figure(figsize=(10, 6), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto')
    plt.colorbar(hfig1)
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    
plot_model(inp_ano)

resize_ref_model = resize_model(nz,nx,inp_ano)

plot_model(resize_ref_model)

resize_ref_model_sm = gaussian_filter(resize_ref_model, 15)
plot_model(resize_ref_model_sm)


# flname = '../input/72_thick_marm_ano_born_mig_flat/marm_full_ext_flat.dat'
# gt.writebin(resize_ref_model, flname)

# flname = '../input/72_thick_marm_ano_born_mig_flat/marm_sm_ext_flat.dat'
# gt.writebin(resize_ref_model_sm, flname)

#%%
fl_full = '../input/org_full/marm2_full.dat'
inp_full = gt.readbin(fl_full,nz,nx)[:,:255]

x1 = 144
x2 = 200
z1 = 120
z2 = 135

# x1 = 295
# x2 = 317
# z1 = 75
# z2 = 100

inp_cut    = inp_full[z1:z2,x1:x2]     # Zone B - Cut the model in the area of the anomaly
old_vel    = np.max(inp_cut)                # Find the value where the anomaly wants to be changed    


plot_model(inp_full)
  

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
    print(np.mean(inp1_before))
    new_vel = nv
    inp1[index1] = new_vel
    
    # print(new_vel - inp1_before)
    # print(np.mean(new_vel - inp1_before))
    return inp1,index1

    
# inp_mod, ind_mod = modif_layer(inp_org, 2.55, 2.649, 4.5)

inp_mod_org, ind_mod = modif_layer(inp_full, 3.5, 4.1, 3.8280766)
plot_model(inp_mod_org)
resize_mod_org = resize_model(nz,nx,inp_mod_org)

flnam_out = '../input/72_thick_marm_ano_born_mig_flat/inp_mig_flat_extent_org.dat'
gt.writebin(resize_mod_org, flnam_out)
  


inp_mod_ano, ind_mod = modif_layer(inp_full, 3.5, 4.1, 3.8280766*1.14)
plot_model(inp_mod_ano)
resize_mod_ano = resize_model(nz,nx,inp_mod_ano)

flnam_out = '../input/72_thick_marm_ano_born_mig_flat/inp_mig_flat_extent_ano.dat'
gt.writebin(resize_mod_ano, flnam_out)
  