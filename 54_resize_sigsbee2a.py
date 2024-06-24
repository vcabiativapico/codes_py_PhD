#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 09:17:35 2024

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

path_mig = '/home/vcabiativapico/local/madagascar/Sigsbee2A/sigsbee2a_migration_velocity.sgy'
path_stra = '/home/vcabiativapico/local/madagascar/Sigsbee2A/sigsbee2a_stratigraphy.sgy'
path_ref = '/home/vcabiativapico/local/madagascar/Sigsbee2A/sigsbee2a_reflection_coefficients.sgy'


vel_mig_model = input.segy_reader(path_mig).dataset.T
vel_stra_model =  input.segy_reader(path_stra).dataset.T
vel_ref_model =  input.segy_reader(path_ref).dataset.T

plt.figure(figsize=(10,8))
plt.imshow(vel_mig_model)

plt.figure(figsize=(10,8))
plt.imshow(vel_stra_model)

hmax = np.max(vel_ref_model)/10
hmin = -hmax
plt.figure(figsize=(10,8))
plt.imshow(vel_ref_model,vmin=hmin,vmax=hmax,cmap='seismic')


# target images size
def resize_model(new_nz,new_nx,model):
    '''Modifies the model to the desired dimensions'''
    images = Image.fromarray(model)
    resized_images = images.resize((new_nx, new_nz), Image.LANCZOS)
    resized_array = np.array(resized_images)
    print(resized_array.shape)
    return resized_array

nz, nx = 151, 601
resize_mig_model = resize_model(nz,nx,vel_mig_model)/4000+276/1000
resize_stra_model = resize_model(nz,nx,vel_stra_model)/4000+276/1000
resize_ref_model = resize_model(nz,nx,vel_ref_model)

hmax = np.max(resize_mig_model[:,:])
hmin = np.min(resize_mig_model[:,:])
plt.figure(figsize=(14,7))
plt.imshow(resize_mig_model[:,:],aspect='auto',vmin=hmin,vmax=hmax)
plt.colorbar()

hmax = np.max(resize_stra_model[:,:])
hmin = np.min(resize_stra_model[:,:])
plt.figure(figsize=(14,7))
plt.imshow(resize_stra_model[:,:],aspect='auto',vmin=hmin,vmax=hmax)
plt.colorbar()


hmax = np.max(resize_ref_model[:,:])
hmin = -hmax
plt.figure(figsize=(14,7))
plt.imshow(resize_ref_model[:,:],cmap='seismic',aspect='auto',vmin=hmin,vmax=hmax)
plt.colorbar()


# Smooth model

mig_model_sm = gaussian_filter(resize_mig_model,15)
hmax = np.max(mig_model_sm[:,:])
hmin = np.min(mig_model_sm[:,:])
plt.figure(figsize=(14,7))
plt.imshow(mig_model_sm[:,:],aspect='auto',vmin=hmin,vmax=hmax)
plt.colorbar()



# diff = 1/mig_model_sm[:,:]**2-1/resize_stra_model[:,:]**2
# hmax = np.max(diff)
# hmin = np.min(diff)
# plt.figure(figsize=(14,7))
# plt.imshow(diff,aspect='auto',vmin=hmin,vmax=hmax)
# plt.colorbar()

# gt.writebin(resize_mig_model,'/home/vcabiativapico/local/madagascar/Sigsbee2A/sigsbee2a_migration_velocity.dat')
# gt.writebin(resize_stra_model,'/home/vcabiativapico/local/madagascar/Sigsbee2A/sigsbee2a_stratigraphy.dat')
# gt.writebin(mig_model_sm,'/home/vcabiativapico/local/madagascar/Sigsbee2A/sigsbee2a_migration_velocity_sm.dat')



#%%

fl2 = '../input/marm2_full.dat'
inp_org   = gt.readbin(fl2,nz,nx)
hmax = np.max(inp_org)
hmin = np.min(inp_org)
plt.figure(figsize=(15,12))
plt.imshow(inp_org,aspect=1,vmin=hmin,vmax=hmax)
plt.colorbar()