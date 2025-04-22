#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 10:44:58 2025

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

def convert_slowness_to_vel(inp,nx,nz):
    inp = inp.reshape(nz*nx)
    inp_corr_amp = [0]*(nz*nx) 
    for i,x in enumerate(inp):
        inp_corr_amp[i] = 1/np.sqrt(inp[i])
    inp_corr_amp = np.reshape(inp_corr_amp,(nz,nx))
    return inp_corr_amp

def resize_model(new_nz,new_nx,model):
    '''Modifies the model to the desired dimensions'''
    images = Image.fromarray(model)
    resized_images = images.resize((new_nx, new_nz), Image.LANCZOS)
    resized_array = np.array(resized_images)
    print(resized_array.shape)
    return resized_array

#%%
# path_mig = "/home/vcabiativapico/local/real_data/akerBP/aker_bp_edg_for_victor_rtm_seis.segy"
# path_vel = "/home/vcabiativapico/local/real_data/akerBP/aker_bp_edg_for_victor_velmodel.segy"

path_mig = "/home/vcabiativapico/local/real_data/akerBP/aker_bp_edg_for_victor_rtm_seis_sline_1707.segy"
path_vel = "/home/vcabiativapico/local/real_data/akerBP/aker_bp_edg_for_victor_velmodel_sline_1707.segy"


nx_mig = 600
dx_mig = 12.5
ax_mig = nx_mig*dx_mig


nz_mig = 450
dz_mig = 4
az_mig = dz_mig*nz_mig


mig = input.segy_reader(path_mig).dataset.T[:nz_mig,:nx_mig]


# # nz, nx = vel_model.shape[0]*5, vel_model.shape[1]*2
# # resize_vel_model = resize_model(nz,nx,vel_model)[:nz_mig,:nx_mig]



# plt.figure(figsize=(10,8))
# plt.imshow(mig, vmin = np.min(mig), vmax= -np.min(mig),cmap='seismic')
# plt.imshow(resize_vel_model,alpha=0.5,
#            vmin = 1500, vmax= 2500,cmap='jet')


# plt.figure(figsize=(10,8))
# plt.imshow(mig, vmin = np.min(mig), vmax= -np.min(mig),cmap='seismic')

# plt.figure(figsize=(10,8))
# plt.imshow(resize_vel_model,alpha=1,vmin = 1500, vmax= 2500,cmap='jet')



#%%
'''Resize mig nx and nz'''

'''First : interpolate to obtain a round ax_mig number'''
new_nx_mig = nx_mig*4
new_dx_mig = dx_mig/4
new_ax_mig = new_nx_mig*new_dx_mig
 
rs_mig = resize_model(nz_mig,new_nx_mig,mig)

'''Second : according to the new_ax_mig calculate the needed new_dx_mig = 4'''

new_dx_mig2 = 4
new_nx_mig2 = int(new_ax_mig//new_dx_mig2)
new_ax_mig2 = new_nx_mig2 * new_dx_mig2

rs_mig2 = resize_model(nz_mig,new_nx_mig2,rs_mig)

plt.figure(figsize=(10,8))
plt.imshow(rs_mig2, vmin = np.min(rs_mig), vmax= -np.min(rs_mig),
           cmap='seismic',extent=[0, new_ax_mig2, az_mig, 0])


plt.figure(figsize=(10,8))
plt.imshow(rs_mig, vmin = np.min(rs_mig), vmax= -np.min(rs_mig),
           cmap='seismic',extent=[0, new_ax_mig, az_mig, 0])


plt.figure(figsize=(10,8))
plt.imshow(mig, vmin = np.min(mig), vmax= -np.min(mig),
           cmap='seismic',extent=[0, ax_mig, az_mig, 0])


#%%

'''Resize vel nx '''

nx_vel = 300
dx_vel = 25
ax_vel = nx_vel * dx_vel

nz_vel = 450
dz_vel = 20
az_vel = dz_vel * nz_vel

vel_model =  input.segy_reader(path_vel).dataset.T[:nz_mig,:nx_mig]



new_nx_vel = nx_vel*4
new_dx_vel = dx_vel/4
new_ax_vel = new_nx_vel*new_dx_vel
 


rs_vel = resize_model(nz_vel,new_nx_vel,vel_model)

'''Second : according to the new_ax_vel calculate the needed new_dx_vel = 4'''

new_dx_vel2 = 4
new_nx_vel2 = int(new_ax_vel//new_dx_vel2)
new_ax_vel2 = new_nx_vel2 * new_dx_vel2

rs_vel2 = resize_model(nz_vel,new_nx_vel2,rs_vel)

plt.figure(figsize=(10,8))
plt.imshow(rs_vel2, vmin = np.min(rs_vel), vmax= 2500,
           cmap='seismic',extent=[0, new_ax_vel2, az_vel, 0])

plt.figure(figsize=(10,8))
plt.imshow(vel_model, vmin = np.min(vel_model), vmax= 2500,
           cmap='seismic',extent=[0, ax_vel, az_vel, 0])


'''Resize vel nz '''

new_nz_vel3 = nz_vel*5
new_dz_vel3 = dz_vel/5
new_az_vel3 = new_nz_vel3*new_dz_vel3
 

rs_vel3 = resize_model(new_nz_vel3,new_nx_vel2,rs_vel2)


plt.figure(figsize=(10,8))
plt.imshow(rs_mig2, vmin = np.min(rs_mig), vmax= -np.min(rs_mig),
           cmap='seismic',extent=[0, new_ax_mig2, az_mig, 0])
plt.imshow(rs_vel3, vmin = np.min(rs_vel), vmax= 2500,alpha=0.5,
           cmap='jet',extent=[0, new_ax_vel2, new_az_vel3, 0])

cut_idx_to_mig = int(az_mig//new_dz_vel3)


plt.figure(figsize=(14,14))
plt.imshow(rs_mig2, vmin = np.min(rs_mig2), vmax= -np.min(rs_mig2),
           cmap='seismic',extent=[0, new_ax_mig2, az_mig, 0])
plt.imshow(rs_vel3[:cut_idx_to_mig], vmin = np.min(rs_vel), vmax= 2500,alpha=0.5,
           cmap='jet',extent=[0, new_ax_vel2, az_mig, 0])


#%%

'''Resizing du modèle pour fortran Mines Paris PSL'''

fnx, fnz = 601,151

first_idx_vel = 150
first_idx_mig = 150



f_mig = resize_model(fnz,fnx,rs_mig2[first_idx_vel:])
f_vel = resize_model(fnz,fnx,rs_vel3[first_idx_mig:cut_idx_to_mig])


plt.figure(figsize=(14,14))
plt.imshow(f_mig, 
           vmin = np.min(rs_mig2), vmax= -np.min(rs_mig2),
           cmap='seismic'
           )

plt.imshow(f_vel, 
           vmin = np.min(rs_vel), vmax= 2500,alpha=0.5,
           cmap='jet'
           )

gt.writebin(f_mig,"/home/vcabiativapico/local/real_data/akerBP/aker_bp_resize_rtm.dat")
gt.writebin(f_vel,"/home/vcabiativapico/local/real_data/akerBP/aker_bp_resize_vel.dat")


#%%

'''Construction du modèle full pour migration'''
sm0_slowness_rs = 1/(f_vel/1000)**2

mig_sum_slow = sm0_slowness_rs + f_mig/5

mig_sum = convert_slowness_to_vel(mig_sum_slow,fnx,fnz)

plt.figure(figsize=(12,12))
plt.imshow(mig_sum, 
          alpha=1,
           cmap='jet')
# plt.colorbar(orientation='horizontal')

gt.writebin(mig_sum,"/home/vcabiativapico/local/real_data/akerBP/aker_bp_resize_full_mix.dat")
