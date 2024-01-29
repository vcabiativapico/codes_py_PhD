#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:20:25 2023

@author: vcabiativapico
"""

import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt

# data_obs = np.load('/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_co2_official/kimberlina_co2_train_data/data_sim0420_t100.npz')
data_vel = np.load('/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_co2_official/kimberlina_co2_train_label/label_sim0380_t80.npz')


# data_obs.files
data_vel.files

# obs = data_obs['data']
vel = data_vel['label']

# obs_0 = obs[0]

# plt.imshow(obs[0])
# extent=[ao[0], ao[-1], at[-1], at[0]],

# hmin = np.min(obs_0)
# hmin = -5
# hmax = -hmin
# fig = plt.figure(figsize=(10, 8), facecolor="white")
# av = plt.subplot(1, 1, 1)
# hfig = av.imshow(obs_0, 
#                  vmin=hmin, vmax=hmax, aspect='auto',
#                  cmap='seismic')
# plt.colorbar(hfig)

vel_ext = np.zeros((601,151))
vel_ext[100:501,0:141] = vel

for i in range(0,100):
    vel_ext[i] = vel_ext[120] 
for i in range(499,601):
    vel_ext[i] = vel_ext[120] 
    
for k in range(139,151):
    vel_ext[:,k] = vel_ext[:,135]
    
hmin = np.min(vel)
# hmin = 0
hmax = np.max(vel)

fig = plt.figure(figsize=(16, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(vel_ext.T, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)

#%%
import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.ndimage import gaussian_filter

# data_vel = pickle.load('/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_3d/year0_cut10.bin')
data_vel0 = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_3d/vp_year0/year0_cut10.bin'
data_vel35 = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_3d/vp_year35/year35_cut10.bin'
# 1,350,400,400

nx = 350
nz = 400
def readbin(flnam,nz,nx):
    # Read binary file (32 bits)
    with open(flnam,"rb") as fl:
        im = np.fromfile(fl, dtype=np.float32)
    im = im.reshape(1,nx,nz,400,order='F')
    return im

kim_model_y0 = readbin(data_vel0,nz,nx)
kim_model_y0 = kim_model_y0[0][:,200]

kim_model_y35 = readbin(data_vel35,nz,nx)
kim_model_y35 = kim_model_y35[0][:,200]

#%%
diff_y0_y35 = kim_model_y0-kim_model_y35

kim_model_sm0 = gaussian_filter(kim_model_y0,10)


hmin = np.min(kim_model_y0)
hmax = np.max(kim_model_y0)
fig = plt.figure(figsize=(16, 15), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_model_sm0, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


hmin = np.min(kim_model_y0)
hmax = np.max(kim_model_y0)
fig = plt.figure(figsize=(16, 15), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_model_y0, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


fig = plt.figure(figsize=(16, 15), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_model_y35, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


gt.writebin(kim_model_y0,'../input/28_kimberlina3d/kim_model_y0_full.dat')
gt.writebin(kim_model_y35,'../input/28_kimberlina3d/kim_model_y35_full.dat')