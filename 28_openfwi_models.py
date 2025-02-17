#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 11:20:25 2023

@author: vcabiativapico
"""

import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from PIL import Image 
from scipy.ndimage import gaussian_filter

labelsize = 16
nt = 1501
dt = 1.14e-3
ft = -100.11e-3
nz = 151
fz = 0.0
dz = 12.0/1000.
nx = 601
fx = 0.0
dx = 12.0/1000.
no = 251
# no        = 2002
do = dx
fo = -(no-1)/2*do
ao = fo + np.arange(no)*do
at = ft + np.arange(nt)*dt
az = fz + np.arange(nz)*dz
ax = fx + np.arange(nx)*dx

def plot_model(inp):
    plt.rcParams['font.size'] = 20
    hmin = np.min(inp)
    hmax = -hmin
    # hmax = np.max(inp)
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', cmap='seismic')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig1, format='%1.1f',label='m/s')
    # fig.tight_layout()
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    return inp, fig

def convert_slowness_to_vel(inp):
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


data_vel0_2d = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_2d/vp_year0/vp_year0_slide10.bin'
data_vel5_2d = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_2d/vp_year5/vp_year5_slide10.bin'
data_vel10_2d = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_2d/vp_year10/vp_year10_slide10.bin'
data_vel20_2d = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_2d/vp_year20/vp_year20_slide10.bin'
data_vel30_2d = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_2d/vp_year30/vp_year30_slide10.bin'

# 1,350,400,400

nx = 350
nz = 600

# nx = 601
# nz = 151

def readbin(flnam,nz,nx):
    # Read binary file (32 bits)
    with open(flnam,"rb") as fl:
        im = np.fromfile(fl, dtype=np.float32)
    print(im.shape)
    im = im.reshape(1,nx,nz,order='F')
    return im

kim2d_model_y0 = readbin(data_vel0_2d,nz,nx)[0]/1000

kim2d_model_y5 = readbin(data_vel5_2d,nz,nx)[0]/1000

kim2d_model_y10 = readbin(data_vel10_2d,nz,nx)[0]/1000

kim2d_model_y20 = readbin(data_vel20_2d,nz,nx)[0]/1000

kim2d_model_y30 = readbin(data_vel30_2d,nz,nx)[0]/1000

kim2d_model_sm0 = gaussian_filter(kim2d_model_y0,10)



new_nz, new_nx = 151, 601

def resize_model(new_nz,new_nx,model):
    '''Modifies the model to the desired dimensions'''
    images = Image.fromarray(model)
    resized_images = images.resize((new_nx, new_nz), Image.LANCZOS)
    resized_array = np.array(resized_images)
    print(resized_array.shape)
    return resized_array

kim2d_model_y0_rs  = resize_model(new_nz,new_nx,kim2d_model_y0)

kim2d_model_y5_rs  = resize_model(new_nz,new_nx,kim2d_model_y5)

kim2d_model_y10_rs = resize_model(new_nz,new_nx,kim2d_model_y10)

kim2d_model_y20_rs = resize_model(new_nz,new_nx,kim2d_model_y20)

kim2d_model_y30_rs = resize_model(new_nz,new_nx,kim2d_model_y30)

kim2d_model_sm0_rs = resize_model(new_nz,new_nx,kim2d_model_sm0)


hmin = np.min(kim2d_model_y0)
hmax = np.max(kim2d_model_y0)
fig  = plt.figure(figsize=(12, 8), facecolor="white")
av   = plt.subplot(1, 1, 1)
hfig = av.imshow(kim2d_model_y0_rs, 
                 vmin=hmin, vmax=hmax,extent=[ax[0], ax[-1], az[-1], az[0]], aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


# hmin = np.min(kim2d_model_y35)
# hmax = np.max(kim2d_model_y35)
fig  = plt.figure(figsize=(12, 8), facecolor="white")
av   = plt.subplot(1, 1, 1)
hfig = av.imshow(kim2d_model_y20_rs, 
                 vmin=hmin, vmax=hmax,extent=[ax[0], ax[-1], az[-1], az[0]], aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


hmin = np.min(kim2d_model_y30_rs-kim2d_model_y20_rs)
hmax = np.max(kim2d_model_y30_rs-kim2d_model_y20_rs)
fig  = plt.figure(figsize=(12, 8), facecolor="white")
av   = plt.subplot(1, 1, 1)
hfig = av.imshow(kim2d_model_y30_rs-kim2d_model_y20_rs, 
                 vmin=hmin, vmax=hmax,extent=[ax[0], ax[-1], az[-1], az[0]], aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)





fig  = plt.figure(figsize=(12, 8), facecolor="white")
av   = plt.subplot(1, 1, 1)
hfig = av.imshow(kim2d_model_sm0_rs, 
                 vmin=hmin, vmax=hmax,extent=[ax[0], ax[-1], az[-1], az[0]], aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)



gt.writebin(kim2d_model_y0_rs,'../input/83_kimberlina2d/kim_model_y0_resize.dat')
gt.writebin(kim2d_model_y5_rs,'../input/83_kimberlina2d/kim_model_y5_resize.dat')
gt.writebin(kim2d_model_y10_rs,'../input/83_kimberlina2d/kim_model_y10_resize.dat')
gt.writebin(kim2d_model_y20_rs,'../input/83_kimberlina2d/kim_model_y20_resize.dat')
gt.writebin(kim2d_model_y30_rs,'../input/83_kimberlina2d/kim_model_y30_resize.dat')
gt.writebin(kim2d_model_sm0_rs,'../input/83_kimberlina2d/kim_model_sm0_resize.dat')


test = gt.readbin('../input/83_kimberlina2d/kim_model_y0_resize.dat',151,601)
# test = gt.readbin('../input/78_marm_sm8_thick_sum_pert/full_org.dat',151,601)
plot_model(test)
#%%


# data_vel = pickle.load('/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_3d/year0_cut10.bin')
data_vel0 = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_3d/vp_year0/year0_cut10.bin'
# data_vel15 = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_3d/vp_year15/vp_year15_slide10.bin'
data_vel35 = '/media/vcabiativapico/NTFS_seagate4TB/Victor/FWIOpenData/kimberlina_3d/vp_year35/year35_cut10.bin'

# 1,350,400,400

nx = 350
nz = 400

# nx = 601
# nz = 151


kim_model_y0 = readbin(data_vel0,nz,nx)
kim_model_y0 = kim_model_y0[0][:,200]

# kim_model_y15 = readbin(data_vel15,nz,nx)
# kim_model_y15 = kim_model_y15[0][:,200]


kim_model_y35 = readbin(data_vel35,nz,nx)
kim_model_y35 = kim_model_y35[0][:,200]



#%%

nz, nx = 151, 601


kim_model_y0_rs = resize_model(nz,nx,kim_model_y0)/1000

# kim_model_y15_rs = resize_model(nz,nx,kim_model_y15)/1000

kim_model_y35_rs = resize_model(nz,nx,kim_model_y35)/1000

kim_model_sm0_rs = gaussian_filter(kim_model_y0_rs,10)


diff = kim_model_y35_rs-kim_model_y0_rs

# hmin = np.min(kim_model_y0_rs)
# hmax = np.max(kim_model_y0_rs)
# plot_model(kim_model_y0_rs)

fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_model_y0_rs, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


hmin = np.min(kim_model_y35_rs)
hmax = np.max(kim_model_y35_rs)
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_model_y35_rs, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)

hmin = np.min(kim_model_sm0_rs)
hmax = np.max(kim_model_sm0_rs)
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_model_sm0_rs, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)



hmin = np.min(kim_model_sm0_rs-kim_model_y0_rs)
hmax = np.max(kim_model_sm0_rs-kim_model_y0_rs)
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_model_sm0_rs-kim_model_y0_rs, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)

hmin = np.min(diff)
hmax = np.max(diff)
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(diff, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


# gt.writebin(kim_model_y0_rs,'../input/82_kimberlina_rs/kim_model_y0_resize.dat')
# gt.writebin(kim_model_y35_rs,'../input/82_kimberlina_rs/kim_model_y35_resize.dat')
# gt.writebin(kim_model_sm0_rs,'../input/82_kimberlina_rs/kim_m


fl3 = '../output/82_kimberlina_rs/org/inv_betap_x_s.dat'
org = gt.readbin(fl3,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_model(org)

fl4 = '../output/82_kimberlina_rs/ano/inv_betap_x_s.dat'
ano = gt.readbin(fl4,nz,nx)
plot_model(ano)


#%%

kim_model_sm0_slowness_rs = 1/kim2d_model_sm0_rs**2


nx = 601
nz = 151

fl3 = '../output/83_kimberlina2d/org/inv_betap_x_s.dat'
org = gt.readbin(fl3,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_model(org)

fl4 = '../output/83_kimberlina2d/ano/inv_betap_x_s.dat'
ano = gt.readbin(fl4,nz,nx)
plot_model(ano)


kim_org_sum_slow = kim_model_sm0_slowness_rs + org
kim_ano_sum_slow = kim_model_sm0_slowness_rs + ano

kim_org_sum = convert_slowness_to_vel(kim_org_sum_slow)
kim_ano_sum = convert_slowness_to_vel(kim_ano_sum_slow)

kim_diff_sum = kim_org_sum - kim_ano_sum


hmin = np.min(kim_org_sum)
hmax = np.max(kim_org_sum)
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_org_sum, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_ano_sum, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


hmin = np.min(kim_diff_sum)
hmax = np.max(kim_diff_sum)
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_diff_sum, 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)



gt.writebin(kim_org_sum,'../input/83_kimberlina2d/full_sum/sum_kim_model_y0_resize.dat')
gt.writebin(kim_ano_sum,'../input/83_kimberlina2d/full_sum/sum_kim_model_y20_resize.dat')


