#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 15:01:56 2025

@author: vcabiativapico
"""


import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from PIL import Image 
from scipy.ndimage import gaussian_filter
from scipy import signal
import pandas as pd

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


def plot_test_model(inp,hmin=-9999,hmax=9999):
    if hmax==9999:
        hmin = np.min(inp)
        hmax = np.max(inp)
    fig  = plt.figure(figsize=(14, 7), facecolor="white")
    av   = plt.subplot(1, 1, 1)
    hfig = av.imshow(inp, 
                     vmin=hmin, vmax=hmax, aspect='auto',
                      extent=[ax[0], ax[-1], az[-1], az[0]],
                     cmap='viridis')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig, format='%1.2f',label='m/s')
    fig.tight_layout()
    return fig
        
def crop_fill(inp1,inp2,z_limit):
    inp_overthrust_rs = resize_model(z_limit, 600, inp1[:135])
    # inp_overthrust_rs = resize_model(z_limit, 600, inp1)
    # plot_test_model(inp_overthrust_rs)
    inp_kim2d_crop = np.copy(inp2)
    inp_kim2d_crop[:z_limit] = 0
    inp_mix = np.copy(inp_kim2d_crop)
    inp_mix[:z_limit] = inp_overthrust_rs
    return inp_mix

def create_diagonal_mask(height, width, angle, taper=10):
    """
    Creates a central diagonal mask that tapers off from the main diagonal.
    
    Parameters:
        height (int): Height of the matrix.
        width (int): Width of the matrix.
        angle (float): Vertical angle in degrees.
        taper (float): Controls the width of the tapering effect.
        
    Returns:
        np.ndarray: A 2D mask with values tapering from the diagonal.
    """
    # Convert angle to radians
    theta = np.radians(angle)

    # Create coordinate grid
    x = np.arange(width)
    y = np.arange(height)
    X, Y = np.meshgrid(x, y)

    # Compute the central diagonal equation y = mx + b
    m = np.tan(theta)  # Slope from the angle
    center_x = width / 2
    center_y = height / 2

    # Compute distance from each point to the diagonal line
    dist = np.abs((m * (X - center_x)) - (Y - center_y)) / np.sqrt(m**2 + 1)

    # Apply a tapering function (Gaussian-like falloff)
    mask = np.exp(- (dist**2) / (2 * (taper**2)))

    return mask

def model_window(win1=0,win2=200,mode='horizontal'):
    if mode=='horizontal':
        window = np.zeros_like(org)
        win_size = win2-win1
        win_tuk = signal.windows.tukey(win_size,alpha=0.6)
        print(win_tuk.shape)
        for i in range(nz):  
            window[i,win1:win2] = win_tuk
        print(window[i,win1:win2].shape)
    elif mode == 'vertical':
        window = np.zeros_like(org)
        win_size = win2-win1
        win_tuk = signal.windows.tukey(win_size,alpha=0.6)
        for i in range(nx):  
            window[win1:win2,i] = win_tuk
    return window

#%%

# nx = 801
# nz = 187


# flnam = '../input/84_overthrust/overthrust2D'
# # inp_overthrust = gt.readbin(flnam, nz, nx)/2300 +0.52
# inp_overthrust = gt.readbin(flnam, nz, nx) /1500 +0.35


# inp_merge1 = np.zeros_like(inp_overthrust)
# inp_merge1 = inp_overthrust[:,:450] 
# inp_merge2 = inp_overthrust[:,450:] 

# inp_merge12 = np.zeros((nz,450+351))

# inp_merge12[:,351:] = inp_merge1[:,:]
# inp_merge12[:,:351] = inp_merge2[:,:]

# inp_merge3 = np.zeros((nz,nx+351))
# inp_merge3[:,nx:] = inp_merge2
# inp_merge3[:,:nx] = inp_merge12

# plot_test_model(inp_merge3)

# vp_value = []
# year_vp= []
# for i in range(0,51,5):
#     fl_kim_y0 = '../input/83_kimberlina2d/original/vp_year'+str(i)+'/vp_year'+str(i)+'_slide10.bin'
    
#     inp_kim2d_y0 = gt.readbin(fl_kim_y0,350,600)/1000+0.28
    
#     def crop_fill(inp1,inp2,z_limit):
#         inp_overthrust_rs = resize_model(z_limit, 600, inp1[:135])
#         # inp_overthrust_rs = resize_model(z_limit, 600, inp1)
#         # plot_test_model(inp_overthrust_rs)
#         inp_kim2d_crop = np.copy(inp2)
#         inp_kim2d_crop[:z_limit] = 0
#         inp_mix = np.copy(inp_kim2d_crop)
#         inp_mix[:z_limit] = inp_overthrust_rs
#         return inp_mix
    
#     z_limit =230

#     inp_mix = crop_fill(inp_merge3,inp_kim2d_y0,z_limit)
#     inp_mix_rs = resize_model(151, 601, inp_mix)
    
#     plot_test_model(inp_mix_rs)
    
#     vp_value.append(inp_mix_rs[122,217])
#     year_vp.append(str(i))
    
    
#     gt.writebin(inp_mix_rs,'../input/90_kimberlina_mod_v3_high/mix_model_rs_y'+str(i)+'.dat')
 
#     if i==0:
#         sm_0 = gaussian_filter(inp_mix_rs, 8)
        
#         plot_test_model(sm_0)
             
#         gt.writebin(sm_0,'../input/90_kimberlina_mod_v3_high/mix_model_rs_sm0.dat')


# file = '../output/90_kimberlina_mod_v3_high/res_vp_value.csv'

# df = pd.DataFrame({'year':year_vp,'res_vp_value':vp_value})
# df.to_csv(file,index=None)

# plot_test_model(sm_0)



# test = gt.readbin('../input/90_kimberlina_mod_v3_high/mix_model_rs_y10.dat', 151, 601)

# plot_test_model(test)

# hmin = np.min(test)
# hmax = np.max(test)
# fig  = plt.figure(figsize=(12, 8), facecolor="white")
# av   = plt.subplot(1, 1, 1)
# hfig = av.imshow(test, 
#                   vmin=hmin, vmax=hmax, aspect='auto',extent=[ax[0], ax[-1], az[-1], az[0]],
#                   cmap='jet')
# plt.colorbar(hfig)


#%%
fl1 = '../input/94_kimberlina_v4/mix_model_rs_sm0.dat'
sm_0 = gt.readbin(fl1,nz,nx)

# plot_test_model(sm_0)


year = 30


kim_over_sm0_slowness_rs = 1/sm_0**2
# plot_test_model(kim_over_sm0_slowness_rs)

nx = 601
nz = 151

fl3 = '../output/94_kimberlina_v4/y0/inv_betap_x_s.dat'
org = gt.readbin(fl3,nz,nx)
flout = '../png/inv_betap_x_s.png'
# plot_model(org)

fl2 = '../input/94_kimberlina_v4/mix_model_rs_y0.dat'
org_full = gt.readbin(fl2,nz,nx)

fl21 = '../input/94_kimberlina_v4/mix_model_rs_y30.dat'
ano_full = gt.readbin(fl21,nz,nx)


anomaly_full = org_full-ano_full

anomaly_full[anomaly_full < 0] = 0 # eliminate negative values

kim_org_sum_slow = kim_over_sm0_slowness_rs + org

kim_org_sum = convert_slowness_to_vel(kim_org_sum_slow)
   
plot_test_model(kim_org_sum)
plot_test_model(anomaly_full)




# gt.writebin(kim_org_sum,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y0.dat')

angle_max = 90

for angle in range(angle_max,9, -10):
        
    if angle == 90:
        taper = 25  # Width of taper
        roll = -65
        extra_x = 0
        
    elif angle == 70:
        taper = 25  # Width of taper
        roll = -80
        extra_x = 0
        
    elif angle == 50:
        taper = 20  # Width of taper
        roll = -110
        extra_x = 0
        
    elif angle == 40:
        taper = 18  # Width of taper
        roll = -130
        extra_x = 0
        
    elif angle == 30:
        taper = 12  # Width of taper
        roll = -170
        extra_x = 0
        
    elif angle == 20:
       taper = 10  # Width of taper
       roll = -360
       extra_x = 300  
       
    elif angle == 10:
       taper = 5  # Width of taper
       roll = -500
       extra_x = 250 
    
    else : 
        print('no angle data for angle : '+str(angle))
        continue
    mask_fit = np.zeros_like(anomaly_full)
    
    # Generate mask
    mask = create_diagonal_mask(nz, nx+extra_x, angle, taper)
    # Roll to position the mask
    mask_roll = np.roll(mask, roll)
    
    
    diff_kim_masked_model = mask_roll[:nz,:nx] * anomaly_full
    
    if angle == angle_max:
        hmin = np.min(anomaly_full)
        hmax = np.max(anomaly_full)
        plot_test_model(anomaly_full,hmin,hmax)
        plot_test_model(mask_roll[:nz,:nx])

    if angle == angle_max:
        hmin1 = np.min(diff_kim_masked_model)
        hmax1 = np.max(diff_kim_masked_model)
    fig = plot_test_model(diff_kim_masked_model,hmin1,hmax1)
    flout = '../png/95_kimberlina_fault/angle_'+str(angle)+'_y'+str(year)+'.png'
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')
     
    kim_masked_angle  = kim_org_sum - diff_kim_masked_model 
    
    gt.writebin(kim_masked_angle,
                '../input/95_kimberlina_fault/full_sum/sum_kim_model_angle_'+str(angle)+'_y'+str(year)+'.dat')
    
    if angle == angle_max:
        hmin = np.min(kim_masked_angle)
        hmax = np.max(kim_masked_angle)
    fig = plot_test_model(kim_masked_angle)
    flout = '../png/95_kimberlina_fault/model_angle_'+str(angle)+'_y'+str(year)+'.png'
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')     
