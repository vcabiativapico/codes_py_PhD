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
    # hmax = -hmin
    hmax = np.max(inp)
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


def plot_test_model(inp):
    hmin = np.min(inp)
    hmax = np.max(inp)
    fig  = plt.figure(figsize=(14, 7), facecolor="white")
    av   = plt.subplot(1, 1, 1)
    hfig = av.imshow(inp, 
                     vmin=hmin, vmax=hmax, aspect='auto',extent=[ax[0], ax[-1], az[-1], az[0]],
                     cmap='jet')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig, format='%1.2f',label='m/s')
    
#%%

nx = 801
nz = 187


flnam = '../input/84_overthrust/overthrust2D'
# inp_overthrust = gt.readbin(flnam, nz, nx)[:,:450] /2300 +0.52

inp_overthrust = gt.readbin(flnam, nz, nx)[:,:450] /1500 +0.35


for i in range(0,31,5):
    fl_kim_y0 = '../input/83_kimberlina2d/original/vp_year'+str(i)+'/vp_year'+str(i)+'_slide10.bin'
    
    inp_kim2d_y0 = gt.readbin(fl_kim_y0,350,600)/1000+0.28
    
    def crop_fill(inp1,inp2,z_limit):
        inp_overthrust_rs = resize_model(z_limit, 600, inp1[:135])
        # inp_overthrust_rs = resize_model(z_limit, 600, inp1)
        # plot_test_model(inp_overthrust_rs)
        inp_kim2d_crop = np.copy(inp2)
        inp_kim2d_crop[:z_limit] = 0
        inp_mix = np.copy(inp_kim2d_crop)
        inp_mix[:z_limit] = inp_overthrust_rs
        return inp_mix
    
    z_limit =230
    
    
    inp_mix = crop_fill(inp_overthrust,inp_kim2d_y0,z_limit)
    
    inp_mix_rs = resize_model(151, 601, inp_mix)
    
    
    plot_test_model(inp_mix_rs)
        
    
    gt.writebin(inp_mix_rs,'../input/89_kim_mix_overthrust_vhigh/mix_model_rs_y'+str(i)+'.dat')

    if i==0:
        sm_0 = gaussian_filter(inp_mix_rs, 8)
        
        plot_test_model(sm_0)
     
        
        gt.writebin(sm_0,'../input/89_kim_mix_overthrust_vhigh/mix_model_rs_sm0.dat')


plot_model(sm_0)


test = gt.readbin('../input/89_kim_mix_overthrust_vhigh/mix_model_rs_y0.dat', 151, 601)
hmin = np.min(test)
hmax = np.max(test)
fig  = plt.figure(figsize=(12, 8), facecolor="white")
av   = plt.subplot(1, 1, 1)
hfig = av.imshow(test, 
                  vmin=hmin, vmax=hmax, aspect='auto',extent=[ax[0], ax[-1], az[-1], az[0]],
                  cmap='jet')
plt.colorbar(hfig)
    
    
#%%

year = 'y10'
# year = 'y20'

kim_over_sm0_slowness_rs = 1/sm_0**2
plot_model(kim_over_sm0_slowness_rs)

nx = 601
nz = 151

fl3 = '../output/89_kim_mix_overthrust_vhigh/y0/inv_betap_x_s.dat'
org = gt.readbin(fl3,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_model(org)

fl4 = '../output/89_kim_mix_overthrust_vhigh/'+year+'/inv_betap_x_s.dat'
ano = gt.readbin(fl4,nz,nx)
plot_model(ano)


kim_org_sum_slow = kim_over_sm0_slowness_rs + org
kim_ano_sum_slow = kim_over_sm0_slowness_rs + ano

kim_org_sum = convert_slowness_to_vel(kim_org_sum_slow)
kim_ano_sum = convert_slowness_to_vel(kim_ano_sum_slow)

kim_diff_sum = kim_org_sum - kim_ano_sum


hmin = np.min(kim_diff_sum)
hmax = np.max(kim_diff_sum)
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_diff_sum,extent=[ax[0], ax[-1], az[-1], az[0]], 
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='jet')
plt.colorbar(hfig)


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

    
kim_diff_part1 = kim_diff_sum * model_window(70, 150)
kim_diff_part2 = kim_diff_sum * model_window(150, 230)
kim_diff_part3 = kim_diff_sum * model_window(230, 310)
 

kim_y20_part1  = kim_org_sum - kim_diff_part1 
kim_y20_part2  = kim_org_sum - kim_diff_part2 
kim_y20_part3  = kim_org_sum - kim_diff_part3 


kim_diff_part3_v1 = kim_diff_part3 * model_window(110, 130,'vertical')
kim_diff_part3_v2 = kim_diff_part3 * model_window(124, 140,'vertical')

kim_y20_part3_v1  = kim_org_sum - kim_diff_part3_v1 
kim_y20_part3_v2  = kim_org_sum - kim_diff_part3_v2 


kim_diff_part2_v1 = kim_diff_part2 * model_window(110, 130,'vertical')
kim_diff_part2_v2 = kim_diff_part2 * model_window(124, 140,'vertical')


kim_y10_part2_v1  = kim_org_sum - kim_diff_part2_v1 
kim_y10_part2_v2  = kim_org_sum - kim_diff_part2_v2 



# plot_test_model(kim_org_sum)
# plot_test_model(kim_ano_sum)

# plot_test_model(kim_diff_sum)

hmin = -0.1235926076841829
hmax = 0.16841850522783552
fig  = plt.figure(figsize=(14, 7), facecolor="white")
av   = plt.subplot(1, 1, 1)
hfig = av.imshow(kim_diff_part2_v1, 
                 vmin=hmin, vmax=hmax, aspect='auto',extent=[ax[0], ax[-1], az[-1], az[0]],
                 cmap='jet')
plt.xlabel('Distance (km)')
plt.ylabel('Depth (km)')
plt.colorbar(hfig, format='%1.2f',label='m/s')


gt.writebin(kim_org_sum,'../input/89_kim_mix_overthrust_vhigh/full_sum/sum_kim_model_y0.dat')
gt.writebin(kim_ano_sum,'../input/89_kim_mix_overthrust_vhigh/full_sum/sum_kim_model_'+year+'.dat')

gt.writebin(kim_y20_part1,'../input/89_kim_mix_overthrust_vhigh/full_sum/sum_kim_model_'+year+'_p1.dat')
gt.writebin(kim_y20_part2,'../input/89_kim_mix_overthrust_vhigh/full_sum/sum_kim_model_'+year+'_p2.dat')
gt.writebin(kim_y20_part3,'../input/89_kim_mix_overthrust_vhigh/full_sum/sum_kim_model_'+year+'_p3.dat')

gt.writebin(kim_y10_part2_v1,'../input/89_kim_mix_overthrust_vhigh/full_sum/sum_kim_model_'+year+'_p2_v1.dat')
gt.writebin(kim_y10_part2_v2,'../input/89_kim_mix_overthrust_vhigh/full_sum/sum_kim_model_'+year+'_p2_v2.dat')

gt.writebin(kim_y20_part3_v1,'../input/89_kim_mix_overthrust_vhigh/full_sum/sum_kim_model_'+year+'_p3_v1.dat')
gt.writebin(kim_y20_part3_v2,'../input/89_kim_mix_overthrust_vhigh/full_sum/sum_kim_model_'+year+'_p3_v2.dat')



#%%


test = gt.readbin('../input/86_new_mix_kim_overthrust/full_sum/sum_kim_model_y0.dat', 151, 601)
test10 = gt.readbin('../input/86_new_mix_kim_overthrust/full_sum/sum_kim_model_y10.dat', 151, 601)
test20 = gt.readbin('../input/86_new_mix_kim_overthrust/full_sum/sum_kim_model_y20.dat', 151, 601)



diff_test10 = test - test10
diff_test20 = test - test20

hmin = np.min(diff_test20)
hmax = np.max(diff_test20)
fig  = plt.figure(figsize=(12, 8), facecolor="white")
av   = plt.subplot(1, 1, 1)
hfig = av.imshow(diff_test10, 
                  vmin=hmin, vmax=hmax, aspect='auto',extent=[ax[0], ax[-1], az[-1], az[0]],
                  cmap='jet')
plt.colorbar(hfig)
    

hmin = np.min(diff_test20)
hmax = np.max(diff_test20)
fig  = plt.figure(figsize=(12, 8), facecolor="white")
av   = plt.subplot(1, 1, 1)
hfig = av.imshow(diff_test20, 
                  vmin=hmin, vmax=hmax, aspect='auto',extent=[ax[0], ax[-1], az[-1], az[0]],
                  cmap='jet')
plt.colorbar(hfig)
    
