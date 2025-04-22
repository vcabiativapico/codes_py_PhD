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
from spotfunk.res import procs
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

   
def crop_fill(inp1,inp2,z_limit):
    inp_overthrust_rs = resize_model(z_limit, 600, inp1[:135])
    # inp_overthrust_rs = resize_model(z_limit, 600, inp1)
    # plot_test_model(inp_overthrust_rs)
    inp_kim2d_crop = np.copy(inp2)
    inp_kim2d_crop[:z_limit] = 0
    inp_mix = np.copy(inp_kim2d_crop)
    inp_mix[:z_limit] = inp_overthrust_rs
    return inp_mix
#%%

nx = 801
nz = 187


flnam = '../input/84_overthrust/overthrust2D'
# inp_overthrust = gt.readbin(flnam, nz, nx)/2300 +0.52
inp_overthrust = gt.readbin(flnam, nz, nx) /1900 +0.3


inp_merge1 = np.zeros_like(inp_overthrust)
inp_merge1 = inp_overthrust[:,:450] 
inp_merge2 = inp_overthrust[:,450:] 

inp_merge12 = np.zeros((nz,450+351))

inp_merge12[:,351:] = inp_merge1[:,:]
inp_merge12[:,:351] = inp_merge2[:,:]

inp_merge3 = np.zeros((nz,nx+351))
inp_merge3[:,nx:] = inp_merge2
inp_merge3[:,:nx] = inp_merge12

plot_test_model(inp_merge3)

vp_value = []
year_vp= []
for i in range(0,51,5):
    fl_kim_y0 = '../input/83_kimberlina2d/original/vp_year'+str(i)+'/vp_year'+str(i)+'_slide10.bin'
    
    # inp_kim2d_y0 = gt.readbin(fl_kim_y0,350,600)/1000+0.28
    
    inp_kim2d_y0 = gt.readbin(fl_kim_y0,350,600)/1000*1.8-3
 
    
    z_limit =230

    inp_mix = crop_fill(inp_merge3,inp_kim2d_y0,z_limit)
    inp_mix_rs = resize_model(151, 601, inp_mix)
    
    plot_test_model(inp_mix_rs)
    
    vp_value.append(inp_mix_rs[122,186])
    year_vp.append(str(i))
    
    
    # gt.writebin(inp_mix_rs,'../input/94_kimberlina_v4/mix_model_rs_y'+str(i)+'.dat')
 
    if i==0:
        sm_0 = gaussian_filter(inp_mix_rs, 8)
        
        plot_test_model(sm_0)
             
        # gt.writebin(sm_0,'../input/94_kimberlina_v4/mix_model_rs_sm0.dat')


# file = '../output/90_kimberlina_mod_v3_high/res_vp_value_new.csv'

# df = pd.DataFrame({'year':year_vp,'res_vp_value':vp_value})
# df.to_csv(file,index=None)

plot_test_model(sm_0)



# test = gt.readbin('../input/90_kimberlina_mod_v3_high/mix_model_rs_y10.dat', 151, 601)

# plot_test_model(test)

# hmin = np.min(test)
# hmax = np.max(test)
# fig  = plt.figure(figsize=(12, 8), facecolor="white")
# av   = plt.subplot(1, 1, 1)
# hfig = av.imshow(test, 
#                   vmin=hmin, vmax=hmax, aspect='auto',extent=[ax[0], ax[-1], az[-1], az[0]],
#                   cmap='viridis')
# plt.colorbar(hfig)
    
    
#%%
vp_max_diff  = []
year_vp_diff = []
vp_mean_diff = []
vp_extract_full = []
vp_mediane = []
percentage = []

vp_new_min_full = []
vp_original_min_full = []
label = []
# year = 10
# year = 'y20'


for year in range(50,5,-5):
    kim_over_sm0_slowness_rs = 1/sm_0**2
    # plot_model(kim_over_sm0_slowness_rs)
    
    nx = 601
    nz = 151
    
    fl3 = '../output/94_kimberlina_v4/y0/inv_betap_x_s.dat'
    org = gt.readbin(fl3,nz,nx)
    flout = '../png/inv_betap_x_s.png'
    # plot_model(org)
    
    fl2 = '../input/94_kimberlina_v4/mix_model_rs_y0.dat'
    org_full = gt.readbin(fl2,nz,nx)
    
    fl21 = '../input/94_kimberlina_v4/mix_model_rs_y'+str(year)+'.dat'
    ano_full = gt.readbin(fl21,nz,nx)
    
    
    anomaly_full = (org_full - ano_full)*0.7
    
    anomaly_full[anomaly_full < 0] = 0 # eliminate negative values
    
    kim_org_sum_slow = kim_over_sm0_slowness_rs + org
    
    kim_org_sum = convert_slowness_to_vel(kim_org_sum_slow)
       
    
    kim_ano_sum = kim_org_sum -anomaly_full
    
    
    # plot_test_model(kim_org_sum)
    # plt.title('original')
    # plot_test_model(kim_ano_sum)
        
    kim_diff_part1 = anomaly_full * model_window(70, 150)
    kim_diff_part2 = anomaly_full * model_window(150, 230)
    kim_diff_part3 = anomaly_full * model_window(230, 310)
     
    
    kim_y20_part1  = kim_org_sum - kim_diff_part1 
    kim_y20_part2  = kim_org_sum - kim_diff_part2 
    kim_y20_part3  = kim_org_sum - kim_diff_part3 
    
    
    kim_diff_part3_v1 = kim_diff_part3 * model_window(110, 130,'vertical')
    kim_diff_part3_v2 = kim_diff_part3 * model_window(124, 140,'vertical')
    
    kim_y20_part3_v1  = kim_org_sum - kim_diff_part3_v1 
    kim_y20_part3_v2  = kim_org_sum - kim_diff_part3_v2 
    
    
    kim_diff_part2_v1 = kim_diff_part2 * model_window(110, 129,'vertical')
    # kim_diff_part2_v2 = kim_diff_part2 * model_window(124, 140,'vertical')
    
    
    kim_y10_part2_v1  = kim_org_sum - kim_diff_part2_v1 
    # kim_y10_part2_v2  = kim_org_sum - kim_diff_part2_v2 
    
 
    # plot_test_model(kim_org_sum)
    
    if year == 50:
        hmin = np.min(anomaly_full)
        hmax = np.max(anomaly_full)
    
    # plt.figure()
    # plot_test_model(anomaly_full,hmin,hmax)
    # plt.figure()
        
    # plot_test_model(kim_diff_sum)
    if year == 50:
        hmin2 = np.min(kim_diff_part2_v1)
        hmax2 = np.max(kim_diff_part2_v1)
        
    # plt.figure()    
    # fig = plot_test_model(kim_diff_part2_v1,hmin2,hmax2)
    # plt.figure()
    # 
    # plt.title('year = '+str(year))
    # flout = '../png/92_kimberlina_corr_amp/diff_'+str(year)+'.png'
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    
    # if year == 50:
    #     hmin = np.min(kim_y10_part2_v1) 
    #     hmax = np.max(kim_y10_part2_v1)
    # plot_test_model(kim_y10_part2_v1)
    # plt.title('year = '+str(year))
    
    
    # gt.writebin(kim_org_sum,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y0.dat')
    # gt.writebin(kim_ano_sum,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y'+str(year)+'.dat')
    
    # gt.writebin(kim_y20_part1,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y'+str(year)+'_p1.dat')
    # gt.writebin(kim_y20_part2,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y'+str(year)+'_p2.dat')
    # gt.writebin(kim_y20_part3,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y'+str(year)+'_p3.dat')
    
    # gt.writebin(kim_y10_part2_v1,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y'+str(year)+'_p2_v1.dat')
    # gt.writebin(kim_y10_part2_v2,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y'+str(year)+'_p2_v2.dat')
    
    # gt.writebin(kim_y20_part3_v1,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y'+str(year)+'_p3_v1.dat')
    # gt.writebin(kim_y20_part3_v2,'../input/94_kimberlina_v4/full_sum/sum_kim_model_y'+str(year)+'_p3_v2.dat')
    
    vp_max_diff.append(np.max(kim_diff_part2_v1))
    vp_mean_diff.append(np.mean(kim_diff_part2_v1))
    year_vp_diff.append(str(year))
    
    rs_kim_part2_v1_diff = np.reshape(kim_diff_part2_v1,(nx*nz))
    
    
    vp_extract_list = [(index, value) for index, value in enumerate(rs_kim_part2_v1_diff) if value > 0.05]
    
    
    # Select the indexes biggest anomaly in size
    # 
    vp_extract_idx    = [tup[0] for tup in vp_extract_list]
    
    # With a set of indexes extract values for every year
    
    vp_original = np.reshape(org_full,(nx*nz))[vp_extract_idx]

    vp_new = np.reshape(kim_y10_part2_v1,(nx*nz))[vp_extract_idx]
    percentage.append(np.mean(100-vp_new/vp_original*100))


    
    # vp_new_mean_full.append(procs.RMS_calculator(vp_new))
    # vp_original_mean_full.append(procs.RMS_calculator(vp_original))
    
    vp_new_min_full.append(np.min(vp_new))
    vp_original_min_full.append(np.min(vp_original))
    
    
    vp_extract_values = rs_kim_part2_v1_diff[vp_extract_idx]
    
    vp_extract_full.append(np.mean(vp_extract_values))
    vp_mediane.append(np.median(vp_extract_values))

    label.append(year)
    plt.plot(vp_new,'.')
    plt.legend(label)
    
    
plt.rcParams['font.size'] = 20
plt.figure(figsize=(7, 7))
plt.plot(year_vp_diff,percentage,'o-')
plt.xlabel('year')
plt.ylabel('vp mean change (%)')

plt.rcParams['font.size'] = 20
plt.figure(figsize=(7, 7))
plt.plot(year_vp_diff,np.array(vp_new_min_full)*1000,'o-')
plt.xlabel('year')
plt.ylabel('vp mean (m/s)')
    

file = '../output/94_kimberlina_v4/res_vp_value_new.csv'

df = pd.DataFrame({'year':year_vp_diff[::-1],
                   'mean_vp_diff':vp_extract_full[::-1],
                   'max_vp_diff':vp_max_diff[::-1],
                   'vp_percentage':percentage[::-1],
                   'vp_anomalies_min':vp_new_min_full[::-1],
                   'vp_original_min':vp_original_min_full[::-1]
                   })
# df.to_csv(file,index=None)


#%%

name = 'p2_v1'

for i in range(30,5,-5):
    test = gt.readbin('../input/94_kimberlina_v4/full_sum/medium/sum_kim_model_y0.dat', 151, 601)
    test10 = gt.readbin('../input/94_kimberlina_v4/full_sum/medium/sum_kim_model_y'+str(i)+'_'+name+'.dat', 151, 601)
    # test15 = gt.readbin('../input/90_kimberlina_mod_v3_high/full_sum/sum_kim_model_y15_p2_v1.dat', 151, 601)
    # test20 = gt.readbin('../input/90_kimberlina_mod_v3_high/full_sum/sum_kim_model_y20_p2_v1.dat', 151, 601)
    
    
    diff_test10 = test - test10
    
    # diff_test20 = test - test20
    if i == 30:
        hmin = np.min(diff_test10)
        hmax = np.max(diff_test10)
    
    fig  = plt.figure(figsize=(14, 7), facecolor="white")
    av   = plt.subplot(1, 1, 1)
    hfig = av.imshow(diff_test10, 
                      vmin=hmin, vmax=hmax, aspect='auto',
                       extent=[ax[0], ax[-1], az[-1], az[0]],
                      cmap='viridis')
    plt.title(str(i))
    
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig, format='%1.2f',label='m/s')
    fig.tight_layout()
    # flout = '../png/90_kimberlina_mod_v3_high/y'+str(i)+'_'+name+'_diff.png' 
    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    
    print(np.max(diff_test10))

# hmin = np.min(diff_test20)
# hmax = np.max(diff_test20)
# fig  = plt.figure(figsize=(12, 8), facecolor="white")
# av   = plt.subplot(1, 1, 1)
# hfig = av.imshow(diff_test20, 
#                   vmin=hmin, vmax=hmax, aspect='auto',extent=[ax[0], ax[-1], az[-1], az[0]],
#                   cmap='jet')
# plt.colorbar(hfig)
    


