#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 22:27:52 2024

@author: vcabiativapico
"""


import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from spotfunk.res import procs
import csv
from scipy.ndimage import gaussian_filter,maximum_filter
from matplotlib.ticker import FormatStrFormatter, MaxNLocator, StrMethodFormatter
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import pandas as pd
from scipy import signal
from scipy.interpolate import CubicSpline
import seaborn as sns

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


def plot_trace(xmax, inp1, inp2, flout, tr):
    
    axi = np.zeros(np.size(tr))
    fig, (axi) = plt.subplots(nrows=1, ncols=np.size(tr),
                              sharey=True,
                              figsize=(8, 8),
                              facecolor="white")
    
    ratio = np.asarray(tr, dtype='f')
    for i in range(np.size(tr)):
        # xmin = np.min(inp1[:,tr[i]]) + np.min(inp1[:,tr[i]])/1.5
        # xmin = 1.0
        xmax = 1801*0.0012
        xmin = -xmax
        ratio[i] = np.max(inp2[:, tr[i]])/np.max(inp1[:, tr[i]])
        inp1[:, tr[i]] = (inp1[:, tr[i]]/np.max(inp1[:, tr[i]]))
        inp2[:, tr[i]] = (inp2[:, tr[i]]/np.max(inp2[:, tr[i]]))
        axi[i].plot(inp1[:, tr[i]], at, 'r')
    
        axi[i].plot(inp2[:, tr[i]], at, 'b--')
    
        axi[i].set_xlim(xmin, xmax)
        axi[i].set_ylim(2, ft)
    
        axi[i].xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
    
        axi[i].set_xlabel("Ratio = "+str(f'{ratio[i]:.2f}'))
        # plt.colorbar()
        fig.tight_layout()
    
    axi[0].set_ylabel('Time (s)')
    axi[0].legend(['obs', 'syn'], loc='upper left', shadow=True)
    
    # axi[0].legend(['Baseline','Monitor'],loc='upper left',shadow=True)
    fig.text(0.48, -0.01, "Amplitude")
    fig.text(0.48, 1, 'Comparison')
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')
    
    return ratio, inp1, inp2

# # #### TO PLOT SHOTS FROM MODELLING
def plot_shot_gathers(hmin, hmax, inp, flout):
    
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                     vmin=hmin, vmax=hmax, aspect='auto',
                     cmap='seismic')
    
    plt.title('x= '+str(title*12))
    plt.colorbar(hfig, format='%2.2f')
    plt.rcParams['font.size'] = 22
    plt.ylim(at[-1],ft)
    plt.xlabel('Offset (km)')
    plt.ylabel('Time (s)')
    fig.tight_layout()
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')

def read_results(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
        # attr = [x for x in attr if str(x) != 'nan']
    return attr


def first_arrival_detect(dataset,win_length):

    if win_length%2 != 1:
        win_length += 1

    function_tau = [0]*int(win_length)

    lower_integral = np.sum(np.array([dataset[l]*dataset[l] for l in range(0,win_length+int(win_length/2))]))

    max_tau = 0.5
    ind_max_tau = 0

    for k in range(win_length-int(win_length/2),len(dataset)-win_length):
        upper_integral= np.sum(np.array([dataset[l]*dataset[l] for l in range(k,k+win_length)]))
        lower_integral += dataset[k+int(win_length)]*dataset[k+int(win_length)]
        if lower_integral > 0:
            function_tau.append(upper_integral/lower_integral)
            if upper_integral/lower_integral > max_tau:
                max_tau = upper_integral/lower_integral 
                ind_max_tau = k
        else:
            function_tau.append(0)


    return function_tau,ind_max_tau

def find_first_index_greater_than(lst, target):
    for index, value in enumerate(lst):
        if value > target:
            return index
    return -1

def max_in_rec(total_rms,total_src_x_fw,total_rec_x_fw):
    total_rms_T = total_rms.T
    total_rms_obl_max = []
    total_rms_obl_argmax = []
    for i in range(len(total_rms.T)):
        total_rms_obl_max.append(np.max(total_rms_T[i]))
        total_rms_obl_argmax.append(np.argmax(total_rms_T[i]))
    
    # plt.imshow(total_rms_T)
    # plt.plot((np.array(total_rms_obl_argmax)),(np.arange(len(total_rms_T))),'b.')
    
    map_max = np.zeros_like(total_rms_T)
    
    for i in range(no):
        map_max[i][total_rms_obl_argmax[i]] = 1
    
    ones_y, ones_x = np.where(map_max == 1)
    
    y_ax= total_src_x_fw.T[ones_y,ones_x]
    x_ax = total_rec_x_fw.T[ones_y,ones_x]
    return y_ax, x_ax

def truncate_float(float_number, decimal_places):
    multiplier = 10 ** decimal_places
    return int(float_number * multiplier) / multiplier

#%%

path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/074_thick_ano_pert_sm8/depth_demig_out/074_thick_marm_org_sm_2024-11-04_14-09-32/results/depth_demig_output.csv'

src_x = np.array(read_results(path1,1))
src_y = np.array(read_results(path1,2))
src_z = np.array(read_results(path1,3))    
rec_x = np.array(read_results(path1,4))  
rec_y = np.array(read_results(path1,5))    
rec_z = np.array(read_results(path1,6))
spot_x = np.array(read_results(path1,7)) 
spot_y = np.array(read_results(path1,8))
spot_z= np.array(read_results(path1,9))
off_x  = np.array(read_results(path1,16))
tt_inv = np.array(read_results(path1,17))
   

xmax_tr = 0.3
nt =1801
# title = 501
# title = 401
# shot_nb = [211,221,231,241,251]
# shot_nb = np.arange(152,321)
# shot_nb = np.arange(291,450)
shot_nb = np.arange(152,350)

# shot_nb = [302]
# shot_nb  = [402,519]

attr = ['max_rms','idx_max_rms','off','sel_max_rms','idx_sel_max_rms','sel_off','time_amp_max']
dict_shot_nb =  {item: {x:{} for x in attr} for item in shot_nb}

total_rms = []
total_off = []
max_total_cc = []
max_total = []
fb_all =  []


for title in tqdm(shot_nb): 
# if title = 218:
    no = 251
    
    fo = -(no-1)/2*do
    ao = fo + np.arange(no)*do
    
    
    tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/80_smooth_ano_sum_pert/5percent/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    inp1 = -gt.readbin(tr1, no, nt).transpose()
    inp2 = -gt.readbin(tr2, no, nt).transpose()
    
    window = signal.windows.tukey(no,alpha=0.4)
    
    inp1 = inp1 * window
    inp2 = inp2 * window  
    
    diff = inp1 - inp2
    hmax = np.max(diff)*2.5
    hmin = -hmax
    
    
    
    idx_tr_amp_max = []
    tr_rms = [] 
    for i in range(no):
       tr_rms.append(procs.RMS_calculator(diff[:,i]))
       idx_tr_amp_max.append(np.argmax(diff[:1701,i]))  
    total_rms.append(tr_rms)
    
    dict_shot_nb[title]['time_amp_max'] = at[idx_tr_amp_max]
    dict_shot_nb[title]['max_rms']  = np.max(tr_rms)
    dict_shot_nb[title]['idx_max_rms']  = np.argmax(tr_rms)
    dict_shot_nb[title]['off'] = ao[dict_shot_nb[title]['idx_max_rms']]
    
    
    sel_tr_rms = []
    idx_tr_rms = []
    
    for index, x in enumerate(tr_rms):
        if x > np.max(tr_rms) - np.max(tr_rms)/4:
            sel_tr_rms.append(x)
            idx_tr_rms.append(index)
    
    dict_shot_nb[title]['sel_max_rms'] = sel_tr_rms
    dict_shot_nb[title]['idx_sel_max_rms'] = idx_tr_rms
    dict_shot_nb[title]['sel_off'] = ao[dict_shot_nb[title]['idx_sel_max_rms']]
    
    
    flout = '/home/vcabiativapico/local/src/victor/out2dcourse/png/78_marm_sm8_thick_sum_pert/shots/full/'+str(title*12)+'.png'
    

    
    fb_idx = []
    fb_t = []
    for i in range(no):
        perc = 0.05
        fb_idx.append(find_first_index_greater_than(diff[:,i], np.max(diff[:,i])*perc))
        fb_t = np.array(fb_idx)*dt
    
    
       
    window = signal.windows.tukey(no,alpha=0.4)
    
    inp1 = inp1 * window
    inp2 = inp2 * window  
    
    SLD_TS = []
    max_sld = []
    max_cross_corr = []
    
    
    win1_add = 0.25
    win2_add = 0.6
    # win1_array = ao_s*1000
    win1_array = (fb_t+win1_add)*1000
    
    # win2_array = np.array([at[-1]]*no)*1000
    win2_array = (fb_t + win2_add) * 1000
    
    for i in range(no):
        if at[idx_tr_amp_max][i] > 1:
            win1 = win1_array[i]
            win2 = win2_array[i]
            if win1 > at[-1]*1000-100: 
                win1 = at[-1]*1000-100
                
            if win2 > at[-1]*1000: 
                win2 = at[-1]*1000
            
            max_cross_corr.append(procs.max_cross_corr(inp1[:,i],inp2[:,i],win1=win1,win2=win2,thresh=None,si=dt,taper=25))
        else: 
            max_cross_corr.append(0)
    max_total_cc.append(max_cross_corr)
    max_total.append(max_sld)    
    
    fb_all.append(fb_t)      
    if title%250 == 0:
        hmin= np.min(diff)
        hmax= -np.min(diff)
        plt.rcParams['font.size'] = 27
        fig = plt.figure(figsize=(10, 12), facecolor="white")
        av1 = plt.subplot2grid((6, 1), (0, 0),rowspan=4)
        hfig = av1.imshow(diff, extent=[ao[0], ao[-1], at[-1], at[0]],
                          vmin=hmin, vmax=hmax, aspect='auto',
                          cmap='seismic')
        # av1.plot(ao,win1_array/1000,'yellowgreen',linewidth=3)
        # av1.plot(ao,win2_array/1000)
        av1.plot(diff[:,dict_shot_nb[title]['idx_max_rms']]+ao[dict_shot_nb[title]['idx_max_rms']],at)
        # av1.plot(diff[:,dict_shot_nb[title]['idx_sel_max_rms']]+ao[dict_shot_nb[title]['idx_sel_max_rms'] ],at,'gray')
        av1.set_title('x= '+str(title*12))
        av1.set_ylabel('Time (s)')
        av1.set_ylim(at[-1],at[0])
        
        av = plt.subplot2grid((6, 1), (4, 0),rowspan=1)
        av.plot(ao,tr_rms, '-')
        plt.setp(av1.get_xticklabels(), visible=False)
        av.set_xlim(ao[0],ao[-1])
        av.yaxis.set_major_formatter(StrMethodFormatter("{x:1.0e}")) 
        
        av.set_ylabel('Trace RMS')
        av.set_ylim(np.min(tr_rms),np.max(tr_rms))
        
        av2 = plt.subplot2grid((6, 1), (5, 0),rowspan=1)
        av2.plot(ao,max_cross_corr)
        plt.setp(av.get_xticklabels(), visible=False)
        av2.set_xlim(ao[0],ao[-1])
        av2.set_ylabel('TS (ms)')
        av2.set_xlabel('Offset (km)')

src_x_fw = np.array(list(dict_shot_nb.keys()))*12


off_x_fw = []
rms_x_fw = []
idx_x_fw = []

for person, info in dict_shot_nb.items():
    tmp = info.get("off")  # Safely get age from nested dictionary
    tmp2 = info.get("max_rms")
    tmp3 = info.get("idx_max_rms")
    
    off_x_fw.append((tmp))
    rms_x_fw.append((tmp2))
    idx_x_fw.append((tmp3))
    

total_src_x_fw = []
for x in list(dict_shot_nb.keys()): 
    total_src_x_fw.append([x]*no)
    
    

total_off = list(ao)*len(shot_nb)
total_off = np.reshape(total_off,(len(shot_nb),no))
      

total_src_x_fw =  np.reshape(total_src_x_fw,(len(src_x_fw),no))*12
total_rms = np.reshape(total_rms,(len(src_x_fw),no))
    


y_ax, x_ax =  max_in_rec(total_rms,total_src_x_fw,total_off)

max_total_cc = np.array(max_total_cc)
    


  
#%%


delta = 0
idx_t = 1
off = 0.6

for i in range(0,100,100):
    source = 2.400 + i/1000
    
    # off= off_x[20]/1000
    # source = src_x[20]/1000
    
    title = int(source*1000/12)+delta
    
    j = int(off * 1000 // 12 + 125)
    fb  = np.array(fb_all[title-shot_nb[0]][j]) + win1_add
    fb2 = np.array(fb_all[title-shot_nb[0]][j]) + win2_add
    
    tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/78_marm_sm8_thick_sum_pert/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/80_smooth_ano_sum_pert/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/80_smooth_ano_sum_pert/5percent/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    
    inp1 = -gt.readbin(tr1, no, nt).transpose()
    inp2 = -gt.readbin(tr2, no, nt).transpose()
    
    
    CC_TS = max_total_cc[title-shot_nb[0]][j]
    truncate_CC_TS = truncate_float(CC_TS,2)
    
    
    # %matplotlib qt5
    plt.rcParams['font.size'] = 22
    fig = plt.figure(figsize=(5, 10))
    gs = GridSpec(1, 1, figure=fig)
    ax1 = fig.add_subplot(gs[:, 0])
    ax1.plot(inp1[:,j], at[:], label='org',linewidth=2)
    ax1.plot(inp2[:,j], at[:], label='ano',linewidth=2)
    ax1.set_title('Trace \n src = '+str(title*12) + ' m \noff = '+str(int(off*1000))+' m\nTS = '+str(truncate_CC_TS)+' ms')
    ax1.legend(loc='upper left')
    ax1.set_xlim(-0.03, 0.03)
    ax1.axhline(fb,c='r')
    ax1.axhline(fb2,c='r')
    ax1.set_ylim(1.3,fb+0.8)
    ax1.set_ylabel('Time (s)')
    ax1.set_xlabel('Amplitude')
    ax1.grid()
    plt.gca().invert_yaxis()
    fig.tight_layout()
    flout = '../png/80_displays/trace_points/'+str(title*12)+'_off'+str(int(off*1000))+'.png'
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')
    
    
    
    print(CC_TS)
    
    # %matplotlib inline
    

    
    palette = sns.color_palette("viridis",as_cmap=True)
    palette2 = sns.color_palette("Greys", as_cmap=True)
    palette3 = sns.color_palette("bwr", as_cmap=True)
    
    points_x = [2.4,3.223,3.4]
    points_y = [0.908,1.031,0]
    word = str([1,2,3])
    
    if 1==1:
        plt.rcParams['font.size'] = 26
        fig, (ax0) = plt.subplots(figsize=(10,14),nrows=1)
        # levels = MaxNLocator(nbins=15).tick_values(np.min(max_total_cc), np.max(max_total_cc))
        # im = ax0.contourf(total_src_x_fw/1000,
        #                   total_off, max_total_cc, levels=levels,
        #                   cmap=palette)

        im = ax0.pcolormesh(total_src_x_fw/1000, total_off, -max_total_cc,
                              vmin= -np.min(-max_total_cc),vmax=np.min(-max_total_cc), 
                              cmap=palette3,alpha=1)
        # ax0.plot(source,off, 'ko')
        ax0.scatter(src_x[::4]/1000,off_x[::4]/1000,marker='o', c='r',label='RT')
        ax0.scatter(rec_x[::4]/1000,-off_x[::4]/1000,marker='o', c='r')
        ax0.scatter(points_x,points_y,c='white',marker='o',s=180,edgecolors='black',label='PP')
        # ax0.scatter(3.865,0.843,c='orange',marker='o',s=180,edgecolors='black',label='PC')
        ax0.set_title('CC time-shift')
        ax0.set_xlabel('Source x (km)')
        # ax0.set_ylabel('Offset x')  
        ax0.legend()
        cbar= fig.colorbar(im, ax=ax0, format='%1.1f',label='TS (ms)',orientation='horizontal')
        plt.gca().set_aspect('equal')
        fig.tight_layout()
        flout = '../png/80_displays/trace_points/TS_MAP_'+str(title*12)+'_off'+str(int(off*1000))+'.png'
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
        
        
        
        fig, (ax0) = plt.subplots(figsize=(10,14),nrows=1)
        
        im = ax0.pcolor(total_src_x_fw.T/1000, total_off.T, total_rms.T,\
                        vmin= np.min(total_rms),vmax=np.max(total_rms), cmap=palette)
        ax0.scatter(src_x[::4]/1000,off_x[::4]/1000,marker='o', c='r',label='RT')
        ax0.scatter(rec_x[::4]/1000,-off_x[::4]/1000,marker='o', c='r')    
        # ax0.plot(source,off, 'ko')
        ax0.scatter(points_x,points_y,c='white',marker='o',s=180,edgecolors='black',label='PP')
        # ax0.scatter(3.865,0.843,c='orange',marker='o',s=180,edgecolors='black',label='PC')
        ax0.set_title('Max amplitude')
        ax0.set_xlabel('Source x (km)')
        ax0.set_ylabel('Offset x (km)')  
        ax0.legend()
        cbar= fig.colorbar(im, ax=ax0, format='%1.e',orientation='horizontal')
        cbar.set_label('Amplitude')
        plt.gca().set_aspect('equal')
        fig.tight_layout()
        flout = '../png/80_displays/trace_points/AMP_MAP_'+str(title*12)+'_off'+str(int(off*1000))+'.png'
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
        
        
        
        fig, (ax0) = plt.subplots(figsize=(8,14),nrows=1)
           
        im = ax0.pcolor(total_src_x_fw.T/1000, total_off.T, total_rms.T,\
                        vmin= np.min(total_rms),vmax=np.max(total_rms), cmap=palette2)
        im2 = ax0.pcolormesh(total_src_x_fw/1000, total_off, -max_total_cc,
                              vmin= -np.min(-max_total_cc),vmax=np.min(-max_total_cc), 
                              cmap=palette3,alpha=0.5) 
        ax0.scatter(src_x[::4]/1000,off_x[::4]/1000,marker='o', c='r',label='RT')
        ax0.scatter(rec_x[::4]/1000,-off_x[::4]/1000,marker='o', c='r')
        ax0.scatter(points_x,points_y,c='white',marker='o',s=180,edgecolors='black',label='PP')
        # ax0.scatter(3.865,0.843,c='orange',marker='o',s=180,edgecolors='black',label='PC')
        ax0.set_title('Overlay amplitude and TS')
        ax0.set_xlabel('Source x (km)')
        # ax0.set_ylabel('offset x')  
        cbar= fig.colorbar(im, ax=ax0, format='%1.e',label='Amplitude',orientation='horizontal',pad=0.01)
        cbar= fig.colorbar(im2, ax=ax0, format='%1.f',label='TS (ms)',orientation='horizontal',pad=0.1)
        ax0.legend()
        plt.gca().set_aspect('equal')
        fig.tight_layout()



# print(max_total_cc[title-shot_nb[0]][j-125])
#%%

plt.rcParams['font.size'] = 23
fig, (ax1,ax2,ax0) = plt.subplots(figsize=(
    30,10),nrows=1,ncols=3)

plt.suptitle('Attribute maps compared to ray-tracing values')

im1 = ax1.pcolor(total_src_x_fw.T/1000, total_off.T, total_rms.T,\
                vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='viridis')
ax1.scatter(src_x[::4]/1000,off_x[::4]/1000,marker='o', c='r',label='RT')
ax1.scatter(rec_x[::4]/1000,-off_x[::4]/1000,marker='o', c='r')
ax1.set_title('RMS amplitude')
ax1.set_xlabel('Source x')
ax1.set_ylabel('Offset x')  
ax1.legend()
fig.colorbar(im1, ax=ax1, format='%1.e',label='Amplitude')
plt.gca().set_aspect('equal')
fig.tight_layout()

# im2 = ax2.pcolor(total_src_x_fw/1000, total_off, max_total_cc,
#                  vmin= -np.max(max_total_cc),vmax=np.max(max_total_cc), cmap='jet')
im2 = ax2.pcolor(total_src_x_fw/1000, total_off, max_total_cc,
                 vmin= 0,vmax=np.max(max_total_cc), cmap='viridis')
ax2.scatter(src_x[::4]/1000,off_x[::4]/1000,marker='o', c='r',label='RT')
ax2.scatter(rec_x[::4]/1000,-off_x[::4]/1000,marker='o', c='r')
ax2.set_title('CC Time-shift')
ax2.set_xlabel('Source x')
# ax2.set_ylabel('offset x') 
ax2.legend()
fig.colorbar(im2, ax=ax2,format='%1.2f',label='TS (ms)')
plt.gca().set_aspect('equal')
fig.tight_layout()


im = ax0.pcolor(total_src_x_fw.T/1000, total_off.T, total_rms.T,\
                vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='Greys')
im2 = ax0.pcolormesh(total_src_x_fw/1000, total_off, max_total_cc, cmap='viridis',alpha=0.5)  
ax0.scatter(src_x[::4]/1000,off_x[::4]/1000,marker='o', c='r',label='RT')
ax0.scatter(rec_x[::4]/1000,-off_x[::4]/1000,marker='o', c='r')
ax0.set_title('Overlay amplitude and TS')
ax0.set_xlabel('Source x')
# ax0.set_ylabel('offset x')  
cbar= fig.colorbar(im, ax=ax0, format='%1.e',label='Amplitude')
cbar= fig.colorbar(im2, ax=ax0, format='%1.f',label='TS (ms)')
ax0.legend()
plt.gca().set_aspect('equal')
fig.tight_layout()
