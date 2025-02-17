

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 15:33:08 2024

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





#%%

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


for title in shot_nb: 
    
    no = 251
    # title,no = 201,251
    # nomax = 201
    # no = (nomax)+101
    # no = 302 # for 101 & 501
    # no = 401 # for 201 & 401
    # no = 403  # for 301
    # no = 403-abs(301-title)
    fo = -(no-1)/2*do
    ao = fo + np.arange(no)*do
    
    # tr1 = '../output/68_thick_marm_ano/full_org_thick/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/68_thick_marm_ano/full_ano_thick/t1_obs_000'+str(title).zfill(3)+'.dat'
    
  
    # tr1 = '../output/75_marm_sum_pert/full_org/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/75_marm_sum_pert/full_ano/t1_obs_000'+str(title).zfill(3)+'.dat'
   
   
    # tr1 = '../output/72_thick_marm_ano_born_mig_flat/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/72_thick_marm_ano_born_mig_flat/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    

    # tr1 = '../output/74_test_flat/new_vop8/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/74_test_flat/new_vop8/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/78_marm_sm8_thick_sum_pert/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'


    # tr1 = '../output/79_two_ano/full_one_ano_thick_sm8/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/79_two_ano/full_two_ano_thick_sm8/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/80_smooth_ano_sum_pert/t1_obs_000'+str(title).zfill(3)+'.dat'

    
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
 
    # flout = '/home/vcabiativapico/local/src/victor/out2dcourse/png/79_two_ano/shots/two_ano_shot_'+str(title*12)+'.png'
    
    
    flout = '/home/vcabiativapico/local/src/victor/out2dcourse/png/78_marm_sm8_thick_sum_pert/shots/full/'+str(title*12)+'.png'
    
    win1 = np.array(at[idx_tr_amp_max])-0.4
    win1 = win1*0+1.2
    # win2 = np.array(at[idx_tr_amp_max])+0.1
    
    # win1 = [1.2]*251
    win2 = [2.4]*251
    # if title%219 == 0:
    if title%1 == 0:
        plt.rcParams['font.size'] = 18
        fig = plt.figure(figsize=(8, 10), facecolor="white")
        av1 = plt.subplot2grid((5, 1), (0, 0),rowspan=4)
        hfig = av1.imshow(diff, extent=[ao[0], ao[-1], at[-1], at[0]],
                          vmin=hmin, vmax=hmax, aspect='auto',
                          cmap='seismic')
        av1.plot(ao,win1)
        # av1.plot(ao,win2)
        av1.plot(diff[:,dict_shot_nb[title]['idx_max_rms']]+ao[dict_shot_nb[title]['idx_max_rms'] ],at)
        # av1.plot(diff[:,dict_shot_nb[title]['idx_sel_max_rms']]+ao[dict_shot_nb[title]['idx_sel_max_rms'] ],at,'gray')
        av1.set_title('x= '+str(title*12))
        av1.set_ylabel('Time (s)')
        av1.set_ylim(at[-1],at[0])
        av = plt.subplot2grid((5, 1), (4, 0),rowspan=1)
        av.plot(ao,tr_rms, '-')
        plt.setp(av1.get_xticklabels(), visible=False)
        av.set_xlim(ao[0],ao[-1])
        av.yaxis.set_major_formatter(StrMethodFormatter("{x:1.0e}")) 
        av.set_xlabel('Offset (km)')
        # av.set_ylabel('Trace RMS')
        av.set_ylim(np.min(tr_rms),np.max(tr_rms))
        fig.tight_layout()
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
        


# plt.rcParams['font.size'] = 18
# fig = plt.figure(figsize=(8, 10), facecolor="white")
# av1 = plt.subplot2grid((5, 1), (0, 0),rowspan=4)
# hfig = av1.imshow(diff, extent=[ao[0], ao[-1], at[-1], at[0]],
#               vmin=np.min(diff), vmax=-np.min(diff), aspect='auto',
#               cmap='seismic')
# # av1.plot(diff[:,dict_shot_nb[title]['idx_max_rms']]+ao[dict_shot_nb[title]['idx_max_rms'] ],at)
# # av1.plot(diff[:,dict_shot_nb[title]['idx_sel_max_rms']]+ao[dict_shot_nb[title]['idx_sel_max_rms'] ],at,'gray')
# av1.set_title('x= '+str(title*12))
# av1.set_ylabel('Time (s)')
# av = plt.subplot2grid((5, 1), (4, 0),rowspan=1)
# av.plot(ao,tr_rms, '-')
# plt.setp(av1.get_xticklabels(), visible=False)
# av.set_xlim(ao[0],ao[-1])
# av.set_xlabel('Offset (km)')
# av.set_ylim(np.min(tr_rms),np.max(tr_rms))
# fig.tight_layout()
# print("Export to file:", flout)
# fig.savefig(flout, bbox_inches='tight')

        
#%%

    
path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/071_thick_ano_compare_FW/depth_demig_out/068_input_2024-10-10_15-22-34/results/depth_demig_output.csv'


# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/072_thick_ano_compare_FW_flat/depth_demig_out/deeper2_8_2024-10-14_11-37-56/results/depth_demig_output.csv'
    
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
    
    
rec_x_fw = src_x_fw + np.array(off_x_fw)*1000

total_rec_x_fw = []
for x in shot_nb:
    for i in ao: 
        total_rec_x_fw.append(x * dx * 1000 + i*1000)



total_src_x_fw = []
for x in list(dict_shot_nb.keys()): 
    total_src_x_fw.append([x]*no)
    
    

total_off = list(ao)*len(shot_nb)
total_off = np.reshape(total_off,(len(shot_nb),no))
  

total_rec_x_fw =  np.reshape(total_rec_x_fw,(len(src_x_fw),no))
total_src_x_fw =  np.reshape(total_src_x_fw,(len(src_x_fw),no))*12
total_rms = np.reshape(total_rms,(len(src_x_fw),no))


time_amp_max = []
for k, info in dict_shot_nb.items():
    tmp3 = info.get('time_amp_max')
    time_amp_max.append((tmp3))

db_time_amp_max = 20 * np.log(time_amp_max)
db_rms_x_fw = 20 * np.log(rms_x_fw)
db_total_rms = 20 * np.log(total_rms)
   


def max_in_plot(val_in_plot,seuil=1e-3):
    max_in_plot = np.zeros_like(val_in_plot)
        
    for i in range(len(val_in_plot)):
        for idx, x in enumerate(val_in_plot[i]): 
            if x > seuil:
                max_in_plot[i][idx] =  2
            else:
                max_in_plot[i][idx] =  0 
    return max_in_plot
                


total_rms_max = max_in_plot(total_rms,seuil=2e-4)     


def ridge(total_rms,total_src_x_fw,total_rec_x_fw,sigma=2,size=3):
    Z_smooth = gaussian_filter(total_rms, sigma)
    maxima = (Z_smooth == maximum_filter(Z_smooth, size))
    ridge_indices = np.where(maxima)
    ridge_x = total_src_x_fw[ridge_indices]
    ridge_y = total_rec_x_fw[ridge_indices]
    
    ridge_x = ridge_x[8:]
    ridge_y = ridge_y[8:]
    return ridge_x, ridge_y

ridge_x, ridge_y = ridge(total_rms,total_src_x_fw,total_off)


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

y_ax, x_ax =  max_in_rec(total_rms,total_src_x_fw,total_rec_x_fw)


if 1 == 0: 
    'Print the video of the amplitude map'
    total_blank = np.zeros_like(total_rms.T)
    
    # im = ax0.pcolormesh(total_src_x_fw, total_rec_x_fw, total_rms_max, cmap='viridis')
    for i in range(1,198):
        fig, (ax0) = plt.subplots(figsize=(10,10),nrows=1)
        plt.rcParams['font.size'] = 22
        im = ax0.pcolor(total_src_x_fw.T/1000, total_off.T, total_blank,\
                        vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='viridis')
       
        im = ax0.pcolor(total_src_x_fw.T[:,:i]/1000, total_off.T[:,:i], total_rms.T[:,:i],\
                        vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='viridis')
        # ax0.plot(ridge_x, ridge_y-ridge_x, 'r.')
        ax0.set_title('Max amplitude')
        ax0.set_xlabel('source_x')
        ax0.set_ylabel('offset_x')  
        # plt.legend()
        cbar= fig.colorbar(im, ax=ax0, format='%1.e',label='amp')
        plt.gca().set_aspect('equal')
        fig.tight_layout()
        flout2 = '../png/80_displays/maps/amplitude/'+str(i).zfill(3)+'.png'
        print("Export to file:", flout2)
        fig.savefig(flout2, bbox_inches='tight')


'''Amplitude map '''

fig, (ax0) = plt.subplots(figsize=(10,10),nrows=1)
plt.rcParams['font.size'] = 22
im = ax0.pcolor(total_src_x_fw.T/1000, total_off.T, total_rms.T,\
                vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='viridis')
ax0.scatter(src_x[::2]/1000,off_x[::2]/1000,marker='o', c='k',label='RT')
ax0.scatter(rec_x[::2]/1000,-off_x[::2]/1000,marker='o', c='k')
ax0.scatter(y_ax[::4]/1000 , (x_ax[::4]-y_ax[::4])/1000,c='red',label='WE')    
# ax0.plot(ridge_x/1000, ridge_y, 'ro')
ax0.set_title('Max amplitude')
ax0.set_xlabel('source x')
ax0.set_ylabel('offset x')  
# plt.legend()
cbar= fig.colorbar(im, ax=ax0, format='%1.e',label='amp')
plt.gca().set_aspect('equal')
fig.tight_layout()
flout2 = '../png/80_displays/maps/amplitude/'+str(i).zfill(3)+'.png'
print("Export to file:", flout2)
fig.savefig(flout2, bbox_inches='tight')
  

#%%
shot_nb = np.arange(152,350)



max_total = []
max_total_cc = []
for title in tqdm(shot_nb):
    
    
    # tr1 = '../output/68_thick_marm_ano/full_org_thick/t1_obs_000'+str(title)+'.dat'
    # tr2 = '../output/68_thick_marm_ano/full_ano_thick/t1_obs_000'+str(title)+'.dat'
    

    # tr1 = '../output/75_marm_sum_pert/full_org/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/75_marm_sum_pert/full_ano/t1_obs_000'+str(title).zfill(3)+'.dat'
   
    # tr1 = '../output/72_thick_marm_ano_born_mig_flat/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/72_thick_marm_ano_born_mig_flat/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    
    # tr1 = '../output/73_new_flat_sm/full_org/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/73_new_flat_sm/full_ano/t1_obs_000'+str(title).zfill(3)+'.dat'
   
    tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/78_marm_sm8_thick_sum_pert/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    # tr1 = '../output/79_two_ano/full_one_ano_thick_sm8/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/79_two_ano/full_two_ano_thick_sm8/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/80_smooth_ano_sum_pert/t1_obs_000'+str(title).zfill(3)+'.dat'

    
    inp1 = -gt.readbin(tr1, no, nt).transpose()
    inp2 = -gt.readbin(tr2, no, nt).transpose()
    
   
    window = signal.windows.tukey(no,alpha=0.4)
    
    inp1 = inp1 * window
    inp2 = inp2 * window  
    
    SLD_TS = []
    max_sld = []
    max_cross_corr = []
    
    for i in range(no):
        if time_amp_max[title-152][i] > 1:
            # win1 = float(np.array(time_amp_max[title-152][i]-0.4)*1000)
            # win2 = float(np.array(time_amp_max[title-152][i]+0.1)*1000)
            win1 = 1200
            win2 = 2400
            max_cross_corr.append(procs.max_cross_corr(inp1[:,i],inp2[:,i],win1=win1,win2=win2,thresh=None,si=dt,taper=30))
        else: 
        
            max_cross_corr.append(0)
        
        
        # SLD_TS.append(procs.sliding_TS(inp1[800:,i],inp2[800:,i], oplen=300, si=dt, taper=30))
        # max_sld.append(np.max(SLD_TS[-1]))
    max_total_cc.append(max_cross_corr)
    max_total.append(max_sld)


# y_val, x_val =  max_in_rec(np.array(max_total_cc),total_src_x_fw,total_rec_x_fw)

# ridge_x_cc, ridge_y_cc = ridge(np.array(max_total_cc),total_src_x_fw,total_rec_x_fw,sigma=4,size=6)


# max_total_cc_max = max_in_plot(max_total_cc,seuil=2)                


plt.imshow(max_total_cc)
plt.gca().set_aspect('equal')
# flnam2 = '../output/71_thick_marm_ano_born_mig/SLD_max_total_130.dat'
# max_total_in =read_results(flnam2, 0)


fig, (ax0) = plt.subplots(figsize=(10,10), nrows=1)
plt.rcParams['font.size'] = 22
# im = ax0.pcolor(total_src_x_fw, total_rec_x_fw, total_rms,\
#                 vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='viridis')
im = ax0.pcolor(total_src_x_fw/1000, total_off, max_total_cc, cmap='bwr',alpha=1)
fig.colorbar(im, ax=ax0,format='%1.1f',label='TS (ms)')
# ax0.plot(ridge_x, ridge_y, 'b*', label='WE')
# ax0.plot(ridge_x_cc, ridge_y_cc, 'g.', label='WE')
# ax0.scatter(rec_x[::4],-off_x[::4],marker='v', c='k',edgecolor='k',label='RT')
# ax0.scatter(src_x[::4],off_x[::4],marker='v', c='k',edgecolor='k')
# ax0.scatter(y_ax[::4] , x_ax[::4]-y_ax[::4],c='green',label='WE') 
# ax0.scatter(y_val[::4] , x_val[::4]-y_val[::4],c='green',label='WE')  
# ax0.scatter(src_x_fw[::3], rec_x_fw[::3], c=rms_x_fw[::3], cmap='jet', s=50, edgecolor='k')  
ax0.set_title('CC TS')
ax0.set_xlabel('source x')
ax0.set_ylabel('offset x') 
# plt.legend()
plt.gca().set_aspect('equal')
fig.tight_layout()
flout = '/home/vcabiativapico/local/src/victor/out2dcourse/png/78_marm_sm8_thick_sum_pert/maps/CC_TS.png'
# print("Export to file:", flout)
# fig.savefig(flout, bbox_inches='tight')
  

flnam2 = '../output/78_marm_sm8_thick_sum_pert/CC_TS_max_total.dat'
# df = pd.DataFrame(max_total)
# df.to_csv(flnam2, header=False, index=False)


# total_src_x_fw = total_src_x_fw.T
# total_src_x_fw[0] = total_src_x_fw[1] 
# total_src_x_fw[-1]  = total_src_x_fw[-2] 

# total_off = total_off.T
# total_off[0] = total_off[1]
# total_off[-1] = total_off[-2]


# time_amp_max = np.transpose(time_amp_max).T
# time_amp_max[0] = time_amp_max[1]
# time_amp_max[-1] = time_amp_max[-2]



fig, (ax0) = plt.subplots(figsize=(10,10), nrows=1)
plt.rcParams['font.size'] = 22
im = ax0.pcolormesh(total_src_x_fw/1000, total_off, time_amp_max,vmin=1, cmap='bwr',alpha=1)
fig.colorbar(im, ax=ax0,format='%1.1f',label='TS (ms)')
ax0.set_title('CC TS')
ax0.set_xlabel('source x')
ax0.set_ylabel('offset x') 
plt.gca().set_aspect('equal')
fig.tight_layout()

#%%
'''distance calculation'''

dist_cc  = []
dist_amp = []

for i in range(0,len(rec_x),4):
    # dist_cc.append(np.sqrt((src_x[i]-x_val[i])**2+(rec_x[i]-y_val[i])**2))
    dist_amp.append(np.sqrt((src_x[::-1][i]-x_ax[i])**2+(rec_x[::-1][i]-y_ax[i])**2))
    
y_ax[:124:-1]-src_x
# plt.plot(dist_cc[3:],ao)
# plt.plot(dist_amp[3:],ao)


#%%
option =1

if option ==0 :
    idx = 5
    
    title = ridge_x[idx] // 12
    j = ridge_y
    j = int(ridge_y[idx]/0.012+125)

else:
    idx_2 = 65
    title = int(x_ax[idx_2]//12)
    off = (x_ax[idx_2]-y_ax[idx_2])/12
    j = idx_2


delta = 15
idx_t = 60
off = off_x[idx_t]/1000
title = int(src_x[idx_t]//12)+delta
j = int(off_x[idx_t]//12)+125

tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
tr2 = '../output/78_marm_sm8_thick_sum_pert/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'

tr1 = '../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
tr2 = '../output/80_smooth_ano_sum_pert/t1_obs_000'+str(title).zfill(3)+'.dat'


inp1 = -gt.readbin(tr1, no, nt).transpose()
inp2 = -gt.readbin(tr2, no, nt).transpose()



# j= 700//12

plt.rcParams['font.size'] = 22
fig = plt.figure(figsize=(5, 10))
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[:, 0])
ax1.plot(inp1[:,j], at[:], label='org',linewidth=2)
ax1.plot(inp2[:,j], at[:], label='ano',linewidth=2)
ax1.set_title('Trace \n src = '+str(title*12) + ' m \noff = '+str(int(off*1000))+' m')
ax1.legend(loc='upper left')
ax1.set_xlim(-0.03, 0.03)
# ax1.set_ylim(1.4,1.8)
ax1.set_ylim(1.3,2)
ax1.set_ylabel('Time (s)')
ax1.set_xlabel('Amplitude')
plt.gca().invert_yaxis()
fig.tight_layout()

plt.rcParams['font.size'] = 22
fig = plt.figure(figsize=(5, 10))
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[:, 0])
ax1.plot(inp1[:,j]-inp2[:,j],at)
# ax1.set_xlim(-0.03, 0.03)
# ax1.set_ylim(1.4,1.8)
ax1.set_title('Diff \n src = '+str(title*12)  + ' m \nrec = '+str(int(off*1000))+' m')
# ax1.set_ylim(1.8,2.4)
ax1.set_ylabel('Time (s)')
ax1.set_xlabel('Amplitude')
plt.gca().invert_yaxis()
fig.tight_layout()



fig, (ax0) = plt.subplots(figsize=(10,10),nrows=1)
plt.rcParams['font.size'] = 22
# im = ax0.pcolormesh(total_src_x_fw, total_rec_x_fw, total_rms_max, cmap='viridis')
# im = ax0.pcolor(total_src_x_fw.T, total_rec_x_fw.T, total_rms.T,\
#                 vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='viridis')
im=    ax0.pcolormesh(total_src_x_fw/1000, total_off, max_total_cc, cmap='bwr',alpha=1)
# im = ax0.pcolormesh(total_src_x_fw, total_rec_x_fw, max_total_cc, cmap='bwr',alpha=1)    
# ax0.scatter(src_x[::4],rec_x[::4],marker='o', c='k',label='RT')
# ax0.scatter(rec_x[::4],src_x[::4],marker='o', c='k')
# ax0.scatter(y_ax[::4] , x_ax[::4],c='red',label='WE')   
# if option ==0:
#     ax0.plot(ridge_x[idx]/1000, ridge_y[idx], 'ko')
# else: 
#     ax0.plot(x_ax[idx_2]/1000, -off*0.012, 'ko')

ax0.plot(src_x[idx_t]/1000+delta*0.012,off_x[idx_t]/1000, 'ko')
# ax0.plot(title*0.012,off/1000*-1, 'ko')
ax0.set_title('Max TS')
ax0.set_xlabel('source_x')
ax0.set_ylabel('offset x')  
# plt.legend()
cbar= fig.colorbar(im, ax=ax0, format='%1.2f',label='TS (ms)')
plt.gca().set_aspect('equal')
fig.tight_layout()



fig, (ax0) = plt.subplots(figsize=(10,10),nrows=1)
im = ax0.pcolor(total_src_x_fw.T/1000, total_off.T, total_rms.T,\
                vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='viridis')

ax0.plot(src_x[idx_t]/1000+delta*0.012,off_x[idx_t]/1000, 'ko')
ax0.set_title('Max amplitude')
ax0.set_xlabel('source_x')
ax0.set_ylabel('offset x')  
# plt.legend()
cbar= fig.colorbar(im, ax=ax0, format='%1.e',label='amp')
plt.gca().set_aspect('equal')
fig.tight_layout()


fig, (ax0) = plt.subplots(figsize=(10,10),nrows=1)
plt.rcParams['font.size'] = 22
im = ax0.pcolor(total_src_x_fw.T/1000, total_off.T, total_rms.T,\
                vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='Greys')

im=    ax0.pcolormesh(total_src_x_fw/1000, total_off, max_total_cc, cmap='bwr',alpha=0.2)

# ax0.plot(ridge_x[idx]/1000, ridge_y[idx], 'ko')
# ax0.plot(title*0.012,off/1000*-1, 'ko')
# if option ==0:
#     ax0.plot(ridge_x[idx]/1000, ridge_y[idx], 'ko')
# else: 
#     ax0.plot(x_ax[idx_2]/1000, -off*0.012, 'ko')
ax0.plot(src_x[idx_t]/1000+delta*0.012,off_x[idx_t]/1000, 'ko')    
ax0.scatter(src_x[::4]/1000,off_x[::4]/1000,marker='o', c='r',label='RT')
ax0.scatter(rec_x[::4]/1000,-off_x[::4]/1000,marker='o', c='r')
ax0.plot(src_x[idx_t]/1000,off_x[idx_t]/1000)
ax0.set_title('Max TS')
ax0.set_xlabel('source_x')
ax0.set_ylabel('offset x')  
# plt.legend()
# cbar= fig.colorbar(im, ax=ax0, format='%1.2f',label='TS (ms)')
cbar= fig.colorbar(im, ax=ax0, format='%1.e',label='amp')
plt.gca().set_aspect('equal')
fig.tight_layout()

#%%

# fig, (ax0,ax1,ax2) = plt.subplots(figsize=(
#     30,10),nrows=1,ncols=3)

# plt.rcParams['font.size'] = 22
# im1 = ax0.pcolor(total_src_x_fw.T/1000, total_off.T, total_rms.T,\
#                 vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='viridis')
# ax0.set_title('Raytracing optimal positions')
# ax0.set_xlabel('Source x')
# ax0.set_ylabel('Offset x')
# # ax0.set_xlim(1.824,4.188)
# fig.colorbar(im1, ax=ax0, format='%1.e',label='amp')
# plt.gca().set_aspect('equal')
# # fig.tight_layout()


fig, (ax1,ax2) = plt.subplots(figsize=(
    20,10),nrows=1,ncols=2)

plt.suptitle('Attribute maps compared to ray-tracing values')
im1 = ax1.pcolor(total_src_x_fw.T/1000, total_off.T, total_rms.T,\
                vmin= np.min(total_rms),vmax=np.max(total_rms), cmap='viridis')
ax1.scatter(src_x[::4]/1000,off_x[::4]/1000,marker='o', c='r',label='RT')
ax1.scatter(rec_x[::4]/1000,-off_x[::4]/1000,marker='o', c='r')
ax1.set_title('RMS amplitude')
ax1.set_xlabel('source x')
ax1.set_ylabel('offset x')  
ax1.legend()
fig.colorbar(im1, ax=ax1, format='%1.e',label='amp')
# plt.gca().set_aspect('equal')
fig.tight_layout()


im2 = ax2.pcolor(total_src_x_fw/1000, total_off, max_total_cc, cmap='bwr')
ax2.scatter(src_x[::4]/1000,off_x[::4]/1000,marker='o', c='k',label='RT')
ax2.scatter(rec_x[::4]/1000,-off_x[::4]/1000,marker='o', c='k')
ax2.set_title('CC TS')
ax2.set_xlabel('source x')
ax2.set_ylabel('offset x') 
ax2.legend()
fig.colorbar(im2, ax=ax2,format='%1.1f',label='TS (ms)')
# plt.gca().set_aspect('equal')
fig.tight_layout()





