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
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import pandas as pd

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
shot_nb = np.arange(152,321)

# shot_nb = [302]
# shot_nb  = [402,519]

attr = ['max_rms','idx_max_rms','off','sel_max_rms','idx_sel_max_rms','sel_off']
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
    
    tr1 = '../output/68_thick_marm_ano/full_org_thick/t1_obs_000'+str(title)+'.dat'
    tr2 = '../output/68_thick_marm_ano/full_ano_thick/t1_obs_000'+str(title)+'.dat'
    
  
    # tr1 = '../output/75_marm_sum_pert/full_org/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/75_marm_sum_pert/full_ano/t1_obs_000'+str(title).zfill(3)+'.dat'
   
   
    # tr1 = '../output/72_thick_marm_ano_born_mig_flat/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/72_thick_marm_ano_born_mig_flat/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    

    # tr1 = '../output/74_test_flat/new_vop8/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/74_test_flat/new_vop8/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    
    inp1 = -gt.readbin(tr1, no, nt).transpose()
    inp2 = -gt.readbin(tr2, no, nt).transpose()
    
    
    
    diff = inp1 - inp2
    hmax = np.max(diff)*2.5
    hmin = -hmax

    
    tr_rms = [] 
    for i in range(no):
       tr_rms.append(procs.RMS_calculator(diff[:,i]))
    
    total_rms.append(tr_rms)
    
    
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
 

    
 
    
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(8, 10), facecolor="white")
    av1 = plt.subplot2grid((5, 1), (0, 0),rowspan=4)
    hfig = av1.imshow(diff, extent=[ao[0], ao[-1], at[-1], at[0]],
                      vmin=hmin, vmax=hmax, aspect='auto',
                      cmap='seismic')
    av1.plot(diff[:,dict_shot_nb[title]['idx_max_rms']]+ao[dict_shot_nb[title]['idx_max_rms'] ],at)
    # av1.plot(diff[:,dict_shot_nb[title]['idx_sel_max_rms']]+ao[dict_shot_nb[title]['idx_sel_max_rms'] ],at,'gray')
    av1.set_title('x= '+str(title*12))
    av1.set_ylabel('Time (s)')
    av = plt.subplot2grid((5, 1), (4, 0),rowspan=1)
    av.plot(ao,tr_rms, '-')
    plt.setp(av1.get_xticklabels(), visible=False)
    av.set_xlim(ao[0],ao[-1])
    av.set_xlabel('Offset (km)')
    av.set_ylim(np.min(tr_rms),np.max(tr_rms))
    
    
#%%

    
path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/071_thick_ano_compare_FW/depth_demig_out/068_input_2024-10-10_15-22-34/results/depth_demig_output.csv'


# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/072_thick_ano_compare_FW_flat/depth_demig_out/deeper2_8_2024-10-14_11-37-56/results/depth_demig_output.csv'
    
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

total_off = list(ao)*len(src_x_fw)
  

total_rec_x_fw =  np.reshape(total_rec_x_fw,(len(src_x_fw),no))
total_src_x_fw =  np.reshape(total_src_x_fw,(len(src_x_fw),no))*12
total_rms = np.reshape(total_rms,(len(src_x_fw),no))
total_off = np.reshape(total_off,(len(src_x_fw),no))



fig, (ax0) = plt.subplots(figsize=(10,8),nrows=1)
im = ax0.pcolormesh(total_src_x_fw, total_rec_x_fw, total_rms, cmap='viridis')
fig.colorbar(im, ax=ax0)
ax0.set_title('RMS')
ax0.scatter(src_x[::3],rec_x[::3],marker='v', c='w',edgecolor='w')
ax0.scatter(src_x_fw[::3], rec_x_fw[::3], c=rms_x_fw[::3], cmap='jet', s=50, edgecolor='k')   
# ax0.set_ylim(4400,5700)
ax0.set_xlabel('source_x')
ax0.set_ylabel('receiver_x')  



fig, ( ax1) = plt.subplots(figsize=(10,8),nrows=1)
levels = MaxNLocator(nbins=15).tick_values(np.min(total_rms), np.max(total_rms))
# contours are *point* based plots, so convert our bound into point
# centers
cf = ax1.contourf(total_src_x_fw,
                  total_rec_x_fw, total_rms, levels=levels,
                  cmap='viridis')
ax1.scatter(src_x[::3],rec_x[::3],marker='v', c='w',edgecolor='w')
ax1.scatter(src_x_fw[::3], rec_x_fw[::3], c=rms_x_fw[::3], cmap='jet', s=50, edgecolor='k')   
fig.colorbar(cf, ax=ax1)
ax1.set_title('RMS with levels')
# ax1.set_ylim(4400,5700)
ax1.set_xlabel('source_x')
ax1.set_ylabel('receiver_x')  


plt.figure(figsize=(10,8))
plt.title('offset')
hfig = plt.pcolormesh(total_src_x_fw,total_rec_x_fw,total_off,vmin=-1.5,vmax=1.5,cmap='seismic') 
plt.scatter(src_x[::3],rec_x[::3],c=off_x[::3],vmin=-1500,vmax=1500, cmap='seismic', s=70,marker='o', edgecolor='k',label='rt')
plt.colorbar(hfig)
plt.xlabel('source_x')
plt.ylabel('receiver_x')  


# plt.figure(figsize=(10,8))
# plt.title('RMS')
# plt.scatter(src_x,rec_x, edgecolor='b')
# plt.scatter(src_x_fw, rec_x_fw, c=rms_x_fw, cmap='jet', s=50, edgecolor='k')   
# plt.colorbar()
# plt.xlabel('source_x')
# plt.ylabel('receiver_x')  

# plt.figure(figsize=(10,8))
# plt.title('Offset')
# plt.scatter(src_x,rec_x,c=off_x,vmin=-1500,vmax=1500, cmap='seismic', s=70,marker='v', edgecolor='k',label='rt')
# plt.scatter(src_x_fw, rec_x_fw, c=off_x_fw,vmin=-1.5,vmax=1.5, cmap='seismic', s=50, edgecolor='k',label='fw')   
# plt.colorbar()
# plt.legend()
# plt.xlabel('source_x')
# plt.ylabel('receiver_x')  



# fl_sm = '../input/71_thick_marm_ano_born_mig/inp_mig_plus_bg_ano.dat'

# # fl_sm = '../input/72_thick_marm_ano_born_mig_flat/new_flat_org.dat'

# inp_sm = gt.readbin(fl_sm,nz,nx)
# hmin = np.min(inp_sm)
# hmax = np.max(inp_sm)
# fig = plt.figure(figsize=(10, 6), facecolor="white")
# av = plt.subplot(1, 1, 1)
# hfig1 = av.imshow(inp_sm, extent=[ax[0]*1000, ax[-1]*1000, az[-1]*1000, az[0]*1000],
#                   vmin=hmin, vmax=hmax, aspect='auto')
# plt.xlabel('Distance (km)')
# plt.ylabel('Depth (km)')
# plt.scatter(src_x[::10],np.zeros_like(src_x[::10]),c='yellow',marker='*')
# plt.scatter(rec_x[::10],np.zeros_like(rec_x[::10]),c='orange',marker='v')
# plt.scatter(src_x_fw[::10],np.zeros_like(src_x_fw[::10]),c='darkblue', marker='*', cmap='viridis', s=50)
# plt.scatter(rec_x_fw[::10],np.zeros_like(rec_x_fw[::10]), c='blue',marker='v', cmap='viridis', s=50)


#%%
shot_nb = np.arange(152,321)

max_total = []
for title in tqdm(shot_nb):
    
    
    # tr1 = '../output/68_thick_marm_ano/full_org_thick/t1_obs_000'+str(title)+'.dat'
    # tr2 = '../output/68_thick_marm_ano/full_ano_thick/t1_obs_000'+str(title)+'.dat'
    

    tr1 = '../output/75_marm_sum_pert/full_org/t1_obs_000'+str(title).zfill(3)+'.dat'
    tr2 = '../output/75_marm_sum_pert/full_ano/t1_obs_000'+str(title).zfill(3)+'.dat'
   
    # tr1 = '../output/72_thick_marm_ano_born_mig_flat/org_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/72_thick_marm_ano_born_mig_flat/ano_full/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    
    # tr1 = '../output/73_new_flat_sm/full_org/t1_obs_000'+str(title).zfill(3)+'.dat'
    # tr2 = '../output/73_new_flat_sm/full_ano/t1_obs_000'+str(title).zfill(3)+'.dat'
    
    
    inp1 = -gt.readbin(tr1, no, nt).transpose()
    inp2 = -gt.readbin(tr2, no, nt).transpose()
     
    
    SLD_TS = []
    max_sld = []
    for i in range(no):
        SLD_TS.append(procs.sliding_TS(inp1[500:,i],inp2[500:,i], oplen=300, si=dt, taper=30))
        max_sld.append(np.max(SLD_TS[-1]))
    max_total.append(max_sld)
    

max_total_rs = np.reshape(max_total,(len(shot_nb),no))
                          
plt.imshow(max_total_rs,vmax=10,vmin=0)

# flnam2 = '../output/71_thick_marm_ano_born_mig/SLD_max_total_130.dat'
# max_total_in =read_results(flnam2, 0)

fig, (ax0) = plt.subplots(figsize=(10,8),nrows=1)
im = ax0.pcolormesh(total_src_x_fw, total_rec_x_fw, max_total, vmin= 0,vmax=5, cmap='viridis')
fig.colorbar(im, ax=ax0)
ax0.set_title('SLD_TS')
ax0.set_xlabel('source_x')
ax0.set_ylabel('receiver_x')  


flnam2 = '../output/71_thick_marm_ano_born_mig/SLD_max_total.dat'
df = pd.DataFrame(max_total)
df.to_csv(flnam2, header=False, index=False)


plt.rcParams['font.size'] = 22
fig = plt.figure(figsize=(5, 10))
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[:, 0])
ax1.plot(np.array(SLD_TS)[i], at[500:], label='org',linewidth=2)
ax1.set_title('Traces at '+str(title*12) + ' m')
ax1.legend()
ax1.set_xlim(-1.5, 2)
ax1.set_ylim(2.0, ft-0.1)
ax1.set_ylabel('Time (s)')
ax1.set_xlabel('Amplitude')


plt.rcParams['font.size'] = 22
fig = plt.figure(figsize=(5, 10))
gs = GridSpec(1, 1, figure=fig)
ax1 = fig.add_subplot(gs[:, 0])
ax1.plot(inp1[500:,i], at[500:], label='org',linewidth=2)
ax1.plot(inp2[500:,i], at[500:], label='ano',linewidth=2)
ax1.set_title('Traces at '+str(title*12) + ' m')
ax1.legend()
ax1.set_xlim(-0.05, 0.05)
ax1.set_ylim(2.0, ft-0.1)
ax1.set_ylabel('Time (s)')
ax1.set_xlabel('Amplitude')

