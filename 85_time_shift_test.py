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

title = 250
no = 251

fo = -(no-1)/2*do
ao = fo + np.arange(no)*do


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

win1 = np.array(at[idx_tr_amp_max])
# win2 = np.array(at[idx_tr_amp_max])+0.1


%matplotlib qt5
plt.rcParams['font.size'] = 25
fig = plt.figure(figsize=(10, 10), facecolor="white")
av1 = plt.subplot2grid((5, 1), (0, 0),rowspan=4)
hfig = av1.imshow(diff*5, extent=[ao[0], ao[-1], at[-1], at[0]],
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
av.yaxis.set_major_formatter(StrMethodFormatter("{x:1.0e}")) 
av.set_xlabel('Offset (km)')
av.set_ylabel('Trace RMS')
av.set_ylim(np.min(tr_rms),np.max(tr_rms))
fig.tight_layout()
# print("Export to file:", flout)
# fig.savefig(flout, bbox_inches='tight')



pick = plt.ginput(n=-1,timeout=20)
plt.close()
pick_x, pick_y = np.array(pick).T

%matplotlib inline

#%%

cs = CubicSpline(pick_x, pick_y)

ao_s = cs(ao)
# pick = np.asarray(pick).astype(int)



plt.rcParams['font.size'] = 25
fig = plt.figure(figsize=(10, 10), facecolor="white")
av1 = plt.subplot2grid((5, 1), (0, 0),rowspan=4)
hfig = av1.imshow(diff*5, extent=[ao[0], ao[-1], at[-1], at[0]],
                  vmin=hmin, vmax=hmax, aspect='auto',
                  cmap='seismic')
# av1.plot(ao,win1)
# av1.plot(ao,win2)
av1.plot(diff[:,dict_shot_nb[title]['idx_max_rms']]+ao[dict_shot_nb[title]['idx_max_rms'] ],at)
# av1.plot(diff[:,dict_shot_nb[title]['idx_sel_max_rms']]+ao[dict_shot_nb[title]['idx_sel_max_rms'] ],at,'gray')
av1.set_title('x= '+str(title*12))
av1.set_ylabel('Time (s)')
av = plt.subplot2grid((5, 1), (4, 0),rowspan=1)
av.plot(ao,tr_rms, '-')
plt.setp(av1.get_xticklabels(), visible=False)
av.set_xlim(ao[0],ao[-1])
av.yaxis.set_major_formatter(StrMethodFormatter("{x:1.0e}")) 
av.set_xlabel('Offset (km)')
av.set_ylabel('Trace RMS')
av.set_ylim(np.min(tr_rms),np.max(tr_rms))
fig.tight_layout()
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')





inp1 = -gt.readbin(tr1, no, nt).transpose()
inp2 = -gt.readbin(tr2, no, nt).transpose()

   
window = signal.windows.tukey(no,alpha=0.4)

inp1 = inp1 * window
inp2 = inp2 * window  

SLD_TS = []
max_sld = []
max_cross_corr = []


win1_array = ao_s*1000

win2_array = np.array([at[-1]]*no)*1000


for i in range(no):
    if at[idx_tr_amp_max][i] > 1:
        win1 = win1_array[i]
        win2 = win2_array[i]
        if win1 > at[-1]*1000-100: 
            win1 = at[-1]*1000-100
            
        if win2 > at[-1]*1000: 
            win2 = at[-1]*1000
            
        
        max_cross_corr.append(procs.max_cross_corr(inp1[:,i],inp2[:,i],win1=win1,win2=win2,thresh=None,si=dt,taper=30))
    else: 
        max_cross_corr.append(0)
        
      

hmin= np.min(diff)
hmax= -np.min(diff)
plt.rcParams['font.size'] = 25
fig = plt.figure(figsize=(10, 12), facecolor="white")
av1 = plt.subplot2grid((6, 1), (0, 0),rowspan=4)
hfig = av1.imshow(diff, extent=[ao[0], ao[-1], at[-1], at[0]],
                  vmin=hmin, vmax=hmax, aspect='auto',
                  cmap='seismic')
av1.plot(ao,win1_array/1000,'yellowgreen',linewidth=3)
av1.plot(ao,win2_array/1000)
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
av.set_xlabel('Offset (km)')
av.set_ylabel('Trace RMS')
av.set_ylim(np.min(tr_rms),np.max(tr_rms))

av2 = plt.subplot2grid((6, 1), (5, 0),rowspan=1)
av2.plot(ao,max_cross_corr)


#%%

diff_norm = diff/np.min(diff)

Sismique_shotpoint= np.copy(diff_norm).T

first_arrival_function = []

first_arrival_index = []



for k in range(len(Sismique_shotpoint)):
    
    function_tau,ind_max_tau = first_arrival_detect(Sismique_shotpoint[k],100)

    first_arrival_function.append(function_tau)

    first_arrival_index.append(ind_max_tau)
    
first_break = np.array(first_arrival_index) 
    

hmin = np.min(diff_norm)

hmax = -hmin
fig = plt.figure(figsize=(10, 12), facecolor="white")
av1 = plt.subplot2grid((6, 1), (0, 0),rowspan=4)
hfig = av1.imshow(diff_norm, extent=[ao[0], ao[-1], at[-1], at[0]],
                  vmin=hmin, vmax=hmax, aspect='auto',
                  cmap='seismic')
plt.colorbar(hfig)
av1.plot(ao,np.array(first_arrival_index)*dt)
