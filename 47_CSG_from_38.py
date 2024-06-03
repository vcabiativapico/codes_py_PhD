#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 15:31:13 2023

@author: vcabiativapico
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter
from scipy import interpolate
# from scipy.interpolate import splrep, BSpline, interpn, RegularGridInterpolator
from scipy.signal import hilbert
import csv
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from wiggle.wiggle import wiggle
import tqdm
if __name__ == "__main__":
  
  
## Building simple vel and rho models to test modeling
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
    no = 251
   # no        = 2002
    do = dx
    fo = -(no-1)/2*do
    ao = fo + np.arange(no)*do
    at = ft + np.arange(nt)*dt
    az = fz + np.arange(nz)*dz
    ax = fx + np.arange(nx)*dx
## Add y dimension    
    fy = -500 
    ny = 21
    dy = 50
    ay = fy + np.arange(ny)*dy
    
    
def plot_shot_gathers(hmin, hmax, inp, flout):
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                     vmin=hmin, vmax=hmax, aspect='auto',
                     cmap='seismic')
    plt.colorbar(hfig, format='%2.2f')
    plt.rcParams['font.size'] = 16
    plt.xlabel('Offset (km)')
    plt.ylabel('Time (s)')
    fig.tight_layout()
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')
    return fig
    
    
def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr

## Read the results from demigration
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
""" READ AND PLOT SOURCE RECEIVER AND TRAVELTIME """

# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug141223_raytracing/015_marm_flat_v2_test_p0001_v2_test_bp_az_0_4992.csv'
# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/016_flat_2000ms_1188/016_flat_1188_v2.csv'
# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output2/018_2000ms_606/018_flat_606_v2.csv'
path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/results/depth_demig_output.csv'
# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2/binv/040_rt_binv_marm_sm_full.csv'
# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2/badj/040_rt_badj_marm_sm_full.csv'

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


indices = []
for i in range(len(src_x)):
    if str(src_x[i]) != 'nan' or -1476 < off_x[i] > 1476:
        indices.append(i)

# indices = []
# for i in range(len(src_x)):
#     if str(src_x[i]) != 'nan':
#         indices.append(i)
""" PLOT TRAVELTIMES OVER THE SHOTS """



csg_trace= np.zeros((nt,len(indices)))
csg_trace = np.zeros((nt,len(src_x)))
csg_trace_INT = np.zeros((nt,len(indices)))
shot_int = np.zeros((nt,476-126))

dec_off_x = off_x/1000
no = 251
dec_off_x[0]


shot_idx = np.arange(126,476+1)
dec_title = shot_idx*12






def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


sd_val = []
sd_idx = []
for k in range(len(src_x)):
    sd_val.append(find_nearest(dec_title,src_x[k])[0])
    print(src_x[k])
    sd_idx.append(find_nearest(dec_title,src_x[k])[1])


fd_val = []
fd_idx = []
for k in range(len(src_x)):
    fd_val.append(find_nearest(ao, dec_off_x[k])[0])
    fd_idx.append(find_nearest(ao, dec_off_x[k])[1])
    
    




d_dox = dec_off_x[0] - dec_off_x[1]
sd_dox = src_x[0] - src_x[1]

## READ ALL THE SHOTS 
# inp_hilb3 = np.zeros((351,nt, no),dtype = 'complex_')
inp_hilb3 = np.zeros((351,nt, no))





for i in range(351):
    txt = str(i+126)
    title = txt.zfill(3)
    # tr3 = '../output/out_141/t1_obs_000'+str(title)+'.dat'
    tr1 = '../output/45_marm_ano_v3/ano_114_perc_1801TL/t1_obs_000'+str(title)+'.dat'
    tr2 = '../output/45_marm_ano_v3/org_1801TL/t1_obs_000'+str(title)+'.dat'
    inp1 = -gt.readbin(tr1, no, nt).transpose()
    inp2 = -gt.readbin(tr2, no, nt).transpose()
    inp3 = inp2
    # inp3 = inp1
    # inp_hilb3 = np.zeros_like(inp3,dtype = 'complex_')  
    for j in range(no):
        # inp_hilb3[i][:,j] = hilbert(inp3[:,j]) 
        inp_hilb3[i][:,j] = inp3[:,j]
        
        
# inp_hilb3 = inp_hilb3.imag
# inp_hilb3 = inp_hilb3.real
# inp_hilb3 = inp3

## PLOT SHOTS   
hmin, hmax = -0.2,0.2
shot_num = 236
flout_gather = '../png/out_141/obs_'+str(126+shot_num)+'.png'
fig = plot_shot_gathers(hmin, hmax, inp_hilb3[shot_num], flout_gather)
fig = plt.plot(off_x[0]/1000,tt_inv[0]/1000,'ok')


## INTERPOLATION OF SHOTS AND RECEIVERS
rec_to_int = np.zeros((4,nt,4))
tr_INT     = np.zeros((4,nt,5)) 

for k,i in enumerate(indices): 
    
    print('given index is: ',fd_idx[i])
    # Index creation, we extract four values around the desired offset to interpolate
    if fd_idx[i] > 2:
        ind_tr_int = np.arange(fd_idx[i]-2,fd_idx[i]+2)
    else:
        ind_tr_int = [0, 1, 2, 3]
    print(ind_tr_int)
    print('offset is :',dec_off_x[i])
    
    # ind_shot_int = np.arange(sd_idx[i]-2,sd_idx[i]+2)
    if sd_idx[i] < 350:
        ind_shot_int = np.arange(sd_idx[i]-2,sd_idx[i]+2)
    else:  
        ind_shot_int = np.array([347, 348, 349, 350])
        
    # shot_to_int = inp_hilb3[ind_shot_int][:,ind_tr_int]
    
    # Interpolation on the receivers
    for j in range(4):
        
        rec_to_int[j][:,:] = inp_hilb3[ind_shot_int[j]][:,ind_tr_int]
        
        f = interpolate.RegularGridInterpolator((at,ao[ind_tr_int]), rec_to_int[j], method='linear',bounds_error=False, fill_value=None) 
        at_new = np.linspace(at[0], at[-1], nt)
        ao_new = np.linspace(dec_off_x[i]-d_dox*2,dec_off_x[i]+d_dox*2, 5)
        AT, AO = np.meshgrid(at_new, ao_new, indexing='ij')
        tr_INT[j][:,:] = f((AT,AO))
        rec_int = tr_INT[:,:,2]
    # Interpolation on the shots
    f = interpolate.RegularGridInterpolator((at,(ind_shot_int+126)*12), rec_int.T, method='linear',bounds_error=False, fill_value=None) 
    at_new = np.linspace(at[0], at[-1], nt)
    src_new = np.linspace(src_x[i]-sd_dox*2, src_x[i]+sd_dox*2, 5)
    AT, SRC = np.meshgrid(at_new, src_new, indexing='ij')
    src_INT = f((AT,SRC))
    csg_trace[:,i] = src_INT[:,2] 

    
# flout = '../input/csg_raytracing_marm_sm.dat'
# gt.writebin(csg_trace,flout)




'''Tracé de rais analytique'''   


h1     = dz * 98
h2     = dz * 101+dz


v1     = 2.00

t0     = 2*h1 / v1
t1     = np.sqrt(t0**2 + (ao/v1)**2)

ao_conv  = len(ao) # Read the axes to keep same size
at_conv  = len(at)
 

inp_x    = np.zeros((at_conv,ao_conv)) # initilize the matrix with zeros


for i in range(no):
    n = np.round(t1[i]/dt)
    n = n.astype(int)    # Find the index and convert to integer
    inp_x[n,i]   = ((n+1)*dt - t1[i]) / dt 
    inp_x[n+1,i] = (t1[i]- n*dt) / dt
    # n_p1 = np.round(t2[i]/dt)
    # n_p1 = n_p1.astype(int)
    # inp_x[n_p1,i] = ((n+1)*dt - t1[i]) / dt
    # inp_x[n_p1+1,i] = (t1[i]- n*dt) / dt
    

# hmax = np.max(np.abs(inp_w))/2
hmax = 0.01
hmin = -hmax

fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
hfig = av.imshow(inp_hilb3[shot_num], extent=[ao[0], ao[-1], at[-1], at[0]],
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='seismic')
av.plot(ao,t1-0.012,'k')
av.plot(-off_x[0]/1000,tt_inv[0]/1000,'og')
plt.colorbar(hfig, format='%2.2f')
plt.rcParams['font.size'] = 16
plt.xlabel('Offset (km)')
plt.ylabel('Time (s)')
# fig.tight_layout()
# flout = '../input/out_141/csg_raytracing_modeling_2_0.png'
# print("Export to file:", flout)
# fig.savefig(flout, bbox_inches='tight')



fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
# hfig = av.imshow(inp_hilb3[shot_num], extent=[ao[0], ao[-1], at[-1], at[0]],
#                  vmin=hmin, vmax=hmax, aspect='auto',
#                  cmap='seismic')
wiggle(inp_hilb3[shot_num][:,::2],tt=at,xx=ao[::2])
# plt.colorbar(hfig, format='%2.2f')
plt.rcParams['font.size'] = 16
plt.xlabel('Offset (km)')
plt.ylabel('Time (s)')
fig.tight_layout()


#%%

h1     = dz * (50+0.5-1)*2

# h1 = (606-12)/1000*2

v1     = 2.00

t0     = h1 / v1
t1     = np.sqrt(t0**2 + (off_x/1000/v1)**2)

ao_conv  = len(ao) # Read the axes to keep same size
at_conv  = len(at)
 


'''Plots for comparsion'''
## PLOT THE RAYTRACING TRAVELTIMES OVERLAYING COMMON SPOT GATHER TRACES
hmin = -0.1
hmax = 0.1
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1)
# hfig = av.imshow(plot_rec_int, extent=[0, len(indices), at[-1], at[0]],
#                  vmin=hmin, vmax=hmax, aspect='auto',
#                  cmap='seismic')
hfig = av.imshow(csg_trace, extent=[0, len(src_x), at[-1]-ft, at[0]-ft],
                 vmin=hmin, vmax=hmax, aspect='auto',
                 cmap='seismic')   
av.plot(csg_trace[:,62]*80 + 62,at-ft,'k')
# plt.plot(np.arange(len(src_x)),tt_inv[:],'-k',markersize=3)
# plt.plot(np.arange(len(src_x)),t1,'g')
av.xaxis.set_ticks(np.arange(len(indices))) 
av.xaxis.set_ticklabels(np.rint(off_x[:]).astype(int))
plt.colorbar(hfig, format='%2.2f')
xticks = plt.gca().xaxis.get_major_ticks()
for i in range(len(xticks)):
    if i % 10 != 0:
        xticks[i].set_visible(False)
# plt.legend(['raytracing','analytique'])
plt.rcParams['font.size'] = 16
plt.title('Common spot gather Raytracing and Modelling')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')     



## PLOT A WIGGLE OVERLAY
fig = plt.figure(figsize=(8, 8), facecolor="white")
## First and main plot 
av = plt.subplot2grid((5, 1), (0, 0),rowspan=4)

wiggle(csg_trace[:,::3],tt=at-ft,xx=np.arange(len(indices))[::3])

# plt.plot(np.arange(len(indices)),tt_inv[indices],'-r',markersize=3)

# plt.plot(np.arange(len(indices))[::2],t1[::2],'r')

## Define the tick axis
av.xaxis.set_ticks(np.arange(len(indices))) 
av.xaxis.set_ticklabels(np.rint(off_x[indices]).astype(int))
xticks = plt.gca().xaxis.get_major_ticks()
for i in range(len(xticks)):
    if i % 10 != 0:
        xticks[i].set_visible(False)
## Labels
# plt.legend(['raytracing','modeling'])
plt.rcParams['font.size'] = 16
plt.title('Common spot gather Raytracing and Modelling')
plt.xlabel('Offset (m)')
plt.ylabel('Time (s)')       

## Secondary plot 
av1= plt.subplot2grid((5, 1), (4, 0),rowspan=1)
plt.plot(np.arange(len(indices)),off_x[indices],'-')
av1.xaxis.set_ticks(np.arange(len(indices))) 
av1.xaxis.set_ticklabels(np.array(indices))
# av1.xaxis.set_ticklabels(np.rint(off_x[indices]).astype(int))
xticks = plt.gca().xaxis.get_major_ticks()
for i in range(len(xticks)):
    if i % 20 != 0:
        xticks[i].set_visible(False)
plt.ylabel('Offset (m)')
plt.xlabel('Raytrace nb')  
fig.tight_layout()
flout = '../png/out_141/wiggle_csg_'+str(title)+'_interpolated.png'
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')




# plt.figure(figsize=(12, 10), facecolor="white")
# zero_ph_pick2 = np.array(read_results('../input/27_marm/29_pick_csg_flat.csv', 0))
# # zero_ph_pick2 = np.array(read_results('../input/27_marm/29_pick_csg.csv', 0))
# zero_ph_corr = zero_ph_pick2 - 100.11
# plt.plot(off_x,zero_ph_corr)
# plt.plot(off_x,tt_inv[indices],'-r',markersize=3)
# plt.legend(['modeling','raytracing'])
# plt.ylim(2000,0)
# plt.xlabel('offset (m)')
# plt.ylabel('time (ms)')


# decalage = zero_ph_corr - tt_inv[indices]
# plt.figure(figsize=(12, 10), facecolor="white")
# plt.plot(off_x,-decalage,'.')
# plt.xlabel('offset (m)')
# plt.ylabel('decalage (ms)')
# plt.title('Difference between analytical and numerical ray-tracing')
    
    
plt.figure(figsize=(7,4))
# plt.plot(off_x,zero_ph_corr,label='Modeling')
plt.plot(off_x,t1*1000,'-k',label='Analytical')
plt.plot(off_x[::2],tt_inv[indices][::2]*1000,'.r',markersize=3,label='Numerical')
plt.title('Event on the CSG')
plt.xlabel('offset (m)')
plt.ylabel('time (ms)')
plt.legend()
plt.ylim(860,580)

# decalage_ana_pick = zero_ph_corr - (t1*1000 )
# plt.figure(figsize=(12, 10), facecolor="white")
# plt.plot(off_x,-decalage_ana_pick,'.')
# plt.xlabel('offset (m)')
# plt.ylabel('decalage (ms)')
# plt.title('Décalage entre modélisation et analytique')
    


decalage_ana_tt = tt_inv[indices]*1000 - (t1*1000)
plt.figure(figsize=(12, 10), facecolor="white")
plt.plot(off_x,-decalage_ana_tt,'.')
plt.xlabel('offset (m)')
plt.ylabel('decalage (ms)')
plt.title('Difference between ray-tracing numerical and analytical solution')
    



z_max = np.zeros(101)
for i in range(101):
    # path_ray = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/016_flat_2000ms_1188/rays/opti_"+str(i)+"_ray.csv"
    # path_ray = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2/binv/rays/opti_"+str(i)+"_ray.csv"
    path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/047_simple_flat_marm_new_solver/depth_demig_out/out_data/rays/ray_'+str(i)+'.csv'

    ray_z = np.array(read_results(path_ray, 2))
    z_max[i] = np.max(ray_z)

plt.figure(figsize=(12, 10), facecolor="white")
plt.plot(off_x,-12-z_max,'.')
plt.xlabel('offset (m)')
plt.ylabel('difference to the actual 12m depth (m)')
plt.title('Difference between horizon and actual source and receiver position')


""" PLOT THE POSITIONS AND RECEIVERS OBTAINED """
colors = src_x[::2]
fig = plt.figure(figsize= (10,7))
plt.scatter(src_x[::2],src_y[::2],c=colors,marker='*',cmap='jet')
plt.scatter(rec_x[::2],rec_y[::2],c=colors,marker='v',cmap='jet')
plt.title('Map of the source and receiver positions')
plt.legend(['sources','receivers'])
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.ylim(-0.0001,0.0001)