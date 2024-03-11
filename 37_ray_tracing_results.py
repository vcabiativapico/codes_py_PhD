#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:51:08 2024

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
if __name__ == "__main__":
  
  
## Building simple vel and rho models to test modeling
    labelsize = 16
    nt = 1501
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
    attr = np.array(attr)
    attr = np.nan_to_num(attr)
    return attr


path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2/badj/040_rt_badj_marm_sm_full.csv'
path2 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2/binv/040_rt_binv_marm_sm_full.csv'

pick1 = '../input/40_marm_ano/badj_mig_pick_smooth.csv'
pick2 = '../input/40_marm_ano/binv_mig_pick_smooth.csv'

src_x_1  = read_results(path1,1)
src_y_1  = read_results(path1,2)
src_z_1  = read_results(path1,3)    
rec_x_1  = read_results(path1,4) 
rec_y_1  = read_results(path1,5)  
rec_z_1  = read_results(path1,6)
spot_x_1 = read_results(path1,7) 
spot_y_1 = read_results(path1,8)
spot_z_1 = read_results(path1,9)
off_x_1  = read_results(path1,16)
tt_inv_1 = read_results(path1,17)
    
src_x_2  = read_results(path2,1)
src_y_2  = read_results(path2,2)
src_z_2  = read_results(path2,3)    
rec_x_2  = read_results(path2,4) 
rec_y_2  = read_results(path2,5)  
rec_z_2  = read_results(path2,6)
spot_x_2 = read_results(path2,7) 
spot_y_2 = read_results(path2,8)
spot_z_2 = read_results(path2,9)
off_x_2  = read_results(path2,16)
tt_inv_2 = read_results(path2,17)
   
    
""" PLOT THE POSITIONS AND RECEIVERS OBTAINED """
def plot_positions(src_x,src_y,rec_x,rec_y,dec=1):
    colors = src_x[::dec]
    fig = plt.figure(figsize= (10,7))
    plt.scatter(src_x[::dec],src_y[::dec],c=colors,marker='*',cmap='jet')
    plt.scatter(rec_x[::dec],rec_y[::dec],c=colors,marker='v',cmap='jet')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.ylim(-0.1,0.1)
    
plot_positions(src_x_1, src_y_1, rec_x_1, rec_y_1)
plot_positions(src_x_2, src_y_2, rec_x_2, rec_y_2) 

def difference(attr1,attr2):
    diff_attr = np.zeros_like(attr1)
    for i in range(len(attr1)):
        if attr1[i] != 0 and attr2[i] != 0 :   
            diff_attr[i] = attr1[i] - attr2[i]
    return diff_attr

def find_nearest(array, value):
    temp = np.asarray(array)
    array = np.asarray(array)
    for i in range(len(array)):
        temp[i] = np.abs(array[i] - value)
        idx = temp.argmin()
    return array[idx], idx

def extract_trace(src_x, off_x,ax,ao):  
    title = str(find_nearest(ax, src_x/1000)[1])
    title = title.zfill(3)
    print('indice source', title)
    fl = '../output/40_marm_ano/badj/t1_obs_000'+str(title)+'.dat'
    inp = gt.readbin(fl, no, nt).transpose()
    tr = find_nearest(ao,off_x/1000)[1]
    return inp[tr],inp,tr


def plot_shot_gather(hmin, hmax, inp,tr):
    plt.rcParams['font.size'] = 16
    
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    av = plt.subplot(1, 1, 1) 
    ao = fo + np.arange(no)*do 
    hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                     vmin=hmin, vmax=hmax, aspect='auto',
                     cmap='seismic')
    av.plot(inp[:,tr]+ao[tr],at,'k')
    plt.colorbar(hfig, format='%2.2f')
    
    plt.xlabel('Offset (km)')
    plt.ylabel('Time (s)')
    fig.tight_layout()

diff_src_x = difference(src_x_1, src_x_2)
diff_rec_x = difference(rec_x_1, rec_x_2)


diff_ind_max = diff_src_x.argmax()
diff_max = np.max(diff_src_x) 


diff_ind_max = 28

print('SRC_X for the Standard migration : ',src_x_1[diff_ind_max])
print('REC_X for the Standard migration : ',rec_x_1[diff_ind_max])
print('SRC_X for the Quantitative migration : ',src_x_2[diff_ind_max])
print('REC_X for the Quantitative migration : ',rec_x_2[diff_ind_max])

plt.figure(figsize=(10,8))
plt.scatter(src_x_1[diff_ind_max],0,c='k',marker='*')
plt.scatter(rec_x_1[diff_ind_max],0,c='k',marker='v')
plt.scatter(spot_x_1[diff_ind_max],spot_z_1[80],c='k',marker='o')

plt.scatter(src_x_2[diff_ind_max],0,c='r',marker='*')
plt.scatter(rec_x_2[diff_ind_max],0,c='r',marker='v')
plt.scatter(spot_x_2[diff_ind_max],spot_z_2[80],c='r',marker='o')


plt.figure(figsize=(10,8))
plt.plot(src_x_1,'.')
plt.plot(src_x_2,'.')

plt.figure(figsize=(10,8))
plt.plot(rec_x_1,'.')
plt.plot(rec_x_2,'.')




plt.figure(figsize=(10,8))
plt.plot(diff_src_x,'.')
plt.axvline(diff_ind_max)

plt.figure(figsize=(10,8))
plt.plot(diff_rec_x,'.')


trace,gather,tr = extract_trace(src_x_1[diff_ind_max], off_x_1[diff_ind_max],ax,ao)


hmax =np.max(gather)
hmin = -hmax
plot_shot_gather(hmax,hmin, gather,tr)

#%%

nr_src_x, idx_nr_src = find_nearest(ax, src_x_1[diff_ind_max]/1000)
nr_off_x, idx_nr_off = find_nearest(ao, off_x_1[diff_ind_max]/1000)

if idx_nr_src < 2 :
    nb_gathers = np.array([0, 1, 2, 3, 4])
elif idx_nr_src > nx-3:
    nb_gathers = np.array([597, 598, 599, 600, 601])
else:
    nb_gathers = np.arange(idx_nr_src-2, idx_nr_src+3)


if idx_nr_off < 2 :
    nb_traces = np.array([0, 1, 2, 3, 4])
elif idx_nr_off > no-3:
    nb_traces = np.array([247, 248, 249, 250, 251])
else:
    nb_traces = np.arange(idx_nr_off-2, idx_nr_off+3)



gather_path = '../output/40_marm_ano/badj/'

def read_shots_around(nb_gathers,no,nt):
    inp3 = np.zeros((len(nb_gathers),nt, no))
    for k, i in enumerate(nb_gathers):
        txt = str(i)
        title = txt.zfill(3)
        print(title)
        tr3 = gather_path+'/t1_obs_000'+str(title)+'.dat'
        inp3[k][:,:] = -gt.readbin(tr3, no, nt).transpose()
    return inp3

inp3 = read_shots_around(nb_gathers, no, nt)


rec_to_int = np.zeros((4,nt,4))
tr_INT     = np.zeros((5,nt,5)) 

for k, i in enumerate(nb_gathers): 

    # Interpolation on the receivers
    for j in range(5):
        f = interpolate.RegularGridInterpolator((at,ao[nb_traces]), inp3[j][:,nb_traces], method='linear',bounds_error=False, fill_value=None) 
        at_new = at
        ao_new = np.linspace(off_x_1[diff_ind_max]/1000-do*2,off_x_1[diff_ind_max]/1000+do*2, 5)
        AT, AO = np.meshgrid(at_new, ao_new, indexing='ij')
        tr_INT[j][:,:] = f((AT,AO))
        rec_int = tr_INT[:,:,2]
        
        
    # Interpolation on the shots
    f = interpolate.RegularGridInterpolator((at,nb_gathers*12), rec_int.T, method='linear',bounds_error=False, fill_value=None) 
    at_new = at
    src_new = np.linspace(src_x_1[diff_ind_max]/1000 - dx*2, src_x_1[diff_ind_max]/1000 + dx*2, 5)
    AT, SRC = np.meshgrid(at_new, src_new, indexing='ij')
    src_INT = f((AT,SRC))
    finterp_trace = src_INT[:,2] 

plt.plot(rec_int[0])
plt.plot(finterp_trace[0])


hmax =np.max(gather)
hmin = -hmax        
fig = plt.figure(figsize=(10, 8), facecolor="white")
av = plt.subplot(1, 1, 1) 
ao = fo + np.arange(no)*do 
for k, i in enumerate(nb_gathers):
    hfig = av.imshow(inp3[k], extent=[ao[0], ao[-1], at[-1], at[0]],
                  vmin=hmin, vmax=hmax, aspect='auto',
                  cmap='seismic')




#%%



""" PLOT MIGRATED IMAGE AND RAYS """

## Read migrated image

# file = '../output/27_marm/flat_marm/inv_betap_x_s.dat'
file = '../input/27_marm/marm2_sm15.dat'
# file = '../output/27_marm/flat_marm/dbetap_exact.dat'
Vit_model2 = gt.readbin(file,nz,nx).T*1000
file_pick = '../input/40_marm_ano/badj_mig_pick_smooth.csv'
pick_hz = read_pick(file_pick,0)

## Read rays from raytracing

def plot_rays(src_x,rec_x,spot_x,spot_z,off_x,hz):
    path_ray = [0]*102
    indices = []
    for i in range(28,101,50):
        if src_x[i] > 0.: 
            indices.append(i)
            
            # path_ray[i] = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug141223_raytracing/rays/opti_"+str(i)+"_ray.csv"
            path_ray[i] = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2/badj/rays/opti_"+str(i)+"_ray.csv"
            ray_x = np.array(read_results(path_ray[i], 0))
            ray_z = np.array(read_results(path_ray[i], 2))
    
            x_disc = np.arange(nx)*12.00
            z_disc = np.arange(nz)*12.00
            
            fig1 = plt.figure(figsize=(18, 8), facecolor="white")
            av1 = plt.subplot(1, 1, 1)
            hmin = np.min(Vit_model2)
            # hmax = -hmin
            hmax = np.max(Vit_model2)
            hfig = av1.imshow(Vit_model2.T[:,270:560],vmin = hmin,
                        vmax = hmax,aspect = 1, 
                        extent=(x_disc[270],x_disc[560],z_disc[-1],z_disc[0]),cmap='jet')
            plt.plot(x_disc[270:560],hz[270:560])
            plt.colorbar(hfig)
            
            
            av1.plot(src_x[i],24,'*')
            av1.plot(rec_x[i],24,'v')
            av1.plot(spot_x,-spot_z,'.k')
            av1.scatter(ray_x,-ray_z, c="r", s=0.1)
            av1.set_title('ray number '+str(i)+'; offset = ' +str(int(off_x[i])))
            # flout1 = "../png/27_marm/flat_marm/corr_az_pert/ray_sm_img_"+str(i)+"_f_p0007.png"
            # print("Export to file:", flout1)
            # fig1.savefig(flout1, bbox_inches='tight')
            
            
     
            
plot_rays(src_x_1, rec_x_1, spot_x_1, spot_z_1, off_x_1, pick_hz)
