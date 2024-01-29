#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 13 09:51:39 2023

@author: vcabiativapico
"""
import os
import numpy as np
from math import log, sqrt, log10, pi, cos, sin, atan
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from scipy.ndimage import gaussian_filter
from scipy.signal import hilbert
import csv
import matplotlib.animation as manimation


if __name__ == "__main__":

    os.system('mkdir -p png_course')

    # Global parameters
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
    no =251
   # no        = 2002
    do = dx
    fo = -(no-1)/2*do
    ao = fo + np.arange(no)*do
    at = ft + np.arange(nt)*dt
    az = fz + np.arange(nz)*dz
    ax = fx + np.arange(nx)*dx



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
 
path1 ='/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug141223_raytracing/015_marm_flat_v2_test_p0001_v2_test_bp_az_0_4992.csv'

src_x  = np.array(read_results(path1,1))
src_y  = np.array(read_results(path1,2))
src_z  = np.array(read_results(path1,3))    
rec_x  = np.array(read_results(path1,4))  
rec_y  = np.array(read_results(path1,5))    
rec_z  = np.array(read_results(path1,6))
spot_x = np.array(read_results(path1,7)) 
spot_y = np.array(read_results(path1,8))
spot_z = np.array(read_results(path1,9))
off_x  = np.array(read_results(path1,16))
tt_inv = np.array(read_results(path1,17))


indices = []
for i in range(0,101):
    if str(src_x[i]) != 'nan':
        indices.append(i)
        
# ## Calculate the index of the shot
shot = np.round(src_x/12)

shot = np.array((np.rint(shot)).astype(int))
     
        
fl1 = '../input/27_marm/marm2_sm15.dat'
bg = gt.readbin(fl1, nz, nx)  # model


fl2 = '../output/29_sim_flat_marm/419/p2d_lsm_000001.dat'

# nxl = bites/4/nz/nt = bites/4/151/1501
# position central (301-1)*dx = 3600
# 291 = 1+2*145
# point à gauche = 3600-145*dx 
# point à droite = 3600+145*dx

# nxl = 443
nxl = 291
h_nxl = int((nxl-1)/2)
born = -gt.readbin(fl2, nz, nxl*nt)   


left_p = 300-h_nxl  # left point
right_p = 300+h_nxl  # right point

left_p = 155+115 # left point
right_p = 445+115  # right point


left_p2 = 200-h_nxl  # left point
right_p2 = 200+h_nxl  # right point

# print("size",np.shape(born))

born = np.reshape(born, (nz, nt, nxl))



## Read rays from raytracing
path_ray = [0]*102
plt.figure(figsize= (8,6))



        
path_ray = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/baptiste_corr_v181223_y0.01/rays/opti_"+str(indices[-1])+"_ray.csv"
ray_x = np.array(read_results(path_ray, 0))
ray_z = np.array(read_results(path_ray, 2))


x_disc = np.arange(nx)*12.00
z_disc = np.arange(nz)*12.00

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='a red circle following a blue sine wave')
writer = FFMpegWriter(fps=3, metadata=metadata)


def plot_sim_wf(bg, inp1):
    hmax = np.max(inp1)
    print('hmax: ', hmax)
    hmax = 1
    hmin = -hmax
    
    hmin1 = np.min(bg)
    hmax1 = np.max(bg)

    fig = plt.figure(figsize=(13, 6), facecolor="white")
    fig.tight_layout()
    with writer.saving(fig, "../png/29_sim_flat_marm/wf_sim.mp4", 100):
        for i in range(200, 1100, 50):
        
            av = plt.subplot(1, 1, 1)
            hfig1 = av.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                              vmin=hmin1, vmax=hmax1,aspect='auto', alpha=1,
                              cmap='jet')
            hfig = av.imshow(inp1[:, i, :], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                              vmin=hmin, vmax=hmax, aspect='auto', alpha=0.5,
                              cmap='Greys')
            # av.set_title('t = '+str(i)+' m/s')
            plt.colorbar(hfig)
            
            # flout2 = '../png/29_sim_flat_marm/born_'+str(i)+'.png'
            # print("Export to file:", flout2)
            # fig.savefig(flout2, bbox_inches='tight')
            plt.rcParams['font.size'] = 18
            # fig = plt.figure(figsize=(13, 6), facecolor="white")
            # av = plt.subplot(1, 1, 1)
            av.axhline(1.190)
            av.plot(src_x[indices[-1]]/1000,0.024,'*')
            av.plot(rec_x[indices[-1]]/1000,0.024,'v')
            av.plot(spot_x/1000,-spot_z/1000,'.k')
            av.scatter(ray_x/1000,-ray_z/1000, c="r", s=0.1)
            av.set_title('ray number '+str(indices[-1])+'; t = '+str(i)+' ms')
            
            # flout1 = "../png/29_sim_flat_marm/ray_plots_over_wf_img_"+str(i)+"_f_p0007.png"
            # print("Export to file:", flout1)
            # fig.savefig(flout1, bbox_inches='tight')
            writer.grab_frame() 




plot_sim_wf(bg, born)