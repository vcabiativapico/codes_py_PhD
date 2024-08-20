#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 16:07:00 2024

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

def read_results(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
        # attr = [x for x in attr if str(x) != 'nan']
    return attr

def find_nearest(array, value):
    val = np.zeros_like(array)
    for i in range(len(array)):
        val[i] = np.abs(array[i] - value)
        idx = val.argmin()
    return array[idx], idx

gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'


path_inv = gen_path + '048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/results/depth_demig_output.csv'

class Param_class:
    "Class for the parameters definition"
    def __init__(self,path):
        self.src_x_ = read_results(path,1)
        self.src_y_ = read_results(path,2)
        self.src_z_ = read_results(path,3)
        self.rec_x_ = read_results(path,4)
        self.rec_y_ = read_results(path,5)
        self.rec_z_ = read_results(path,6)
        self.spot_x_ = read_results(path,7)
        self.spot_y_ = read_results(path,8)
        self.spot_z_ = read_results(path,9)
        self.off_x_ = read_results(path,16)
        self.tt_ = read_results(path,17)
        self.nt_ = 1801
        self.dt_ = 1.41e-3
        self.ft_ = -100.11e-3
        self.nz_ = 151
        self.fz_ = 0.0
        self.dz_ = 12.0/1000.
        self.nx_ = 601
        self.fx_ = 0.0
        self.dx_ = 12.0/1000.
        self.no_ = 251
        self.do_ = self.dx_
        self.fo_ = -(self.no_-1)/2*self.do_
        self.ao_ = self.fo_ + np.arange(self.no_)*self.do_
        self.at_ = self.ft_ + np.arange(self.nt_)*self.dt_
        self.az_ = self.fz_ + np.arange(self.nz_)*self.dz_
        self.ax_ = self.fx_ + np.arange(self.nx_)*self.dx_
   
        
p_inv = Param_class(path_inv)
 
fl2 = '../output/47_marm2/fwi/p2d_fwi_000001.dat'


# nxl = 443
nxl = 291
h_nxl = int((nxl-1)/2)
born = -gt.readbin(fl2, p_inv.nz_, nxl*p_inv.nt_)   


left_p = 300-h_nxl  # left point
right_p = 300+h_nxl  # right point

# left_p = 155+115 # left point
# right_p = 445+115  # right point


left_p2 = 0  # left point
right_p2 = 232+h_nxl+104  # right point


        
fl1 = '../input/27_marm/marm2_sm15.dat'
bg = gt.readbin(fl1, p_inv.nz_, p_inv.nx_)  # model

born = np.reshape(born, (p_inv.nz_, p_inv.nt_, nxl))*2


'''Read the raypath'''
# spot lancé dans le modele 3510,-1210
path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/054_TS_deeper/depth_demig_out/050_TS_analytiquedeep_2024-07-01_14-27-46/rays/ray_0.csv'
# path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/052_TS_deep/depth_demig_out/052_TS_analytique_deep_2024-06-20_12-02-07/rays/ray_0.csv'
ray_x = np.array(read_results(path_ray, 0))
ray_z = np.array(read_results(path_ray, 2))
ray_tt = np.array(read_results(path_ray, 8))


x_disc = np.arange(p_inv.nx_)*12.00
z_disc = np.arange(p_inv.nz_)*12.00

FFMpegWriter = manimation.writers['ffmpeg']
metadata = dict(title='Movie Test', artist='Matplotlib',
                comment='a red circle following a blue sine wave')
writer = FFMpegWriter(fps=3, metadata=metadata)




hmax = np.max(born)
print('hmax: ', hmax)
hmax = 1
hmin = -hmax

hmin1 = np.min(bg)
hmax1 = np.max(bg)



# fig = plt.figure(figsize=(14, 6), facecolor="white")
# fig.tight_layout()
# with writer.saving(fig, "../png/wf_sim.mp4", 100):
for i in range(100, 1000,100):
    fig = plt.figure(figsize=(14, 6), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(bg[:, left_p:right_p], extent=[p_inv.ax_[left_p2], p_inv.ax_[right_p2], p_inv.az_[-1], p_inv.az_[0]],
                      vmin=hmin1, vmax=hmax1,aspect='auto', alpha=1,
                      cmap='jet')
    hfig = av.imshow(born[:, i, :], extent=[p_inv.ax_[left_p2], p_inv.ax_[right_p2], p_inv.az_[-1], p_inv.az_[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', alpha=0.5,
                      cmap='Greys')
    
    plt.colorbar(hfig)
    
    plt.rcParams['font.size'] = 18
    # fig = plt.figure(figsize=(13, 6), facecolor="white")
    # av = plt.subplot(1, 1, 1)
    # av.axhline(1.190)
    idx = find_nearest(ray_tt, i/1000)[1]
    print(i)
    # av.plot(spot_x/1000,-spot_z/1000,'.k')
    av.scatter(ray_x[:idx]/1000,-ray_z[:idx]/1000, c="r", s=0.1)
    av.set_title('ray number ; t = '+str(i)+' ms')
    
        # flout1 = "../png/29_sim_flat_marm/ray_plots_over_wf_img_"+str(i)+"_f_p0007.png"
        # print("Export to file:", flout1)
        # fig.savefig(flout1, bbox_inches='tight')
        # writer.grab_frame() 
