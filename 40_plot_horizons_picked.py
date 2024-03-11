#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 11:13:07 2024

@author: vcabiativapico
"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.ndimage import gaussian_filter

import csv

from wiggle.wiggle import wiggle
if __name__ == "__main__":

        
    labelsize = 16
    nt        = 1501
    dt        = 2.08e-3
    ft        = -99.84e-3
    nz        = 151
    fz        = 0.0
    dz        = 12.00/1000.
    nx        = 601
    fx        = 0.0
    dx        = 12.00/1000.
    no        = 251
    do        = dx
    fo        = -(no-1)/2*do
    ao        = fo + np.arange(no)*do
    at        = ft + np.arange(nt)*dt
    az        = fz + np.arange(nz)*dz
    ax        = fx + np.arange(nx)*dx

        
    def read_pick(path,srow):
        attr = []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            # header = next(spamreader)
            for row in spamreader:
                attr.append(float(row[srow]))
        return attr
    
    def find_nearest(array, value):
        val = np.zeros_like(array)
        for i in range(len(array)):
            val[i] = np.abs(array[i] - value)
            idx = val.argmin()
        return array[idx], idx

    
    file_pick_badj = '../input/40_marm_ano/badj_mig_pick_smooth.csv'
    pick_hz_badj = np.array(read_pick(file_pick_badj,0))
    
      
    file_pick_binv = '../input/40_marm_ano/binv_mig_pick_smooth.csv'
    pick_hz_binv = np.array(read_pick(file_pick_binv,0))
    
    diff_inv_adj = pick_hz_binv-pick_hz_badj
    
    spot_x = 4550
    spot_z = pick_hz_binv[find_nearest(ax*1000,spot_x)[1]]
    
    idx_diff_spot = find_nearest(ax*1000,spot_x)[1]
    
    diff_at_spot = diff_inv_adj[idx_diff_spot]
    
    plt.figure(figsize=(10,8))
    plt.plot(ax*1000, pick_hz_binv, label='inv')
    plt.plot(ax*1000, pick_hz_badj, label='adj')
    plt.scatter(spot_x,spot_z,c='k', label = 'spot')
    plt.gca().invert_yaxis()
    plt.legend()
    plt.xlim(3000,6000)
    plt.ylabel('Depth (m)')
    plt.xlabel('Distance (m)')
    plt.title('Comparison horizon pick inv vs adj')
    
    plt.figure(figsize=(10,8))
    plt.plot(ax*1000,diff_inv_adj,'.', label='diff')
    plt.scatter(spot_x,diff_inv_adj[idx_diff_spot],c='k',label='spot')
    plt.legend()
    plt.ylabel('Difference (m)')
    plt.xlabel('Distance (m)')
    plt.title('Difference horizon pick inv vs adj')