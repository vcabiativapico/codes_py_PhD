#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 14:41:48 2023

@author: vcabiativapico
"""

import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel


if __name__ == "__main__":
    
## Building simple vel and rho models to test modeling
    labelsize = 16
    nt        = 1001
    dt        = 2.08e-3
    ft        = -99.84e-3
    nz        = 151
    fz        = 0.0
    dz        = 12.49/1000.
    nx        = 601
    fx        = 0.0
    dx        = 12.49/1000.
    no        = 251
    do        = dx
    fo        = -(no-1)/2*do
    ao        = fo + np.arange(no)*do
    at        = ft + np.arange(nt)*dt
    az        = fz + np.arange(nz)*dz
    ax        = fx + np.arange(nx)*dx

    hmin,hmax = 1.5,4.0
    fl1       = './input/15_picked_models/vel_full_0_CO2.dat'
    inp1      = gt.readbin(fl1,nz,nx)
    perc      = 0.9
        
    hmin = 1.0
    hmax = 4.0
    velf = np.zeros(inp1.shape) + 2.0
    
    vels = np.zeros(inp1.shape) + 2.0
    
    rhof = np.zeros(inp1.shape) + 1.0
    # rhof[50,300]=4.0
    rhos = np.zeros(inp1.shape) + 1.0
    
    
    fig = plt.figure(figsize=(10,5), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(velf, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect='auto', \
                      cmap='jet')
    plt.colorbar(hfig)
    fig.tight_layout()
    flout3 = './png/vel_full.png'
    print("Export to file:",flout3)
    fig.savefig(flout3, bbox_inches='tight')
    av.title.set_text('vel full')
    gt.writebin(velf,'./input/vel_full.dat')