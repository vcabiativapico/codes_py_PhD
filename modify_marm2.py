# -*- coding: utf-8 -*-

"""
Display the results
"""

import os
import numpy as np
from math import log, sqrt, log10, pi, cos, sin, atan

import matplotlib.pyplot as plt
from matplotlib import use
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel
import pickle as pk

if __name__ == "__main__":
    
    # Global parameters
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
    fl1       = './input/org/marm2_sel.dat'
    inp_org   = gt.readbin(fl1,nz,nx)
    inp1      = gt.readbin(fl1,nz,nx)
    perc      = 0.9

#%%
    # # # ######### Modifing initial model anomaly ############

    # Find indexes of the anomaly
    inp1_cut   = inp1[75-15:75+5,300-20:300+20] # Zone A - Cut the model in the area of the anomaly
    #inp1_cut   = inp1[75-25:75+1,500-100:500]   # Zone B - Cut the model in the area of the anomaly
    old_vel    = np.min(inp1_cut)                  # Find the value where the anomaly wants to be changed    
    ind_cut    = np.where(inp1_cut == old_vel)  # Find the index of the to_mod value

    # Replace index with a new value
    ind        = np.where(inp1 == old_vel)      # Find again to_mod now for the whole model

    new_v      = old_vel * perc                # add 10 % to old velocity value
    inp1[ind]  = new_v                         # Replace the zone with another value
    
        #print ("index :", ind)
    print("old velocity : ", old_vel)
    print("new velocity : ", new_v) 
        #print(inp1)
    
    # Find indexes of the anomaly
    # inp1_cut   = inp1[75-15:75+5,300-20:300+20] # Zone A - Cut the model in the area of the anomaly
    inp2_cut    = inp1[75-25:75+1,500-100:500]     # Zone B - Cut the model in the area of the anomaly
    old2_vel    = np.min(inp2_cut)                # Find the value where the anomaly wants to be changed    
    ind2_cut    = np.where(inp2_cut == old2_vel)    # Find the index of the to_mod value

    # Replace index with a new value
    ind2        = np.where(inp1 == old2_vel)      # Find again to_mod now for the whole model

    new2_v      = old2_vel * perc                # add 10 % to old velocity value
    inp1[ind2]   = new2_v                         # Replace the zone with another value
    
        #print ("index :", ind)
    print("old velocity 2 : ", old2_vel)
    print("new velocity 2 : ", new2_v)

   
    
    
    inp3_cut = inp_org[50:100,200:350]
    old3_vel = np.max(inp3_cut)
    ind3_cut = np.where(inp3_cut == old3_vel)
    
    ind3 = np.where(inp1 == old3_vel)
    new3_v = old3_vel
    

    inp1[:]=1.5
    inp1[ind2] = 4.0 
    inp1[ind]  = 4.0
    inp1[ind3] = 4.0
    
    inp1[78:120,340:460] = 1.5
    
#%%   
    ## GAUSSIAN FILTER SMOOTHING
  
    #Input model
    fl2 = './input/18_3_interface/3_interfaces.dat'
    inp = gt.readbin(fl2,nz,nx)
    #smoothing
    inp_sm = gaussian_filter(inp,5)
    
    
    
    flout = './png/18_3_interfaces/3_interfaces_smooth.png'
    
    gt.writebin(inp_sm,'./input/18_3_interface/3_interfaces_smooth.dat')

    fig = plt.figure(figsize=(10,5), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp_sm, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect='auto')
    plt.colorbar(hfig)
    fig.tight_layout()
    print("Export to file:",flout)
    fig.savefig(flout, bbox_inches='tight')

   
#%%    
    #### Create a new constant velocity model at 2.0 km/s
    flout2 = './png/simple_2060.png'
    inp_new = np.zeros(inp1.shape)
    inp_new = inp_new + 2.0
    print(inp_new[0,0])
    fig = plt.figure(figsize=(10,5), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp_new, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect='auto', \
                      cmap='jet')
    plt.colorbar(hfig)
    fig.tight_layout()
    print("Export to file:",flout2)
    fig.savefig(flout2, bbox_inches='tight')
    
    
    # gt.writebin(inp_new,'./input/simple/simple_2060.dat')
     
    #### Create a new constant velocity model with a layer of 2.06 km/s in between 
    
  