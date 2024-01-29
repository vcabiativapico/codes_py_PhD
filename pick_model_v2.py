#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:30:57 2023

@author: vcabiativapico
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 11:54:19 2023

@author: vcabiativapico
"""

"""
Display the results
"""

#%matplotlib inline
import os
import numpy as np
import pandas as pd
from math import log, sqrt, log10, pi, cos, sin, atan
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt
from matplotlib import use
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel
import pickle as pk
from scipy.interpolate import splrep, BSpline

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


   
    
#%%

def pick_interface_model(inp):
 # Create a plot only with integers by using the extension of the grid ax az
     
     ax = np.arange(fx,nx,1)
     az = np.arange(fz,nz,1)
     vbg = 1
 #### Plot initial model 
     # use('TkAgg')
     use('Qt5Agg')
     hmax = np.max(inp)
     hmin = -hmax
     fig = plt.figure(figsize=(15,7), facecolor = "white")
     av  = plt.subplot(1,1,1)
     
     hfig = av.imshow(inp,vmin=hmin,vmax=hmax,aspect='auto', cmap='seismic')
     plt.colorbar(hfig)
     # plt.colorbar(hfig)
     fig.tight_layout()

     
 # Extraction of indexes and arrangement to insert them
     pick = plt.ginput(n=-1,timeout=30)
     plt.close()
     pick = np.asarray(pick).astype(int)
     pickt = pick.transpose()
     pick_f = [pickt[1],pickt[0]] 
     pick_f = tuple(pick_f)  
 

 ### Interpolation
 # Create variable to interpolate
     pickt_x = pickt[0]
     pickt_y = pickt[1]
 # Find the max and min of x
     minp_x = pickt[0,0]
     maxp_x = pickt[0,-1]
 # Or extend to the limits of the grid     
     minp_x = int(fx)
     maxp_x = int(nx)
     
 # Create a tck for interpolation with bspline
     tck = interpolate.splrep(pickt_x,pickt_y, s=2)
 # A new x and y is needed to fill with tck    
     xnew = np.arange(minp_x,maxp_x,1)
     ynew = interpolate.splev(xnew, tck, der=0)
 
 # Plot the points and its bspline 

 # Convert into a matrix    
     # ynew= np.asarray(ynew).astype(int)
     ynew= np.asarray(ynew)
     index = (ynew,xnew)
     rhof_int=np.zeros(inp.shape) + vbg
     rhof_int = inp

     
     return rhof_int, index, xnew, ynew, pickt_x, pickt_y
 
    
def plot_pick(inp,pick_x,pick_y,xnew,ynew):
    hmax = np.max(inp)
    hmin = -hmax
    fig = plt.figure(figsize=(15,8), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp,vmin=hmin,vmax=hmax,aspect='auto', alpha =0.7,\
                      cmap='seismic')
    plt.colorbar(hfig)
    
    plt.plot(xnew,ynew,'k')
    plt.plot(pick_x,pick_y,'r*')
    fig.tight_layout()
    
    plt.ylim(151,0) 
    flout2 = '../png/24_for_ray_tracing/picking_model.png'
    print("Export to file:",flout2)
    fig.savefig(flout2, bbox_inches='tight')
 
#%%

fl3 = '../output/23_mig/org/nh10_is4/dens_corr/inv_betap_x.dat'
inp3 = gt.readbin(fl3,nz,nx) 

model3, interface3, xnew, ynew, pick_x, pick_y = pick_interface_model(inp3)

ynew= ynew*12.49

# print(xnew)
# print(ynew)

table_n = [xnew+1, ynew]

df = pd.DataFrame(table_n)
df.to_csv('../png/24_for_ray_tracing/table_pick_testing.csv',header=False,index=False)

print("pick y :", pick_y)
print(pick_x)
plot_pick(inp3,pick_x,pick_y,xnew,ynew/12.49)