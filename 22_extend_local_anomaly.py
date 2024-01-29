#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 15:18:12 2023

@author: vcabiativapico
"""

"""
Display the results
"""

#%matplotlib inline
import os
import numpy as np
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
    
    
    def plot_model(inp,flout):
        # hmax = np.max(np.abs(inp2))
        # hmin = -hmax
        
        # print(hmin,hmax)
        
        # inp = gt.readbin(fl1,nz,nx)
        hmax = np.max(inp)
        hmin = np.min(inp)
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig1 = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto', alpha=1,\
                        )
        # av.plot(ax[p1[0]],az[p1[1]],'or')
        # av.plot(ax[p2[0]],az[p2[1]],'or')
        # av.plot(ax[p3[0]],az[p3[1]],'.r')
        # av.plot(ax[p4[0]],az[p4[1]],'.r')
        
        # av.set_xlim([2.5,4.5])
        # av.set_ylim([0.8,0.4])
        plt.colorbar(hfig1)
        fig.tight_layout()
            
        print("Export to file:",flout)
        fig.savefig(flout, bbox_inches='tight')
        return inp[::25,301]                

    fl1 = './input/19_anomaly_4_layers/3_interfaces_anomaly_171.dat'
    flout1 ='./png/test_model.png'
    
    inp1 = gt.readbin(fl1,nz,nx)
    plot_model(inp1,flout1)
    
    def modif_layer(inp1,r1,r2,nv): 
        area      = np.zeros(inp1.shape)
        for i in range(nz):
            for j in range(nx):
                if inp1[i,j] > r1 and inp1[i,j] < r2 : 
                    area[i,j] = 1
                else: 
                    area[i,j] = 0
                
                
        index1     = np.where(area == 1)
        new_vel   = nv
        # new_vel   = 1.75*1.14
        inp1[index1] = new_vel
        
        return inp1,index1
    
    
    def calculate_slope(nv):
        inp2,index2 = modif_layer(inp1,1.7,1.72,1.71)
        
        idx_max_x0 = np.max(index2[1])
        idx_min_x0 = np.min(index2[1])
        
        idx_max_y0 = np.max(index2[0])
        idx_min_y0 = np.min(index2[0])
        
        p1 = [idx_min_x0+1,idx_min_y0-1]
        p2 = [idx_max_x0+9,idx_min_y0+2]
        
        # p3 = [idx_min_x0+16,idx_max_y0-2]
        # p4 = [idx_max_x0-13,idx_max_y0]
        
        a_p1 = [ax[p1[0]],az[p1[1]]]
        a_p2 = ax[p2[0]],az[p2[1]]
        
        # a_p3 = ax[p3[0]],az[p3[1]]
        # a_p4 = ax[p4[0]],az[p4[1]]
        
        
        m1 = (a_p1[1]-a_p2[1])/(a_p1[0]-a_p2[0])
        b1 = a_p1[1]-a_p1[0]*m1
        
        # m2 = (a_p3[1]-a_p4[1])/(a_p3[0]-a_p4[0])
        # b2 = a_p3[1]-a_p3[0]*m2
        
        
        pente_1 = (b1+0.03 + m1+0.007 * ax)
        pente_2 = pente_1+0.04
        
        
        fl2 = './input/19_anomaly_4_layers/3_interfaces_org.dat'
        inp2 = gt.readbin(fl2,nz,nx)
        
        for k in range(inp1.shape[1]):
            for i,z in enumerate(az): 
                if z >= pente_1[k] and z <= pente_2[k] :
                    inp2[i,k] = nv
                # elif z < pente_2[k]:
                #     inp1[i,k] = inp1    
        return inp2
    
    nv = 250
    inp2 = calculate_slope(nv/100)
    
    flout2 ='./png/22_extend_anomaly/extend_ano_'+str(nv)+'.png'
    plot_model(inp2,flout2)       
    datout = './input/22_extend_anomaly/extend_ano_'+str(nv)+'.dat'
    gt.writebin(inp2,datout)         
    
    # # Create a tck for interpolation with bspline
    # tck = interpolate.splrep(pickt_x,pickt_y, s=10)
    # # A new x and y is needed to fill with tck    
    # xnew = np.arange(idx_min_x,idx_max_x,1)
    # ynew = interpolate.splev(xnew, tck, der=0)
    
    
    
    