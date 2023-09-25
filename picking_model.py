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

    hmin,hmax = 1.5,4.0
    fl1       = './input/org/marm2_sel.dat'
    inp1      = gt.readbin(fl1,nz,nx)
    perc      = 0.9
        
    hmin = 1.0
    hmax = 4.0
    velf = np.zeros(inp1.shape) + 2.0
    velf[50,300] = 4.0
    vels = np.zeros(inp1.shape) + 2.0
    
    rhof = np.zeros(inp1.shape) + 1.0
    # rhof[50,300]=4.0
    rhos = np.zeros(inp1.shape) + 1.0
    
    
    #### RHO_FULL
   
    # ax = np.around(ax*1000)/12
    # az = np.around(az*1000)/12
    # ax = ax.astype(int)
    # az = az.astype(int)
    
    vbg = 1
    
    inp = np.zeros(inp1.shape) + vbg
    #

    
    
    def pick_interface(inp):
    # Create a plot only with integers by using the extension of the grid ax az
        
        ax = np.arange(fx,nx,1)
        az = np.arange(fz,nz,1)
        
    #### Plot initial model 
        # fig = plt.figure(figsize=(10,5), facecolor = "white")
        # av  = plt.subplot(1,1,1)
        # hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
        #                   vmin=hmin,vmax=hmax,aspect='auto', \
        #                   cmap='jet')
        # plt.colorbar(hfig)
        # fig.tight_layout()
        # flout3 = './png/rho_full_200.png'
        # print("Export to file:",flout3)
        # fig.savefig(flout3, bbox_inches='tight')
        # av.title.set_text('rho full')
        # gt.writebin(rhof,'./input/rho_full.dat')
        
        use('TkAgg')
        fig = plt.figure(figsize=(10,8), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(inp)
        plt.colorbar(hfig)
        fig.tight_layout()
        # flout3 = './png/rho_picks.png'
        # print("Export to file:",flout3)
        # fig.savefig(flout3, bbox_inches='tight')
        # av.title.set_text('rho full')
    
    # Extraction of indexes and arrangement to insert them
        pick = plt.ginput(n=-1,timeout=30)
        plt.close()
        pick = np.asarray(pick).astype(int)
        pickt = pick.transpose()
        pick_f = [pickt[1],pickt[0]] 
        pick_f = tuple(pick_f)  
    
    # Modification of the velocity for the picked values 
        #inp[pick_f]  = 5
    
    # Plot the points in a imshow az has to be recalculated for the 
        # az        = fz + np.arange(nz)*dz
        # ax        = fx + np.arange(nx)*dx
        # fig = plt.figure(figsize=(10,5), facecolor = "white")
        # av  = plt.subplot(1,1,1)
        # hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
        #                   vmin=hmin,vmax=hmax,aspect='auto', \
        #                   cmap='jet')
        # plt.colorbar(hfig)
        # fig.tight_layout()
    
    ### Interpolation
    # Convert floating picks to int 
        pickt = np.asarray(pickt).astype(int)
        pickt_x = pickt[0]
        pickt_y = pickt[1]
    # Find the max and min of x
        minp_x = pickt[0,0]
        maxp_x = pickt[0,-1]
    # Or extend to the limits of the grid     
        minp_x = int(fx)
        maxp_x = int(nx)
        
    # Create a tck for interpolation with bspline
        tck = interpolate.splrep(pickt_x,pickt_y, s=10)
    # A new x and y is needed to fill with tck    
        xnew = np.arange(minp_x,maxp_x,1)
        ynew = interpolate.splev(xnew, tck, der=0)
    
    # Plot the points and its bspline 
        plt.figure(figsize=(10,5))
        plt.plot(pickt_x,-pickt_y,'*')
        plt.plot(xnew,-ynew,'x')
        plt.show()
        plt.ylim(-151,0) 
    # Convert into a matrix    
        # ynew= np.asarray(ynew).astype(int)
        ynew= np.asarray(ynew)
        index = (ynew,xnew)
        rhof_int=np.zeros(inp1.shape) + vbg
        rhof_int = inp
        # rhof_int[index] = 5
        
        # rhof_int[75,300] = 10
    # Plot final interpolated interface    
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(rhof_int, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto', \
                          cmap='jet')
        plt.colorbar(hfig)
        fig.tight_layout()
        
        
        return rhof_int, index,xnew,ynew
    
    

    # ## Region extension
    # n = 10 
    # ynew_ext = np.zeros(325*n)
    # # number of pixels
    
    # ind_ext = np.arrange
    # for i in range (n):    
    #     ind_ext[1,0] = index[0]

#%%  


# pente_1 = interface2[0] *12.49/1000
# pente_2 = interface2[1] *12.49/1000
    
pente_1 = 0.4 + 0.02*ax
pente_2 = 1.2 + 0.04*ax

v_1 = 1500
v_2 = 1750
v_3 = 2500

V_model = np.zeros((nx,nz))

def fill_custom_model(V_model,v_1,v_2,v_3,pente_1,pente_2):   
    for k in range(V_model.shape[0]):
        for i,z in enumerate(az):
            
            if z < pente_1[k]:
                V_model[k,i] = v_1
                
                if  z+dz > pente_1[k]:
                    
                    V_model[k,i] = (v_1*(pente_1[k]-z)+v_2*(dz-pente_1[k]+z))/dz
                # print(pente_1[k],z,-pente_1[k]+dz+z)
                
            elif z < pente_2[k]:
                V_model[k,i] = v_2 + z
                
                if z+dz > pente_2[k]:
                    
                    V_model[k,i] = ((v_2 + z)*(pente_2[k]-z)+v_3*(dz-pente_2[k]+z))/dz
                
                
            else:
                V_model[k,i] = v_3

    return V_model
#%%

# model,interface = pick_interface(inp)
# model2,interface2 = pick_interface(model)

#%%

fl3 = './input/15_picked_models/vel_full_3_CO2.dat'
inp3 = gt.readbin(fl3,nz,nx) 

model3, interface3 = pick_interface(inp3)

pente_3 = interface3[0] * 12.49/1000
  
for k in range(model3.shape[1]):
    for i,z in enumerate(az):
        
        if z >= pente_3[k]:
            model3[i,k] = 3.0


outmodel = model3
gt.writebin(outmodel,'./input/24_pick/3_interfaces.dat')      
model_export = './input/24_pick/3_interfaces.dat'
inp3 = gt.readbin(model_export,nz,nx)            
            
fig = plt.figure(figsize=(10,5))            
hfig = plt.pcolor(ax,az,inp3)   
plt.colorbar(hfig) 
plt.gca().invert_yaxis()       
flout3 = './png/test3_vel_full_model.png'
fig.savefig(flout3, bbox_inches='tight')





#%%
# pente_1 = interface[0] * 12.49/1000
# pente_2 = interface2[0] * 12.49/1000     

      
# plt.figure(figsize=(10,5))
# plt.plot(ax,pente_1,'r')
# plt.plot(ax,pente_2,'b')

# plt.xlim(ax[0],ax[-1])
# plt.ylim(az[0],az[-1])
# plt.gca().invert_yaxis()
            

# model_fill = fill_custom_model(V_model,v_1,v_2,v_3,pente_1,pente_2)/1000
 
# fig = plt.figure(figsize=(10,5))
# hfig = plt.pcolor(ax,az,model_fill.T)
# plt.gca().invert_yaxis()
# plt.colorbar(hfig)
# flout3 = './png/test_vel_full_model.png'
# print("Export to file:",flout3)
# fig.savefig(flout3, bbox_inches='tight')
# # gt.writebin(model_fill.T,'./input/15_picked_models/vel_full_model.dat')


# model_fill_sm = gaussian_filter(model_fill.T,2)

# fig = plt.figure(figsize=(10,5))
# hfig = plt.pcolor(ax,az,model_fill_sm)
# plt.gca().invert_yaxis()
# plt.colorbar(hfig)
# flout3 = './png/test_vel_smooth_model.png'
# print("Export to file:",flout3)
# fig.savefig(flout3, bbox_inches='tight')
# gt.writebin(model_fill_sm,'./input/15_picked_models/vel_sm_model.dat')




#%% PICK SEISMIC

fl3 = './output/23_mig/org/nh10_is4/dens_corr/inv_betap_x.dat'
inp3 = gt.readbin(fl3,nz,nx) 

model3, interface3, xnew, ynew = pick_interface(inp3)

ynew= ynew*12.49

print(xnew)
print(ynew)
table_n = [xnew+1, ynew]

df = pd.DataFrame(table_n)
df.to_csv('./png/24_for_ray_tracing/table_pick.csv',header=False,index=False)
  

    