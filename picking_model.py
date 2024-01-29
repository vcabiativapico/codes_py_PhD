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
    hmin,hmax = 1.5,4.0
    fl1       = '../input/org/marm2_sel.dat'
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
        # plt.colorbar(hfig)
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
        # minp_x = int(fx)
        # maxp_x = int(nx)
        
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
        ynew= np.asarray(ynew).astype(int)
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
        
        
        return rhof_int, index
    
    

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

fl3 = '../input/25_v2_4_layers/4_interfaces_new_02.dat'
inp3 = gt.readbin(fl3,nz,nx) 

model3, interface3 = pick_interface(inp3)

# pente_3 = interface3[0] * 12.49/1000
  
# for k in range(model3.shape[1]):
#     for i,z in enumerate(az):
        
#         if z >= pente_3[k]:
#             model3[i,k] = 3.0


outmodel = model3
gt.writebin(outmodel,'../input/25_v2_4_layers/4_interfaces_ano.dat')      
model_export = '../input/25_v2_4_layers/4_interfaces_ano.dat'
inp3 = gt.readbin(model_export,nz,nx)            
            
fig = plt.figure(figsize=(10,5))            
hfig = plt.pcolor(ax,az,inp3)   
plt.colorbar(hfig) 
plt.gca().invert_yaxis()       
flout3 = '../png/25_v2_4_layers/4_interfaces_ano.png'
fig.savefig(flout3, bbox_inches='tight')


df = pd.DataFrame(interface3)
df.to_csv('../png/25_v2_4_layers/table_pick_ano.csv',header=False,index=False)

pente_1 = interface3[0] * 12.49/1000
pente_2 = interface3[0] * 12.49/1000+0.5

      
for k in range(model3.shape[1]):
    for i,z in enumerate(az):
        
        if z >= pente_1[k] and z < pente_2[k]:
            model3[i,k] = 3.0


#%%    

def plot_model(inp, flout):
    # hmax = np.max(np.abs(inp2))
    # hmin = -hmax

    # print(hmin,hmax)

    hmax = 3.3
    hmin = 1.5
    # hmin = np.max(inp)
    # hmax = -hmin
    # hmin = 0
    fig = plt.figure(figsize=(10, 5), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', alpha=1
                      )
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig1) 
    print("Export to file:",flout)
    fig.savefig(flout, bbox_inches='tight')
    return inp[::11, 301]
    
fl3 = '../input/25_v2_4_layers/4_interfaces_ano_rc_norm.dat'
inp3 = gt.readbin(fl3,nz,nx) 
flout3 = '../png/4_interfaces_rc_norm.png'
plot_model(inp3,flout3)


# spot_x [m]	spot_y [m]	spot_z [m]
# # 4543.246474	0.010000	-1457.886875
# inp3[117,364]= 8
# flout3 = '../png/25_v2_4_layers/spot_in_model.png'
# plot_model(inp3,flout3)
# model_export = '../input/25_v2_4_layers/spot_in_model.dat'
# gt.writebin(inp3,model_export)     

inp3[107:108,286:378] = 2.613
inp3[107:109,290:374] = 2.613
inp3[107:110,294:370] = 2.613
inp3[107:111,298:366] = 2.613
inp3[107:112,302:362] = 2.613
# fig = plt.figure(figsize=(10,5))            
# hfig = plt.pcolor(ax,az,inp3)   

# plt.colorbar(hfig) 
# plt.gca().invert_yaxis()       

flout3 = '../png/25_v2_4_layers/4_interfaces_rc_norm.png'
plot_model(inp3,flout3)
model_export = '../input/25_v2_4_layers/4_interfaces_rc_norm.dat'
gt.writebin(inp3,model_export)     

# sm_ano = gaussian_filter(inp3,8)
# flout_sm = '../png/25_v2_4_layers/4_interfaces_ano_smooth_rc_norm_2926.png'
# plot_model(sm_ano,flout_sm)
# gt.writebin(sm_ano,'../input/25_v2_4_layers/4_interfaces_ano_smooth_rc_norm_2926.dat')


fl4 = '../input/25_v2_4_layers/4_interfaces_rc_norm.dat'
inp4 = gt.readbin(fl4,nz,nx) 
sm4 = gaussian_filter(inp4,8)
flout_sm4 = '../png/25_v2_4_layers/4_interfaces_smooth_rc_norm.png'
plot_model(sm4,flout_sm4)
model_export = '../input/25_v2_4_layers/4_interfaces_smooth_rc_norm.dat'
# gt.writebin(sm4,model_export)     


# plt.figure(figsize=(10,5))
# plt.plot(ax,pente_1,'r')
# plt.plot(ax,pente_2,'b')

# plt.xlim(ax[0],ax[-1])
# plt.ylim(az[0],az[-1])
# plt.gca().invert_yaxis()



#%%
# pente_1 = interface3[0] * 12.49/1000
# pente_2 = interface3[0] * 12.49/1000   

      
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

 

 # ## Region extension
 # n = 10 
 # ynew_ext = np.zeros(325*n)
 # # number of pixels
 
 # ind_ext = np.arrange
 # for i in range (n):    
 #     ind_ext[1,0] = index[0]
    


#%% PICK HORIZON FOR DEMIGRATION

fl3 = './output/23_mig/org/nh10_is4/dens_corr/inv_betap_x.dat'
inp3 = gt.readbin(fl3,nz,nx) 

model3, interface3, xnew, ynew = pick_interface(inp3)

ynew= ynew*12.49

print(xnew)
print(ynew)
table_n = [xnew+1, ynew]

df = pd.DataFrame(table_n)
df.to_csv('./png/24_for_ray_tracing/table_pick.csv',header=False,index=False)
  

  