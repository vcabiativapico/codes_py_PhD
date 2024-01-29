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
from scipy import interpolate


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
    fl1       = '../input/org/marm2_sel.dat'
    inp_org   = gt.readbin(fl1,nz,nx)
    inp1      = gt.readbin(fl1,nz,nx)
    perc      = 0.9
    
#%%

    new_sm = np.asarray(inp_org)
    new_sm = new_sm * 0+ 2.0
    
    gt.writebin(new_sm,'../input/30_marm_flat/2_0_sm_constant.dat')
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
    
    
    fig = plt.figure(figsize=(15,8), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp1, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect='auto', \
                      cmap='jet')
    plt.colorbar(hfig)
    
    
    #%% MODIFY MARMOUSI2 FOR DEMIGRATION
    fl1       = '../input/org_full/marm2_full.dat'
    inp_org   = gt.readbin(fl1,nz,nx)
    
    inp_cut    = inp_org[70:120,355:400]     # Zone B - Cut the model in the area of the anomaly
    old_vel    = np.max(inp_cut)                # Find the value where the anomaly wants to be changed    
    
    hmin = 1.5
    hmax = 4.5
    
    fig = plt.figure(figsize=(15,8), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp_org, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect='auto' \
                      )
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig,format='%1.1f')
    plt.rcParams['font.size'] = 16
    fig.tight_layout()

    
    def modif_layer(inp1,r1,r2,nv): 
        area      = np.zeros(inp1.shape)
        for i in range(70,110): 
            for j in range(355,400):
                if inp1[i,j] > r1 and inp1[i,j] < r2 : 
                    area[i,j] = 1
                else: 
                    area[i,j] = 0
                
                
        index1     = np.where(area == 1)
        new_vel    = nv
        # new_vel   = 1.75*1.14
        inp1[index1] = new_vel
        
        return inp1,index1
    
        
    inp_mod, ind_mod = modif_layer(inp_org, 3.5, 4.0, 4.0)
    
    fl1       = '../input/org_full/marm2_full.dat'
    inp_org   = gt.readbin(fl1,nz,nx)
    inp_diff = inp_mod - inp_org
      
    # fig = plt.figure(figsize=(15,8), facecolor = "white")
    # av  = plt.subplot(1,1,1)
    # hfig = av.imshow(inp_cut, extent=[ax[0],ax[-1],az[-1],az[0]], \
    #                   vmin=hmin,vmax=hmax,aspect='auto', \
    #                   cmap='jet')
    # plt.colorbar(hfig)
    
    inp_diff10 = inp_diff+1
    
    inp_diff10[ind_mod] = 4.0
    
    az[ind_mod[0][63]]
    ax[ind_mod[1][63]]
    
    fig = plt.figure(figsize=(15,8), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp_diff10, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect='auto' \
                      )
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig,format='%1.1f')
    plt.rcParams['font.size'] = 16
    fig.tight_layout()
    flout = '../png/27_marm/diff_marm.png'
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')
      
    gt.writebin(inp_diff10,'../input/27_marm/diff_marm.dat')
    
    
#%% FLAT INTERFACE
    # fl1       = '../input/org_full/marm2_full.dat'
    # fl2       = '../input/27_marm/marm2_sm15.dat'
    
    # fl1 = '../input/vel_full.dat'
    # fl2 = '../input/vel_smooth.dat'
    
    fl1 = '../input/27_marm/inp_flat.dat'
    fl2 = '../input/marm2_sm15.dat'
    

    inp_org   = gt.readbin(fl1,nz,nx)
    inp_smooth= gt.readbin(fl2,nz,nx)
    inp_flat  = inp_org*0
    
    # inp_flat[0:100] = 1.5
    inp_flat[51:100] = 0.05
    
    # inp_flat_corr   = inp_flat + 1/np.sqrt(inp_smooth)
    # inp_flat_tap = inp_flat 
    # # inp_flat_corr = inp_flat+inp_smooth
    
    # abetap = 1/inp_flat_corr**2
    
    # # adbetap_exact = 1/inp_flat_corr**2 - 1/inp_smooth**2
    
    # adbetap_exact = 1/inp_org**2 - 1/inp_smooth**2
    
    def taper(ntap, sh,nx):
        # nx =50
        # ntap = 10
        # sh = 10
        
        tapmin = np.max([sh,1])
        tapmax = np.min([ntap+sh,nx])
        tap = [1]*nx
        for i in range(tapmin,tapmax):
            val = np.sin((i-sh) / ntap * np.pi/2.)
            val = val**2
            tap[i] = tap[i]*val
        for j in range(np.min([sh,nx])):
            tap[j]= 0.0
        # plt.plot(tap,'.')
        return tap
    
 
        
    def plot_model_t(inp):
        hmax = np.max(inp)
        # hmax = 4.5
        # hmin = 1.5
        hmin = np.min(inp)
        # hmin = -hmax
        fig = plt.figure(figsize=(15,8), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto' \
                          )
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
        plt.colorbar(hfig,format='%1.1f')
        plt.rcParams['font.size'] = 16
        fig.tight_layout()
        flout = '../png/30_marm_flat/inp_flat.png'
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
    
    
    tap_x = taper(100,10,nx)
    tap_z = taper(15,5,nz)
    inp_taper_l = inp_flat * tap_x
    inp_taper_l_r = inp_taper_l * tap_x[::-1]
    inp_taper_l_r_top =  np.transpose(inp_taper_l_r.T *tap_z)
    inp_taper_all =  np.transpose(inp_taper_l_r_top.T *tap_z[::-1])  
    
   
    
    # inp_taper_adbetap = 1/inp_taper_all**2 - 1/inp_smooth**2
    
    # inp_taper_corr = inp_taper_all + inp_smooth
    # inp_taper_adbetap_exact = 1/inp_taper_corr**2 - 1/inp_smooth**2
    
   
    # plot_model_t(inp_taper_corr)
    # plot_model_t(inp_taper_all)
    # plot_model_t(inp_taper_adbetap)
    
    # plot_model_t(inp_taper_adbetap_exact)
       
    # gt.writebin(inp_taper_all,'../input/30_marm_flat/inp_flat_taper.dat')
    
    
    inp_const = new_sm + inp_taper_all
    inp_adbetap_const = 1/inp_const**2 - 1/new_sm**2
    plot_model_t(inp_flat)
    plot_model_t(inp_const)
    plot_model_t(inp_adbetap_const)
    
    # gt.writebin(inp_const,'../input/31_const_flat_tap/inp_flat_2050_const.dat')

    
    
#%%
    fl1       = '../input/30_marm_flat/inp_flat_taper_corr_org.dat'
    fl2       = '../input/16_densite_constante/rho_smooth_2x.dat'
    inp_taper_corr  = gt.readbin(fl1,nz,nx)
    inp_smooth= gt.readbin(fl2,nz,nx)
    
    
    f = interpolate.RegularGridInterpolator((az,ax), inp_smooth,method='cubic',bounds_error=False, fill_value=None) 
    az_new = np.linspace(az[0], az[-1], 301)
    ax_new = np.linspace(ax[0], ax[-1], 1201)
    AZ, AX = np.meshgrid(az_new, ax_new, indexing='ij')
    INT_inp_smooth = f((AZ,AX))
    
    
    plot_model_t(INT_inp_smooth)
    gt.writebin(INT_inp_smooth,'../input/30_marm_flat/rho_smooth_2x.dat')
    
    # plot_model_t(INT_inp_taper_corr)
    # gt.writebin(INT_inp_taper_corr,'../input/30_marm_flat/rho_full_2x.dat')
    
#%%
    fl1       = '../input/27_marm/diff_marm.dat'
    fl2       = '../input/27_marm/marm2_sm15.dat'
    inp_diff  = gt.readbin(fl1,nz,nx)
    inp_smooth= gt.readbin(fl2,nz,nx)
    
    
    
    
    inp_ano   = inp_diff+inp_smooth-1
    
    hmin = 0
    hmax = 4.5
    
    fig = plt.figure(figsize=(15,8), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp_ano, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect='auto', cmap='jet'
                      )
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig,format='%1.2f')
    plt.rcParams['font.size'] = 16
    fig.tight_layout()
    flout = '../png/27_marm/marm_ano_corr.png'
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')

    gt.writebin(inp_ano,'../input/27_marm/marm_ano_corr.dat')
    
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
    
  