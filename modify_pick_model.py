#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 10:02:29 2023

@author: vcabiativapico
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel


if __name__ == "__main__":
    
    # Global parameters
    labelsize = 16
    nt        = 1001
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

    hmin,hmax = 1.5,2.5



    # fl1       = './input/18_3_interface/3_interfaces.dat'   
  
    # inp1      = gt.readbin(fl1,nz,nx)
    
   
    
  
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
        
        return inp1
    


    def plot_model(inp,hmax,hmin,flout):
        fig = plt.figure(figsize=(11,6), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto')
        plt.colorbar(hfig,format='%1.1f')
        plt.rcParams['font.size'] = 14
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
        fig.tight_layout() 
        print("Export to file:",flout)
        fig.savefig(flout, bbox_inches='tight')
        return inp[::11,301]

    hmin,hmax = 1.5,3.0

    fl1       = '../input/25_v2_4_layers/4_interfaces_rc_norm.dat'   
    inp1      = gt.readbin(fl1,nz,nx)
    flout     = '../png/25_v2_4_layers/4_interfaces_rc_norm.png'
    plot_model(inp1,3.5,1.5,flout)
    
    
    r1 = 3.3
    r2 = 3.4
    nv1 = 3.238
    inp_modif = modif_layer(inp1,r1,r2,nv1)
    print(inp_modif[::11,301])
    # nv2 = 1.71
    # inp_modif = modif_layer(inp_modif,2.2,2.4,nv2)
    
    flout     = '../png/25_v2_4_layers/4_interfaces_rc_norm.png'
    plot_model(inp_modif,3.5,1.5,flout)
    
    datout = '../input/25_v2_4_layers/4_interfaces_rc_norm.dat'
    gt.writebin(inp_modif,datout) 
    
    # rc = [1.5, 1.8333333333333333, 2.2407407407407405, 2.738683127572016, 3.347279378143575, 4.091119239953258]
    # rc_norm = [1.5, 1.715217391304348, 2.1150406504065042, 2.6135419841124716, 3.2380501871772447]
    # inp_modif[::7,301]
    # inp_modif = modif_layer(inp1,1.6,2.0)
    # imout     = './png/15_picked_models/vel_full_30_CO2.png'
    
    # plot_model(inp_modif,4.0,hmin,imout)
    # flout = './input/15_picked_models/vel_full_30_CO2.dat'
    # # gt.writebin(inp_modif,flout)  
    
    # inp_modif_smoo = gaussian_filter(inp1,5)
    # imout = './png/19_anomaly_4_layers/3_interfaces_org_smooth.png'
    # plot_model(inp_modif_smoo,3.0,1.5,imout)
    
    # flout = './input/19_anomaly_4_layers/3_interfaces_org_smooth.dat'
    # gt.writebin(inp_modif_smoo,flout) 
    
#%% Gardner equation to build rho models

    def rho_from_vel(title,hmin,hmax):
        ## Converts velocity models into density models with the Gardner equation
        ## Plots model and exports .dat and .png
        
        # INPUT VELOCITY MODEL AS .DAT
        fl1   = './input/19_anomaly_4_layers/3_interfaces_'+str(title)+'.dat'
        inp1      = gt.readbin(fl1,nz,nx)
        # DENSITY MODEL
        rho_f     = 0.31 * (inp1 * 1000) ** 0.25
        
        # OUTPUT VELOCITY MODEL AS .DAT
        # datout     = './input/20_density_4_layers/rho_full_'+str(title)+'.dat'
        # gt.writebin(rho_f,datout)
        # PLOT VELOCITY
        fig       = plt.figure(figsize=(10,5), facecolor = "white")
        av        = plt.subplot(1,1,1)
        hfig      = av.imshow(rho_f, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto')
        flout     = './png/20_density_4_layers/rho_full_'+str(title)+'.png'
        plt.colorbar(hfig)
        fig.tight_layout() 
        print("Export to file:",flout)
        fig.savefig(flout, bbox_inches='tight')
        gt.writebin(rho_f,'./input/20_density_4_layers/rho_full_'+str(title)+'.dat') 
        return rho_f
    

    
    def smooth_rho(rho,title,hmin,hmax):
        
        rho_sm = gaussian_filter(rho, 5)
        # datout     = './input/20_density_4_layers/rho_smooth_'+str(title)+'.dat'
        # gt.writebin(rho_sm,datout)
        # PLOT VELOCITY
        fig       = plt.figure(figsize=(10,5), facecolor = "white")
        av        = plt.subplot(1,1,1)
        hfig      = av.imshow(rho_sm, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto')
        # flout     = './png/20_density_4_layers/rho_smooth_'+str(title)+'.png'
        # plt.colorbar(hfig)
        # fig.tight_layout() 
        # print("Export to file:",flout)
        # fig.savefig(flout, bbox_inches='tight')
       
        
    hmin  = 1.9
    hmax  = 2.3
    
    
    # # rho_full_org   = rho_from_vel('org',hmin,hmax)
    # rho_full_150   = rho_from_vel('anomaly_150',hmin,hmax)
    # rho_full_160   = rho_from_vel('anomaly_160',hmin,hmax)
    # rho_full_175   = rho_from_vel('anomaly_175',hmin,hmax)
    # rho_full_180   = rho_from_vel('anomaly_180',hmin,hmax)

   
    # rho_full_190   = rho_from_vel('anomaly_190',hmin,hmax)
    # rho_full_200   = rho_from_vel('anomaly_200',hmin,hmax)
    # rho_full_210   = rho_from_vel('anomaly_210',hmin,hmax)
    # rho_full_230   = rho_from_vel('anomaly_230',hmin,hmax)
    # rho_full_250   = rho_from_vel('anomaly_250',hmin,hmax)
    
    
    
    # rho_smooth_org  = smooth_rho(rho_full_org,'org',hmin,hmax)

    # rho_smooth_171  = smooth_rho(rho_full_171,171,hmin,hmax)
    # rho_smooth_180  = smooth_rho(rho_full_180,180,hmin,hmax)
    # rho_smooth_190  = smooth_rho(rho_full_190,190,hmin,hmax)
    # rho_smooth_210  = smooth_rho(rho_full_210,210,hmin,hmax)
    # rho_smooth_230  = smooth_rho(rho_full_230,230,hmin,hmax)
    # rho_smooth_250  = smooth_rho(rho_full_250,250,hmin,hmax)
    
 
#%% Creation of density model according to Gardner's equation    
    # r1 = 1.6
    # r2 = 2.0       
    # inp_org   = gt.readbin(fl1,nz,nx)
    # inp_modif = modif_layer(inp_org,r1,r2)    
    # rho_full_modif = 0.31 * (inp1 * 1000) ** 0.25
    # # rfm_out = './png/15_picked_models/rho_full_3_CO2.png'
    # # gt.writebin(rho_full_modif,'./input/15_picked_models/rho_full_3_CO2.dat' )   
    # # plot_model(rho_full_modif,1.8,2.2,rfm_out)
    
  
    
    
    # rho_full = 0.31 * (inp_org * 1000) ** 0.25
    # # rf_out = './png/15_picked_models/rho_full_0_CO2.png'
    # # gt.writebin(rho_full,'./input/15_picked_models/rho_full_0_CO2.dat' ) 
    # # plot_model(rho_full,1.8,2.2,rf_out)
    
    
    # vel_full_3_CO2 = inp_modif
    # # vf3_out = './png/15_picked_models/vel_full_3_CO2.png'
    # # gt.writebin(vel_full_3_CO2,'./input/15_picked_models/vel_full_3_CO2.dat' ) 
    # # plot_model(vel_full_3_CO2,1.7,2.5,vf3_out)
    
    # fl1 = './input/15_picked_models/vel_full_3_CO2.dat'
    # inp_test   = gt.readbin(fl1,nz,nx)
    # vel_sm_3_CO2 = gaussian_filter(inp_test,2)
    # vf3_out = './png/15_picked_models/vel_smooth_3_CO2.png'
    # gt.writebin(vel_sm_3_CO2,'./input/15_picked_models/vel_smooth_3_CO2.dat' ) 
    # plot_model(vel_sm_3_CO2,1.7,2.5,vf3_out)

    
    
    # flout = './png/15_picked_models/vel_full_0_CO2.png'
    # gt.writebin(inp1,'./input/15_picked_models/vel_full_0_CO2.dat' )
    
    # flout = './png/15_picked_models/vel_full_3_CO2.png'
    # gt.writebin(inp1,'./input/15_picked_models/vel_full_3_CO2.dat' )
    
   
   
    
    

    
#%% Smoothing of models    
    
    
    # rho_sm = gaussian_filter(rho_full, 2)
    # rho_sm_modif = gaussian_filter(rho_full_modif, 2)
    
    # # rfs_out = './png/15_picked_models/rho_smooth_0_CO2.png'
    # # gt.writebin(rho_sm,'./input/15_picked_models/rho_smooth_0_CO2.dat' ) 
    # # plot_model(rho_sm,1.8,2.2,rfs_out)
    
    
    # rfms_out = './png/15_picked_models/rho_smooth_3_CO2.png'
    # gt.writebin(rho_sm_modif,'./input/15_picked_models/rho_smooth_3_CO2.dat' )   
    # plot_model(rho_sm_modif,1.8,2.2,rfms_out)
    
    #%%
    fl1       = './input/18_3_interface/3_interfaces.dat'   
    fl2       = './input/18_3_interface/3_interfaces_smooth.dat'    
    inp_f    = gt.readbin(fl1,nz,nx)
    inp_sm   = gt.readbin(fl2,nz,nx)
    
    inp_dp    = inp_f - inp_sm
    
    gt.writebin(inp_dp,'./input/18_3_interface/3_interfacess_dp.dat' )
    
    