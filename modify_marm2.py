# -*- coding: utf-8 -*-

"""
Display the results
"""

import os
import numpy as np
from math import log, sqrt, log10, pi, cos, sin, atan
import csv
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
    nt        = 1801
    dt        = 2.08e-3
    ft        = -100.11e-3
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

    hmin,hmax = 1.5,4.0
    fl1       = '../input/org/marm2_sel.dat'
    inp_org   = gt.readbin(fl1,nz,nx)
    inp1      = gt.readbin(fl1,nz,nx)
    perc      = 0.9
    
        
    def plot_model(inp,hmin,hmax):
        plt.rcParams['font.size'] = 20
        fig = plt.figure(figsize=(16,8), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto'
                         )
        plt.colorbar(hfig)
        plt.xlabel('Distance (Km)')
        plt.ylabel('Profondeur (Km)')
        fig.tight_layout()
        return fig
        
    def export_model(inp,fig,imout,flout):
        fig.savefig(imout, bbox_inches='tight')
        gt.writebin(inp,flout)  
    
#%%

    new_sm = np.asarray(inp_org)
    new_sm = new_sm * 0 + 2.0
    
    gt.writebin(new_sm,'Solution_analytique_Hankel/input/2000_sm_constant.dat')
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
    
    plot_model(inp1,hmin,hmax)

    
    #%% MODIFY MARMOUSI2 FOR DEMIGRATION
    
    fl1       ='../input/45_marm_ano_v3/fwi_org.dat'
    # fl1       = '../input/org_full/marm2_full.dat'
    # fl1       = '../input/45_marm_ano_v3/fwi_ano_45.dat'
    
    fl2       = '../input/marm2_sm15.dat'
    
    fl2 = '../input/vel_smooth.dat'
    inp_org   = gt.readbin(fl1,nz,nx)
    inp_sm    = gt.readbin(fl2,nz,nx)
    plot_model(inp_sm,hmin,hmax)
    
    inp_sm15 = gaussian_filter(inp_org,15)
    plot_model(inp_org,hmin,hmax)
    plot_model(inp_sm15,hmin,hmax)
    
    
    x1 = 220
    x2 = 300
    z1 = 78
    z2 = 100
    
    # x1 = 295
    # x2 = 317
    # z1 = 75
    # z2 = 100
    
    inp_cut    = inp_org[z1:z2,x1:x2]     # Zone B - Cut the model in the area of the anomaly
    old_vel    = np.max(inp_cut)                # Find the value where the anomaly wants to be changed    
    
    hmin = 1.5
    hmax = 4.5
    
    plot_model(inp_org,hmin,hmax)
      
    
    def modif_layer(inp1,r1,r2,nv): 
        area      = np.zeros(inp1.shape)
        for i in range(z1,z2): 
            for j in range(x1,x2):
                if inp1[i,j] > r1 and inp1[i,j] < r2 : 
                    area[i,j] = 1
                else: 
                    area[i,j] = 0
                
                
        index1     = np.where(area == 1)
        # new_vel    = nv
        inp1_before = inp1[index1]
        new_vel   = inp1[index1]*1.14
       
        inp1[index1] = new_vel
        
        # print(new_vel - inp1_before)
        # print(np.mean(new_vel - inp1_before))
        return inp1,index1
    
        
    # inp_mod, ind_mod = modif_layer(inp_org, 2.55, 2.649, 4.5)
    
    inp_mod, ind_mod = modif_layer(inp_org, 2.5, 2.65, 5)
    
    fig_mod = plot_model(inp_mod,hmin,hmax)
    imout_mod = '../png/50_ts_model/full_mod.png'
    flout_mod = '../input/50_ts_model/full_mod.dat'
    # export_model(np.array(inp_mod),fig_mod,imout_mod,flout_mod)
    
    
    inp_mod_sm = gaussian_filter(inp_mod,3)
    fig_mod_sm = plot_model(inp_mod_sm,hmin,hmax)
    imout1 = '../png/50_ts_model/sm3_ano.png'
    flout1 = '../input/50_ts_model/sm3_ano.dat'
    # export_model(inp_mod_sm,fig_mod_sm,imout1,flout1)
    
    inp_sm5 = gaussian_filter(inp_org,3)
    fig_sm5 = plot_model(inp_sm5,hmin,hmax)
    imout_sm5 = '../png/50_ts_model/sm3_org.png'
    flout_sm5 = '../input/50_ts_model/sm3_org.dat'
    # export_model(inp_sm5,fig_sm5,imout_sm5,flout_sm5)
    
    plt.plot(inp_mod_sm[:,266]); plt.plot(inp_sm5[:,266])
    
    fl1       = '../input/org_full/marm2_full.dat'
    inp_org   = gt.readbin(fl1,nz,nx)
    inp_diff = (inp_mod - inp_org)
    inp_diff = (inp_mod - inp_org)*1.14
    plot_model(inp_cut,hmin,hmax)
    
    
    
    inp_diff10 = inp_diff + 1
    
    inp_diff10[ind_mod] = 4.0
      
    # az[ind_mod[0][63]]
    # ax[ind_mod[1][63]] 
    
    hmax = np.max(inp_diff)
    hmin = np.min(inp_diff)
    fig1 = plot_model(inp_diff,hmin,hmax)
    imout1 = '../png/45_marm_ano_v3/fwi_diff_ano.png'
    flout1 = '../input/45_marm_ano_v3/fwi_diff_ano.dat'
    # export_model(inp_diff,fig1,imout1,flout1)
    
    
    inp_diff_sm = inp_diff+inp_sm
    
    hmax = np.max(inp_diff_sm)
    hmin = np.min(inp_diff_sm)
    fig1 = plot_model(inp_diff_sm,hmin,hmax)
    imout1 = '../png/45_marm_ano_v3/fwi_diff_sm_ano.png'
    flout1 = '../input/45_marm_ano_v3/fwi_diff_sm_ano.dat'
    # export_model(inp_diff_sm,fig1,imout1,flout1)
    
    # adbetap_exact = np.copy(inp_diff)
    
    # for i in range(nz):
    #     for j in range(nx): 
    #         if inp_diff[i,j] == 0: 
    #             adbetap_exact[i,j] = 0
    #         else:
    #             adbetap_exact[i,j] = 1/inp_diff_sm[i,j]**2 - 1/inp_sm[i,j]**2
    
    # hmin = np.min(adbetap_exact)
    # hmax = np.max(adbetap_exact)
    # fig1 = plot_model(adbetap_exact,hmin,hmax)
    # imout1 = '../png/45_marm_ano_v3/adbetap_diff_sm_ano.png'
    # flout1 = '../input/45_marm_ano_v3/adbetap_diff_sm_ano.dat'
    # export_model(adbetap_exact,fig1,imout1,flout1)
    
    
    # fig1 = plot_model(inp_mod,hmin,hmax)
    # # imout1 = '../png/45_marm_ano_v3/fwi_ano_full_114_percent.png'
    # # flout1 = '../input/45_marm_ano_v3/fwi_ano_fault_114_percent.dat'
    # # export_model(inp_mod,fig1,imout1,flout1)
    
    # fig1 = plot_model(inp_org,hmin,hmax)
    # # imout1 = '../png/45_marm_ano_v3/fwi_ano_org.png'
    # # flout1 = '../input/45_marm_ano_v3/fwi_org.dat'
    # # export_model(inp_org,fig1,imout1,flout1)
    
    
    # inp_sm = gaussian_filter(inp_org,15)
    # fig1 = plot_model(inp_sm,hmin,hmax)
    # imout1 = '../png/45_marm_ano_v3/fwi_sm.png'
    # flout1 = '../input/45_marm_ano_v3/fwi_sm.dat'
    # export_model(inp_sm,fig1,imout1,flout1)
    
    # import pandas as pd
    # df = pd.DataFrame(np.transpose(ind_mod))
    # df.to_csv('../input/45_marm_ano_v3/fwi_ano_114_percent.csv',header=False,index=False)
    
    def depth_to_time(ray_z, ray_x, vel_ray):
        '''Transform profile from depth to time using the velocity '''
        ray_time = []
        dz = []
        v0 = []
        time = [0]
        dz0 = []
        for i in range(len(ray_x)//2-1):
            dz = np.sqrt((ray_z[i] - ray_z[i+1])**2 + (ray_x[i] - ray_x[i+1])**2)
            v0 = (vel_ray[i]+vel_ray[i+1])/2
            time.append(dz/v0)
            print('dz: ', dz, 'v0: ', v0, 'time: ', time[-1]*2)
        ray_time = np.cumsum(time)
    
        ray_time = np.array(ray_time)*2
        return ray_time
    
    
    from spotfunk.res import procs, visualisation
    
    # at_z = np.zeros_like(inp_sm)
    # nx = len(inp_sm[0])
    # nz = len(inp_sm[:,0])
    
    # for i in range(nz): 
    #     for j in range(nx): 
    #         at_z[i] = az[i]/inp_sm[i,j]
            
    # at_z = np.reshape(at_z,(nz,nx))
    
    
    
    # hmin = np.min(inp_rms)
    # hmax = np.max(inp_rms)
    # plot_model(inp_rms,hmin,hmax)
    
    
    
    # plot_model(inp_mod_sm, hmin, hmax)
 #%%   
    # az[ind_mod[0][63]]
    # ax[ind_mod[1][63]]
    
    
    fig1 = plot_model(inp_mod,hmin,hmax)
    imout1 = '../png/41_marm_ano_new/marm_ano_full_55.png'
    flout1 = '../input/41_marm_ano_new/marm_ano_full_55.dat'
    
    def read_pick(path,srow):
        attr = []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            # header = next(spamreader)
            for row in spamreader:
                attr.append(float(row[srow]))
        return attr
    
    def pente_function(path_slope,ax):
        path_demig = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023'
        file = path_demig + path_slope
        point_x = np.array(read_pick(file,0))/1000
        point_z = -np.array(read_pick(file,2))/1000
        
        print(point_z, point_x)
        m = (point_z[1] - point_z[0])/(point_x[1] - point_x[0])
        print(m)
        b = point_z[0] - point_x[0] * m
        print(b)
        z = ax * m + b
        return z, m
    
    
    def fill_custom_model(V_model,pente_1,v_couche_1,v_couche_2,az):   
        print('shape',V_model.shape)
        ''' 
        Calculates the custom model with a slope function 
        The input V_model has shape : (601, 151)
        '''
        
        for k in range(V_model.shape[0]):
            
            for i,z in enumerate(az):
                
                if z < pente_1[k]:
                    V_model[k,i] = v_couche_1
                    
  
                else:
        
                    V_model[k,i] = v_couche_2
        
        return V_model   
    
      
    inp_anomaly = inp_org * 0 
    inp_anomaly[ind_mod] = 4.0
    inp_anomaly_sm = inp_anomaly + inp_sm
    plot_model(inp_anomaly_sm,hmin,hmax)
    
    
    # imout_2 = '../png/41_marm_ano_new/marm_ano_only_40.png'
    # flout_2 = '../input/41_marm_ano_new/marm_ano_only_40.dat'
    # export_model(inp_mod,fig1,imout_2,flout_2)
    
    ## Construct a slope model
    path_slope_inv = '/output/041_marm2_slope_binv_2946/slope_binv.csv'
    pente_1, m = pente_function(path_slope_inv, ax)
    inp_slope_model = fill_custom_model(inp_org.T, pente_1,1.5,2.5,az).T
    
    
    # Create the model 
    plot_model(inp_slope_model,hmin,hmax)
    imout_3 = '../png/41_marm_ano_new/slope_only_25.png'
    flout_3 = '../input/41_marm_ano_new/slope_only_25.dat'
    # export_model(inp_slope_model,fig1,imout_3,flout_3)
    
    
#%% FLAT INTERFACE
    # fl1       = '../input/org_full/marm2_full.dat'
    # fl2       = '../input/27_marm/marm2_sm15.dat'
    
    # fl1 = '../input/vel_full.dat'
    # fl2 = '../input/vel_smooth.dat'
    
    # fl1 = '../input/27_marm/inp_flat.dat'
    fl1 = '../input/33_report_model/vel_full.dat'
    fl2 = '../input/marm2_sm15.dat'

    inp_org   = gt.readbin(fl1,nz,nx)
    inp_smooth= gt.readbin(fl2,nz,nx)
    inp_flat  = inp_org * 0
    
  
    
 
    
    # inp_flat[0:100] = 1.5
    # inp_flat[51:100] = inp_flat[51:100]+0.05
    
    f_idx = 51
    l_idx = 100
    # l_idx = 52
    
    # for i in range(0,49,4): 
    #     inp_flat[f_idx+i:l_idx+i] = inp_flat[f_idx+i:l_idx+i] + 0.015*(i+1)/4  *1.14
    

    inp_flat[f_idx:l_idx] = inp_flat[f_idx:l_idx] + 0.1*1.14
        
    thick = (l_idx - f_idx) * dz
    
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
        fig = plt.figure(figsize=(14,7), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto' \
                          )
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
        plt.colorbar(hfig,format='%1.1f')
        plt.rcParams['font.size'] = 20
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
    
    
    # inp_const = new_sm + inp_taper_all
    # inp_adbetap_const = 1/inp_const**2 - 1/new_sm**2
    # plot_model_t(inp_flat)
    # plot_model_t(inp_const-2)
    # plot_model_t(inp_adbetap_const)
    
    # gt.writebin(inp_const,'../input/31_const_flat_tap/inp_flat_2050_const.dat')
    # gt.writebin(inp_const*1000,'Solution_analytique_Hankel/input/2050_dp_inp_flat.dat')
    


    '''Modification of the flat model of two interfaces with marm_smooth15'''
    
    # inp_corr = inp_org+inp_smooth-2
    # plot_model_t(inp_corr)
    # plot_model_t(inp_smooth)
    # gt.writebin(inp_corr,'../input/39_mig_marm_flat/vel_marm_plus_flat_corr.dat')
    
    ''' Creation of models for artificial traces'''
    
    plot_model(inp_flat, hmax, hmin)
    
    inp_taper_all = inp_taper_all+2.0
    inp_taper_all[0:50] = 2.0
    inp_taper_all[101:] = 2.0
    
    rho_f     = 0.31 * (inp_taper_all * 1000) ** 0.25
    rho_sm = np.zeros_like(rho_f) +  np.min(rho_f)
    
    hmax  = np.max(inp_taper_all)
    hmin  = np.min(inp_taper_all)
    fig1   = plot_model(inp_taper_all, hmax, hmin)
    imout1 = '../png/73_new_flat_sm/vel_'+str(int(thick*1000))+'_ano.png'
    flout1 = '../input/73_new_flat_sm/vel_'+str(int(thick*1000))+'_ano.dat'
    # imout1 = '../png/63_evaluating_thickness/vel_degrade_ano.png'
    # flout1 = '../input/63_evaluating_thickness/vel_degrade_ano.dat'
    # export_model(inp_taper_all,fig1,imout1,flout1)
    
    hmax  = np.max(rho_f)
    hmin  = np.min(rho_f)
    fig1   = plot_model(rho_f, hmax, hmin)
    imout1 = '../png/73_new_flat_sm/rho_'+str(int(thick*1000))+'_ano.png'
    flout1 = '../input/73_new_flat_sm/rho_'+str(int(thick*1000))+'_ano.dat'
    export_model(rho_f,fig1,imout1,flout1)
    
    
    hmax  = np.max(rho_sm)
    hmin  = np.min(rho_sm)
    fig1   = plot_model(rho_sm, hmax, hmin)
    imout1 = '../png/73_new_flat_sm/rho_sm_'+str(int(thick*1000))+'_ano.png'
    flout1 = '../input/73_new_flat_sm/rho_sm_'+str(int(thick*1000))+'_ano.dat'
    export_model(rho_sm,fig1,imout1,flout1)
    
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
    
  