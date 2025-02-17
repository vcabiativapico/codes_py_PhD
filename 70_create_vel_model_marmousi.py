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
import pandas as pd


if __name__ == "__main__":
    
    # Global parameters
    labelsize = 16
    nt        = 1801
    dt        = 1.14e-3
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

        plt.rcParams['font.size'] = 20
        hmax = np.max(inp)
        hmin = np.min(inp)

        fig = plt.figure(figsize=(14,7), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto' \
                          )
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
        plt.colorbar(hfig,format='%1.2f')
        
        fig.tight_layout()
        # flout = '../png/30_marm_flat/inp_flat.png'
        # print("Export to file:", flout)
        # fig.savefig(flout, bbox_inches='tight')
        return fig
    
   
    def apply_taper(tap_x,tap_z,inp):   
        inp_taper_l = inp * tap_x
        inp_taper_l_r = inp_taper_l * tap_x[::-1]
        inp_taper_l_r_top =  np.transpose(inp_taper_l_r.T *tap_z)
        inp_taper_all =  np.transpose(inp_taper_l_r_top.T *tap_z[::-1])  
        return inp_taper_all
    
    # fl1 = '../input/org_full/marm2_full.dat'
    # fl2 = '../input/marm2_sm15.dat'

    # inp_org    = gt.readbin(fl1,nz,nx)
    # inp_smooth = gt.readbin(fl2,nz,nx)
    # inp_ano    = np.copy(inp_org) 
    
    # tap_x = taper(100,10,nx)
    # tap_z = taper(15,5,nz)
    
   

    # fig1 = plot_model_t(inp_org)
    
    # imout1 = '../png/67_TS_graded_flat/betap_graded_ano_.png'
    # flout1 = '../input/67_TS_graded_flat/betap_graded_ano_.dat'

    
    # export_model(inp_org,fig1,imout1,flout1)
    
#%%
 
    # fl1       ='../input/45_marm_ano_v3/fwi_org.dat'
    fl1       = '../input/org_full/marm2_full.dat'
    # fl1       = '../input/45_marm_ano_v3/fwi_ano_45.dat'
    
    fl2       = '../input/marm2_sm15.dat'
    inp_org   = gt.readbin(fl1,nz,nx)
    # inp_sm    = gt.readbin(fl2,nz,nx)
    
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
    
    # plot_model(inp_sm,hmin,hmax)
      
    
    def modif_layer(inp1,r1,r2,nv, new_vel):
        fl1       = '../input/org_full/marm2_full.dat'
        inp_org   = gt.readbin(fl1,nz,nx)
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
        # new_vel   = inp1[index1] 
        # print(inp1_before, np.mean(new_vel))
        # # new_vel = 2.9425097 # -> np.mean(new_vel)
        # new_vel = 2.6585832
        inp_mod = np.copy(inp_org)
        inp_mod[index1] = new_vel
        
        # print(new_vel - inp1_before)
        # print(np.mean(new_vel - inp1_before))
        return inp_mod,index1
    
    def create_ano(inp_org,new_vel):  
        
        inp_mod, idx_mod = modif_layer(inp_org, 2.5, 2.65, 5, new_vel)
        
        
        new_idx1 = (idx_mod[0]+2,idx_mod[1]+7)
        inp_ano_thick = np.copy(inp_mod)
        inp_ano_thick[new_idx1] = new_vel
        
        new_idx = (idx_mod[0]+1,idx_mod[1]+3)
        inp_ano_thick[new_idx] = new_vel
        
        # inp_ano_thick =np.copy(inp_mod)
        return inp_ano_thick, new_idx1, new_idx,idx_mod

    new_vel_org = 2.6585832
    new_vel_ano = 2.6585832*1.03
    
    inp_org_thick,new_idx2,new_idx,idx_mod = create_ano(inp_org, new_vel_org) 
    inp_ano_thick,new_idx2,new_idx, idx_mod= create_ano(inp_org, new_vel_ano)    
    
    
    
    def plot_model2(inp,hmin,hmax):
        plt.rcParams['font.size'] = 20
        fig = plt.figure(figsize=(16,8), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(inp, 
                          vmin=hmin,vmax=hmax,aspect='auto'
                         )
        plt.colorbar(hfig)
        plt.xlabel('Distance (Km)')
        plt.ylabel('Profondeur (Km)')
        fig.tight_layout()
        return fig
  
    
    inp_two_ano = np.copy(inp_ano_thick)
    
    
    vel_to_org = new_vel_org
    for i in range(7):
        inp_two_ano[86,279+i] =vel_to_org
    for i in range(12):    
        inp_two_ano[87,279+i-1] = vel_to_org
    for i in range(16):
        
        
        inp_two_ano[88,279+i-2] = vel_to_org
        inp_two_ano[89,279+i-4] = vel_to_org
        inp_two_ano[90,279+i-5] = vel_to_org
        inp_two_ano[91,279+i-6] = vel_to_org
        inp_two_ano[91,279+i-7] = vel_to_org
    for i in range(14):   
        inp_two_ano[92,279+i-5] = vel_to_org
    for i in range(7):
        inp_two_ano[93,279+i] = vel_to_org
    
    
    
    
    fig_org_thick = plot_model2(inp_org_thick,hmin,hmax)
    
    fig_two_ano = plot_model2(inp_two_ano,hmin,hmax)
    imout_two_ano = '../png/68_thick_marm_ano/marm_thick_two_ano.png'
    flout_two_ano = '../input/68_thick_marm_ano/marm_thick_two_ano.dat'
    # export_model(inp_two_ano,fig_org_thick,imout_two_ano,flout_two_ano)
    
    
    # flnam2 = '../input/68_thick_marm_ano/new_idx.dat'
    # df = pd.DataFrame(new_idx).T
    # df.to_csv(flnam2, header=False, index=False)

    # flnam2 = '../input/68_thick_marm_ano/new_idx2.dat'
    # df = pd.DataFrame(new_idx2).T
    # df.to_csv(flnam2, header=False, index=False)

    # flnam2 = '../input/68_thick_marm_ano/idx_mod.dat'
    # df = pd.DataFrame(idx_mod).T
    # df.to_csv(flnam2, header=False, index=False)


    # plot_model( -inp_org_thick+inp_ano_thick ,0,0.08)
    
    
    ## Anomaly thick full org
    fig_org_thick = plot_model(inp_org_thick,hmin,hmax)
    imout_org_thick = '../png/68_thick_marm_ano/marm_thick_org.png'
    flout_org_thick = '../input/68_thick_marm_ano/marm_thick_org.dat'
    # export_model(inp_org_thick,fig_org_thick,imout_org_thick,flout_org_thick)
 
    
    ## Anomaly thick full ano
    fig_mod = plot_model(inp_ano_thick,hmin,hmax)
    imout_mod = '../png/68_thick_marm_ano/marm_thick_ano.png'
    flout_mod = '../input/68_thick_marm_ano/marm_thick_ano.dat'
    # export_model(inp_ano_thick,fig_mod,imout_mod,flout_mod)
    
    ## Anomaly thick full smooth
    inp_ano_thick_sm = gaussian_filter(inp_ano_thick,7)
    fig_ano_thick_sm = plot_model(inp_ano_thick_sm,hmin,hmax)
    imout_ano_thick_sm = '../png/69_thin_marm_ano/marm_thick_ano_sm7.png'
    flout_ano_thick_sm = '../input/69_thin_marm_ano/marm_thick_ano_sm7.dat'
    # export_model(inp_ano_thick_sm,fig_ano_thick_sm,imout_ano_thick_sm,flout_ano_thick_sm)
    
    # ## Original smooth 5
    inp_org_thick_sm = gaussian_filter(inp_org_thick,7)
    fig_org_thick_sm = plot_model(inp_org_thick_sm,hmin,hmax)
    imout_org_thick_sm = '../png/69_thin_marm_ano/marm_thick_org_sm7.png'
    flout_org_thick_sm = '../input/69_thin_marm_ano/marm_thick_org_sm7.dat'
    # export_model(inp_org_thick_sm,fig_org_thick_sm,imout_org_thick_sm,flout_org_thick_sm)
    
    plt.figure()
    plt.plot(inp_org_thick_sm[:,266])
    plt.plot(inp_ano_thick_sm[:,266])
    
    plt.figure()
    plt.plot(inp_org_thick_sm[:,266]-inp_ano_thick_sm[:,266])
    
    # ## Original betap original smooth 5
    # adbetap_exact_org = 1/inp_org_thick**2 - 1/inp_org_thick_sm**2
    # fig_adbetap_exact_org = plot_model(adbetap_exact_org,np.min(adbetap_exact_org),np.max(adbetap_exact_org))
    # imout_adbetap_exact_org = '../png/68_thick_marm_ano/adbetap_exact_org6.png'
    # flout_adbetap_exact_org = '../input/68_thick_marm_ano/adbetap_exact_org6.dat'
    # export_model(adbetap_exact_org,fig_adbetap_exact_org,imout_adbetap_exact_org,flout_adbetap_exact_org)

    # ## Original betap anomaly thick smooth 5
    # adbetap_exact_ano = 1/inp_ano_thick**2 - 1/inp_ano_thick_sm**2
    # fig_adbetap_exact_ano = plot_model(adbetap_exact_ano,np.min(adbetap_exact_ano),np.max(adbetap_exact_ano))
    # imout_adbetap_exact_ano = '../png/68_thick_marm_ano/adbetap_exact_ano6.png'
    # flout_adbetap_exact_ano = '../input/68_thick_marm_ano/adbetap_exact_ano6.dat'
    # export_model(adbetap_exact_ano,fig_adbetap_exact_ano,imout_adbetap_exact_ano,flout_adbetap_exact_ano)

    