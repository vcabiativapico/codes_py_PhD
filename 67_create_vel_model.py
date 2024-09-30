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
    
    
    
    
    
    
#%% FLAT INTERFACE
    # fl1       = '../input/org_full/marm2_full.dat'
    # fl2       = '../input/27_marm/marm2_sm15.dat'
    
    # fl1 = '../input/vel_full.dat'
    # fl2 = '../input/vel_smooth.dat'
    
    # fl1 = '../input/27_marm/inp_flat.dat'
   
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
        
        # hmax = 4.5
        # hmin = 1.5
        plt.rcParams['font.size'] = 20
        hmax = np.max(inp)
        hmin = np.min(inp)
        # hmin = -hmax
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
    
    fl1 = '../input/33_report_model/vel_full.dat'
    fl2 = '../input/marm2_sm15.dat'

    inp_org   = gt.readbin(fl1,nz,nx)
    inp_smooth= gt.readbin(fl2,nz,nx)*0+2
    inp_flat  = np.copy(inp_org) * 0
    
    def apply_taper(tap_x,tap_z,inp):
        
        
        inp_taper_l = inp * tap_x
        inp_taper_l_r = inp_taper_l * tap_x[::-1]
        inp_taper_l_r_top =  np.transpose(inp_taper_l_r.T *tap_z)
        inp_taper_all =  np.transpose(inp_taper_l_r_top.T *tap_z[::-1])  
        return inp_taper_all
    
    tap_x = taper(100,10,nx)
    tap_z = taper(15,5,nz)
    
   
    # inp_taper_all=apply_taper(tap_x,tap_z,inp_flat)

    
    # plot_model_t(inp_taper_all)
   
    f_idx = 51
    # l_idx = 100
    l_idx =52
    
    # for i in range(0,49,4): 
    #     inp_flat[f_idx+i:l_idx+i] = inp_flat[f_idx+i:l_idx+i] + 0.015*(i+1)/4  *1.14
    inp_flat  = np.copy(inp_org) * 0 
    for i in range(0,52,4):
        
        
        inp_flat[f_idx+i-4:l_idx+i] = inp_flat[f_idx+i:l_idx+i] +0.05+ 0.015*(i+1)/4 *1.14
        
        thick = (l_idx - f_idx) * dz
        
        inp_taper_all=apply_taper(tap_x,tap_z,inp_flat)+2
    
        ''' Creation of models for artificial traces'''
        
        plot_model_t(inp_taper_all)
        plot_model_t(inp_smooth)
    
        inp_taper_adbetap = 1/inp_taper_all**2 - 1/inp_smooth**2
        fig1 = plot_model_t(inp_taper_adbetap)
        
        
        # imout1 = '../png/67_TS_graded_flat/vel_ano_'+str(int(thick*1000+i*12))+'.png'
        # flout1 = '../input/67_TS_graded_flat/vel_ano_'+str(int(thick*1000+i*12))+'.dat'
        
        imout1 = '../png/67_TS_graded_flat/betap_graded_ano_'+str(int(thick*1000+i*12))+'.png'
        flout1 = '../input/67_TS_graded_flat/betap_graded_ano_'+str(int(thick*1000+i*12))+'.dat'

        
        export_model(inp_taper_adbetap,fig1,imout1,flout1)
    
  
    
  