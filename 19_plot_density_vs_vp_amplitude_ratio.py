#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:26:30 2023

@author: vcabiativapico
"""


import os
import numpy as np
from math import log, sqrt, log10, pi, cos, sin, atan
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from tabulate import tabulate

if __name__ == "__main__":

    os.system('mkdir -p png_course')
    
    # Global parameters
    labelsize = 16
    nt        = 1501
    dt        = 1.41e-3
    ft        = -100.11e-3
    nz        = 151
    fz        = 0.0
    dz        = 12.0/1000.
    nx        = 601
    fx        = 0.0
    dx        = 12.0/1000.
    no        = 403
   # no        = 2002
    do        = dx
    fo        = -(no-1)/2*do
    ao        = fo + np.arange(no)*do
    at        = ft + np.arange(nt)*dt
    az        = fz + np.arange(nz)*dz
    ax        = fx + np.arange(nx)*dx
#%% 
    def plot_trace(inp1,inp2,flout,tr):
        
        axi = np.zeros(np.size(tr))
        fig,(axi)  = plt.subplots(nrows=1,ncols=np.size(tr),
                                      sharey=True,
                                      figsize=(8,8),
                                      facecolor = "white")
        
        amp = (3,3)
        amp = np.zeros(amp)
        for i in range(np.size(tr)):    
            # xmin = np.min(inp1[:,tr[i]]) + np.min(inp1[:,tr[i]])/2
            # xmax = -xmin
            # xmin = -xmax   
            tr_inp1 = inp1[:,tr[i]]
            theset = np.min(tr_inp1)
            idx = np.where(theset == tr_inp1)
            axi[i].plot(theset,at[idx],'o')
            
            
            tr_inp2 = tr_inp1[int(idx[0])+100:]
            theset2 = np.min(tr_inp2)#[::-1]
            idx2 = np.where(theset2 == tr_inp1)
            axi[i].plot(theset2,at[idx2],'o')    
            print(idx2)    
            
            tr_inp3 = tr_inp1[int(idx2[0])+50:]
            theset3 = np.min(tr_inp3)#[::-1]
            idx3 = np.where(theset3 == tr_inp1)
            axi[i].plot(theset3,at[idx3],'o')    
            print(idx3)   
            
            
            amp[i] = [theset,theset2,theset3]
            
            if i >-1: xmax,xmin = 0.2,-0.2
            axi[i].plot(inp1[:,tr[i]],at,'r')
            # axi[i].plot(inp2[:,tr[i]],at,'r--')
            # axi[i].plot(inp3[:,tr[i]],at,'r--')
            axi[i].set_xlim(xmin,xmax)
            axi[i].set_ylim(2,ft)  
            axi[i].xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
            # Calculate the peak
            
            # print(np.where(theset[0] == trace_inp1))
            
            # plt.colorbar()
            fig.tight_layout()
            axi[0].set_ylabel('Time (s)')
        
        
        # axi[0].legend(['Base','Monitor'],loc='upper left',shadow=True)
        # axi[0].legend(['Born','FWI'],loc='upper left',shadow=True)
        # # axi[0].legend(['Diff'],loc='upper left',shadow=True)
        fig.text(0.48, -0.01, "Amplitude")
        fig.text(0.48, 1, 'Difference')
        print("Export to file:",flout)
        fig.savefig(flout, bbox_inches='tight')
        return amp
#%%


   # tr1  = './output/15_picked_models/4_CO2/born/t1_obs_000301.dat'     
   # tr2  = './output/15_picked_models/4_CO2/fwi/t1_obs_000301.dat' 
   # tr3  = './output/18_3_interfaces/org/born/t1_obs_000301.dat' 
   
   # tr1  = './output/18_3_interfaces/org/fwi/t1_obs_000301.dat'
   # tr2  = './output/18_3_interfaces/soft_anomaly/fwi/t1_obs_000301.dat'
    
    num = 250
    
    tr1 = './output/19_anomaly_4_layers/org/born/t1_obs_000301.dat'
    tr2 = './output/19_anomaly_4_layers/ano_'+str(num)+'/born/t1_obs_000301.dat'
    
    # tr1  = './output/t1_obs_000301.dat'     
    # tr2  = './output/17_picked_models_rho/rho_0/born/t1_obs_000301.dat'
    
    inp1 = gt.readbin(tr1,no,nt).transpose()
    inp2 = gt.readbin(tr2,no,nt).transpose()
    # inp3 = gt.readbin(tr3,no,nt).transpose()
    
    diff = inp1-inp2
    
    tr  = [71,135,201]
    
    
    
       
    # flout  = './png/19_anomaly_4_layers/difference_fwi_vs_born_ano_trace.png' 
    # flout  = './png/19_anomaly_4_layers/born_trace_diff_ano_vs_org.png'  
    # plot_trace(diff,diff,flout,tr)
    
    flout  = './png/19_anomaly_4_layers/trace_'+str(num)+'.png' 
    amp=plot_trace(inp2,inp2,flout,tr)
    
    ratio = amp[0]/amp[1]
    ratio2 = amp[0]/amp[2]
    # print('Ratio amplitude 1st and 2nd reflector : ',ratio)
    # print('Ratio amplitude 1st and 3nd reflector : ',ratio2)
    
    
    # hmin,hmax = -0.1,0.1
    tabledim = (3,3)
    table_n = np.zeros(tabledim)
    for i in range(np.size(tr)):
        table_n[i] = [str(ao[tr[i]]),str(ratio[i]),str(ratio2[i])]
        
    # Tableau avec rien comme principale     
    # col_names = ["Offset", "Ratio 1st/2nd reflector", "Ratio 1st/3rd reflector"]
    # with open('./txt/'+str(num)+'.txt', 'w') as f:
    #     f.writelines('            Anomaly at '+str(num/100)+' km/s : \n')
    #     f.writelines(tabulate(table_n,headers=col_names, tablefmt="fancy_grid"))
    
    # Tableau avec offset comme principal
    table_n = [None]*2 
    table_n[0] = ["Ratio 1st/2nd reflector",ratio[0],ratio[1],ratio[2]]
    table_n[1] = ["Ratio 1st/3nd reflector",ratio2[0],ratio2[1],ratio2[2]]
    ao_m = ao*1000
    ao_m = ao_m.astype(int)   
    col_names = ["Offset (m/s)", ao_m[tr[0]], ao_m[tr[1]],ao_m[tr[2]]]  
    with open('./txt/'+str(num)+'.txt', 'w') as f:
        f.writelines('            Anomaly at '+str(num*10)+' m/s : \n')
        f.writelines(tabulate(table_n,headers=col_names, tablefmt="fancy_grid"))

    #%%
    
    def plot_shot_gathers(hmin,hmax,inp,flout):   
        fig  = plt.figure(figsize=(10,8), facecolor = "white")
        av   = plt.subplot(1,1,1)
        hfig = av.imshow(inp, extent=[ao[0],ao[-1],at[-1],at[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto', \
                          cmap='seismic')
        for i in range(np.size(tr)):
            plt.axvline(x=ao[tr[i]], color='k',ls='--')
        plt.colorbar(hfig)
        fig.tight_layout()
        print("Export to file:",flout)
        fig.savefig(flout, bbox_inches='tight')
    
    hmin,hmax = -0.1,0.1
    flout ='./png/gather_test.png' 
    plot_shot_gathers(hmin, hmax, inp2, flout)
    
    
    #%%
   
    
    # tr_71 = inp1[:,71]
    # inp_g1 =tr_71[tr_71 != 0]
    
   
    
    
    