#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  8 18:26:30 2023

@author: vcabiativapico
"""


import os
import numpy as np
import pandas as pd
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
import csv



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
            
            tr_inp_or = inp1[:,tr[i]]
            
            # if i == 1: tr_inp1 = inp1[:500,tr[i]]
            # else: tr_inp1 = inp1[:,tr[i]]
            
            if i == 1: tr_inp1 = inp1[550:,tr[i]]
            elif i == 2: tr_inp1 = inp1[430:700,tr[i]]   
            else: tr_inp1 = inp1[:,tr[i]]
            
            # tr_inp1 = tr_inp_or
            theset = np.max(tr_inp1)
            idx = np.where(theset == tr_inp_or)
            axi[i].plot(theset,at[idx],'o')
            print('index reflecteur 1 ',idx, 'trace', i+1)
            
            tr_inp2 = tr_inp_or[int(idx[0]):]
            if i == 0 : 
                tr_inp2 = tr_inp_or[int(idx[0])+108:int(idx[0])+150]
            elif i == 1 : 
                tr_inp2 = tr_inp_or[int(idx[0])+170:int(idx[0])+250]
            else : tr_inp2 = tr_inp_or[int(idx[0])+220:int(idx[0])+270]
                
            theset2 = np.max(tr_inp2)#[::-1]
            idx2 = np.where(theset2 == tr_inp_or)
            axi[i].plot(theset2,at[idx2],'o')    
            print('index reflecteur 2',idx2, 'trace', i+1)    
            
            tr_inp3 = tr_inp_or[int(idx2[0]):]
            if i == 0:
                tr_inp3 = tr_inp_or[int(idx2[0])+65:int(idx2[0])+110]
            elif i ==1:
                tr_inp3 = tr_inp_or[int(idx2[0])+90:]
            else : tr_inp3 = tr_inp_or[int(idx2[0])+170:int(idx2[0])+190]
            
            theset3 = np.max(tr_inp3)#[::-1]
            idx3 = np.where(theset3 == tr_inp_or)
            axi[i].plot(theset3,at[idx3],'o')    
            print('index reflecteur 3',idx3, 'trace', i+1)   
            
            
            amp[i] = [theset,theset2,theset3]
            
            if i >-1: xmax,xmin = 0.3,-0.3
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



   # tr1  = './output/15_picked_models/4_CO2/born/t1_obs_000301.dat'     
   # tr2  = './output/15_picked_models/4_CO2/fwi/t1_obs_000301.dat' 
   # tr3  = './output/18_3_interfaces/org/born/t1_obs_000301.dat' 
   
   # tr1  = './output/18_3_interfaces/org/fwi/t1_obs_000301.dat'
   # tr2  = './output/18_3_interfaces/soft_anomaly/fwi/t1_obs_000301.dat'
    
    num = 200
    
    tr1 = './output/21_vel_rho_anomaly/org/fwi/t1_obs_000301.dat'
    tr2 = './output/21_vel_rho_anomaly/'+str(num)+'/fwi/t1_obs_000301.dat'
   
    # tr2 = './output/t1_obs_000301.dat'     
    # tr2  = './output/17_picked_models_rho/rho_0/born/t1_obs_000301.dat'
    
    inp1 = gt.readbin(tr1,no,nt).transpose()
    inp2 = gt.readbin(tr2,no,nt).transpose()
    # inp3 = gt.readbin(tr3,no,nt).transpose()
    
    diff = inp1-inp2
    
    tr  = [71,135,201]
    
           
    # flout  = './png/19_anomaly_4_layers/difference_fwi_vs_born_ano_trace.png' 
    # flout  = './png/19_anomaly_4_layers/born_trace_diff_ano_vs_org.png'  
    # plot_trace(diff,diff,flout,tr)
    
    flout  = './png/21_vel_rho_anomaly/trace_'+str(num)+'.png' 
    amp=plot_trace(diff,diff,flout,tr)
    
    ratio = amp[:,0]/amp[:,1]
    ratio2 = amp[:,0]/amp[:,2]
    # print('Ratio amplitude 1st and 2nd reflector : ',ratio)
    # print('Ratio amplitude 1st and 3nd reflector : ',ratio2)
    
    
    # hmin,hmax = -0.1,0.1
    tabledim = (3,3)
    table_n = np.zeros(tabledim)
    for i in range(np.size(tr)):
        table_n[i] = [str(ao[tr[i]]),str(ratio[i]),str(ratio2[i])]
    
    print(table_n)
    # Tableau avec rien comme principale     
    # col_names = ["Offset", "Ratio 1st/2nd reflector", "Ratio 1st/3rd reflector"]
    # with open('./txt/'+str(num)+'.txt', 'w') as f:
    #     f.writelines('            Anomaly at '+str(num/100)+' km/s : \n')
    #     f.writelines(tabulate(table_n,headers=col_names, tablefmt="fancy_grid"))
    
    # Tableau avec offset comme principal
    table_n2 = [None]*3 
    table_n2[1] = ["Ratio 1st/2nd reflector",ratio[0],ratio[1],ratio[2]]
    table_n2[2] = ["Ratio 1st/3nd reflector",ratio2[0],ratio2[1],ratio2[2]]
    ao_m = ao
    # ao_m = ao_m.astype(int)   
    table_n2[0] = ["Offset (km)", ao_m[tr[0]], ao_m[tr[1]],ao_m[tr[2]]]  
    
       
    df = pd.DataFrame(table_n2)
    df.to_csv('./txt/21_vel_rho_anomaly/fwi_'+str(num)+'.csv',header=False,index=False)
    
    


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
    
    num = 250
    tr1 = './output/19_anomaly_4_layers/org/fwi/t1_obs_000301.dat'
    tr2 = './output/22_extend_anomaly/'+str(num)+'/fwi/t1_obs_000301.dat'
    tr  = [71,135,201]
    
    inp1 = gt.readbin(tr1,no,nt).transpose()
    inp2 = gt.readbin(tr2,no,nt).transpose()
    diff = inp2-inp1
    hmin,hmax = -0.1,0.1
    # flout ='./png/22_extend_anomaly/gather_diff_'+str(num)+'.png' 
    flout1 ='./png/22_extend_anomaly/gather_org.png' 
    flout2 ='./png/22_extend_anomaly/gather_ano_'+str(num)+'.png' 
    plot_shot_gathers(hmin, hmax, diff, flout)
    plot_shot_gathers(hmin, hmax, inp1, flout1)
    plot_shot_gathers(hmin, hmax, inp2, flout2)
    
    #%%
    ano = np.array([150,160,171,180,190,200,210,230,250])
    def read_ratio_tables(ano):
        vel = [0]*int(np.size(ano))
        rho = [0]*int(np.size(ano))
        
        for i in range(np.size(ano)):
            rho[i] = pd.read_csv("./txt/20_density_4_layers/fwi_"+str(ano[i])+".csv", sep=',', header=None)
            rho[i] = rho[i].values
            
            vel[i] = pd.read_csv("./txt/19_anomaly_4_layers/fwi_"+str(ano[i])+".csv", sep=',', header=None)
            vel[i] = vel[i].values
        return vel,rho
        ### vel[anomaly_value][numero_reflecteur,numero_off]
        ### vel_ref_"which_ratio_number"
    
    
    def organize_ratio_tables(ano,values):            
        mat_shp = (np.size(ano),3)
        val_1 = np.zeros(mat_shp)
        val_2 = np.zeros(mat_shp)
        for j in range(1,4):
            for i in range(np.size(ano)):
                val_1[i][j-1] = values[i][1,j]
                val_2[i][j-1] = values[i][2,j]
        val_1 = val_1.T
        val_2 = val_2.T
        return val_1, val_2
    
    
    def calculate_constrast(ano):
        vel_cont       = ano/100 - 1.5
        vel_perc       = vel_cont/1.5*100
    
        ivel_values    = np.append([1.5] ,ano/100)
        rho_ano_values = (0.31 * (ivel_values*10)**0.25)
        rho_cont       = rho_ano_values[1:] - rho_ano_values[0]
        rho_perc       = rho_cont/1.5*100
        return vel_cont,rho_cont,vel_perc,rho_perc
    
    vel,rho = read_ratio_tables(ano)
    
    
    
    vel_ref_1,vel_ref_2 = organize_ratio_tables(ano,vel) # ratio amplitudes pour les deux reflecteurs en dessous
    rho_ref_1,rho_ref_2 = organize_ratio_tables(ano,rho) # ratio amplitudes pour les deux reflecteurs en dessous
        
    vel_cont,rho_cont,vel_perc,rho_perc = calculate_constrast(ano)
    
    
    
    
    def plot_amplitude_ratios(val_ref,ano,mod,flout,off):
        offset_2f=[0]*3
        for i in range(3):
            offset_2f[i] = "{:.2f}".format(abs(off[0][0,i+1]))
        fig = plt.figure(figsize=(10,8), facecolor = "white")
        val_ref = val_ref[:]
        vel_cont = ano - 1.5
        vel_perc1 = vel_cont/1.5*100
        vel_cont2 = ano - 2.0
        vel_perc2 = vel_cont2/2.0*100
        
        rho_ano_values = (0.31 * (ano*1000)**0.25)
        print(rho_ano_values)
        rho_cont1       = rho_ano_values[:] - rho_ano_values[0]
        print(rho_cont1)
        rho_perc1       = rho_cont1/rho_ano_values[0]*100
        rho_cont2       = rho_ano_values[:] - rho_ano_values[5]
        
        rho_perc2       = rho_cont2/rho_ano_values[5]*100
        # plt.title(title)
        
        if mod == 'vel':
            
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()  
            ax3 = ax1.twiny()
            ax1.plot(vel_perc1,val_ref.T,'o-')
            ax2.plot(ano,val_ref.T)
            ax3.plot(vel_perc2,val_ref.T)
            ax3.spines.top.set_position(("axes", -0.4))
            
            ax1.set_title('Amplitude ratio : VELOCITY anomaly \n')
            ax1.set_xlabel('% velocity from 1.5 km/s')
            ax2.set_xlabel('Velocity values (Km/s)')
            ax3.set_xlabel('% velocity from 2.0 km/s')
            ax1.set_ylabel('$\dfrac{A 1st_{Reflector}}{A 3rd_{Reflector}}$')
            ax1.legend(offset_2f,title="Offset (km)")
        else:
            ax1 = fig.add_subplot(111)
            ax2 = ax1.twiny()  
            ax3 = ax1.twiny()
            ax1.plot(rho_perc1,val_ref.T,'o-')
            ax2.plot(rho_ano_values,val_ref.T)
            ax3.plot(rho_perc2,val_ref.T)
            ax3.spines.top.set_position(("axes", -0.4))
            
            ax1.set_title('Amplitude ratio : DENSITY anomaly \n')
            ax1.set_xlabel('% density from 1st layer')
            ax2.set_xlabel('Density values (Km/s)')
            ax3.set_xlabel('% density from 2nd layer')
            ax1.set_ylabel('$\dfrac{A 1st_{Reflector}}{A 2nd_{Reflector}}$')
            ax1.legend(offset_2f,title="Offset (km)")
        # plt.ylabel('$\dfrac{A 1st_{Reflector}}{A 2nd_{Reflector}}$')
        plt.rcParams['font.size'] = 18
        fig.tight_layout()
        print("Export to file:",flout)
        fig.savefig(flout, bbox_inches='tight')
    
    
    # flout ='./png/19_anomaly_4_layers/ratio_vel_2nd_reflector_percentage.png'
    # plot_amplitude_ratios(vel_ref_1,ano/100,'vel',flout,vel)
    
    # flout ='./png/21_vel_rho_anomaly/ratio_vel_2nd_reflector_percentage.png'
    # plot_amplitude_ratios(vel_ref_1,ano/100,'vel',flout,vel)
    
    
    flout ='./png/20_density_4_layers/ratio_rho_2nd_reflector_percentage.png'
    plot_amplitude_ratios(rho_ref_1,ano/100,'rho',flout,rho)   
    
    
    # flout = './png/19_anomaly_4_layers/ratio_vel_3rd_reflector_percentage.png'
    # plot_amplitude_ratios(vel_ref_2,ano/100,'vel',flout,vel)
    
    # flout ='./png/21_vel_rho_anomaly/ratio_rho_3rd_reflector_percentage.png'
    # plot_amplitude_ratios(vel_ref_2,ano/100,'vel',flout,vel)
    
    # flout ='./png/20_density_4_layers/ratio_rho_3rd_reflector_percentage.png'
    # plot_amplitude_ratios(rho_ref_2,ano/100,'rho',flout,rho)
    

    
    