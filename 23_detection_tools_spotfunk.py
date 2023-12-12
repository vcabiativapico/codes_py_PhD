#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 15:31:10 2023

@author: vcabiativapico
"""

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
from scipy.signal import hilbert
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from spotfunk.res import procs,visualisation
import csv

if __name__ == "__main__":
  
  
## Building simple vel and rho models to test modeling
    labelsize = 16
    nt = 1501
    dt = 1.41e-3
    ft = -100.11e-3
    nz = 151
    fz = 0.0
    dz = 12.0/1000.
    nx = 601
    fx = 0.0
    dx = 12.0/1000.
    no =251
   # no        = 2002
    do = dx
    fo = -(no-1)/2*do
    ao = fo + np.arange(no)*do
    at = ft + np.arange(nt)*dt
    az = fz + np.arange(nz)*dz
    ax = fx + np.arange(nx)*dx
    
    def plot_trace(xmax, inp1, inp2, flout, tr, ano_nb):

        axi = np.zeros(np.size(tr))
        fig, (axi) = plt.subplots(nrows=1, ncols=np.size(tr),
                                  sharey=True,
                                  figsize=(12, 8),
                                  facecolor="white")

        ratio = np.asarray(tr, dtype='f')
        for i in range(np.size(tr)):
            # xmin = np.min(inp1[:,tr[i]]) + np.min(inp1[:,tr[i]])/1.5
            # xmin = 1.0
            xmax = 1.2
            xmin = -xmax
            ratio[i] = np.max(inp2[:, tr[i]])/np.max(inp1[:, tr[i]])
            inp1[:, tr[i]] = (inp1[:, tr[i]]/np.max(inp1[:, tr[i]]))
            inp2[:, tr[i]] = (inp2[:, tr[i]]/np.max(inp2[:, tr[i]]))
            
            axi[i].plot(inp1[:, tr[i]], at, 'r')
            axi[i].plot(inp2[:, tr[i]], at, 'b--')

            axi[i].set_xlim(xmin, xmax)
            axi[i].set_ylim(2, ft)

            axi[i].xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

            # axi[i].set_xlabel("Ratio = "+str(f'{ratio[i]:.2f}'))
            # plt.colorbar()
            fig.tight_layout()

        axi[0].set_ylabel('Time (s)')
        # axi[0].legend(['org', ano_nb[i]], loc='upper left', shadow=True)

        axi[0].legend(['Baseline','Monitor '+str(ano_nb)],loc='upper left',shadow=True)
        fig.text(0.48, -0.01, "Amplitude")
        fig.text(0.48, 1, 'Comparison')
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')

        return ratio, inp1, inp2,axi

    # # #### TO PLOT SHOTS FROM MODELLING
    def plot_shot_gathers(hmin, hmax, inp, flout):

        fig = plt.figure(figsize=(10, 8), facecolor="white")
        av = plt.subplot(1, 1, 1)
        hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                         vmin=hmin, vmax=hmax, aspect='auto',
                         cmap='seismic')
        for i in range(np.size(tr)):
            plt.axvline(x=ao[tr[i]], color='k', ls='--')
        plt.colorbar(hfig,format='%2.2f')
        plt.rcParams['font.size'] = 16
        plt.xlabel('Offset (km)')
        plt.ylabel('Time (s)')
        fig.tight_layout()
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
        return fig

    def read_results(path,srow):
        travel_t = []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            header = next(spamreader)
            for row in spamreader:
                travel_t.append(float(row[srow]))
        return travel_t
   
    xmax_tr = 0.3

    # title = 501
    # title = 201
    # title = 301
    shot,no = 333,251
    ano_nb = 2926
    ano_nb = 2979
    ano_nb = 3083
    
    
    # nomax = 201
    # no = (nomax)+101
    # no = 302 # for 101 & 501
    # no = 402 # for 201 & 401
    # no = 403  # for 301
    # no = 403-abs(301-title)
    fo = -(no-1)/2*do
    ao = fo + np.arange(no)*do
    
    # tr1  = '../output/21_vel_rho_anomaly/org/fwi/t1_obs_000301.dat'
    # tr2  = './output/21_vel_rho_anomaly/'+str(title)+'/born/t1_obs_000301.dat'
    # tr1  = './output/17_picked_models_rho/rho_'+str(title)+'/born/t1_obs_000301.dat'
    # tr2  = './output/17_picked_models_rho/rho_'+str(title)+'/fwi/t1_obs_000301.dat'
    # tr1 = '../output/24_mig_stack/binv/t1_obs_000'+str(title)+'.dat'
    # tr2 = '../output/24_mig_stack/binv/t1_syn_000'+str(title)+'.dat'
    # tr1  = './output/17_picked_models_rho/rho_'+str(title)+'/born/t1_obs_000301.dat'
    # tr2  = './output/17_picked_models_rho/rho_'+str(title)+'/fwi/t1_obs_000301.dat'
    # tr1 = '../output/26_mig_4_interfaces/binv_rc_norm_3083/t1_obs_000'+str(title)+'.dat'
   
#%% Spotfunk tools
    ano_nb = 2926
    # ano_nb = 2979
    # ano_nb = 3083
    tr_nb = 126
    
    tr1 = '../output/26_mig_4_interfaces/badj_rc_norm/t1_syn_000'+str(shot)+'.dat'
    
    tr2 = '../output/26_mig_4_interfaces/badj_rc_norm_'+str(ano_nb)+'/t1_syn_000'+str(shot)+'.dat'
    
    # inp1 = gt.readbin(tr1, no, nt).transpose()
    inp2 = -gt.readbin(tr2, no, nt).transpose()
    inp1 = -gt.readbin(tr1, no, nt).transpose() 
    tr_ex1 = inp1[:,tr_nb].T
    tr_ex2 = inp2[:,tr_nb].T
    
    
    CC_TS = procs.max_cross_corr(tr_ex1,tr_ex2,win1=1500,win2=1700,si=dt)
    SLD_TS = procs.sliding_TS(tr_ex1,tr_ex2,si=dt)
    RMS = procs.sliding_NRMS(tr_ex1,tr_ex2,si=dt)
    envelope = procs.enveloppe_ratio(tr_ex1,tr_ex2,si=dt)
    
    v_ov = visualisation.overlay(tr_ex1,tr_ex2,si=dt,
                          legend=["Base", "Monitor "+str(ano_nb/1000)],
                          clist=['k', 'r'])
    plt.title('Baseline vs Monitor '+str(ano_nb/1000))
    plt.xlim(-25,25)
    
    v_sld_ts = visualisation.overlay(SLD_TS,si=dt)
    plt.title('Sliding time shift baseline vs monitor anomaly '+str(ano_nb)+' m/s')
    plt.xlim(-5,0.5)
    
    v_rms = visualisation.overlay(RMS,si=dt)
    plt.title('RMS baseline vs monitor anomaly '+str(ano_nb)+' m/s')
    plt.xlim(-0.1,1.3)
    
    v_envelope = visualisation.overlay(envelope,si=dt)
    plt.title('Envelope ratio baseline vs monitor anomaly '+str(ano_nb)+' m/s')
    plt.xlim(-2,28)
#%% SHOT GATHERS WAVE-EQUATION VS RAY-TRACING travel-time RESULTS

    # tr = np.array([63,126,189])
    tr = np.array([42,84,126,168,210])
    # tr = [71, 135, 201, 260, 333]
    # tr   = [260,268]
    # tr   = [71,135,201,267]
    # diff = inp1-inp2
    rec_x = (shot*dx+ao[tr])*1000

    # flout  = './png/22_extend_anomaly/born_trace_'+str(title)+'.png'
    # plot_trace(xmax_tr,inp2,inp2,flout,tr)
    
    # tr_adj = '../output/26_mig_4_interfaces/badj_rc_norm/t1_obs_000'+str(shot)+'.dat'
    # tr_inv = '../output/26_mig_4_interfaces/binv_rc_norm/t1_obs_000'+str(shot)+'.dat'
    
    tr_adj = '../output/27_marm/mod_marm_inv/t1_obs_000'+str(shot)+'.dat'
    tr_inv = '../output/27_marm/mod_marm_inv/t1_obs_000'+str(shot)+'.dat'
    
    
    inp_adj = -gt.readbin(tr_adj, no, nt).transpose()
    inp_inv = -gt.readbin(tr_inv, no, nt).transpose()
    

    inp_hilb_adj = np.zeros_like(inp_adj,dtype = 'complex_')   
    inp_hilb_inv = np.zeros_like(inp_inv,dtype = 'complex_')  
    
    for i in range(no):
        inp_hilb_adj[:,i] = hilbert(inp_adj[:,i])    
        inp_hilb_inv[:,i] = hilbert(inp_inv[:,i])    
    
    imag_hilb_adj = inp_hilb_adj.imag
    imag_hilb_inv = inp_hilb_inv.imag
        
    
    # path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/008_demig_sm_SR_az_12_all5_iz_INV_PP21.csv'
    # path2 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/008_demig_sm_SR_az_12_all5_iz_ADJ_PP21.csv'
    
    # path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/010_marm_sm_binv_PP21_P005_hz02.csv'
    # path2 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/010_marm_sm_badj_PP21_P005_hz02.csv'
    
    path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/012_demig_AO_INV_PP21.csv'
    path2 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/012_demig_AO_ADJ_PP21.csv'

    path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/011_marm_sm_binv_PP21_P021_hz02.csv'
    path2 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/011_marm_sm_badj_PP21_P021_hz02.csv'


    tt_inv = read_results(path1,17)
    tt_adj = read_results(path2,17)
    
    tt_inv = np.array(tt_inv)-16
    tt_adj = np.array(tt_adj)-16
    # tt_inv[0] = 1000

    def plot_gather_scatter(inp,t_time,lgd):
        fig = plt.figure(figsize=(10, 8), facecolor="white")
        av = plt.subplot(1, 1, 1)
        # print(np.max(inp))
        c_point = ['yellow','b','r','k','greenyellow']
        if lgd == 'badj' :
            hmin, hmax = -0.10,0.10
            hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                             vmin=hmin, vmax=hmax, aspect='auto',
                             cmap='seismic')
            flout = '../png/27_marm/diff_marm_inv/compare_RT_OBS_'+str(shot)+'.png'
            plt.title('Observed shot with the ADJOINT demigrated tt data \n')
            plt.colorbar(hfig,format='%1.e')
        else:
            hmin, hmax =-0.10,0.10
            hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                             vmin=hmin, vmax=hmax, aspect='auto',
                             cmap='seismic')
            flout = '../png/27_marm/diff_marm_inv/compare_RT_OBS_'+str(shot)+'.png'
            plt.title('Observed shot with the INVERSE demigrated tt data \n')
            plt.colorbar(hfig,format='%1.e')
        for i in range(tr.size):
            plt.axvline(x=ao[tr[i]], color='k', ls='--',alpha=0.8)
            plt.scatter(ao[tr[i]],t_time[i]/1000, color=c_point[3], ls='--')
            
        
        fig.tight_layout()
        
        plt.rcParams['font.size'] = 16
        plt.xlabel('Offset (km)')
        plt.ylabel('Time (s)')
        fig.tight_layout()
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
        
    
    def plot_trace_scatter(inp,t_time,lgd,flout):
        axi = np.zeros(np.size(tr))
        fig, (axi) = plt.subplots(nrows=1, ncols=np.size(tr),
                                  sharey=True,
                                  figsize=(14, 8),
                                  facecolor="white")

        c_point = ['yellow','b','r','k','greenyellow']
        # ratio = np.asarray(tr, dtype='f')
        for i in range(np.size(tr)):
            # xmin = np.min(inp1[:,tr[i]]) + np.min(inp1[:,tr[i]])/1.5
            # xmin = 1.0
            xmax = np.max(inp[:, tr[i]])
            xmin = -xmax
            
            # inp[:, tr[i]] = (inp[:, tr[i]]/np.max(inp[:, tr[i]]))
            axi[i].plot(inp[:, tr[i]], at, 'b')
        

            axi[i].set_xlim(xmin, xmax)
            axi[i].set_ylim(2, ft)
            axi[i].xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
            axi[i].set_xlabel('Offset: '+str(f'{ao[tr[i]]:.2f}'))
            # print('iter: '+str(i),'time: '+str(t_time[i]))
            axi[i].axhline(t_time[i]/1000, color=c_point[3], ls='--')
            fig.tight_layout()
            axi[i].grid()
        axi[0].set_ylabel('Time (s)')
        axi[0].legend([lgd,'RayTT'], loc='upper left', shadow=True)

        # axi[0].legend(['Baseline','Monitor'],loc='upper left',shadow=True)
        fig.text(0.48, -0.01, "Amplitude")
        fig.text(0.48, 1, 'Traces from observed data vs tt '+str(lgd))
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
            
    
### MINIMUM PHASE ORIGINAL
    lgd_inv = 'binv'    
    lgd_adj = 'badj'
    # plot_gather_scatter(inp_inv,tt_inv,lgd_inv)
    # plot_gather_scatter(inp_adj,tt_adj,lgd_adj)
    
   
    
    # flout1 = '../png/26_mig_4_interfaces/traces_inv'
    # plot_trace_scatter(inp_inv,tt_inv,lgd_inv,flout1) 
    # flout2 = '../png/26_mig_4_interfaces/traces_adj'
    # plot_trace_scatter(inp_adj,tt_adj,lgd_adj,flout2)
    
### ZERO PHASE CORRECTION

    plot_gather_scatter(imag_hilb_inv,tt_inv,lgd_inv)
    plot_gather_scatter(imag_hilb_adj,tt_adj,lgd_adj)
    
    # flout1 = '../png/26_mig_4_interfaces/traces_inv_hilb'
    # plot_trace_scatter(imag_hilb_inv,tt_inv,lgd_inv,flout1) 
    # flout2 = '../png/26_mig_4_interfaces/traces_adj_hilb'
    # plot_trace_scatter(imag_hilb_adj,tt_adj,lgd_adj,flout2)
   
    
    flout3 = '../png/27_marm/diff_marm_inv/traces_phase_inv'
    plot_trace_scatter(imag_hilb_inv,tt_inv,lgd_inv,flout3) 
    flout4 = '../png/27_marm/diff_marm_inv/traces_phase_adj'
    plot_trace_scatter(imag_hilb_adj,tt_adj,lgd_adj,flout4)
    
        
    ## Calculate the difference
   
    tt_delta = np.array(tt_inv)-np.array(tt_adj)

    print('mean difference :',np.mean(tt_delta))
    plt.figure(figsize=(16,8))
    c_point = ['yellow','b','r','k','greenyellow']
    for i in range(tt_delta.size):
        plt.plot(ao[tr[i]],-tt_delta[i],'-o',color=c_point[3])
    plt.grid()
    plt.title('Difference between raytracing with hz in the badj or binv image')
    plt.xlabel('Offset (km)')
    plt.ylabel('Time difference (ms)')
    
#%% PLOT TRACES WITH ANOMALY CHANGES

    shot,no = 333,251
    fo = -(no-1)/2*do
    ao = fo + np.arange(no)*do
    tr_nb = 126
    ano_nb = ['','_2926','_2979','_3083']
    
    tr_nam = [0]*np.size(ano_nb)
    tr_ex  = np.zeros((np.size(ano_nb),nt),float)
    inp = [np.zeros((nt,no))]*np.size(ano_nb)
    
    for i in range(np.size(ano_nb)):
        tr_nam[i] = '../output/26_mig_4_interfaces/binv_rc_norm'+str(ano_nb[i])+'/t1_syn_000'+str(shot)+'.dat'
        inp[i]    = -gt.readbin(tr_nam[i], no, nt).transpose() 
        tr_ex[i]  = inp[i][:,tr_nb].T
    
    


    for i in range(np.size(ano_nb)):
        hmin, hmax = -20, 20
        flout_gather = '../png/26_mig_4_interfaces/binv_rc_norm'+str(ano_nb[i])+'_syn_'+str(shot)+'.png'
        fig_inp1 = plot_shot_gathers(hmin, hmax, inp[i], flout_gather)
        plt.title('Synthetic shot from migrated data '+str(ano_nb[i])+'_'+str(shot)+' \n')
    xmax = np.max(tr_ex)
    flout = '../png/26_mig_4_interfaces/binv_rc_norm'+str(ano_nb[3])+'traces_333.png'
    fig_inp2 = plot_trace(xmax, inp[0], inp[1], flout, tr, ano_nb[3])    
    
    def plot_overlay_traces(tr_ex):
        axi = np.zeros(np.size(tr_ex))
        plt.figure(figsize=(6,15))
        for j in range(np.size(ano_nb)):
            xmin = np.min(tr_ex) 
            xmax = np.max(tr_ex)
            plt.plot(tr_ex[j],at) 
            plt.plot(tr_ex[j],at) 
            plt.xlim(xmin, xmax)
            plt.ylim(2, ft)
            plt.grid(axis="y")
        # plt.gca().invert_yaxis()
        
        
    def plot_parallel_traces(tr_ex):
        axi = np.zeros(np.size(ano_nb))
        fig, (axi) = plt.subplots(nrows=1, ncols=np.size(ano_nb),
                                  sharey=True,
                                  figsize=(12, 8),
                                  facecolor="white")
        xmin = np.min(tr_ex) 
        xmax = np.max(tr_ex)
        for i in range(np.size(ano_nb)): 
            axi[i].plot(tr_ex[i],at) 
            axi[i].set_xlim(xmin, xmax)
            axi[i].set_ylim(2, ft)
            axi[i].title.set_text('Value'+ano_nb[i])
            axi[i].xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
            axi[i].grid(axis="x")
        fig.tight_layout()
     
    plot_overlay_traces(tr_ex)
    plot_parallel_traces(tr_ex)
        # ratio[i] = np.max(inp2[:, tr[i]])/np.max(inp1[:, tr[i]])
        # inp1[:, tr[i]] = (inp1[:, tr[i]]/np.max(inp1[:, tr[i]]))
        # inp2[:, tr[i]] = (inp2[:, tr[i]]/np.max(inp2[:, tr[i]]))
        
        # axi[i].plot(inp1[:, tr[i]], at, 'r')
        # axi[i].plot(inp2[:, tr[i]], at, 'b--')