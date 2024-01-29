#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:09:47 2023

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
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from spotfunk.res import procs,visualisation
import csv
from scipy.signal import ricker, hilbert, hilbert2

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
    no = 251
   # no        = 2002
    do = dx
    fo = -(no-1)/2*do
    ao = fo + np.arange(no)*do
    at = ft + np.arange(nt)*dt
    az = fz + np.arange(nz)*dz
    ax = fx + np.arange(nx)*dx
    
    
    def ricker_creation(freq_principale,si=0.001):
        
       points = 1501 #prendre un nombre impair pour centrer l'ondelette
       a1=np.sqrt(2)/(2*np.pi*freq_principale*si)
       le_ricker = ricker(points, a1)
    
       return le_ricker
    
    
    
    h1     = dz * 49+dz
    h2     = dz * 101+dz
    deg1   = np.linspace(-50,50,50)
    
    v1     = 2.00
    v2     = 2.06
    
    t0     = 2*h1 / v1
    t1     = np.sqrt(t0**2 + (ao/v1)**2)-ft
        
    
    ao_conv  = len(ao) # Read the axes to keep same size
    at_conv  = len(at)
     

    inp_x    = np.zeros((at_conv,ao_conv)) 

    wlet = ricker_creation(10,si=dt)
    
    inp_w = np.zeros_like(inp_x)
    
    for i in range(no):
        n = np.round(t1[i]/dt)
        n = n.astype(int)    # Find the index and convert to integer
        inp_x[n,i]   = ((n+1)*dt - t1[i]) / dt 
        inp_x[n+1,i] = (t1[i]- n*dt) / dt
        # n_p1 = np.round(t2[i]/dt)
        # n_p1 = n_p1.astype(int)
        # inp_x[n_p1,i] = ((n+1)*dt - t1[i]) / dt
        # inp_x[n_p1+1,i] = (t1[i]- n*dt) / dt
    
    for i in range(no):
        inp_w[:,i] = np.convolve(inp_x[:,i],wlet,mode='same')
    
    f_inp_w = np.fft.fft(inp_w[:,126])
    N = f_inp_w.size
    sign_inp_w = np.zeros(N)
    sign_inp_w[0] = 1
    sign_inp_w[1:(N + 1) // 2] = 2
    
    # v_hil  = -sign_inp_w*f_inp_w 
    # h_inp_w = np.fft.ifft(v_hil.imag)
    # plt.plot(h_inp_w.imag)
    # plt.plot(inp_w[:,126])
    # plt.xlim(350,600)
    # plt.plot(h_inp_w.imag)
    
    r_hilb = np.zeros_like(inp_w,dtype = 'complex_')
    real_inp = np.zeros_like(inp_w)
    imag_inp = np.zeros_like(inp_w,dtype = 'complex_')
    for i in range(251):
        r_hilb[:,i] = hilbert(inp_w[:,i])
    real_inp = r_hilb.real
    imag_inp = r_hilb.imag
    
    
    # ## Hilbert deux fois sur une trace
    # deux_hil = np.zeros_like(inp_w[:,126])
    
    org = inp_w[:,126]
    un_hil = hilbert(org)
    # deux_hil = hilbert(un_hil.imag)
    
    plt.plot(org)
    plt.plot(un_hil.imag)
    # plt.plot(deux_hil)
    plt.xlim(300,700)
    
    def plot_gather(inp):
        hmax = np.max(np.abs(inp))
        hmin = -hmax
        fig  = plt.figure(figsize=(10,8), facecolor = "white")
        av   = plt.subplot(1,1,1)
        hfig = av.imshow(inp, extent=[ao[0],ao[-1],at[-1],at[0]], aspect='auto',\
                          vmin=hmin,vmax=hmax,cmap='seismic')
        
    plot_gather(inp_w)
    plot_gather(real_inp)
    plot_gather(imag_inp)
        
    plt.figure()        
    plt.plot(at,real_inp[:,130]/np.max(real_inp[:,130]), alpha=0.8)
    plt.plot(at,imag_inp[:,130]/np.max(imag_inp[:,130]), alpha=0.8)
    plt.xlim(0.4,1)


    
#%%
    shot   = 333
    ano_nb = 2926
    ano_nb = 2979
    ano_nb = 3083
    
    
    tr1 = '../output/26_mig_4_interfaces/binv_rc_norm/t1_obs_000'+str(shot)+'.dat'
    # tr2 = '../output/26_mig_4_interfaces/binv_rc_norm_'+str(ano_nb)+'/t1_syn_000'+str(shot)+'.dat'
    
    inp1 = -gt.readbin(tr1, no, nt).transpose() 
    # inp2 = -gt.readbin(tr2, no, nt).transpose()
    
    def plot_shot_gathers(hmin, hmax, inp, flout):

        fig = plt.figure(figsize=(10, 8), facecolor="white")
        av = plt.subplot(1, 1, 1)
        hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                         vmin=hmin, vmax=hmax, aspect='auto',
                         cmap='seismic')
        # for i in range(np.size(tr)):
        #     plt.axvline(x=ao[tr[i]], color='k', ls='--')
        plt.colorbar(hfig, format='%2.2f')
        plt.rcParams['font.size'] = 16
        plt.xlabel('Offset (km)')
        plt.ylabel('Time (s)')
        fig.tight_layout()
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
       
        
    inp1_hilb = np.zeros_like(inp1,dtype = 'complex_')   
    for i in range(251):
        inp1_hilb[:,i] = hilbert(inp1[:,i])    
    
    hmin, hmax = -0.1, 0.1
    flout = '../png/syn_'+str(shot)+'.png'    
    plot_shot_gathers(hmin,hmax,inp1,flout)
    plot_shot_gathers(hmin,hmax,inp1_hilb.imag,flout)
    
    # hmin, hmax = -20, 20
    # plot_shot_gathers(hmin,hmax,inp2,flout)
    
    
