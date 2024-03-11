#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 28 11:13:13 2024

@author: vcabiativapico
"""


#%matplotlib inline

import numpy as np

import matplotlib.pyplot as plt

import geophy_tools as gt

if __name__ == "__main__":
  
  
## Building simple vel and rho models to test modeling
    labelsize = 16
    nt        = 1501
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

    fl1 = '../input/33_report_model/vel_full.dat'
    fl2 = '../input/marm2_sm15.dat'
    
    inp1      = gt.readbin(fl1,nz,nx)
    inp_smooth= gt.readbin(fl2,nz,nx)
    
   
    
  
    
    c_val = 300
    c_ext = 50
    c_depth = 8
    step = 2
    
    def create_ano(inp):
        for i in range(c_depth):
            inp1[51+i, c_val-c_ext+(i*step) : c_val+c_ext-(i*step)  ] = 2.05+0.025
        return inp1
    
    
    inp1_ano = create_ano(inp1)
    
    hmax =  2.05+0.025
   
    hmin = 2.0
    
    def plot_model(inp,hmin,hmax):
        plt.rcParams['font.size'] = 18
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        av  = plt.subplot(1,1,1)
        hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                          vmin=hmin,vmax=hmax,aspect='auto' \
                         )
        plt.colorbar(hfig)
        fig.tight_layout()
        return fig
        
    def export_model(inp,fig,imout,flout):
        fig.savefig(imout, bbox_inches='tight')
        gt.writebin(inp,flout)  
    
    fig1 = plot_model(inp1_ano,hmin,hmax)
    imout1 = '../png/33_report_model/vel_ano_full.png'
    flout1 = '../input/33_report_model/vel_ano_full.dat'
    export_model(inp1_ano,fig1,imout1,flout1)
    
    
    inp_add = inp1_ano + inp_smooth -2
    hmax2 = np.max(inp_add)
    hmin2 = np.min(inp_add)
    

    
    fig2 = plot_model(inp_add,hmin2,hmax2)
    imout2 = '../png/39_mig_marm_flat/vel_ano_full.png'
    flout2 = '../input/39_mig_marm_flat/vel_marm_plus_flat_ano.dat'
    export_model(inp_add,fig2,imout2,flout2)
    
    