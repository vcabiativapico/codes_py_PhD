#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 15:59:47 2024

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
from scipy.signal import hilbert
from scipy import signal
from scipy.ndimage import laplace
from scipy.sparse.linalg import LinearOperator, cg

if __name__ == "__main__":

    os.system('mkdir -p png_course')

    # Global parameters
    labelsize = 16
    nt = 1801
    dt = 1.14e-3
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
    
    def plot_model(inp, flout):
    
        plt.rcParams['font.size'] = 22
 
        hmin = np.min(inp)
        hmax = -hmin
        # hmax = np.max(inp)
        if np.shape(inp)[1] > 60:
            fig = plt.figure(figsize=(14, 7), facecolor="white")
            av = plt.subplot(1, 1, 1)
            hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                              vmin=hmin, vmax=hmax, aspect='auto',cmap= 'seismic')
            plt.xlabel('Distance (km)')
            plt.ylabel('Depth (km)')
        plt.colorbar(hfig1, format='%1.1f',label='m/s')
        fig.tight_layout()
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
        print('max',np.max(inp))
        return inp, fig
    
    def plot_mig_min_max(inp, hmin,hmax):
   
        plt.rcParams['font.size'] = 22
     
        fig = plt.figure(figsize=(14, 7), facecolor="white")
        av = plt.subplot(1, 1, 1)
        hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                          vmin=hmin, vmax=hmax, aspect='auto', cmap='viridis')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)') 
        plt.colorbar(hfig1, format='%1.1f',label='m/s')
        print('max',np.max(inp))
        return inp, fig
    
    
  # # #### TO PLOT SHOTS FROM MODELLING
    def plot_shot_gathers(hmin, hmax, inp, flout):
        plt.rcParams['font.size'] = 22
        fig = plt.figure(figsize=(10, 8), facecolor="white")
        av = plt.subplot(1, 1, 1)
        hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                         vmin=hmin, vmax=hmax, aspect='auto',
                         cmap='seismic')
        # for i in range(np.size(tr)):
        #     plt.axvline(x=ao[tr[i]], color='k', ls='--')
        plt.title('x= '+str(title*12))
        plt.colorbar(hfig, format='%2.2f')
        
        plt.ylim(at[-1],ft)
        plt.xlabel('Offset (km)')
        plt.ylabel('Time (s)')
        fig.tight_layout()
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')    


#%%        
    title = 253
    
    flout_gather = '../png/obs_'+str(title)+'.png'
    tr1_wsrc ='../output/78_marm_sm8_thick_sum_pert/org_full/t1_obs_000'+str(title)+'.dat'
    tr2_wsrc ='../output/78_marm_sm8_thick_sum_pert/ano_full/t1_obs_000'+str(title)+'.dat'
    
    
    inp1_wsrc = -gt.readbin(tr1_wsrc, no, nt).transpose()
    inp2_wsrc = -gt.readbin(tr2_wsrc, no, nt).transpose()
    
    
    hmin = np.max(inp1_wsrc)/100
    hmax = -hmin   
    
    plot_shot_gathers(hmin, hmax, inp1_wsrc, flout_gather)
    plot_shot_gathers(hmin, hmax, inp2_wsrc, flout_gather)
    diff = inp1_wsrc - inp2_wsrc
  
    
    
    diff_rec = np.zeros_like(diff)
    for i in range(no): 
        diff_rec[:,i] = diff[:,i] * (ao[i]+title*12/1000)
    
  
    
    
    flout_gather = '../png/obs_'+str(title)+'.png'
    plot_shot_gathers(hmin, hmax, diff*10, flout_gather)
    flnam = '../input/78_marm_sm8_thick_sum_pert/res/t1_obs_000'+str(title)+'_I.dat'
    # gt.writebin(diff.transpose(), flnam)
    
   
    plot_shot_gathers(hmin, hmax, diff_rec*10, flout_gather)
    flnam = '../input/78_marm_sm8_thick_sum_pert/res/t1_obs_000'+str(title)+'_IR.dat'
    # gt.writebin(diff_rec.transpose(), flnam)
    
    
    
    window = signal.windows.tukey(no,alpha=0.8)
 
    inp1_window = inp1_wsrc * window
    inp2_window = inp2_wsrc * window    
        
    plot_shot_gathers(hmin, hmax, inp1_window, flout_gather)
    plot_shot_gathers(hmin, hmax, inp2_window, flout_gather)
    
    diff_win = inp1_window - inp2_window
    
    diff_rec_win = np.zeros_like(diff)
    for i in range(no): 
        diff_rec_win[:,i] = diff_win[:,i] * (ao[i]+title*12/1000)
    
 
    flout_gather = '../png/obs_'+str(title)+'.png'
    plot_shot_gathers(hmin, hmax, diff_win*10, flout_gather)
    flnam = '../input/78_marm_sm8_thick_sum_pert/res_apod/t1_obs_000'+str(title)+'_I.dat'
    # gt.writebin(diff_win.transpose(), flnam)
    
   
    plot_shot_gathers(hmin, hmax, diff_rec_win*10, flout_gather)
    flnam = '../input/78_marm_sm8_thick_sum_pert/res_apod/t1_obs_000'+str(title)+'_IR.dat'
    # gt.writebin(diff_rec_win.transpose(), flnam)
  
    
    
    tr_mig_I = '../output/78_marm_sm8_thick_sum_pert/shot_I_apod/inv_betap_x_s.dat'
    inp_mig_I = gt.readbin(tr_mig_I, nz, nx)
    flout = '../png/inv_betap_x_s.png'
    plot_model(inp_mig_I,flout)
    
    
    tr_mig_IR = '../output/78_marm_sm8_thick_sum_pert/shot_IR_apod/inv_betap_x_s.dat'
    inp_mig_IR = gt.readbin(tr_mig_IR, nz, nx)
    flout = '../png/inv_betap_x_s.png'
    plot_model(inp_mig_IR,flout)
    
    epsilon = 1e-13
    r0 = title*12/1000
    sp_img =  (inp_mig_IR*inp_mig_I+epsilon*r0)/(inp_mig_I**2 + epsilon)
    
  
    hmin = title*12/1000 - 1.5
    hmax = title*12/1000 + 1.5
    print('hmax-hmin = ', hmax-hmin)
    plot_mig_min_max(sp_img,hmin,hmax)
    
#%%    
    '''Nouvelle image speculaire regularisée'''
    
    
    A = np.copy(inp_mig_I).reshape(np.size(inp_mig_I))
    B = np.copy(inp_mig_IR).reshape(np.size(inp_mig_IR))
    
    # AtB =  (inp_mig_I*inp_mig_IR).reshape(np.size(inp_mig_I))
    # AtA = (inp_mig_I*inp_mig_IR).reshape(np.size(inp_mig_I))
      
    epsilon = 1e-7
    lam = 1e-2
    
    r = np.ones_like(A)*title*12/1000*epsilon
      
    # r = np.copy(abs(sp_img)).reshape(np.size(inp_mig_IR))
    plot_model(r.reshape((nz,nx)),flout)
    
    
    r0 = title*12/1000
    

    D = A*B + epsilon * r0
    
    # hmin = np.min(A* B)
    # hmax = np.max(A* B)
    # plot_mig_min_max((A* B).reshape((nz,nx)),hmin,hmax)
    
    # hmin = np.min(D)
    # hmax = np.max(D)
    # plot_mig_min_max((D).reshape((nz,nx)),hmin,hmax)
    
    def mv(r_pad_N):
        temp = np.array([A**2 * r + epsilon * r - lam * laplace(r)])
        print(np.shape(temp))
        return temp
    
    C_r = LinearOperator((nx*nz,nx*nz), matvec=mv)
    
    r_cg, exit_code = cg(C_r, D, atol=1e-5)
    
    
    # print(exit_code)
    # np.allclose(C_r.dot(r_cg), D)
    
    # r_found = C_r.matvec(D)
    
    # r_found = C_r*D
    
    r_cg = r_cg
    
    
    hmin = np.min(r_cg)
    hmax = np.max(r_cg)
    print('hmax-hmin = ', hmax-hmin)
    # hmin = title*12/1000 - 1.5
    # hmax = title*12/1000 + 1.5
    plot_mig_min_max((r_cg).reshape((nz,nx)),hmin,hmax)
