# -*- coding: utf-8 -*-

"""
Display the results
"""

import os
import numpy as np
from math import log, sqrt, log10, pi, cos, sin, atan
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel

def cbar(av,labelsize,fig,hfig):
    """Add (nice) colorbar"""
    divider = make_axes_locatable(av)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    cb = fig.colorbar(hfig, cax=cax)
    cb.ax.tick_params(labelsize=labelsize)
    cb.set_ticks([])

def disp_imshow1(inp1,az,ax,title1,title2,flout,vsym,vscale,label1,ccbar=False):
    """1 plot on the same subplot()"""
    fig = plt.figure(figsize=(10,5), facecolor = "white")
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    # Min and max values (for the axis)
    if vsym == 'sym':
        hmax = vscale*np.max(np.abs(inp1))
        hmin = -hmax
    elif vsym == 'dys':
        hmax = vscale*np.max(inp1)
        hmin = vscale*np.min(inp1)
    else:
        hmax = vhmax
        hmin = vhmin
    
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp1, extent=[ax[0],ax[-1],az[-1],az[0]], \
                     vmin=hmin,vmax=hmax,aspect='auto', \
                     cmap='bwr')
    av.tick_params(axis='both', which='major', labelsize=labelsize)
    av.set_xlabel(title1, fontsize = labelsize)
    av.set_ylabel(title2, fontsize = labelsize)
    #av.set_title(label1, \
    #             fontsize = labelsize, fontweight="bold")
    if ccbar:
        cbar(av,labelsize,fig,hfig)
    #plt.xlim(at[0],at[-1])
    #plt.ylim(hmin,hmax)
    #av.legend(fontsize=labelsize)
    fig.tight_layout()
    #plt.show()
    print("Export to file:",flout)
    fig.savefig(flout, bbox_inches='tight')

def disp_imshow1_alpha(inp1,inp2,az,ax,title1,title2,flout,vsym,vsym2,vscale,vscale2,label1,valpha,ccbar=False):
    """1 plot on the same subplot()"""
    fig = plt.figure(figsize=(10,5), facecolor = "white")
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    # Min and max values (for the axis)
    if vsym == 'sym':
        hmax = vscale*np.max(np.abs(inp1))
        hmin = -hmax
    elif vsym == 'dys':
        hmax = vscale*np.max(inp1)
        hmin = vscale*np.min(inp1)
    else:
        hmax = vhmax
        hmin = vhmin

    # Min and max values (for the axis)
    if vsym == 'sym':
        hmax2 = vscale2*np.max(np.abs(inp2))
        hmin2 = -hmax2
    elif vsym == 'dys':
        hmax2 = vscale2*np.max(inp2)
        hmin2 = vscale2*np.min(inp2)
    else:
        hmax2 = vhmax2
        hmin2 = vhmin2

    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp2, extent=[ax[0],ax[-1],az[-1],az[0]], \
                     vmin=hmin2,vmax=hmax2,aspect='auto', \
                     cmap='gray')
    av.imshow(inp1, extent=[ax[0],ax[-1],az[-1],az[0]], \
                     vmin=hmin,vmax=hmax,aspect='auto', \
                     cmap='bwr', alpha = valpha)
    av.tick_params(axis='both', which='major', labelsize=labelsize)
    av.set_xlabel(title1, fontsize = labelsize)
    av.set_ylabel(title2, fontsize = labelsize)
    #av.set_title(label1, \
    #             fontsize = labelsize, fontweight="bold")
    if ccbar:
        cbar(av,labelsize,fig,hfig)
    #plt.xlim(at[0],at[-1])
    #plt.ylim(hmin,hmax)
    #av.legend(fontsize=labelsize)
    fig.tight_layout()
    #plt.show()
    print("Export to file:",flout)
    fig.savefig(flout, bbox_inches='tight')

def disp_imshow2(inp1,inp2,az,ax,title1,title2,flout,vsym,vscale,label1,ccbar=False):
    """1 plot on the same subplot()"""
    fig = plt.figure(figsize=(10,5), facecolor = "white")
    params = {'mathtext.default': 'regular' }          
    plt.rcParams.update(params)

    # Min and max values (for the axis)
    if vsym == 'sym':
        hmax = vscale*np.max(np.abs(inp1))
        hmin = -hmax
    elif vsym == 'dys':
        hmax = vscale*np.max(inp1)
        hmin = vscale*np.min(inp1)
    else:
        hmax = vhmax
        hmin = vhmin
    
    av  = plt.subplot(1,2,1)
    hfig = av.imshow(inp1, extent=[ax[0],ax[-1],az[-1],az[0]], \
                     vmin=hmin,vmax=hmax,aspect='auto', \
                     cmap='bwr')
    av.tick_params(axis='both', which='major', labelsize=labelsize)
    av.set_xlabel(title1, fontsize = labelsize)
    av.set_ylabel(title2, fontsize = labelsize)
    #av.set_title(label1, \
    #             fontsize = labelsize, fontweight="bold")
    if ccbar:
        cbar(av,labelsize,fig,hfig)
    #plt.xlim(at[0],at[-1])
    #plt.ylim(hmin,hmax)
    #av.legend(fontsize=labelsize)
    
    av  = plt.subplot(1,2,2)
    av.imshow(inp2, extent=[ax[0],ax[-1],az[-1],az[0]], \
                     vmin=hmin,vmax=hmax,aspect='auto', \
                     cmap='bwr')
    av.tick_params(axis='both', which='major', labelsize=labelsize)
    av.set_xlabel(title1, fontsize = labelsize)
    # Empty but with some space
    av.set_ylabel(" ", fontsize = labelsize)
    #av.set_title(label1, \
    #             fontsize = labelsize, fontweight="bold")
    if ccbar:
        cbar(av,labelsize,fig,hfig)
    #plt.xlim(at[0],at[-1])
    #plt.ylim(hmin,hmax)
    #av.legend(fontsize=labelsize)
    fig.tight_layout()
    #plt.show()
    print("Export to file:",flout)
    fig.savefig(flout, bbox_inches='tight')

if __name__ == "__main__":

    os.system('mkdir -p png_course')
    
    # Global parameters
    labelsize = 16
    nt        = 1001
    nwsrc     = 97
    dt        = 2.08e-3
    ft        = -99.84e-3
    nz        = 151
    fz        = 0.0
    dz        = 12.49/1000.
    nx        = 601
    fx        = 0.0
    dx        = 12.49/1000.
    niter     = 20
    no        = 251
    do        = dx
    fo        = -(no-1)/2*do
    ao        = fo + np.arange(no)*do
    at        = ft + np.arange(nt)*dt
    awsrc     = ft + np.arange(nwsrc)*dt
    az        = fz + np.arange(nz)*dz
    ax        = fx + np.arange(nx)*dx
    aiter     = np.arange(1,niter+1)

    # Checking the solution
    csel = [False,False,False,False,False,False,False,False,False,False,True]

    # Export migrated sections (binv and badj modes)
    if csel[0]:
        vhmin,vhmax = 1.5,4.0
        fl1 = './input/marm2_sel.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/vel_exact.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        fl1 = './input/marm2_smo.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/vel_smo.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        fl1 = './reflectivity/dbetap_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/betap_refl.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'sym',1.0,labelsize)

        vhmin,vhmax = -0.1,0.1
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/betap_reflexact.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        # Same as for previous image
        fl1 = './output/RES01/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        fl1 = './output/RES01/adj_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/adj_stack_betap.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'sym',0.5,labelsize)

        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES01/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/obs_0301.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'ddd',0.8,labelsize)

        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES01/t1_obs_000321.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/obs_0321.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'ddd',0.8,labelsize)

        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES01/t1_isy_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/isy_0301.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'ddd',0.8,labelsize)

        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES01/t1_isy_000321.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/isy_0321.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'ddd',0.8,labelsize)

        fl1 = './output/RES01/t1_asy_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/asy_0301.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'sym',0.8,labelsize)
        
        # Vertical profiles (data) ==============================
        fl1 = './output/RES01/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES01/t1_isy_000301.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces.png")
        fig.savefig('./png_course/obs_isy_traces.png', bbox_inches='tight')
        
        # Vertical profiles (inverse)
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        fl2 = './output/RES01/inv_stack_betap.dat'
        inp2 = gt.readbin(fl2,nz,nx)

        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.08*fact
        for ind in range(100,600,100):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,az,color='b')
            av.plot(fact*tr2,az,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Depth (km)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            #av.set_xlabel(r"\delta\beta_p", fontsize = labelsize)
            plt.ylim(az[0],az[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/inv_traces.png")
        fig.savefig('./png_course/inv_traces.png', bbox_inches='tight')
        
        # Vertical profiles (data) ==============================
        fl1 = './output/RES01/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES01/t1_asy_000301.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            tr2 *= 1/np.max(np.abs(tr2))*np.max(np.abs(tr1))
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces.png")
        fig.savefig('./png_course/obs_asy_traces.png', bbox_inches='tight')
        
        # Vertical profiles (adjoint)
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        fl2 = './output/RES01/adj_stack_betap.dat'
        inp2 = gt.readbin(fl2,nz,nx)

        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.08*fact
        for ind in range(100,600,100):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            tr2 *= 1/np.max(np.abs(tr2))*np.max(np.abs(tr1))
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,az,color='b')
            av.plot(fact*tr2,az,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Depth (km)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            #av.set_xlabel(r"\delta\beta_p", fontsize = labelsize)
            plt.ylim(az[0],az[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/adj_traces.png")
        fig.savefig('./png_course/adj_traces.png', bbox_inches='tight')
        
        
    # Export migrated sections (binv and badj modes)
    if csel[1]:

        # rick instead of sinv
        fl1 = './output/RES03/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_rick.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'sym',0.7,labelsize)

        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES03/t1_isy_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/isy_0301_rick.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'sym',0.5,labelsize)

        # Vertical profiles (data) ==============================
        fl1 = './output/RES03/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES03/t1_isy_000301.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            tr2 *= 1/np.max(np.abs(tr2))*np.max(np.abs(tr1))
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces_rick.png")
        fig.savefig('./png_course/obs_isy_traces_rick.png', bbox_inches='tight')
        
        # Vertical profiles (inverse)
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        fl2 = './output/RES03/inv_stack_betap.dat'
        inp2 = gt.readbin(fl2,nz,nx)

        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.08*fact
        for ind in range(100,600,100):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            tr2 *= 1/np.max(np.abs(tr2))*np.max(np.abs(tr1))
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,az,color='b')
            av.plot(fact*tr2,az,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Depth (km)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            #av.set_xlabel(r"\delta\beta_p", fontsize = labelsize)
            plt.ylim(az[0],az[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/inv_traces.png")
        fig.savefig('./png_course/inv_traces_rick.png', bbox_inches='tight')
        
    # dxsrc = 20 (and not 10)
    if csel[2]:

        #vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES04/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_dxsrc20.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'sym',0.7,labelsize)

        fl1 = './output/RES04/t1_isy_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/isy_0301_dxsrc20.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'sym',0.7,labelsize)

        fl1 = './output/RES06/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_dxsrc40.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'sym',0.7,labelsize)

        fl1 = './output/RES07/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_dxsrc80.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'sym',0.7,labelsize)

        fl1 = './output/RES08/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_dxsrc160.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'sym',0.7,labelsize)

        fl1 = './output/RES09/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_dxsrc320.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'sym',0.7,labelsize)

        # Vertical profiles (data) ==============================
        fl1 = './output/RES01/t1_obs_000321.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES01/t1_isy_000321.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces321_dxsrc10.png")
        fig.savefig('./png_course/obs_isy_traces321_dxsrc10.png', bbox_inches='tight')
        
        # Vertical profiles (data) ==============================
        fl1 = './output/RES04/t1_obs_000321.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES04/t1_isy_000321.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces321_dxsrc20.png")
        fig.savefig('./png_course/obs_isy_traces321_dxsrc20.png', bbox_inches='tight')
        
        # Vertical profiles (data) ==============================
        fl1 = './output/RES06/t1_obs_000321.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES06/t1_isy_000321.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces_dxsrc40.png")
        fig.savefig('./png_course/obs_isy_traces321_dxsrc40.png', bbox_inches='tight')
        
        # Vertical profiles (data) ==============================
        fl1 = './output/RES07/t1_obs_000321.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES07/t1_isy_000321.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces_dxsrc80.png")
        fig.savefig('./png_course/obs_isy_traces321_dxsrc80.png', bbox_inches='tight')
        
        # Vertical profiles (data) ==============================
        fl1 = './output/RES08/t1_obs_000321.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES08/t1_isy_000321.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces_dxsrc160.png")
        fig.savefig('./png_course/obs_isy_traces321_dxsrc160.png', bbox_inches='tight')
        
        # Vertical profiles (data) ==============================
        fl1 = './output/RES09/t1_obs_000321.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES09/t1_isy_000321.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces_dxsrc320.png")
        fig.savefig('./png_course/obs_isy_traces321_dxsrc320.png', bbox_inches='tight')
        
        # Vertical profiles (inverse)
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        fl2 = './output/RES04/inv_stack_betap.dat'
        inp2 = gt.readbin(fl2,nz,nx)

        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.08*fact
        for ind in range(100,600,100):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            #tr2 *= 1/np.max(np.abs(tr2))*np.max(np.abs(tr1))
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,az,color='b')
            av.plot(fact*tr2,az,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Depth (km)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            #av.set_xlabel(r"\delta\beta_p", fontsize = labelsize)
            plt.ylim(az[0],az[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/inv_traces_dxsrc20.png")
        fig.savefig('./png_course/inv_traces_dxsrc20.png', bbox_inches='tight')
        
    # Iterative solution
    if csel[3]:
        
        vhmin,vhmax = -0.1,0.1
        # Same as for previous image
        fl1 = './output/RES05/xi_000030.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/ite_stack_betap.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        # Not here!!!
        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES05/t1_ite_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/ite_0301.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'ddd',0.8,labelsize)

        # Vertical profiles (data) ==============================
        fl1 = './output/RES05/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES05/t1_ite_000301.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces.png")
        fig.savefig('./png_course/obs_ite_traces.png', bbox_inches='tight')
        
        # Vertical profiles (inverse)
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        fl2 = './output/RES05/xi_000030.dat'
        inp2 = gt.readbin(fl2,nz,nx)

        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.08*fact
        for ind in range(100,600,100):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,az,color='b')
            av.plot(fact*tr2,az,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Depth (km)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            #av.set_xlabel(r"\delta\beta_p", fontsize = labelsize)
            plt.ylim(az[0],az[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/inv_traces.png")
        fig.savefig('./png_course/ite_traces.png', bbox_inches='tight')
        
    # New background velocity field
    if csel[4]:
        vhmin,vhmax = 1.5,4.0
        fl1 = './output/RES10/vel_modif.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/vel_velmodif.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        vhmin,vhmax = -0.1,0.1
        fl1 = './output/RES10/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_velmodif.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES10/t1_isy_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/isy_0301_velmodif.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'ddd',0.8,labelsize)

        # Vertical profiles (data) ==============================
        fl1 = './output/RES10/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES10/t1_isy_000301.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces_velmodif.png")
        fig.savefig('./png_course/obs_isy_traces_velmodif.png', bbox_inches='tight')
        
        # Vertical profiles (inverse)
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        fl2 = './output/RES10/inv_stack_betap.dat'
        inp2 = gt.readbin(fl2,nz,nx)

        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.08*fact
        for ind in range(100,600,100):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,az,color='b')
            av.plot(fact*tr2,az,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Depth (km)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            #av.set_xlabel(r"\delta\beta_p", fontsize = labelsize)
            plt.ylim(az[0],az[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/inv_traces_velmodif.png")
        fig.savefig('./png_course/inv_traces_velmodif.png', bbox_inches='tight')
        
    # New background velocity field -- version 2
    if csel[5]:
        vhmin,vhmax = 1.5,4.0
        fl1 = './output/RES11/vel_modif.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/vel_velmodif2.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        vhmin,vhmax = -0.1,0.1
        fl1 = './output/RES11/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_velmodif2.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES11/t1_isy_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/isy_0301_velmodif2.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'ddd',0.8,labelsize)

        # Vertical profiles (data) ==============================
        fl1 = './output/RES11/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES11/t1_isy_000301.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces_velmodif2.png")
        fig.savefig('./png_course/obs_isy_traces_velmodif2.png', bbox_inches='tight')
        
        # Vertical profiles (inverse)
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        fl2 = './output/RES11/inv_stack_betap.dat'
        inp2 = gt.readbin(fl2,nz,nx)

        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.08*fact
        for ind in range(100,600,100):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,az,color='b')
            av.plot(fact*tr2,az,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Depth (km)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            #av.set_xlabel(r"\delta\beta_p", fontsize = labelsize)
            plt.ylim(az[0],az[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/inv_traces_velmodif2.png")
        fig.savefig('./png_course/inv_traces_velmodif2.png', bbox_inches='tight')
        
    # Truncation in the input data
    if csel[6]:
        vhmin,vhmax = -0.1,0.1
        fl1 = './output/RES12/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_trunc.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES11/t1_isy_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/isy_0301_velmodif2.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'ddd',0.8,labelsize)

        # Vertical profiles (data) ==============================
        fl1 = './output/RES01/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES12/t1_isy_000301.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces_trunc.png")
        fig.savefig('./png_course/obs_isy_traces_trunc.png', bbox_inches='tight')
        
        # Vertical profiles (inverse)
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        fl2 = './output/RES12/inv_stack_betap.dat'
        inp2 = gt.readbin(fl2,nz,nx)

        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.08*fact
        for ind in range(100,600,100):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,az,color='b')
            av.plot(fact*tr2,az,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Depth (km)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            #av.set_xlabel(r"\delta\beta_p", fontsize = labelsize)
            plt.ylim(az[0],az[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/inv_traces_trunc.png")
        fig.savefig('./png_course/inv_traces_trunc.png', bbox_inches='tight')
        
    # Lower central frequency (12.5 Hz)
    if csel[7]:
        vhmin,vhmax = -0.1,0.1
        fl1 = './output/RES13/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_freq.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

        vhmin,vhmax = -0.04,0.04
        fl1 = './output/RES13/t1_isy_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/isy_0301_freq.png'
        disp_imshow1(inp1,at,ao,title1,title2,flout,'ddd',0.8,labelsize)

        # Vertical profiles (data) ==============================
        fl1 = './output/RES13/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES13/t1_isy_000301.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        
        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.07*fact
        for ind in range(0,300,50):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,at,color='b')
            av.plot(fact*tr2,at,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Time (s)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            av.set_xlabel("Amplitude", fontsize = labelsize)
            plt.ylim(at[0],at[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/obs_isy_traces_trunc.png")
        fig.savefig('./png_course/obs_isy_traces_freq.png', bbox_inches='tight')
        
        # Vertical profiles (inverse)
        fl1 = './reflectivity/refl_exact.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        fl2 = './output/RES13/inv_stack_betap.dat'
        inp2 = gt.readbin(fl2,nz,nx)

        fig = plt.figure(figsize=(10,5), facecolor = "white")
        params = {'mathtext.default': 'regular' }          
        plt.rcParams.update(params)

        ipos = 0
        fact = 100
        vvmax = 0.08*fact
        for ind in range(100,600,100):
            ipos += 1
            tr1 = inp1[:,ind]
            tr2 = inp2[:,ind]
            av  = plt.subplot(1,6,ipos)
            av.plot(fact*tr1,az,color='b')
            av.plot(fact*tr2,az,color='r')
            av.tick_params(axis='both', which='major', labelsize=labelsize)
            if ipos == 1:
                av.set_ylabel("Depth (km)", fontsize = labelsize)
            else:
                av.axes.get_yaxis().set_visible(False)
            #av.set_xlabel(r"\delta\beta_p", fontsize = labelsize)
            plt.ylim(az[0],az[-1])
            av.invert_yaxis()
            plt.xlim(-vvmax,vvmax)
            #av.legend(fontsize=labelsize)
        fig.tight_layout()
        print("Export to ./png_course/inv_traces_trunc.png")
        fig.savefig('./png_course/inv_traces_freq.png', bbox_inches='tight')

    # More or less offsets
    if csel[8]:
        vhmin,vhmax = -0.1,0.1
        # from 25 to 300 every 25 + last at 300 with all offsets
        k = 0
        for ind in [17,16,15,1,19,20,21,22,23,24,25,26]:
            k += 1
            ss = str(ind).zfill(2)
            vv = str(k).zfill(3)
            fl1 = './output/RES' + ss +  '/inv_stack_betap.dat'
            inp1 = gt.readbin(fl1,nz,nx)
            title1,title2='Position x (km)','Depth (km)'
            flout = './png_course/inv_stack_betap_off_' + vv + '.png'
            disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',1.0,labelsize)

    # More or less input traces (single shot)
    if csel[9]:
        fl1 = './reflectivity/dbetap_exact.dat'
        inp0 = gt.readbin(fl1,nz,nx)
        #inps = gaussian_filter(inp0,3)
        #inp0 -= inps
        #inp0 = sobel(inp0)
        inp0 /= np.max(np.abs(inp0))*0.5
        fl1 = './input/marm2_smo.dat'
        inp2 = gt.readbin(fl1,nz,nx)
        
        k = 0
        for ind in [27,28,29,30,31,32,33,34]:
            k += 1
            ss = str(ind).zfill(2)
            vv = str(k).zfill(3)
            fl1 = './output/RES' + ss +  '/adj_stack_betap.dat'
            inp1 = gt.readbin(fl1,nz,nx)
            dd = np.max(np.abs(inp1))
            inp1 /= dd
            title1,title2='Position x (km)','Depth (km)'
            flout = './png_course/inv_stack_betap_traces_' + vv + '.png'
            disp_imshow1_alpha(inp1,inp0,az,ax,title1,title2,flout,'sym','sym',0.7,0.7,labelsize,0.8)
        # For reference, without alpha
        for ind in [27,34]:
            k += 1
            ss = str(ind).zfill(2)
            vv = str(k).zfill(3)
            fl1 = './output/RES' + ss +  '/adj_stack_betap.dat'
            inp1 = gt.readbin(fl1,nz,nx)
            dd = np.max(np.abs(inp1))
            title1,title2='Position x (km)','Depth (km)'
            flout = './png_course/inv_stack_betap_traces_ref_' + vv + '.png'
            disp_imshow1(inp1,az,ax,title1,title2,flout,'sym',0.7,labelsize) 

            
    # SpotLight logo
    if csel[10]:
        # New scale in x
        nx        = 660
        ax        = fx + np.arange(nx)*dx

        vhmin,vhmax = -0.1/2,0.1/2
        fl1 = './output/RES35/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_spt1.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',0.7,labelsize)

        vhmin,vhmax = -0.1/4,0.1/4
        fl1 = './output/RES37/inv_stack_betap.dat'
        inp1 = gt.readbin(fl1,nz,nx)
        title1,title2='Position x (km)','Depth (km)'
        flout = './png_course/inv_stack_betap_spt2.png'
        disp_imshow1(inp1,az,ax,title1,title2,flout,'ddd',0.7,labelsize)

        vhmin,vhmax = -0.04/2,0.04/2
        fl1 = './output/RES35/t1_obs_000301.dat'
        inp1 = gt.readbin(fl1,no,nt).transpose()
        fl2 = './output/RES35/t1_obs_000401.dat'
        inp2 = gt.readbin(fl2,no,nt).transpose()
        title1,title2='Offset (km)','Time (s)'
        flout = './png_course/obs_0301_0401_spt.png'
        disp_imshow2(inp1,inp2,at,ao,title1,title2,flout,'ddd',0.6,labelsize)
