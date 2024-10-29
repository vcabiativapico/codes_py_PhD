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
from matplotlib import gridspec
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from scipy.signal import hilbert
if __name__ == "__main__":

    os.system('mkdir -p png_course')

    # Global parameters
    labelsize = 16
    nt = 1501
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

    # hmin,hmax = 2.0,2.1
    # # fl1 = './input/marm2_smo.dat'
    # # fl1 = './input/vel_mod_bin.dat'
    # fl1 = './output/simple/born/dbetap_exact.dat'
    # #fl1 = './input/marm2_smooth.dat'
    # inp2 = gt.readbin(fl1,nz,nx)
    # #flout = './png/born/born_init_full.png'

    # fig = plt.figure(figsize=(10,5), facecolor = "white")
    # av  = plt.subplot(1,1,1)
    # hfig = av.imshow(inp2, extent=[ax[0],ax[-1],az[-1],az[0]], \
    #                   vmin=hmin,vmax=hmax,aspect='auto', \
    #                   cmap='bwr')
    # #hfig = av.imshow(inp2, extent=[ax[0],ax[0],az[0],az[0]], \
    # #                 vmin=hmin,vmax=hmax,aspect='auto', \
    # #                 cmap='bwr')

    # plt.colorbar(hfig)
    # fig.tight_layout()
    #print("Export to file:",flout)
    #fig.savefig(flout, bbox_inches='tight')

    # ================================================

    # ###data
# %%
# figsize=(11,6)
    def plot_shot_gathers_traces(hmax, inp1, inp2, flout, tr):
        hmin = -hmax
        axi = np.zeros(np.size(tr)+1)
        fig, (axi) = plt.subplots(nrows=1, ncols=np.size(tr)+1,
                                  sharey=True,
                                  figsize=(11, 6),
                                  facecolor="white",
                                  gridspec_kw=dict(width_ratios=[5, 1, 1, 1]))

        axi[0].imshow(inp1, extent=[ao[0], ao[-1], at[-1], at[0]],
                      vmin=hmin, vmax=hmax, aspect='auto',
                      cmap='seismic')

        for i in range(np.size(tr)):
            axi[0].axvline(x=ao[tr[i]], color='k', ls='--')
        # plt.colorbar(ax0)
        axi[0].set_ylabel('Time (s)')
        axi[0].set_xlabel('Offset (km)')

        fig.tight_layout()
        #
        #

        for i in range(np.size(tr)):
            xmin = np.min(inp1[:, tr[i]]) + np.min(inp1[:, tr[i]])/1.5
            xmin = 0.2
            xmax = -xmin

            axi[i+1].plot(inp1[:, tr[i]], at, 'k')
            axi[i+1].plot(inp2[:, tr[i]], at, 'r--')
            axi[i+1].set_xlim(xmin, xmax)
            axi[i+1].set_ylim(2, ft)
            axi[i+1].xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

            # plt.colorbar()
            fig.tight_layout()
        axi[1].legend(['Born', 'FWI'], loc='upper left', shadow=True)

        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
        
    
    tr   = [71,135,201]
    tr1 = '../output/48_const_2000_ano/ano/t1_obs_000209.dat'
    inp1 = gt.readbin(tr1,no,nt).transpose()
    flout  = '../png/t1_obs_000209.png'
    hmax = np.max(inp1)
    plot_shot_gathers_traces(hmax,inp1,inp1,flout,tr)
    
    # hmax = 0.1
    # tr1 = './output/23_mig/nh10_is4_2/t1_obs_000301.dat'
    # tr2 = './output/23_mig/nh10_is4_2/t1_syn_000301.dat'
    # # tr1  = './output/15_picked_models/4_CO2/born/t1_obs_000301.dat'
    # # tr2  = './output/15_picked_models/4_CO2/fwi/t1_obs_000301.dat'
    # inp1 = gt.readbin(tr1,no,nt).transpose()
    # inp2 = -gt.readbin(tr2,no,nt).transpose()
    # tr   = [71,135,201]
    # flout  = './png/trace_mig.png'
    # plot_shot_gathers_traces(hmax,inp1,inp2,flout,tr)


# %%


    def plot_trace(xmax, inp1, inp2, flout, tr):

        axi = np.zeros(np.size(tr))
        fig, (axi) = plt.subplots(nrows=1, ncols=np.size(tr),
                                  sharey=True,
                                  figsize=(8, 8),
                                  facecolor="white")

        ratio = np.asarray(tr, dtype='f')
        for i in range(np.size(tr)):
            # xmin = np.min(inp1[:,tr[i]]) + np.min(inp1[:,tr[i]])/1.5
            # xmin = 1.0
            xmax = 1801*0.0012
            xmin = -xmax
            ratio[i] = np.max(inp2[:, tr[i]])/np.max(inp1[:, tr[i]])
            inp1[:, tr[i]] = (inp1[:, tr[i]]/np.max(inp1[:, tr[i]]))
            inp2[:, tr[i]] = (inp2[:, tr[i]]/np.max(inp2[:, tr[i]]))
            axi[i].plot(inp1[:, tr[i]], at, 'r')

            axi[i].plot(inp2[:, tr[i]], at, 'b--')

            axi[i].set_xlim(xmin, xmax)
            axi[i].set_ylim(2, ft)

            axi[i].xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))

            axi[i].set_xlabel("Ratio = "+str(f'{ratio[i]:.2f}'))
            # plt.colorbar()
            fig.tight_layout()

        axi[0].set_ylabel('Time (s)')
        axi[0].legend(['obs', 'syn'], loc='upper left', shadow=True)

        # axi[0].legend(['Baseline','Monitor'],loc='upper left',shadow=True)
        fig.text(0.48, -0.01, "Amplitude")
        fig.text(0.48, 1, 'Comparison')
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')

        return ratio, inp1, inp2

    # # #### TO PLOT SHOTS FROM MODELLING
    def plot_shot_gathers(hmin, hmax, inp, flout):

        fig = plt.figure(figsize=(10, 8), facecolor="white")
        av = plt.subplot(1, 1, 1)
        hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                         vmin=hmin, vmax=hmax, aspect='auto',
                         cmap='seismic')
        # for i in range(np.size(tr)):
        #     plt.axvline(x=ao[tr[i]], color='k', ls='--')
        plt.title('x= '+str(title*12))
        plt.colorbar(hfig, format='%2.2f')
        plt.rcParams['font.size'] = 22
        plt.ylim(at[-1],ft)
        plt.xlabel('Offset (km)')
        plt.ylabel('Time (s)')
        fig.tight_layout()
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')

    xmax_tr = 0.3
    nt =1801
    # title = 501
    # title = 401
    title = 301
    no = 251
    # title,no = 201,251
    # nomax = 201
    # no = (nomax)+101
    # no = 302 # for 101 & 501
    # no = 401 # for 201 & 401
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
    # tr1 = '../output/26_mig_4_interfaces/binv_rc_norm/t1_obs_000'+str(title)+'.dat'
    # tr2 = '../output/26_mig_4_interfaces/binv_rc_norm/t1_syn_000'+str(title)+'.dat'
    
    
    # tr1 = '../output/38_mig_workflow/binv/t1_obs_000'+str(title)+'.dat'
    # tr2 = '../output/27_marm/binv/t1_obs_000'+str(title)+'.dat'
    # tr3 = '../output/27_marm/diff_marm_corr/t1_obs_000'+str(title)+'.dat'
    # tr1 = '../output/27_marm/binv/t1_obs_000'+str(title)+'.dat'
   
    # tr1 = '../output/68_thick_marm_ano/org_thick/t1_obs_000'+str(title)+'.dat'
    # tr2 = '../output/68_thick_marm_ano/ano_thick/t1_obs_000'+str(title)+'.dat'
    
    # tr1 = '../output/71_thick_marm_ano_born_mig/simulations/new_pert_mig_sm_org/t1_obs_000'+str(title)+'.dat'
    # tr2 = '../output/71_thick_marm_ano_born_mig/simulations/new_pert_mig_sm_ano/t1_obs_000'+str(title)+'.dat'
    
    tr1 = '../output/72_thick_marm_ano_born_mig_flat/org/t1_obs_000'+str(title)+'.dat'
    
    # tr1 = '../output/63_evaluating_thickness/vel_thick_204/t1_obs_000'+str(title)+'.dat'
    tr1 ='../output/73_new_flat_sm/born_org_rho/t1_obs_000361.dat'
    tr2 ='../output/73_new_flat_sm/born_ano_rho/t1_obs_000361.dat'


    tr1 ='../output/74_test_flat/op_inv_lsm_aco/t1_obs_000361.dat'
    tr2 ='../output/74_test_flat/op_inv_lsm_aco/t1_syn_000361.dat'
    
    
    tr1 ='../output/75_marm_sum_pert/org/t1_obs_000361.dat'
    
    # tr1 ='../output/74_test_flat/new_vop8/org_full/t1_obs_000302.dat'
    # tr2 ='../output/74_test_flat/new_vop8/ano_full/t1_obs_000302.dat'
    
    
    # tr1 ='../output/73_new_flat_sm/org/t1_obs_000361.dat'
    # tr2 ='../output/73_new_flat_sm/org/t1_syn_000361.dat'

    inp1 = -gt.readbin(tr1, no, nt).transpose()
    inp2 = -gt.readbin(tr2, no, nt).transpose()
    
    
    
    src_nm = '../output/74_test_flat/op_inv_lsm_aco/wsrc.dat'
    wsrc = gt.readbin(src_nm,177,1)
    plt.plot(wsrc)
    
    src_nm = '../output/74_test_flat/op_inv_lsm_aco/winvsrc.dat'
    wsrc = gt.readbin(src_nm,nt,1)
    plt.plot(wsrc)
     

    
    # from spotfunk.res import procs
    # tr_rms = [] 
    # for i in range(no):
    #     tr_rms.append(procs.RMS_calculator(inp1[:,i]))

    
    hmin = np.min(inp1)
    hmax = -hmin
    flout_gather = '../png/obs_'+str(title)+'.png'
    plot_shot_gathers(hmin, hmax, -inp1, flout_gather)
 
    
    # hmin = np.max(inp1)
    # hmax = -hmin
    flout_gather = '../png/obs_'+str(title)+'.png'
    plot_shot_gathers(hmin, hmax, -inp2, flout_gather)
   
    
    diff = inp1 - inp2
    # hmin = np.min(diff)*10
    # hmax = -hmin
    flout_gather = '../png/obs_'+str(title)+'.png'
    plot_shot_gathers(hmin, hmax, diff, flout_gather)
    
    # plt.plot(ao[idx_nr_off]+ao[idx_nr_off],at)
    
    # plt.figure(figsize=(3, 8))
    # plt.plot(inp1[:,151]*10,at)
    # plt.ylim(nt*dt,ft)
    
    # hmin, hmax = -0.1,0.1
    # flout_gather = '../png/27_marm/syn_'+str(title)+'.png'
    # plot_shot_gathers(hmin, hmax, inp_hilb2, flout_gather)
    
    
    # hmin, hmax = -0.1,0.1
    # flout_gather = '../png/27_marm/obs_'+str(title)+'.png'
    # plot_shot_gathers(hmin, hmax, inp_hilb3, flout_gather)

    
    # xmax = 0.2
    # # tr   = [71,135,201]
    # flout = '../png/26_mig_4_interfaces/badj_rc_norm_2979/traces_'+str(title)+'.png'
    # r, i1, i2 = plot_trace(xmax, inp1, inp2, flout, tr)

   
    # fl_ref = '../output/74_test_flat/op_inv_lsm_aco/inv_betap_x_s.dat'
    # inp_ref = gt.readbin(fl_ref,nz,nx)
    # flout = '../png/inv_betap_x_s.png'
    # plot_mig(inp_ref,flout)
    
    
        
    # fl_ref = '../output/74_test_flat/voption8_true/inv_betap_x_s.dat'
    # inp_ref = gt.readbin(fl_ref,nz,nx)
    # flout = '../png/inv_betap_x_s.png'
    # plot_mig(inp_ref,flout)
    
 


    # flout_gather = './png/19_anomaly_4_layers/190_ano/gather_fwi_190_ano.png'
    # plot_shot_gathers(hmin,hmax,inp2,flout_gather)

    # diff_born_fwi = (inp_hilb1-inp_hilb2)*6
    # hmin, hmax = -0.1,0.1
    # flout_gather = '../png/27_marm/diff_org_ano_'+str(title)+'.png'
    # plot_shot_gathers(hmin,hmax,diff_born_fwi,flout_gather)
# hmax = np.max(np.abs(inp2))/2
# hmin = -hmax
    # hmin,hmax = -0.1,0.1

    # # ## To plot shots gathers after modelling

    # fl1  = './output/15_picked_models/0_CO2/born/t1_obs_000301.dat'
    # inp1 = gt.readbin(fl1,no,nt).transpose()
    # flout  = './png/15_picked_models/fwi3_obs_0301_4p0_org.png'

    # fl2  = './output/18_3_interfaces/anomaly/fwi/t1_obs_000301.dat'
    # inp2 = gt.readbin(fl2,no,nt).transpose()
    # flout2  = './png/18_3_interfaces/ano_fwi_obs_0301_4p0_org.png'

    # fl3  = './output/18_3_interfaces/anomaly/born/t1_obs_000301.dat'
    # inp3 = -gt.readbin(fl3,no,nt).transpose()
    # flout3  = './png/18_3_interfaces/ano_born_obs_0301_4p0_org.png'

    # diff_born_fwi = inp3-inp2
    # flout_diff = './png/18_3_interfaces/ano_diff_obs_0301_4p0_org.png'

    # # fl1  = './output/t1_obs_000301.dat'
    # # inp1 = gt.readbin(fl1,no,nt).transpose()
    # # flout  = './png/15_picked_models/testborn_obs_0301_4p0_org.png'

    # plot_shot_gathers(hmin, hmax, inp1, flout)
    # plot_shot_gathers(hmin, hmax, inp2, flout2)
    # # plot_shot_gathers(hmin, hmax, inp3, flout3)
    # plot_shot_gathers(hmin, hmax, diff_born_fwi, flout_diff)

    # fl1  = './output/t1_obs_000301.dat'
    # inp1 = gt.readbin(fl1,no,nt).transpose()
    # flout  = './png/born_rho.png'

    # plot_shot_gathers(hmin, hmax, inp1, flout)
# %%
# FOR MODEL SMOOTHING
    # if True:
    #    fl1 = './input/marm2_full.dat'
    #    inp2 = gaussian_filter(gt.readbin(fl1,nz,nx),15)
    #    fl1 = './input/marm2_smooth_s15.dat'
    #    gt.writebin(inp2,fl1)
# To plot SMOOTH MODELS


    def read_results(path):
        spot_x = []
        spot_y = []
        spot_z = []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            header = next(spamreader)
            for row in spamreader:
                spot_x.append(float(row[7]))
                spot_y.append(float(row[8]))
                spot_z.append(float(row[9]))
        return spot_x, spot_y, spot_z
    name = 0
    hmin, hmax = 0.1, 2.2

    def plot_model(inp, flout):
        # hmax = np.max(np.abs(inp2))
        # hmin = -hmax

        # print(hmin,hmax)
        plt.rcParams['font.size'] = 20
        # hmax = 5.0
        # hmin = 1.5
        hmin = np.min(inp)
        # hmax = -hmin
        hmax = np.max(inp)
        # hmin = 0
        if np.shape(inp)[1] > 60:
            fig = plt.figure(figsize=(14, 7), facecolor="white")
            av = plt.subplot(1, 1, 1)
            hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                              vmin=hmin, vmax=hmax, aspect='auto')
            plt.xlabel('Distance (km)')
            plt.ylabel('Depth (km)')
        else:
            fig = plt.figure(figsize=(5, 5), facecolor="white")
            av = plt.subplot(1, 1, 1)
            hfig1 = av.imshow(inp, extent=[-nh, nh, az[-1], az[0]],
                              vmin=hmin, vmax=hmax, aspect='auto', cmap='seismic')
        # av.set_xlim([2.5,4.5])
        # av.set_ylim([0.8,0.4])
        # plt.axvline(x=ax[tr], color='k',ls='--')
        # plt.axhline(0.606, color='w')
        plt.colorbar(hfig1, format='%1.1f',label='m/s')
        # plt.colorbar(hfig1, format='%1.1f',label='m/s')
        fig.tight_layout()

        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
        return inp, fig
    
    def plot_mig(inp, flout):
        # hmax = np.max(np.abs(inp2))
        # hmin = -hmax

        # print(hmin,hmax)
        plt.rcParams['font.size'] = 20
        # hmax = 5.0
        # hmin = 1.5
        hmin = np.min(inp)
        hmax = -hmin
        # hmax = np.max(inp)
        # hmin = 0
        if np.shape(inp)[1] > 60:
            fig = plt.figure(figsize=(14, 7), facecolor="white")
            av = plt.subplot(1, 1, 1)
            hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                              vmin=hmin, vmax=hmax, aspect='auto')
            plt.xlabel('Distance (km)')
            plt.ylabel('Depth (km)')
        else:
            fig = plt.figure(figsize=(5, 5), facecolor="white")
            av = plt.subplot(1, 1, 1)
            hfig1 = av.imshow(inp, extent=[-nh, nh, az[-1], az[0]],
                              vmin=hmin, vmax=hmax, aspect='auto', cmap='seismic')
        # av.set_xlim([2.5,4.5])
        # av.set_ylim([0.8,0.4])
        # plt.axvline(x=ax[tr], color='k',ls='--')
        # plt.axhline(0.606, color='w')
        plt.colorbar(hfig1, format='%1.2f',label='m/s')
        # plt.colorbar(hfig1, format='%1.1f',label='m/s')
        fig.tight_layout()

        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')
        return inp, fig
    
   
    # fl1 = './output/smooth_test/smooth'+str(name)+'/abetap.dat'
    
    # # fl1 = '../output/45_marm_ano_v3/mig_binv_sm8_TL1801/dbetap_exact.dat'
    # fl1='../input/46_flat_simple_taper/inp_vel_taper_all_ano.dat'
    # # fl1='../input/45_marm_ano_v3/fwi_sm.dat'
    # fl1 = '../input/vel_smooth.dat'
    
    # fl1 = '../input/smooth_input/marm2_smooth_s15.dat'
    
    # fl1 = '../input/vel_full.dat'
    # fl1 = '../input/marm2_sm15.dat'
    # fl1 ='../input/vel_smooth.dat'
    # fl1 = '../input/68_thick_marm_ano/marm_thick_ano.dat'
    # fl1 = '../output/48_const_2000_ano/flat_born_ano/adbetap.dat'
    # fl1 = '../input/46_flat_simple_taper/inp_vel_taper_all_ano.dat'
    # fl1 = '../output/45_marm_ano_v3/mig_binv_sm8_TL1801/dbetap_exact.dat'
    # fl1 = '../output/63_evaluating_thickness/vel_thick_588/avp_init.dat'
    # fl1 = '../input/68_thick_marm_ano/marm_thick_org_sm8.dat'
    
    
    # fl1 = '../input/63_evaluating_thickness/vel_492.dat'
    # fl1= '../output/27_marm/binv'
    # fl1 = '../input/vel_smooth.dat'
    # fl1 = '../output/abetap.dat'
    # fl1 = '../input/45_marm_ano_v3/fwi_ano_114_percent.dat'
    # fl1 = '../output/40_marm_ano/binv/abetap.dat'
    # fl1 =  '../input/vel_full.dat'
    # fl1 = '../input/39_mig_marm_flat/vel_marm_plus_flat_corr.dat'

    # fl1 = '../input/31_const_flat_tap/inp_flat_2050_const.dat'
    # # # fl1 = './input/13_4_ano_smoo/4_ano_4p0_smoo5.dat'
    # # fl1 = '../input/10_onl_two_ano/1p5_two_ano_4p0.dat'
    
    # fl_full = '../input/org_full/marm2_full.dat'
    # inp_full = gt.readbin(fl_full,nz,nx)
    # plot_model(inp_full, flout)
    
    # fl_sm = '../input/org_full/marm2_sm15.dat'
    
        # # fl_sm = '../input/vel_full.dat'
        # inp_sm = gt.readbin(fl_sm,nz,nx)
        # plot_model(inp_sm, flout)
        
        # inp_betap = 1/inp_full**2 - 1/inp_sm**2
        # plot_mig(inp_betap, flout)
        
        # fl = '../output/68_thick_marm_ano/ano_thick/avp_exact.dat'
        # inp = gt.readbin(fl,nz,nx)
        # plot_mig(inp, flout)
        
        # fl1 = '../output/68_thick_marm_ano/ano_thick/abetap.dat'
        # fl1 ='../input/org_full/marm2_full.dat'
        # inp1 = gt.readbin(fl1,nz,nx)
        # print(np.max(inp1))
        # print(np.min(inp1))
        # # flout = '../png/avp_exact.png'
        # flout = '../png/dbetap_exact.png'
        # plot_model(inp1, flout)
        
    
    # plot_mig(1 / sqrt(inp1 - 1/(inp**2)), flout)
    fl1 = '../input/vel_smooth.dat'    
    inp1 = gt.readbin(fl1,nz,nx)
    flout = '../png/adbetap.png'
    plot_model(inp1,flout)
    
    

    fl1 = '../output/72_thick_marm_ano_born_mig_flat/org_full/avp_exact.dat'    
    inp1 = gt.readbin(fl1,nz,nx)
    flout = '../png/adbetap.png'
    plot_model(inp1,flout)
    
    fl2 = '../output/72_thick_marm_ano_born_mig_flat/ano_full/avp_exact.dat'    
    inp2 = gt.readbin(fl2,nz,nx)
    flout = '../png/adbetap.png'
    plot_model(inp2,flout)
    
    inp_diff = inp1-inp2
    plot_mig(inp_diff,flout)
    
    
    # fl3 = '../output/32_2050_flat_tap/binv/dbetap_exact.dat'
    fl3 = '../output/72_thick_marm_ano_born_mig_flat/org/inv_betap_x.dat'
   
    
   
    fl3 = '../output/73_new_flat_sm/org/abetap.dat'
    inp3 = gt.readbin(fl3,nz,nx)
    flout = '../png/inv_betap_x_s.png'
    plot_model(inp3,flout)
    
    fl4 = '../output/73_new_flat_sm/org/adbetap.dat'
    inp4 = gt.readbin(fl4,nz,nx)
    flout = '../png/inv_betap_x_s.png'
    plot_model(inp4,flout)
    
    
    somme = 1/np.sqrt(inp3+inp4)
    plot_model(somme,flout)
    
    
    
    fl_ref = '../output/74_test_flat/op_inv_lsm_aco/inv_betap_x_s.dat'
    inp_ref = gt.readbin(fl_ref,nz,nx)
    flout = '../png/inv_betap_x_s.png'
    plot_mig(inp_ref,flout)
    
    
        
    fl_ref = '../output/73_new_flat_sm/org/inv_betap_x_s.dat'
    inp_ref = gt.readbin(fl_ref,nz,nx)
    flout = '../png/inv_betap_x_s.png'
    plot_mig(inp_ref,flout)
    
    
    fl4 = '../output/73_new_flat_sm/born_ano_rho/inv_betap_x_s.dat'
    inp4= gt.readbin(fl4,nz,nx)
    flout = '../png/inv_betap_x_s.png'
    plot_mig(inp4,flout)
    
    
    inp_diff = inp3-inp4
    plot_mig(inp_diff,flout)
   
    
    
    # fl3 ='../input/45_marm_ano_v3/fwi_ano.dat'
    # inp3= gt.readbin(fl3,nz,nx)
    
    # diff = inp1-inp3
    
    # hmax= np.max(inp4)
    # hmin= np.min(inp4)
    # plt.figure(figsize=(18,10))
    # # plt.imshow(inp4, extent=[ax[0], ax[-1], az[-1], az[0]],
    # #                   vmin=hmin, vmax=hmax, aspect='auto', cmap='seismic')
    # plt.imshow(inp4,extent=[1, 601, az[-1], az[0]], vmin=hmin, vmax=hmax, aspect='auto', cmap='seismic')
    
    
    # hmax= np.max(diff)
    # hmin= np.min(diff)
    
    # plt.figure()
    # plt.imshow(diff, extent=[ax[0], ax[-1], az[-1], az[0]],
    #                   vmin=hmin, vmax=hmax, aspect='auto', cmap='seismic',alpha=1)
    
   # plt.axhline(50*dz,color='k')
    # plt.axhline(99*dz,color='k')

    # plt.figure(figsize=(5,10))
    # plt.plot(inp4[:,300],az)
    # plt.plot((inp1[:,300]-2)*25,az)
    # plt.axhline(50*dz,color='r')
    # # plt.axhline(99.5*dz,color='r')
    # # plt.ylim(1.1,1.35)
    # plt.ylim(0.75,0.5)
    
    # plt.figure(figsize=(5,10))
    # plt.plot(inp4[:,300],az)
    # # plt.plot((inp1[:,300]-2)*25,az)
    # # plt.axhline(50.5*dz,color='r')
    # plt.axhline(99*dz,color='r')
    # plt.ylim(1.1,1.35)
    
    
    
    # car = inp1*inp4/-4
    # flout = '../png/26_mig_4_interfaces/badj_rc_norm/inv_betap_norm.png'
    # plot_model(car,flout)    
    
    # fl5 = '../output/26_mig_4_interfaces/binv/inv_betap_h.dat'
    # inp5= gt.readbin(fl5,nz,nh2)
    # flout = '../png/26_mig_4_interfaces/binv/inv_betap_h.png'
    # plot_model(-inp5,flout)

    def plot_trace(xmax, fl, flout, tr, title):

        inp1 = gt.readbin(fl[0], nz, nx)
        inp2 = gt.readbin(fl[1], nz, nx)
        inp3 = gt.readbin(fl[2], nz, nx)
        axi = np.zeros(np.size(tr))
        fig, (axi) = plt.subplots(nrows=1, ncols=np.size(tr),
                                  sharey=True,
                                  figsize=(4, 8),
                                  facecolor="white")

        # xmin = np.min(inp1[:,tr[i]]) + np.min(inp1[:,tr[i]])/1.5
        # xmin = 0.4
        xmin = -xmax

        axi.plot(inp1[:, tr], az, 'r')
        axi.plot(inp2[:, tr], az, 'k')
        axi.plot(inp3[:, tr], az, 'b')
        axi.set_xlim(xmin, xmax)
        axi.set_ylim(az[-1], fz)
        axi.xaxis.set_major_formatter(FormatStrFormatter('%1.2f'))
        plt.rcParams['font.size'] = 14
        # plt.colorbar()
        fig.tight_layout()
        axi.set_ylabel('Time (s)')
        print(np.shape(inp1))

        axi.legend(title, loc='upper left', shadow=True)
        fig.text(0.48, -0.01, "Amplitude")
        fig.text(0.5, 1, '$\delta p$')
        print("Export to file:", flout)
        fig.savefig(flout, bbox_inches='tight')

    title = ['org', 171, 210]
    fl = [0]*(np.size(title))
    for i in range(np.size(title)-1):
        fl[i+1] = '../output/19_anomaly_4_layers/ano_' + \
            str(title[i+1])+'/born/dbetap_exact.dat'
        fl[0] = '../output/19_anomaly_4_layers/org/born/dbetap_exact.dat'
    flout = '../png/19_anomaly_4_layers/born_trace_adbetap.png'
    xmax_tr = 0.3
    tr = [301]
    # plot_trace(xmax_tr,fl,flout,tr,title)


# TO compute the difference
    #number = 'A9'
    # hmin,hmax = -0.06,0.06

    # fl1 = './output/two_ano/Born/A10/t1_obs_000301.dat'
    # inp2 = -gt.readbin(fl1,no,nt).transpose()
    # # hmax = np.max(np.abs(inp2))/2
    # # hmin = -hmax

    # fl2 = './output/smooth_test/smooth4/t1_obs_000301.dat'
    # inp2 = -gt.readbin(fl2,no,nt).transpose()

    # diff = inp2 - inp2

    # flout = './png/two_ano/full_born_sm4_diff_0301.png'

    # fig = plt.figure(figsize=(10,5), facecolor = "white")
    # av  = plt.subplot(1,1,1)
    # hfig = av.imshow(diff, extent=[ao[0],ao[-1],at[-1],at[0]], \
    #                   vmin=hmin,vmax=hmax,aspect='auto', \
    #                   cmap='seismic')
    # plt.colorbar(hfig)
    # fig.tight_layout()
    # print("Export to file:",flout)
    # fig.savefig(flout, bbox_inches='tight')
# %%

#  To plot WAVEFIELD SNAPSHOTS
    #name = '4'
    # hmin,hmax = 1.5,4.0

    #nxl = nx
    #fl1 = './output/smooth_test/smooth'+str(name)+'/avp_exact.dat'
    # fl1 = './input/marm2_smooth_s15.dat'
    # fl1 = './output/10_wf_sim_two_ano/4p0/p2d_000004.dat'
    # fl2 = './output/10_wf_sim_two_ano/1p5_all/p2d_000004.dat'
    # fl1 = './output/10_wf_sim_two_ano/4p0_1001_off/p2d_000004.dat'
    # fl2 = './output/10_w    name = '4'
    hmin, hmax = 1.5, 4.0
    fl1 = './output/smooth_test/smooth'+str(name)+'/avp_exact.dat'
    fl1 = './input/marm2_smooth_s15.dat'
    fl1 = './input/marm2_full.dat'

    # hmax = np.max(np.abs(inp2))
    # hmin = -hmax
    # hmax = np.max(inp2)
    # hmin = np.min(inp2)
    # print(hmin,hmax)

    flout = './png/smooth_model_s_org.png'

    fig = plt.figure(figsize=(10, 5), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp2, extent=[ax[0], ax[-1], az[-1], az[0]],
                      aspect='auto', alpha=0.7,
                      cmap='Greys')
    plt.colorbar(hfig1)
    fig.tight_layout()
    print("Export to file:", flout)
    fig.savefig(flout, bbox_inches='tight')


# INPUT FOR MARMOUSI ANOMALIES
#     fl1 = './input/10_onl_two_ano/blw_1p5_4ano_4p0.dat'
#     inp1 = gt.readbin(fl1,nz,nx) # model
#     # fl2 = './output/13_4_ano_smoo/sm10/born/p2d_lsm_000004.dat'
#     # fl3 = './output/13_4_ano_smoo/sm10/fwi/p2d_fwi_000004.dat'
#     # fl5 = './output/10_wf_onl_two_ano/fwi/1p5_all_201_off/p2d_fwi_000004.dat'


#     fl2 = './output/p2d_lsm_000004.dat'
# ######## INPUT FOR SIMPLE MODEL


# %%

# ######## INPUT FOR SIMPLE MODEL

    # fl1 = './input/19_anomaly_4_layers/3_interfaces_org_smooth5.dat'
    # fl1 = '../input/19_anomaly_4_layers/3_interfaces_anomaly_250.dat'
    # fl1 = '../input/18_3_interface/3_interfaces_dp.dat'
    # fl1 = '../input/vel_full.dat'
    
    fl1 = '../output/30_marm_flat/adbetap.dat'
    inp1 = gt.readbin(fl1, nz, nx)  # model
    
    # fl2 = './output/12_wf_simple/sm10_2060/born/p2d_lsm_000004.dat'
    # fl3 = './output/12_wf_simple/sm10_2060/fwi/p2d_fwi_000004.dat'
    # fl5 = './output/12_wf_simple/sm15_4000/p0_fwi/p2d_fwi_000004.dat'

    # fl2 = '../output/18_3_interfaces/org/born/p2d_lsm_000004.dat'
    # fl3 = '../output/19_anomaly_4_layers/ano_190/fwi/p2d_fwi_000004.dat'
    # fl5 = '../output/19_anomaly_4_layers/p0/fwi/p2d_fwi_000004.dat'


    fl2 = '../output/30_marm_flat/p2d_lsm_000001.dat'
    nt = 1501
    # nxl = 443
    nxl = 291
    h_nxl = int((nxl-1)/2)
    born = -gt.readbin(fl2, nz, nxl*nt)   #
    # fwi = gt.readbin(fl3, nz, nxl*nt)    #
    # p0_fwi = gt.readbin(fl5, nz, nxl*nt)

    # nxl = bites/4/nz/nt = bites/4/151/1501
    # position central (301-1)*dx = 3600
    # 291 = 1+2*145
    # point à gauche = 3600-145*dx
    # point à droite = 3600+145*dx

    left_p = 300-h_nxl  # left point
    right_p = 300+h_nxl  # right point

    left_p2 = 200-h_nxl  # left point
    right_p2 = 200+h_nxl  # right point

    # print("size",np.shape(born))

    born = np.reshape(born, (nz, nt, nxl))
    # fwi = np.reshape(fwi, (nz, nt, nxl))
    # p0_fwi = np.reshape(p0_fwi, (nz, nt, nxl))

    # # print("size",np.shape(born))

    # dm_fwi = p0_fwi-fwi  # Reflected FWI wavefield

    # diff = born-dm_fwi

    # diff = born+fwi
    # hmax = 10
    # hmin = -10
    # hmax = np.max(born)
    # hmin = np.min(born)
    # print(hmin,hmax)

    # TO PRODUCE SIMULATIONS OF THE FULL WAVEFIELD OVERLAYING THE MODEL

    def plot_sim_wf(bg, inp1):
        hmax = np.max(inp1)
        print('hmax: ', hmax)
        hmax = 1
        hmin = -hmax
        for i in range(100, 1500, 100):
            fig = plt.figure(figsize=(13, 6), facecolor="white")
            av = plt.subplot(1, 1, 1)
            hfig1 = av.imshow(bg[:, left_p+50:right_p-50], extent=[ax[left_p+50], ax[right_p-50], az[-1], az[0]],
                              aspect='auto', alpha=1,
                              cmap='Greys')
            hfig = av.imshow(inp1[:, i, 50:383], extent=[ax[left_p+50], ax[right_p-50], az[-1], az[0]],
                             vmin=hmin, vmax=hmax, aspect='auto', alpha=0.7,
                             cmap='jet')
            av.set_title('t = '+str(i)+' m/s')
            plt.colorbar(hfig)
            fig.tight_layout()
            flout2 = '../png/29_sim_flat_marm/born_'+str(i)+'.png'
            print("Export to file:", flout2)
            fig.savefig(flout2, bbox_inches='tight')
            plt.rcParams['font.size'] = 18
        print(np.shape(bg))
        print(np.shape(inp1))

    plot_sim_wf(inp1, born)

#%%

# ######## INPUT FOR SIMPLE MODEL

    # fl1 = './input/19_anomaly_4_layers/3_interfaces_org_smooth5.dat'
    # fl1 = '../input/19_anomaly_4_layers/3_interfaces_anomaly_250.dat'
    # fl1 = '../input/18_3_interface/3_interfaces_dp.dat'
    fl1 = '../input/vel_full.dat'
    
    # fl1 = '../output/68_thick_marm_ano/org_thick/adbetap.dat'
    inp1 = gt.readbin(fl1, nz, nx)  # model
    
    # fl2 = './output/12_wf_simple/sm10_2060/born/p2d_lsm_000004.dat'
    # fl3 = './output/12_wf_simple/sm10_2060/fwi/p2d_fwi_000004.dat'
    # fl5 = './output/12_wf_simple/sm15_4000/p0_fwi/p2d_fwi_000004.dat'

    # fl2 = '../output/18_3_interfaces/org/born/p2d_lsm_000004.dat'
    # fl3 = '../output/19_anomaly_4_layers/ano_190/fwi/p2d_fwi_000004.dat'
    # fl5 = '../output/19_anomaly_4_layers/p0/fwi/p2d_fwi_000004.dat'

    fl1   = '../output/68_thick_marm_ano/org_thick/p2d_fwi_000001.dat'
    fl2   = '../output/68_thick_marm_ano/ano_thick/p2d_fwi_000001.dat'
    nt    = 1801
    # nxl = 443
    nxl   = 291
    h_nxl = int((nxl-1)/2)
    
    org =  -gt.readbin(fl1, nz, nxl*nt) 
    ano  = -gt.readbin(fl2, nz, nxl*nt)   #
    # fwi = gt.readbin(fl3, nz, nxl*nt)    #
    # p0_fwi = gt.readbin(fl5, nz, nxl*nt)

    # nxl = bites/4/nz/nt = bites/4/151/1501
    # position central (301-1)*dx = 3600
    # 291 = 1+2*145
    # point à gauche = 3600-145*dx
    # point à droite = 3600+145*dx

    left_p  = 300-75 - h_nxl  # left point
    right_p = 300-75 + h_nxl  # right point
    # right_p = 471
    # left_p2  = 300 - h_nxl  # left point
    # right_p2 = 300 + h_nxl  # right point

    # print("size",np.shape(born))

    org = np.reshape(org, (nz, nt, nxl))
    ano = np.reshape(ano, (nz, nt, nxl))
    # fwi = np.reshape(fwi, (nz, nt, nxl))
    # p0_fwi = np.reshape(p0_fwi, (nz, nt, nxl))

    # # print("size",np.shape(born))

    # dm_fwi = p0_fwi-fwi  # Reflected FWI wavefield

    # diff = born-dm_fwi

    # diff = born+fwi
    # hmax = 10
    # hmin = -10
    # hmax = np.max(born)
    # hmin = np.min(born)
    # print(hmin,hmax)

    # TO PRODUCE SIMULATIONS OF THE FULL WAVEFIELD OVERLAYING THE MODEL
    #shot 221
    
    def plot_sim_wf(bg, inp1):
        hmax = np.max(inp1)
        print('hmax: ', hmax)
        hmax = 1
        hmin = -hmax
        for i in range(500, 1800, 100):
            fig = plt.figure(figsize=(13, 6), facecolor="white")
            av = plt.subplot(1, 1, 1)
            hfig = av.imshow(inp1[:, i, :]*100, extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                             vmin=hmin, vmax=hmax, aspect='auto', alpha=1,
                             cmap='jet')
            hfig1 = av.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                              aspect='auto', alpha=0.3,
                              cmap='gray')
           
            av.scatter(2.7,0.012,marker='*')
            av.set_title('t = '+str(i*dt*1000)+' s')
            plt.colorbar(hfig)
            fig.tight_layout()
            flout2 = '../png/68_thick_marm_ano/sim_fwi_'+str(i)+'.png'
            print("Export to file:", flout2)
            fig.savefig(flout2, bbox_inches='tight')
            plt.rcParams['font.size'] = 18
        print(np.shape(bg))
        print(np.shape(inp1))

    plot_sim_wf(inp1, org-ano)

    def plot_sim_2wf(bg, inp1, inp2):
        hmax = np.max(inp1)
        print('hmax: ', hmax)
        hmax = 0.01
        hmin = -hmax
        hmax2 = 0.01
        hmin2 = -hmax2
        for i in range(0, 1500, 100):
            fig = plt.figure(figsize=(15, 11), facecolor="white")
            av1 = plt.subplot(2, 1, 1)
            av1.title.set_text('FWI')
            hfig1 = av1.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                               aspect='auto', alpha=1,
                               cmap='Greys')
            hfig = av1.imshow(inp1[:, i, :], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                              vmin=hmin, vmax=hmax, aspect='auto', alpha=0.7,
                              cmap='jet')
            plt.colorbar(hfig)

            av2 = plt.subplot(2, 1, 2)
            av2.title.set_text('Born')
            hfig1 = av2.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                               aspect='auto', alpha=1,
                               cmap='Greys')
            hfig = av2.imshow(inp2[:, i, :], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                              vmin=hmin2, vmax=hmax2, aspect='auto', alpha=0.7,
                              cmap='jet')
            plt.colorbar(hfig)

            fig.tight_layout()

            # flout2 = './png/13_4_ano_smoo/sm10_born_fwi'+str(i)+'.png'
            # print("Export to file:",flout2)
            # fig.savefig(flout2, bbox_inches='tight')

    # plot_sim_2wf(inp1,dm_fwi,born)

    def plot_sim_3wf(bg, inp1, inp2, inp3):
        hmax = np.max(inp1)
        print('hmax: ', hmax)
        hmax = 0.01
        hmin = -hmax
        hmin2 = hmin
        hmax2 = hmax
        # hmax2 = 0.01
        # hmin2 = -hmax2
        for i in range(0, 1500, 100):
            fig = plt.figure(figsize=(12, 15), facecolor="white")
            av1 = plt.subplot(3, 1, 1)
            av1.title.set_text('FWI')
            hfig1 = av1.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                               aspect='auto', alpha=1,
                               cmap='Greys')
            hfig = av1.imshow(inp1[:, i, :], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                              vmin=hmin, vmax=hmax, aspect='auto', alpha=0.7,
                              cmap='jet')
            plt.colorbar(hfig)

            av2 = plt.subplot(3, 1, 2)
            av2.title.set_text('Born')
            hfig1 = av2.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                               aspect='auto', alpha=1,
                               cmap='Greys')
            hfig = av2.imshow(inp2[:, i, :], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                              vmin=hmin2, vmax=hmax2, aspect='auto', alpha=0.7,
                              cmap='jet')
            plt.colorbar(hfig)

            fig.tight_layout()

            av3 = plt.subplot(3, 1, 3)
            av3.title.set_text('Difference')
            hfig3 = av3.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                               aspect='auto', alpha=1,
                               cmap='Greys')
            hfig3 = av3.imshow(inp3[:, i, :], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                               vmin=hmin2, vmax=hmax2, aspect='auto', alpha=0.7,
                               cmap='jet')
            plt.colorbar(hfig)

            fig.tight_layout()
            flout2 = './png/12_wf_simple/sim/sm10_2060/sm_2060_'+str(i)+'.png'
            print("Export to file:", flout2)
            fig.savefig(flout2, bbox_inches='tight')
    # plot_sim_3wf(inp1,dm_fwi,born,diff)
