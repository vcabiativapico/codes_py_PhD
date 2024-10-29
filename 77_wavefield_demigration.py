#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 10 15:23:53 2024

@author: vcabiativapico
"""


import csv
import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate import interpolate,InterpolatedUnivariateSpline
import matplotlib.patches as patches

if __name__ == "__main__":

    
    class Param_class:
        "Class for the parameters definition"
        def __init__(self,path):
            self.src_x_ = read_results(path,1)
            self.src_y_ = read_results(path,2)
            self.src_z_ = read_results(path,3)
            self.rec_x_ = read_results(path,4)
            self.rec_y_ = read_results(path,5)
            self.rec_z_ = read_results(path,6)
            self.spot_x_ = read_results(path,7)
            self.spot_y_ = read_results(path,8)
            self.spot_z_ = read_results(path,9)
            self.off_x_ = read_results(path,16)
            self.tt_ = read_results(path,17)
            self.nt_ = 1801
            self.dt_ = 1.14e-3
            self.ft_ = -100.32e-3
            self.nz_ = 151
            self.fz_ = 0.0
            self.dz_ = 12.0/1000.
            self.nx_ = 601
            self.fx_ = 0.0
            self.dx_ = 12.0/1000.
            self.no_ = 251
            self.do_ = self.dx_
            self.fo_ = -(self.no_-1)/2*self.do_
            self.ao_ = self.fo_ + np.arange(self.no_)*self.do_
            self.at_ = self.ft_ + np.arange(self.nt_)*self.dt_
            self.az_ = self.fz_ + np.arange(self.nz_)*self.dz_
            self.ax_ = self.fx_ + np.arange(self.nx_)*self.dx_
       
            
    


    
    # Read the results from demigration
    def read_results(path,srow):
        attr = []
        with open(path, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=',')
            header = next(spamreader)
            for row in spamreader:
                attr.append(float(row[srow]))
            # attr = [x for x in attr if str(x) != 'nan'] 
        attr = np.array(attr)
        attr = np.nan_to_num(attr)
        return attr
    
   
    def find_nearest(array, value):
        val = np.zeros_like(array)
        for i in range(len(array)):
            val[i] = np.abs(array[i] - value)
            idx = val.argmin()
        return array[idx], idx
    
    
    def extract_trace(p,idx_rt, path_shot):  
        ax = p.fx_ + np.arange(p.nx_)*p.dx_
        ao = p.fo_ + np.arange(p.no_)*p.do_
        title = str(find_nearest(ax, p.src_x_[idx_rt]/1000)[1])
        title = title.zfill(3)
        print('indice source', title)
        fl = path_shot+'/t1_obs_000'+str(title)+'.dat'
        if int(title) > int(p.nx_ - p.no_//2):
            noff = p.no_ // 2 + p.nx_ - int(title) +1
        else: 
            noff = p.no_
        inp = gt.readbin(fl, noff, p.nt_).transpose()
        tr = find_nearest(ao,p.off_x_[idx_rt]/1000)[1]
        return inp, tr
    
    def create_nodes(diff_ind_max,idx_nr_off, idx_nr_src,nx,no):
        """
        Find the indexes of the traces that will be used as nodes for the interpolation
        """
        if idx_nr_src < 2 :
            nb_gathers = np.array([0, 1, 2, 3, 4])
        elif idx_nr_src > nx-3:
            nb_gathers = np.array([597, 598, 599, 600, 601])
        else:
            nb_gathers = np.arange(idx_nr_src-2, idx_nr_src+3)
        
        
        if idx_nr_off < 2 :
            nb_traces = np.array([0, 1, 2, 3, 4])
        elif idx_nr_off > no-3:
            nb_traces = np.array([247, 248, 249, 250, 251])
        else:
            nb_traces = np.arange(idx_nr_off-2, idx_nr_off+3)
        
        return nb_gathers, nb_traces
    
    
    def read_shots_around(gather_path,nb_gathers,param):
        """
        Reads the shots around the closest shot from raytracing to the numerical born modelling grid
        """
        
        inp3 = np.zeros((len(nb_gathers),param.nt_, param.no_))
        
        for k, i in enumerate(nb_gathers):
            txt = str(i)
            title = txt.zfill(3)
            
            tr3 = gather_path+'/t1_obs_000'+str(title)+'.dat'
         
            inp3[k][:,:] = -gt.readbin(tr3, param.no_, param.nt_).transpose()
         
        return inp3
    
    
    def interpolate_src_rec(nb_traces,nb_gathers,at,ao,inp3,off_x,src_x,do,dx,diff_ind_max):
        """
        Performs interpolation between selected shots and traces
        @nb_traces : index of the reference traces to use as nodes for the interpolation
        @nb_gathers : index of the gathers to use as nodes for the interpolation
        @diff_ind_max : index of the AO result for source, receiver and spot from the raytracing file
        """
        nt = 1801
        tr_INT     = np.zeros((len(nb_gathers),nt,5)) 
        
        for k, i in enumerate(nb_gathers): 
        
            # Interpolation on the receivers
            for j in range(len(nb_gathers)):
               
                f = interpolate.RegularGridInterpolator((at,ao[nb_traces]), inp3[j][:,nb_traces], method='linear',bounds_error=False, fill_value=None) 
                at_new = at
                ao_new = np.linspace(off_x[diff_ind_max]/1000-do*2,off_x[diff_ind_max]/1000+do*2, 5)
                AT, AO = np.meshgrid(at_new, ao_new, indexing='ij')
                tr_INT[j][:,:] = f((AT,AO))
                rec_int = tr_INT[:,:,2]
                  
            # Interpolation on the shots
            f = interpolate.RegularGridInterpolator((at,nb_gathers*12), rec_int.T, method='linear',bounds_error=False, fill_value=None) 
            at_new = at
            src_new = np.linspace(src_x[diff_ind_max] - dx*2000, src_x[diff_ind_max] + dx*2000, 5)
            AT, SRC = np.meshgrid(at_new, src_new, indexing='ij')
            src_INT = f((AT,SRC))
            interp_trace = src_INT[:,2] 
        # print(ao_new)
        # print(src_new)
        
        # print(ao[nb_traces])
        
        # """Verification des interpolations"""
        # added_int=np.zeros((nt,6))
    
        # added_int[:,:-1] = inp3[2][:,nb_traces]
        # added_int[:,-1] = interp_trace
        
        # test = np.zeros((nt,3))
        # for k, i in enumerate([2,5,3]):
        #     test[:,k] = added_int[:,i]
        # plt.figure(figsize=(4,10))
        # # wiggle(added_int)
        # wiggle(test)
        # plt.axhline(500)
        return interp_trace
        # return interp_trace, src_INT,tr_INT
    
    def trace_from_rt(diff_ind_max,gather_path,p):
        '''
        Finds the nearest trace to the source and offset given by the index
        Interpolates the traces so that the calculation is exact
        Exports the trace found by raytracing from the modelling data
        '''
        nr_src_x, idx_nr_src = find_nearest(p.ax_, p.src_x_[diff_ind_max]/1000)
        nr_off_x, idx_nr_off = find_nearest(p.ao_, p.off_x_[diff_ind_max]/1000)
        print(idx_nr_src)
        print(idx_nr_off)
        nb_gathers, nb_traces = create_nodes(diff_ind_max,idx_nr_off, idx_nr_src,p.nx_,p.no_)
        
        
        inp3 = read_shots_around(gather_path, nb_gathers, p)
       
        # fin_trace = interpolate_src_rec(nb_traces,nb_gathers,at,ao,inp3,-p.off_x_,p.src_x_,do,dx,diff_ind_max) 
        
        fin_trace = interpolate_src_rec(nb_traces,nb_gathers,p.at_,p.ao_,inp3,p.off_x_,p.src_x_,p.do_,p.dx_,diff_ind_max)    
        return fin_trace
    
    def plot_sim_wf(bg, inp1,ray_x,ray_z,idx):
        if idx == 221:
            plt.rcParams['font.size'] = 18
            hmax = np.max(inp1)
            print('hmax: ', hmax)
            hmax = 1
            hmin = -hmax
            for i in range(100, 1800, 100):
                fig = plt.figure(figsize=(13, 6), facecolor="white")
                av = plt.subplot(1, 1, 1)
                # hfig = av.imshow(inp1[:, i, :-10]*100, extent=[ax[left_p], ax[right_p-10], az[-1], az[0]],
                #                   vmin=hmin, vmax=hmax, aspect='auto', alpha=1,
                #                   cmap='jet')
                # hfig1 = av.imshow(bg[:, left_p-10:right_p-10], extent=[ax[left_p-10], ax[right_p-10], az[-1], az[0]],
                #                   aspect='auto', alpha=0.3,
                #                   cmap='gray')
                hfig = av.imshow(inp1[:, i, :]*100, extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                                  vmin=hmin, vmax=hmax, aspect='auto', alpha=1,
                                  cmap='jet')
                hfig1 = av.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                                  aspect='auto', alpha=0.3,
                                  cmap='gray')
                av.plot(ray_x,ray_z)
                # arrow1 = patches.Arrow(3.6, 0.2, -0.15, 0.18, width=0.1,color='black')
                # arrow2 = patches.Arrow(4.0, 0.7, 0.15, 0.15, width=0.1,color='white')
                # av.add_patch(arrow1)
                # av.add_patch(arrow2)
                # av.scatter(2.640,0.012,marker='*')
                av.set_title('t = '+str(i*dt*1000)+' s')
                plt.xlabel('Distance (km)')
                plt.ylabel('Depth (km)')
                plt.colorbar(hfig)
                fig.tight_layout()
                flout2 = '../png/68_thick_marm_ano/sim_fwi_diff_'+str(i)+'_shot_'+str(idx)+'.png'
                print("Export to file:", flout2)
                fig.savefig(flout2, bbox_inches='tight')
            
        elif idx==231:
            plt.rcParams['font.size'] = 22
            hmax = np.max(inp1)
            print('hmax: ', hmax)
            hmax = 1
            hmin = -hmax
            for i in range(100, 1800, 100):
                fig = plt.figure(figsize=(13, 6), facecolor="white")
                av = plt.subplot(1, 1, 1)
                hfig = av.imshow(inp1[:, i, :]*100, extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                                  vmin=hmin, vmax=hmax, aspect='auto', alpha=1,
                                  cmap='jet')
                hfig1 = av.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                                  aspect='auto', alpha=0.3,
                                  cmap='gray')
                plt.xlabel('Distance (km)')
                plt.ylabel('Depth (km)')
                arrow1 = patches.Arrow(3.6, 0.2, -0.15, 0.18, width=0.1,color='black')
                arrow2 = patches.Arrow(4.45, 0.6, -0.15, 0.2, width=0.1,color='white')
                av.add_patch(arrow1)
                av.add_patch(arrow2)
                av.plot(ray_x,ray_z)
                # av.scatter(2.640,0.012,marker='*')
                av.set_title('t = '+str(i*dt*1000)+' s')
                plt.colorbar(hfig)
                fig.tight_layout()
                flout2 = '../png/68_thick_marm_ano/sim_fwi_'+str(i)+'_shot_'+str(idx)+'.png'
                print("Export to file:", flout2)
                fig.savefig(flout2, bbox_inches='tight')
        else:
            plt.rcParams['font.size'] = 22
            hmax = np.max(inp1)
            print('hmax: ', hmax)
            hmax = 1
            hmin = -hmax
            for i in range(100, 1800, 100):
                fig = plt.figure(figsize=(13, 6), facecolor="white")
                av = plt.subplot(1, 1, 1)
                hfig = av.imshow(inp1[:, i, :]*500, extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                                  vmin=hmin, vmax=hmax, aspect='auto', alpha=1,
                                  cmap='jet')
                hfig1 = av.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                                  aspect='auto', alpha=0.3,
                                  cmap='gray')
                plt.xlabel('Distance (km)')
                plt.ylabel('Depth (km)')
                # arrow1 = patches.Arrow(3.6, 0.2, -0.15, 0.18, width=0.1,color='black')
                # arrow2 = patches.Arrow(4.45, 0.6, -0.15, 0.2, width=0.1,color='white')
                # av.add_patch(arrow1)
                # av.add_patch(arrow2)
                av.plot(ray_x,ray_z)
                # av.scatter(2.640,0.012,marker='*')
                av.set_title('t = '+str(i*dt*1000)+' s')
                plt.colorbar(hfig)
                fig.tight_layout()
                flout2 = '../png/72_thick_marm_ano_born_mig_flat/sim_fwi_'+str(i)+'_shot_'+str(idx)+'.png'
                print("Export to file:", flout2)
                fig.savefig(flout2, bbox_inches='tight')    
            
                
                
            print(np.shape(bg))
            print(np.shape(inp1))
        
#%%            
    # Global parameters
    labelsize = 16
    nt = 1801
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
    
    gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'
    
    path_inv = gen_path + '068_TS_marm_ano_thick/depth_demig_out/068_thick_marm_org_sm6_2024-09-03_15-26-45/results/depth_demig_output.csv'
    path_adj = gen_path + '068_TS_marm_ano_thick/depth_demig_out/068_thick_marm_org_sm6_badj_2024-09-05_12-09-21/results/depth_demig_output.csv'
    
    # path_inv = gen_path + '072_thick_ano_compare_FW_flat/depth_demig_out/deeper2_8_2024-10-14_11-37-56/results/depth_demig_output.csv'
    
    
    gather_path_fwi_org = '../output/68_thick_marm_ano/org_thick'
    gather_path_fwi_ano = '../output/68_thick_marm_ano/ano_thick'

    
    p_adj = Param_class(path_adj)
    p_inv = Param_class(path_inv)
    
    
    fl3 = '../input/vel_full.dat'
    inp1 = gt.readbin(fl3, nz, nx)
    
    # shot_mod_idx = 402
    shot_mod_idx = 221
    fl1   = '../output/68_thick_marm_ano/org_thick/p2d_fwi_000'+str(shot_mod_idx)+'.dat'
    fl2   = '../output/68_thick_marm_ano/ano_thick/p2d_fwi_000'+str(shot_mod_idx)+'.dat'
    
    # fl1 = '../output/72_thick_marm_ano_born_mig_flat/org_full/sim/p2d_fwi_000'+str(shot_mod_idx)+'.dat'
    # fl2 = '../output/72_thick_marm_ano_born_mig_flat/ano_full/sim/p2d_fwi_000'+str(shot_mod_idx)+'.dat'
      

    
    fl_org = gather_path_fwi_org+'/t1_obs_000'+str(shot_mod_idx)+'.dat'
    fl_ano = gather_path_fwi_ano+'/t1_obs_000'+str(shot_mod_idx)+'.dat'
    
    nt    = 1801
  
    nxl   = 291
    h_nxl = int((nxl-1)/2)
    
    org  =  -gt.readbin(fl1, nz, nxl*nt) 
    ano  = -gt.readbin(fl2, nz, nxl*nt)   #
   

    # nxl = bites/4/nz/nt = bites/4/151/1501
    # position central (301-1)*dx = 3600
    # 291 = 1+2*145
    # point à gauche = 3600-145*dx
    # point à droite = 3600+145*dx
    value = 300 - shot_mod_idx
    
    # value = 300-221
    
    left_p  = 300 - value - h_nxl  # left point
    right_p = 300 - value + h_nxl  # right point
 
    org = np.reshape(org, (nz, nt, nxl))
    ano = np.reshape(ano, (nz, nt, nxl))


    # TO PRODUCE SIMULATIONS OF THE FULL WAVEFIELD OVERLAYING THE MODEL
    #shot 221
    
    # path_ray_org = gen_path + '072_thick_ano_compare_FW_flat/depth_demig_out/deeper2_8_2024-10-14_11-37-56/rays/ray_90.csv'
    
    
    path_ray_org = gen_path + '068_TS_marm_ano_thick/depth_demig_out/068_thick_marm_org_sm6_2024-09-03_15-26-45/rays/ray_0.csv'
    path_ray_ano = gen_path + '068_TS_marm_ano_thick/depth_demig_out/068_thick_marm_ano_sm6_2024-09-03_15-26-51/rays/ray0.csv'
    
    
    
    ray_x  = np.array(read_results(path_ray_org, 0))
    ray_z  = np.array(read_results(path_ray_org, 2))
    ray_tt = np.array(read_results(path_ray_org, 8))
    
    # trace_from_rt(0, gather_path_fwi_org, p_inv)
    # trace_from_rt(0, gather_path_fwi_ano, p_inv)

    
    # fl_org = gather_path_fwi_org+'/t1_obs_000'+str(shot_mod_idx)+'.dat'
    # fl_ano = gather_path_fwi_ano+'/t1_obs_000'+str(shot_mod_idx)+'.dat'
    
    # inp_org = -gt.readbin(fl_org, p_inv.no_, p_inv.nt_).transpose()
    # inp_ano = -gt.readbin(fl_ano, p_inv.no_, p_inv.nt_).transpose()
    
    
    def plot_shot_gathers(hmin, hmax, inp,idx):
        plt.rcParams['font.size'] = 16
        fig = plt.figure(figsize=(10, 8), facecolor="white")
        av = plt.subplot(1, 1, 1)
        hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                         vmin=hmin, vmax=hmax, aspect='auto',
                         cmap='seismic')
        av.plot(inp[:,125],at)
        plt.colorbar(hfig, format='%1.3f')
        plt.title('shot_x = '+str(idx*12))
        plt.xlabel('Offset (km)')
        plt.ylabel('Time (s)')
        
        # fig.tight_layout()
        # print("Export to file:", flout)
        # fig.savefig(flout, bbox_inches='tight')

    # hmin = np.min(inp_org-inp_ano)
    # hmax = -hmin
    # plot_shot_gathers(hmin, hmax, inp_org-inp_ano,shot_mod_idx)
    
    plot_sim_wf(inp1, org-ano,ray_x/1000,-ray_z/1000,shot_mod_idx)
