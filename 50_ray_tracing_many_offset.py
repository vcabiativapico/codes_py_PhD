#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 14:51:08 2024

@author: vcabiativapico
"""



import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate import interpolate
import csv
from wiggle.wiggle import wiggle
from spotfunk.res import procs,visualisation
import pandas as pd
import os
import tqdm
if __name__ == "__main__":
  
  
# Building simple vel and rho models to test modeling
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
# ## Add y dimension    
    fy = -500 
    ny = 21
    dy = 50
    ay = fy + np.arange(ny)*dy


def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr

## Read the results from demigration
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

def norm(inp):
    """
    Normalize the data according to its maximum
    Normalization is no longer necessary due to the correction of amplitude on the formula
    """
    norm_inp = inp/np.max(abs(inp))
#     return inp
    return norm_inp

spot_pos = 2946

gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'



path_adj = gen_path + '048_sm8_correction_new_solver/STD/depth_demig_out/STD/results/depth_demig_output.csv'
path_inv = gen_path + '048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/results/depth_demig_output.csv'

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
        self.dt_ = 1.41e-3
        self.ft_ = -100.11e-3
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
   
        
p_adj = Param_class(path_adj)
p_inv = Param_class(path_inv)



    

def plot_positions(p,dec=1):
    """ PLOT THE POSITIONS AND RECEIVERS OBTAINED """
    colors = p.src_x_[::dec]
    fig = plt.figure(figsize= (10,7))
    plt.scatter(p.src_x_[::dec],p.src_y_[::dec],c=colors,marker='*',cmap='jet')
    plt.scatter(p.rec_x_[::dec],p.rec_y_[::dec],c=colors,marker='v',cmap='jet')
    plt.xlabel('x')
    plt.ylabel('y')
    # plt.ylim(-0.1,0.1)



def difference(attr1,attr2):
    diff_attr = np.zeros_like(attr1)
    for i in range(len(attr1)):
        if attr1[i] != 0 and attr2[i] != 0 :   
            diff_attr[i] = attr1[i] - attr2[i]
    return diff_attr

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



def plot_shot_gather(hmin, hmax, inp,tr,at,fo,no,do,title=''):
    plt.rcParams['font.size'] = 16
    
    fig = plt.figure(figsize=(10, 8), facecolor="white")
    av = plt.subplot(1, 1, 1) 
    ao = fo + np.arange(no)*do 
    hfig = av.imshow(inp, extent=[ao[0], ao[-1], at[-1], at[0]],
                     vmin=hmin, vmax=hmax, aspect='auto',
                     cmap='seismic')
    av.plot(inp[:,tr]+ao[tr],at,'k')
    plt.colorbar(hfig, format='%2.2f')
    plt.title(title)
    plt.xlabel('Offset (km)')
    plt.ylabel('Time (s)')
    fig.tight_layout()


def plot_trace(tr1, tr2, legend = ['STD', 'QNT']):
    # axi = np.zeros(2)
    fig, (axi) = plt.subplots(nrows=1, ncols=1,
                              sharey=True,
                              figsize=(4, 12),
                              facecolor="white")
    tr = [tr1,tr2]
    
    axi.plot(tr[0], at, 'r')
    axi.plot(tr[1], at, 'b')
    

    axi.set_ylabel('Time (s)')
    axi.legend(legend,loc='upper left', shadow=True)
    plt.gca().invert_yaxis()
  
    return tr1, tr2


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
    
    nb_gathers, nb_traces = create_nodes(diff_ind_max,idx_nr_off, idx_nr_src,p.nx_,p.no_)
    
    
    inp3 = read_shots_around(gather_path, nb_gathers, p)
   
    # fin_trace = interpolate_src_rec(nb_traces,nb_gathers,at,ao,inp3,-p.off_x_,p.src_x_,do,dx,diff_ind_max) 
    
    fin_trace = interpolate_src_rec(nb_traces,nb_gathers,p.at_,p.ao_,inp3,p.off_x_,p.src_x_,p.do_,p.dx_,diff_ind_max)    
    return fin_trace






def extreme_value(max_val, min_val):
    extremes = np.max(abs(np.array([max_val,min_val])))
    arg = np.argmax(abs(np.array([max_val, min_val])))
    if arg == 0:
        ext_value = extremes * np.sign(max_val)
    else :
        ext_value = extremes * np.sign(min_val)
    return ext_value


#%%    

def att_mean(tt,at,attr):    
    '''Calculates the average starting from the traveltime given by raytracing'''
    min_tt = find_nearest(at, tt)[1]
    mean = np.mean(attr[min_tt+355:min_tt+355*2])     
    print(min_tt)
    return mean


binv_SLD_TS_fwi_total = []
badj_SLD_TS_fwi_total = []
binv_CC_total = []
badj_CC_total = []
binv_NRMS_total = []
badj_NRMS_total = []

out_idx = []
mean_TS_binv = []
mean_TS_badj = []
mean_NRMS_binv = []
mean_NRMS_badj = []
mean_CC_binv = []
mean_CC_badj = []


gather_path_fwi_org = '../output/45_marm_ano_v3/org_1801TL'
# gather_path_fwi45 = '../output/45_marm_ano_v3/ano'
gather_path_fwi45 = '../output/45_marm_ano_v3/ano_114_perc_1801TL'
for i in tqdm.tqdm(range(0,101,1)):
    if i > 99: break
    tr_binv_fwi_org = trace_from_rt(i,gather_path_fwi_org,p_inv)
    tr_binv_fwi_45 = trace_from_rt(i,gather_path_fwi45,p_inv)
    
    tr_badj_fwi_org = trace_from_rt(i,gather_path_fwi_org,p_adj)
    tr_badj_fwi_45 = trace_from_rt(i,gather_path_fwi45,p_adj)
    
    binv_SLD_TS_fwi = procs.sliding_TS(tr_binv_fwi_org,tr_binv_fwi_45,oplen= 500,si=p_inv.dt_, taper= 30)
    badj_SLD_TS_fwi = procs.sliding_TS(tr_badj_fwi_org,tr_badj_fwi_45,oplen= 500,si=p_adj.dt_, taper= 30)
    
    binv_CC = procs.sliding_corrcoeff(tr_binv_fwi_org, tr_binv_fwi_45,oplen=500,si=p_inv.dt_)
    badj_CC = procs.sliding_corrcoeff(tr_badj_fwi_org, tr_badj_fwi_45,oplen=500,si=p_inv.dt_)
    
    binv_NRMS = procs.sliding_NRMS(tr_binv_fwi_org,tr_binv_fwi_45,oplen=500,si=p_inv.dt_)    
    badj_NRMS = procs.sliding_NRMS(tr_badj_fwi_org,tr_badj_fwi_45,oplen=500,si=p_inv.dt_)
    
    
    binv_SLD_TS_fwi_total.append(binv_SLD_TS_fwi.T)
    badj_SLD_TS_fwi_total.append(badj_SLD_TS_fwi.T)
    
    binv_CC_total.append(binv_CC)
    badj_CC_total.append(badj_CC)
    
    binv_NRMS_total.append(binv_NRMS)
    badj_NRMS_total.append(badj_NRMS)
    
    out_idx.append(i)
    
    
    mean_TS_binv.append(att_mean(p_inv.tt_[i],at,binv_SLD_TS_fwi))
    mean_TS_badj.append(att_mean(p_adj.tt_[i],at,badj_SLD_TS_fwi))

    mean_NRMS_binv.append(att_mean(p_inv.tt_[i],at,binv_NRMS))
    mean_NRMS_badj.append(att_mean(p_adj.tt_[i],at,badj_NRMS))

    mean_CC_binv.append(att_mean(p_inv.tt_[i],at,binv_CC))
    mean_CC_badj.append(att_mean(p_adj.tt_[i],at,badj_CC))



ratio_mean_TS = np.array(mean_TS_binv)/ np.array(mean_TS_badj)
ratio_mean_CC =  np.array(mean_CC_binv)/ np.array(mean_CC_badj)
ratio_mean_NRMS = np.array(mean_NRMS_binv)/np.array(mean_NRMS_badj)


plt.figure(figsize=(10,8))
plt.title('Mean Timeshift vs Offset for QTV and STD')
plt.plot(p_inv.off_x_,mean_TS_binv,'.')
plt.plot(p_adj.off_x_,mean_TS_badj,'.')
plt.ylabel('Mean Timeshift (ms)')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])

plt.figure(figsize=(10,8))
plt.title('Mean CC vs Offset for QTV and STD')
plt.plot(p_inv.off_x_,mean_CC_binv,'.')
plt.plot(p_adj.off_x_,mean_CC_badj,'.')
plt.ylabel('Mean CC (ms)')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])

plt.figure(figsize=(10,8))
plt.title('Mean NRMS vs Offset for QTV and STD')
plt.plot(p_inv.off_x_,mean_NRMS_binv,'.')
plt.plot(p_adj.off_x_,mean_NRMS_badj,'.')
plt.ylabel('Mean NRMS (ms)')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])

plt.figure(figsize=(10,8))
plt.title('Ratio mean Timeshift vs Offset for QTV and STD')
plt.plot(p_inv.off_x_,ratio_mean_TS,'.')
plt.ylabel('Mean Timeshift (ms)')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])
#%%
'''Plot Time-shifts panel'''

def plot_panel_att(attr_binv,attr_badj,p,out_idx,xmin,xmax,title):
    ncols = int(np.shape(attr_binv)[0]/8)
    axi = np.zeros(ncols)
    fig, (axi) = plt.subplots(nrows=1, ncols=ncols,
                              sharey=True,
                              figsize=(27, 8),
                              facecolor="white")
    
    axi[0].set_ylabel('Time (s)')
    axi[0].set_xlabel('TS (ms)')
    
    for i,k in enumerate(range(0,100,8)):
            if i>= ncols: break
            axi[i].plot(attr_binv[k],at,c='tab:blue')
            axi[i].plot(attr_badj[k],at,c='tab:orange')
            axi[i].set_title(str(int(p.off_x_[k])))
            axi[i].set_xlim(xmin,xmax)
            axi[i].set_ylim(at[-1],at[0])
            
            axi[i].axhline(p.tt_[out_idx[k]],c='k')
            axi[i].axhline(p.tt_[out_idx[k]]+0.500,c='g', ls='--')
            axi[i].axhline(p.tt_[out_idx[k]]+1.000,c='g',ls='--')
    axi[i-1].legend(['QTV','STD']) 
    fig.suptitle(title)
      
plot_panel_att(binv_SLD_TS_fwi_total,badj_SLD_TS_fwi_total,p_inv,out_idx,-0.5,4,'TS')
plot_panel_att(binv_CC_total,badj_CC_total,p_inv,out_idx,0.9,1.01,'CC')
plot_panel_att(binv_NRMS_total,badj_NRMS_total,p_inv,out_idx,-0.1,0.5,'NRMS')






# off_id = 2 

# plt.figure(figsize=(6, 10), facecolor="white")
# plt.plot(binv_SLD_TS_fwi_total[off_id],at,c='tab:orange')
# plt.plot(badj_SLD_TS_fwi_total[off_id],at,c='tab:blue')
# plt.axhline(p_inv.tt_[out_idx[off_id]],c='tab:orange',ls='--')
# plt.axhline(p_adj.tt_[out_idx[off_id]],c='tab:blue',ls='--')
# plt.xlim(-0.5,4)
# plt.ylim(at[-1],at[0])


# tr_binv_fwi_org_off = trace_from_rt(off_id,gather_path_fwi_org,p_inv)
# tr_binv_fwi_45_off = trace_from_rt(off_id,gather_path_fwi45,p_inv)

# visualisation.overlay(tr_binv_fwi_org_off,tr_binv_fwi_45_off,si=p_inv.dt_,
#                   legend=["QTV", "STD"],
#                   fontsize = 14,figsize=(5,10)) 
# plt.axhline(p_inv.tt_[off_id]*1000,c='k')
# plt.xlim(-0.05,0.05)


