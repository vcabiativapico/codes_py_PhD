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



# path1 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2/badj/040_rt_badj_marm_sm_full.csv'
# path2 = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2/binv/040_rt_binv_marm_sm_full.csv'
spot_pos = 2946

# gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/042_slope_comp/05_pick_deep_flat_event/'

# gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/042_slope_comp/06_pick_deep_with_angle/'


gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'

# path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug_110424_raytracing_new_solver/output/demig.csv'


path_adj = gen_path + 'output/Debug_110424_raytracing_new_solver/depth_demig_out2/37_degrees/results/depth_demig_output.csv'
path_inv = gen_path + 'output/Debug_110424_raytracing_new_solver/depth_demig_out2/35_degrees/results/depth_demig_output.csv'



path_adj = gen_path + 'output/046_37_35_degrees_sm8/depth_demig_out/STD/results/depth_demig_output.csv'
path_inv = gen_path + 'output/046_37_35_degrees_sm8/depth_demig_out/QTV/results/depth_demig_output.csv'

path_adj = gen_path + '048_sm8_correction_new_solver/STD/depth_demig_out/STD/results/depth_demig_output.csv'
path_inv = gen_path + '048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/results/depth_demig_output.csv'


# path_adj = gen_path + 'badj/042_rt_badj_marm_slope_function.csv'
# path_inv = gen_path + 'binv/042_rt_binv_marm_slope_function.csv'



# pick1 = '../input/40_marm_ano/badj_mig_pick_smooth.csv'
# pick2 = '../input/40_marm_ano/binv_mig_pick_smooth.csv'

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
        self.tt_inv_ = read_results(path,17)
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

plot_positions(p_adj,dec=1)

diff_src_x = difference(p_inv.src_x_, p_adj.src_x_)
diff_rec_x = difference(p_inv.rec_x_, p_adj.rec_x_)


diff_ind_max = diff_src_x.argmax()
diff_max = np.max(diff_src_x) 


diff_ind_max = int(input('write an index: \n'))

print('SRC_X for the Standard migration : ', p_adj.src_x_[diff_ind_max])
print('REC_X for the Standard migration : ', p_adj.rec_x_[diff_ind_max])
print('SRC_X for the Quantitative migration : ', p_inv.src_x_[diff_ind_max])
print('REC_X for the Quantitative migration : ', p_inv.rec_x_[diff_ind_max])
print('Distance between src_x of STD vs QTV', diff_src_x[diff_ind_max])
print('Distance between rec_x of STD vs QTV', diff_rec_x[diff_ind_max])



 


print('difference is : ',diff_src_x[diff_ind_max])

gather_path_fwi_org = '../output/45_marm_ano_v3/org_1801TL'
# gather_path_fwi45 = '../output/45_marm_ano_v3/ano'
gather_path_fwi45 = '../output/45_marm_ano_v3/ano_114_perc_1801TL'


gthr_org1,tr = extract_trace(p_adj,diff_ind_max, gather_path_fwi_org)
gthr_org2,tr = extract_trace(p_inv,diff_ind_max, gather_path_fwi_org)


gather_ano1,tr_ano1 = extract_trace(p_adj, diff_ind_max, gather_path_fwi45)
gather_ano2,tr_ano2 = extract_trace(p_inv, diff_ind_max, gather_path_fwi45)

hmax = np.max(gthr_org1)/10
hmin = -hmax
plot_shot_gather(hmax,hmin, gthr_org1,tr,p_inv.at_,p_inv.fo_,p_inv.no_,p_inv.do_,title='original gather')
plot_shot_gather(hmax,hmin, gthr_org2,tr,p_inv.at_,p_inv.fo_,p_inv.no_,p_inv.do_,title='original gather')

plot_shot_gather(hmin, hmax, gather_ano1, tr_ano1,p_inv.at_,p_inv.fo_,p_inv.no_,p_inv.do_,title='STD monitor gather')
plot_shot_gather(hmin, hmax, gather_ano2, tr_ano2,p_inv.at_,p_inv.fo_,p_inv.no_,p_inv.do_,title='QTV monitor gather')

plot_shot_gather(hmin/20, hmax/20, gather_ano1-gthr_org1, tr_ano1,p_inv.at_,p_inv.fo_,p_inv.no_,p_inv.do_,title='STD monitor gather')
plot_shot_gather(hmin/20, hmax/20, gather_ano2-gthr_org2, tr_ano1,p_inv.at_,p_inv.fo_,p_inv.no_,p_inv.do_,title='QTV monitor gather')


#%%


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
    
    """Verification des interpolations"""
    added_int=np.zeros((nt,6))

    added_int[:,:-1] = inp3[2][:,nb_traces]
    added_int[:,-1] = interp_trace
    
    test = np.zeros((nt,3))
    for k, i in enumerate([2,5,3]):
        test[:,k] = added_int[:,i]
    plt.figure(figsize=(4,10))
    # wiggle(added_int)
    wiggle(test)
    plt.axhline(500)
    return interp_trace
    # return interp_trace, src_INT,tr_INT



# nr_src_x, idx_nr_src = find_nearest(p_inv.ax_, p_inv.src_x_[diff_ind_max]/1000)
# nr_off_x, idx_nr_off = find_nearest(p_inv.ao_, p_inv.off_x_[diff_ind_max]/1000)

# nb_gathers, nb_traces = create_nodes(diff_ind_max,idx_nr_off, idx_nr_src,p_inv.nx_,p_inv.no_)


# inp3 = read_shots_around(gather_path_fwi_org, nb_gathers, p_inv)


# fin_trace, src_INT,tr_INT = interpolate_src_rec(nb_traces,nb_gathers,p_inv.at_,p_inv.ao_,inp3,-p_inv.off_x_,p_inv.src_x_,p_inv.do_,p_inv.dx_,diff_ind_max)    

# added_int=np.zeros((p_inv.nt_,6))

# added_int[:,:-1] = src_INT
# added_int[:,-1] = fin_trace
# wiggle(added_int)
# test = np.zeros((p_inv.nt_,3))
# for k, i in enumerate([2,5,3]):
#     test[:,k] = added_int[:,i]
# plt.figure(figsize=(4,10))
# wiggle(test)
# plt.axhline(500)
    
# wiggle(src_INT)
# wiggle(tr_INT)
# plt.plot(tr_INT[0])
# plt.plot(fin_trace)

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





tr_binv_fwi_org = trace_from_rt(diff_ind_max,gather_path_fwi_org,p_inv)
tr_binv_fwi_45 = trace_from_rt(diff_ind_max,gather_path_fwi45,p_inv)

tr_badj_fwi_org = trace_from_rt(diff_ind_max,gather_path_fwi_org,p_adj)
tr_badj_fwi_45 = trace_from_rt(diff_ind_max,gather_path_fwi45,p_adj)




"""FWI 45_marm_ano_v3 """
visualisation.overlay(tr_badj_fwi_org,tr_badj_fwi_45,si=p_adj.dt_,
                      legend=["Base", "Monitor 4.5"],
                      clist=['k', 'r'],
                      fontsize= 26)
plt.title('Base vs monitor STD')
plt.ylim(2500,0)
plt.xlim(-.05,.05)

visualisation.overlay(tr_binv_fwi_org,tr_binv_fwi_45,si=p_inv.dt_,
                      legend=["Base", "Monitor 4.5"],
                      clist=['k', 'r'],
                      fontsize= 26)
plt.title('Base vs monitor QTV')
plt.ylim(2500,0)
plt.xlim(-.05,.05)

diff = tr_badj_fwi_org-tr_badj_fwi_45
plt.plot(diff)

max_diff = np.max(diff[1500:2000])
max_ano = np.max(tr_badj_fwi_45[1500:2000])
ratio = max_diff/max_ano


binv_SLD_TS_fwi = procs.sliding_TS(tr_binv_fwi_org,tr_binv_fwi_45,oplen= 500,si=p_inv.dt_, taper= 30)
badj_SLD_TS_fwi = procs.sliding_TS(tr_badj_fwi_org,tr_badj_fwi_45,oplen= 500,si=p_inv.dt_, taper= 30)


# hann_window = procs.hann(100,taper_ind=15,n=3)
# correlation = np.correlate(tr_binv_fwi_org[1000:1100]*hann_window,tr_binv_fwi_45[1000:1100]*hann_window,mode='same')
# time,value = procs.extremum_func(correlation,maxi=True,si=dt)
# plt.figure(figsize=(10, 8), facecolor="white")
# plt.plot(correlation)
# plt.title('Cross-correlation of the base and monitor for the quantitative trace')
# plt.xlabel('Time (ms)')
# plt.ylabel('Amplitude')


visualisation.overlay(binv_SLD_TS_fwi,badj_SLD_TS_fwi,si=p_inv.dt_,
                      legend=["QTV", "STD"],
                      fontsize = 14,figsize=(5,10))
# plt.axhline(p_inv.tt_inv_[diff_ind_max]*1000,c='k')
plt.title('Sliding time shift QTV vs STD \n Offset = '+str(p_inv.off_x_[diff_ind_max].astype(int))+' m')
plt.xlabel('miliseconds')
plt.xlim(-1,4)
plt.ylim(2500,0)


binv_NRMS = procs.sliding_NRMS(tr_binv_fwi_org,tr_binv_fwi_45,oplen=500,si=p_inv.dt_)
binv_coeff_corr = procs.sliding_corrcoeff(tr_binv_fwi_org, tr_binv_fwi_45,oplen=500,si=p_inv.dt_)

badj_NRMS = procs.sliding_NRMS(tr_badj_fwi_org,tr_badj_fwi_45,oplen=500,si=p_inv.dt_)
badj_coeff_corr = procs.sliding_corrcoeff(tr_badj_fwi_org, tr_badj_fwi_45,oplen=500,si=p_inv.dt_)


visualisation.overlay(binv_NRMS,badj_NRMS,si=p_inv.dt_,
                      legend=["QTV", "STD"],
                      fontsize = 14,figsize=(5,10))
plt.title('NRMS')


mean_NRMS_badj = np.mean(badj_NRMS[1000:2300])
mean_NRMS_binv = np.mean(binv_NRMS[1000:2300])

max_NRMS_badj = np.max(badj_NRMS[1000:2300])
max_NRMS_binv = np.max(binv_NRMS[1000:2300])

min_NRMS_badj = np.min(badj_NRMS[1000:2300])
min_NRMS_binv = np.min(binv_NRMS[1000:2300])

ext_NRMS_badj = extreme_value(max_NRMS_badj, min_NRMS_badj)
ext_NRMS_binv = extreme_value(max_NRMS_binv, min_NRMS_binv)




visualisation.overlay(binv_coeff_corr,badj_coeff_corr,si=p_inv.dt_,
                      legend=["QTV", "STD"],
                      fontsize = 14,figsize=(5,10))
plt.title('Correlation coefficient')

mean_corr_coef_badj = np.mean(badj_coeff_corr[1000:2300])
mean_corr_coef_binv = np.mean(binv_coeff_corr[1000:2300])


max_corr_coeff_badj = np.max(badj_coeff_corr[1000:2300])
max_corr_coeff_binv = np.max(binv_coeff_corr[1000:2300])

min_corr_coeff_badj = np.min(badj_coeff_corr[1000:2300])
min_corr_coeff_binv = np.min(binv_coeff_corr[1000:2300])

ext_corr_coef_badj = extreme_value(max_corr_coeff_badj, min_corr_coeff_badj)
ext_corr_coef_binv = extreme_value(max_corr_coeff_binv, min_corr_coeff_binv)


# '''Kevin advice'''

# binv_SLD_TS = procs.sliding_TS(tr_binv_org,tr_binv_ano55,oplen= 200,si=dt,taper= 30)


# badj_SLD_TS = procs.sliding_TS(tr_badj_org,tr_badj_ano55,oplen= 200,si=p_inv.dt_,taper= 30)


# visualisation.overlay(binv_SLD_TS,badj_SLD_TS,si=p_inv.dt_,
#                       legend=["QTV", "STD"],
#                       fontsize = 26)
# plt.title('Sliding time shift QTV vs STD \n Offset = '+str(p_inv.off_x_[diff_ind_max].astype(int))+' m')
# plt.xlim(-2,2) 
# plt.ylim(2050,0)
# """"""


# binv_SLD_TS = procs.sliding_TS(tr_binv_org,tr_binv_ano55,si=dt)
# binv_NRMS = procs.sliding_NRMS(tr_binv_org,tr_binv_ano55,si=dt)
# binv_coeff_corr = procs.sliding_corrcoeff(tr_binv_org, tr_binv_ano55)


# badj_SLD_TS = procs.sliding_TS(tr_badj_org,tr_badj_ano55,si=p_inv.dt_)
# badj_NRMS = procs.sliding_NRMS(tr_badj_org,tr_badj_ano55,si=p_inv.dt_)
# badj_coeff_corr = procs.sliding_corrcoeff(tr_badj_org, tr_badj_ano55)


# plt.figure(figsize=(10,8))
# plt.title('Mean NRMS vs Offset for Angle QTV and STD')
# plt.plot(offset_inv,mean_NRMS_inv,'.')
# plt.plot(offset_adj,mean_NRMS_adj,'.')
# plt.ylabel('Mean NRMS')
# plt.xlabel('Offset')
# plt.legend(['QTV','STD'])
# # plt.ylim(2.9,6.1)
'''Export data for a given offset'''


mean_badj_SLD_TS = np.mean(badj_SLD_TS_fwi[1000:2300])
mean_binv_SLD_TS = np.mean(binv_SLD_TS_fwi[1000:2300])

max_badj_SLD_TS = np.max(badj_SLD_TS_fwi[1000:2300])
max_binv_SLD_TS = np.max(binv_SLD_TS_fwi[1000:2300])

min_badj_SLD_TS = np.min(badj_SLD_TS_fwi[1000:2300])
min_binv_SLD_TS = np.min(binv_SLD_TS_fwi[1000:2300])



ext_badj_SLD_TS = extreme_value(max_badj_SLD_TS, min_badj_SLD_TS)
ext_binv_SLD_TS = extreme_value(max_binv_SLD_TS, min_binv_SLD_TS)


path_demig = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/046_37_35_degrees_sm8'
path_results = '/result_ts_off/047_sm8_37_35_degrees_w500.csv'
attr_list = [p_adj.rec_x_[diff_ind_max],
                        p_inv.rec_x_[diff_ind_max],
                        p_adj.src_x_[diff_ind_max],
                        p_inv.src_x_[diff_ind_max], 
                        p_adj.off_x_[diff_ind_max], 
                        max_badj_SLD_TS,
                        p_inv.off_x_[diff_ind_max],
                        max_binv_SLD_TS,
                        mean_binv_SLD_TS,
                        mean_badj_SLD_TS,
                        mean_corr_coef_badj,
                        mean_corr_coef_binv,
                        min_corr_coeff_badj,
                        min_corr_coeff_binv,
                        mean_NRMS_badj,
                        mean_NRMS_binv,
                        max_NRMS_badj,
                        max_NRMS_binv,
                        ext_badj_SLD_TS,
                        ext_binv_SLD_TS,
                        ext_corr_coef_badj,
                        ext_corr_coef_binv,
                        ext_NRMS_badj,
                        ext_NRMS_binv] 
header_list = ['rec_x_inv',
                'rec_x_adj',
                'src_x_inv',
                'src_x_adj',
                'offset_inv',
                'max_TS_adj',
                'offset_adj',
                'max_TS_inv',
                'mean_TS_inv',
                'mean_TS_adj',
                'mean_CC_adj',
                'mean_CC_inv',
                'min_CC_adj',
                'min_CC_inv',
                'mean_NRMS_adj',
                'mean_NRMS_inv',
                'max_NRMS_adj',
                'max_NRMS_inv',
                'ext_badj_SLD_TS',
                'ext_binv_SLD_TS',
                'ext_corr_coef_badj',
                'ext_corr_coef_binv',
                'ext_NRMS_badj',
                'ext_NRMS_binv']

def export_all_offsets(path_demig,path_results,attr_list,header):
    
    if os.path.exists(path_demig+path_results):
        print("File exists! Writing new lines")
        
        df = pd.read_csv(path_demig+path_results)
        results_in = df.values.tolist()
        results_in.append(attr_list)
        
        df2 = pd.DataFrame(results_in)
        df2.to_csv(path_demig+path_results, 
                    header=header_list, 
                    index=False)
    
    else:
        print("File does not exist. Creating file")
    
    
        results_in = np.array(attr_list).reshape(1,24)
        df = pd.DataFrame(results_in)
        df.to_csv(path_demig+path_results, 
                  header= header_list, 
                  index=False)
        
    

# export_all_offsets(path_demig,path_results,attr_list,header_list)



#%%



""" PLOT MIGRATED IMAGE AND RAYS """

## Read migrated image

# file = '../output/40_marm_ano/badj/inv_betap_x_s.dat'
file = '../output/45_marm_ano_v3/mig_badj_sm8_TL1801/inv_betap_x_s.dat'
file = '../output/45_marm_ano_v3/mig_binv_sm8_TL1801/inv_betap_x_s.dat'
# file = '../input/27_marm/marm2_sm15.dat'
# file = '../output/27_marm/flat_marm/dbetap_exact.dat'
# file = '../input/45_marm_ano_v3/fwi_ano_45.dat'

# file = '../input/45_marm_ano_v3/fwi_sm.dat'

Vit_model2 = gt.readbin(file,p_inv.nz_,p_inv.nx_).T*1000 

path_demig = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023'
# file_slope_adj = path_demig + '/input/042_slope_comp/slope_badj_5.csv'
# file_slope_inv = path_demig + '/input/042_slope_comp/slope_binv_5.csv'

# file_slope_adj = path_demig + '/input/042_slope_comp/degrees/slope_degree_7.csv'
# file_slope_inv = path_demig + '/input/042_slope_comp/degrees/slope_degree_8.csv'

# file_slope_adj = path_demig + '/output/Debug_110424_raytracing_new_solver/depth_demig_out/37_degrees/slope_binv_37.csv'
# file_slope_inv = path_demig + '/output/Debug_110424_raytracing_new_solver/depth_demig_out/35_degrees/slope_binv_35.csv'

file_slope_adj = path_demig + '/048_sm8_correction_new_solver/slope_STD_sm8.csv'
file_slope_inv = path_demig + '/048_sm8_correction_new_solver/slope_QTV_sm8.csv'



p_hz_adj = np.array([read_pick(file_slope_adj,0),read_pick(file_slope_adj,2)])
p_hz_inv = np.array([read_pick(file_slope_inv,0),read_pick(file_slope_inv,2)])

# file_pick_adj = '../input/40_marm_ano/badj_mig_pick_smooth.csv'
# file_pick_inv = '../input/40_marm_ano/binv_mig_pick_smooth.csv'

# pick_hz_adj = read_pick(file_pick_adj,0)   
# pick_hz_inv = read_pick(file_pick_inv,0)


plt.rcParams['font.size'] = 25

hmax = np.max(Vit_model2)
hmin = -hmax
fig = plt.figure(figsize=(10,7), facecolor = "white")
av  = plt.subplot(1,1,1)
# hfig = av.imshow(Vit_model2.T, vmin=hmin,vmax=hmax,extent=[ax[0], ax[-1], az[-1], az[0]],aspect='auto', cmap='seismic')
plt.plot(p_hz_adj[0]/1000, p_hz_adj[1]/1000, label='Slope QTV')
plt.plot(p_hz_inv[0]/1000, p_hz_inv[1]/1000, label='Slope STD')

plt.scatter(p_adj.spot_x_/1000,p_adj.spot_z_/1000)
plt.scatter(p_inv.spot_x_/1000,p_inv.spot_z_/1000)
# plt.scatter(input_val_inv['spot_x_input'],spot_z_adj, label='Spot STD')
# plt.scatter(input_val_inv['spot_x_input'],spot_z_inv, label='Spot QTV')
plt.title('Slopes and spots picked in the Standard and Quantitative images')
plt.legend()
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')

## Read rays from raytracing


def plot_rays(p,hz,diff_ind_max,typ):
    path_ray = [0]*102
    indices = []
    for i in range(diff_ind_max,101,101):
    # for i in range(92,101,5):
        if p.src_x_[i] > 0.: 
        # if 1==1:
            indices.append(i)
            
            
            # path_ray[i] = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug141223_raytracing/rays/opti_"+str(i)+"_ray.csv"
            # path_ray[i] = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2_"+str(spot_pos)+"/"+typ+"/rays/opti_"+str(i)+"_ray.csv"
            # path_ray[i] = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2_4000/out/opti_"+str(i)+"_ray.csv"
            # path_ray[i] = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/041_marm2_slope_'+typ+'_'+str(spot_pos)+'/rays/opti_'+str(i)+'_ray.csv'
            # path_ray[i] = path_demig+'/output/042_degrees_slope/' + typ +'/7/rays/opti_'+str(i)+'_ray.csv'
            # path_ray[i] = path_demig +'/output/046_37_35_degrees_sm8/depth_demig_out/'+str(typ)+'/rays/ray_'+str(i)+'.csv'
            path_ray[i] =     path_demig +'/output/046_37_35_degrees_sm8/depth_demig_out/QTV/rays/ray_'+str(i)+'.csv'
            ray_x = np.array(read_results(path_ray[i], 0))
            ray_z = np.array(read_results(path_ray[i], 2))
    
            x_disc = np.arange(p.nx_)*12.00
            z_disc = np.arange(p.nz_)*12.00
            
            fig1 = plt.figure(figsize=(18, 8), facecolor="white")
            av1 = plt.subplot(1, 1, 1)
            hmin = np.min(Vit_model2)
            hmax = -hmin
            # hmax = np.max(Vit_model2)
            
            hfig = av1.imshow(Vit_model2.T[40:120,230:315],vmin = hmin,
                        vmax = hmax,aspect = 1, 
                        extent=(x_disc[230],x_disc[315],z_disc[120],z_disc[40]),cmap='seismic')
            
            # hfig = av1.imshow(Vit_model2.T[:],vmin = hmin,
            #             vmax = hmax,aspect = 'auto', 
            #             extent=(x_disc[0],x_disc[-1],z_disc[-1],z_disc[0]))
            
            plt.plot(hz[0],-hz[1],'k',linewidth=4.0)
            
            
            # plt.colorbar(hfig,format='%1.2e')
            plt.colorbar(hfig)
            
            # av1.scatter(p.src_x_[i],24,c='y',marker= '*',s=200)
            # av1.scatter(p.rec_x_[i],24,marker= 'v',s=200)
            av1.plot(p.spot_x_,-p.spot_z_,'ok')
            # av1.scatter(ray_x,-ray_z, c="r", s=0.2)
            av1.set_title('Raytracing for '+typ+' in m/s \n ray number '+str(i)+'; offset = ' +str(int(p.off_x_[i])))
            av1.set_xlabel('Distance (m)')
            av1.set_ylabel('Depth (m)')
            av1.legend(['slope','spot','source','receiver'])
            
           # flout1 = "../png/27_marm/flat_marm/corr_az_pert/ray_sm_img_"+str(i)+"_f_p0007.png"
            # print("Export to file:", flout1)
            # fig1.savefig(flout1, bbox_inches='tight')
    
            
     
plot_rays(p_inv, p_hz_inv,diff_ind_max,'QTV')     

plot_rays(p_adj, p_hz_adj,diff_ind_max,'STD')



def plot_2rays(p1,p2,hz1,hz2,diff_ind_max,typ):
    path_ray_inv = [0]*102
    path_ray_adj = [0]*102
    indices = []
    for i in range(diff_ind_max,101,101):
    # for i in range(92,101,5):
        if p1.src_x_[i] > 0.: 
        # if 1==1:
            indices.append(i)
            
            
            # path_ray[i] = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug141223_raytracing/rays/opti_"+str(i)+"_ray.csv"
            # path_ray[i] = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2_"+str(spot_pos)+"/"+typ+"/rays/opti_"+str(i)+"_ray.csv"
            # path_ray[i] = "/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/040_marm2_4000/out/opti_"+str(i)+"_ray.csv"
            # path_ray[i] = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/041_marm2_slope_'+typ+'_'+str(spot_pos)+'/rays/opti_'+str(i)+'_ray.csv'
            # path_ray_inv[i] = path_demig+'/output/042_degrees_slope/binv/7/rays/opti_'+str(i)+'_ray.csv'
            
            path_ray_inv[i] = path_demig +'/output/046_37_35_degrees_sm8/depth_demig_out/QTV/rays/ray_'+str(i)+'.csv'
            
            # path_ray_inv[i] = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/042_slope_comp/06_pick_deep_with_angle/badj/rays/opti_'+str(i)+'_ray.csv'
            
            ray_x_inv = np.array(read_results(path_ray_inv[i], 0))
            ray_z_inv = np.array(read_results(path_ray_inv[i], 2))
            
            
            # path_ray_adj[i] = path_demig+'/output/042_degrees_slope/binv/9/rays/opti_'+str(i)+'_ray.csv'
            # path_ray_adj[i] = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug_050424_raytracing_solver/output/rays/ray_'+str(i)+'.csv'
            path_ray_adj[i] = path_demig +'/output/046_37_35_degrees_sm8/depth_demig_out/STD/rays/ray_'+str(i)+'.csv'
            
            
            ray_x_adj = np.array(read_results(path_ray_adj[i], 0))
            ray_z_adj = np.array(read_results(path_ray_adj[i], 2))
            
            
            x_disc = np.arange(p1.nx_)*12.00
            z_disc = np.arange(p1.nz_)*12.00
            
            fig1 = plt.figure(figsize=(18, 8), facecolor="white")
            av1 = plt.subplot(1, 1, 1)
            hmin = np.min(Vit_model2)
            # hmax = -hmin
            hmax = np.max(Vit_model2)
            
            
            hfig = av1.imshow(Vit_model2.T[:,150:380],vmin = hmin,
                        vmax = hmax,aspect = 1, 
                        extent=(x_disc[150],x_disc[380],z_disc[-1],z_disc[0]))
            
            plt.plot(hz1[0],-hz1[1],'k',linewidth=2.0)
            plt.plot(hz2[0],-hz2[1],'r',linewidth=2.0)
            
            # plt.colorbar(hfig,format='%1.2e')
            plt.colorbar(hfig)
            
            av1.scatter(p1.src_x_[i],12,c='r',marker= '*',s=200)
            av1.scatter(p1.rec_x_[i],12,c='r',marker= 'v',s=200)
            av1.plot(p1.spot_x_,-p1.spot_z_,'ok')
            av1.scatter(ray_x_inv,-ray_z_inv, c='k',s=0.1)
            av1.set_title('Raytracing for '+typ+' in m/s \n ray number '+str(i)+'; offset = ' +str(int(p1.off_x_[i]))+'m')
            
            av1.scatter(p2.src_x_[i],12,c='y',marker= '*',s=200)
            av1.scatter(p2.rec_x_[i],12,c='y',marker= 'v',s=200)
            # av1.plot(p2.spot_x_,-p2.spot_z_,'ok')
            av1.scatter(ray_x_adj,-ray_z_adj, c='r',s=0.1)
            av1.set_title('Raytracing for '+typ+' in m/s \n ray number '+str(i)+'; offset = ' +str(int(p2.off_x_[i]))+'m')
            
            av1.set_xlabel('Distance (m)')
            av1.set_ylabel('Depth (m)')
            av1.legend(['slope std','slope qtv','spot','source','receiver'])
            
           # flout1 = "../png/27_marm/flat_marm/corr_az_pert/ray_sm_img_"+str(i)+"_f_p0007.png"
            # print("Export to file:", flout1)
            # fig1.savefig(flout1, bbox_inches='tight')


# plot_2rays(p_adj,p_inv,p_hz_adj,p_hz_inv,diff_ind_max,'binv')
