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
## Add y dimension    
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


gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/042_degrees_slope/'

# path_adj = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug_050424_raytracing_solver/output/demig.csv'


path_adj = gen_path + 'binv/7/042_rt_binv_marm_slope_function_deg.csv'
path_inv = gen_path + 'binv/8/042_rt_binv_marm_slope_function_deg.csv'


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
        self.nt_ = 1501
        self.dt_ = 1.41e-3
        self.ft_ = -100.11e-3
        self.nz_ = 151
        self.fz_ = 0.0
        self.dz_ = 12.0/1000.
        self.nx_ = 601
        self.fx_ = 0.0
        self.dx_ = 12.0/1000.
        self.no_ = 251
        self.do_ = dx
        self.fo_ = -(no-1)/2*do
        self.ao_ = fo + np.arange(no)*do
        self.at_ = ft + np.arange(nt)*dt
        self.az_ = fz + np.arange(nz)*dz
        self.ax_ = fx + np.arange(nx)*dx
   
        
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
    ax = fx + np.arange(nx)*dx
    ao = fo + np.arange(no)*do
    title = str(find_nearest(ax, p.src_x_[idx_rt]/1000)[1])
    title = title.zfill(3)
    print('indice source', title)
    fl = path_shot+'t1_obs_000'+str(title)+'.dat'
    if int(title) > int(p.nx_ - p.no_//2):
        noff = p.no_ // 2 + p.nx_ - int(title) +1
    else: 
        noff = p.no_
    inp = gt.readbin(fl, noff, nt).transpose()
    tr = find_nearest(ao,-p.off_x_[idx_rt]/1000)[1]
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



 
# plt.ylim(-0.1,np.max(diff_src_x)+0.1)

print('difference is : ',diff_src_x[diff_ind_max])


base_path_org = '../output/45_marm_ano_v3/org/'
base_path_ano = '../output/45_marm_ano_v3/ano/'


# base_path_org = '../output/40_marm_ano/binv/'
# base_path_ano = '../output/40_marm_ano/binv_ano/'


gthr_org1,tr = extract_trace(p_adj,diff_ind_max, base_path_org)
gthr_org2,tr = extract_trace(p_inv,diff_ind_max, base_path_org)


gather_ano1,tr_ano1 = extract_trace(p_adj, diff_ind_max, base_path_ano)
gather_ano2,tr_ano2 = extract_trace(p_inv, diff_ind_max, base_path_ano)

hmax =np.max(gthr_org1)/10
hmin = -hmax
plot_shot_gather(hmax,hmin, gthr_org1,tr,at,fo,no,do,title='original gather')
plot_shot_gather(hmax,hmin, gthr_org2,tr,at,fo,no,do,title='original gather')

plot_shot_gather(hmin, hmax, gather_ano1, tr_ano1,at,fo,no,do,title='STD monitor gather')
plot_shot_gather(hmin, hmax, gather_ano2, tr_ano2,at,fo,no,do,title='QTV monitor gather')

plot_shot_gather(hmin/20, hmax/20, gather_ano1-gthr_org1, tr_ano1,at,fo,no,do,title='STD monitor gather')
plot_shot_gather(hmin/20, hmax/20, gather_ano2-gthr_org2, tr_ano1,at,fo,no,do,title='QTV monitor gather')


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
    # added_int=np.zeros((nt,6))

    # added_int[:,:-1] = inp3[2][:,nb_traces]
    # added_int[:,-1] = interp_trace

    # test = np.zeros((nt,3))
    # for k, i in enumerate([2,5,3]):
    #     test[:,k] = added_int[:,i]
    # plt.figure(figsize=(4,10))
    # wiggle(test)
    # plt.axhline(500)
    
    return interp_trace


def trace_from_rt(diff_ind_max,gather_path,p):
    '''
    Finds the nearest trace to the source and offset given by the index
    Interpolates the traces so that the calculation is exact
    Exports the trace found by raytracing from the modelling data
    '''
    nr_src_x, idx_nr_src = find_nearest(p.ax_, p.src_x_[diff_ind_max]/1000)
    nr_off_x, idx_nr_off = find_nearest(p.ao_, -p.off_x_[diff_ind_max]/1000)
    
    nb_gathers, nb_traces = create_nodes(diff_ind_max,idx_nr_off, idx_nr_src,p.nx_,p.no_)
    
    
    inp3 = read_shots_around(gather_path, nb_gathers, p)
   
    
    fin_trace = interpolate_src_rec(nb_traces,nb_gathers,p.at_,p.ao_,inp3,-p.off_x_,p.src_x_,p.do_,p.dx_,diff_ind_max)    
    return fin_trace


gather_path_inv_org = '../output/40_marm_ano/binv'
gather_path_ano = '../output/40_marm_ano/binv_ano'
gather_path_ano42 = '../output/40_marm_ano/binv_ano_42'
gather_path_ano425 = '../output/40_marm_ano/binv_ano_425'
gather_path_ano45 = '../output/40_marm_ano/binv_ano_45'
gather_path_ano50 = '../output/40_marm_ano/binv_ano_50'
gather_path_ano55 = '../output/40_marm_ano/binv_ano_55'

# gather_path_ano55 = '../output/43_deep_flat_ano'

gather_path_fwi_org = '../output/45_marm_ano_v3/org'
gather_path_fwi45 = '../output/45_marm_ano_v3/ano'


tr_binv_org = trace_from_rt( diff_ind_max,gather_path_inv_org,p_inv)
tr_binv_ano = trace_from_rt( diff_ind_max,gather_path_ano,p_inv)
tr_binv_ano42 = trace_from_rt( diff_ind_max,gather_path_ano42,p_inv)
tr_binv_ano425 = trace_from_rt(diff_ind_max,gather_path_ano425,p_inv)
tr_binv_ano45 = trace_from_rt(diff_ind_max,gather_path_ano45,p_inv)
tr_binv_ano50 = trace_from_rt(diff_ind_max,gather_path_ano50,p_inv)
tr_binv_ano55 = trace_from_rt(diff_ind_max,gather_path_ano55,p_inv)

tr_badj_org = trace_from_rt( diff_ind_max,gather_path_inv_org,p_adj)
tr_badj_ano = trace_from_rt( diff_ind_max,gather_path_ano,p_adj)
tr_badj_ano42 = trace_from_rt( diff_ind_max,gather_path_ano42,p_adj)
tr_badj_ano425 = trace_from_rt(diff_ind_max,gather_path_ano425,p_adj)
tr_badj_ano45 = trace_from_rt(diff_ind_max,gather_path_ano45,p_adj)
tr_badj_ano50 = trace_from_rt(diff_ind_max,gather_path_ano50,p_adj)
tr_badj_ano55 = trace_from_rt(diff_ind_max,gather_path_ano55,p_adj)



tr_binv_fwi_org = trace_from_rt(diff_ind_max,gather_path_fwi_org,p_inv)
tr_binv_fwi_45 = trace_from_rt(diff_ind_max,gather_path_fwi45,p_inv)

tr_badj_fwi_org = trace_from_rt(diff_ind_max,gather_path_fwi_org,p_adj)
tr_badj_fwi_45 = trace_from_rt(diff_ind_max,gather_path_fwi45,p_adj)



def plt_compare_monitor(tr_org,tr_monitor,title=''):
    fig, (axi) = plt.subplots(nrows=1, ncols=2,
                              sharey=True,
                              figsize=(8, 15),
                              facecolor="white")
    tr = [tr_org, tr_monitor]
    
    axi[0].plot(tr[0], at, 'r')
    axi[0].plot(tr[1], at, '--b')
    # axi.plot(tr[2], at, 'k')
    axi[1].plot(tr[1]-tr[0], at, 'k')
    
    axi[0].set_ylabel('Time (s)')
    axi[0].legend(['Base',5.5],loc='upper left', shadow=True)
    axi[0].set_title('Comparison '+title)
    axi[1].set_title('Detection '+title)
    
    plt.gca().invert_yaxis()


# plt_compare_monitor(tr_badj_org, tr_badj_ano55,'STD')
# plt_compare_monitor(tr_binv_org,tr_binv_ano55,'QTTV')


"""FWI 45_marm_ano_v3 """
visualisation.overlay(tr_binv_fwi_org,tr_binv_fwi_45,si=p_inv.dt_,
                      legend=["Base", "Monitor 5.5"],
                      clist=['k', 'r'],
                      fontsize= 26)
plt.title('Base vs monitor QTV')
plt.ylim(2050,0)
plt.xlim(-0.1,0.1)

visualisation.overlay(tr_badj_fwi_org,tr_badj_fwi_45,si=dt,
                      legend=["Base", "Monitor 5.5"],
                      clist=['k', 'r'],
                      fontsize = 26)
plt.title('Base vs monitor STD')
plt.ylim(2050,0)
plt.xlim(-0.1,0.1)


binv_SLD_TS_fwi = procs.sliding_TS(tr_binv_fwi_org,tr_binv_fwi_45,oplen= 200,si=dt,taper= 25)
badj_SLD_TS_fwi = procs.sliding_TS(tr_badj_fwi_org,tr_badj_fwi_45,oplen= 200,si=p_inv.dt_,taper= 25)


visualisation.overlay(binv_SLD_TS_fwi,badj_SLD_TS_fwi,si=p_inv.dt_,
                      legend=["QTV", "STD"],
                      fontsize = 26)
plt.title('Sliding time shift QTV vs STD \n Offset = '+str(p_inv.off_x_[diff_ind_max].astype(int))+' m')
# plt.xlim(-2,2)
plt.ylim(2050,0)




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


'''Export data for a given offset'''

max_badj_SLD_TS = np.max(abs(binv_SLD_TS_fwi[500:]))
max_binv_SLD_TS = np.max(abs(binv_SLD_TS_fwi[500:]))

path_demig = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/input/042_slope_comp'



# results_in = np.zeros(3)
# results_in = np.array([p_adj.rec_x_[diff_ind_max],p_inv.rec_x_[diff_ind_max],p_adj.src_x_[diff_ind_max],p_inv.src_x_[diff_ind_max], p_adj.off_x_[diff_ind_max], max_badj_SLD_TS,p_inv.off_x_[diff_ind_max],max_binv_SLD_TS]).reshape(1,8)
# df = pd.DataFrame(results_in)
# df.to_csv(path_demig+'/result_ts_off/results_both_angle_7_9_2.csv', header=['rec_x_7','rec_x_9','src_x_7','src_x_9','offset_7','max_7','offset_9','max_9'], index=False)


# df = pd.read_csv(path_demig+'/result_ts_off/results_both_angle_7_9_2.csv')
# results_in = df.values.tolist()
# results_in.append([p_adj.rec_x_[diff_ind_max],p_inv.rec_x_[diff_ind_max],p_adj.src_x_[diff_ind_max],p_inv.src_x_[diff_ind_max], p_adj.off_x_[diff_ind_max], max_badj_SLD_TS,p_inv.off_x_[diff_ind_max],max_binv_SLD_TS])

# df2 = pd.DataFrame(results_in)
# df2.to_csv(path_demig+'/result_ts_off/results_both_angle_7_9_2.csv', header=['rec_x_7','rec_x_9','src_x_7','src_x_9','offset_7','max_7','offset_9','max_9'], index=False)



'''Sliding time-shift for born modelling, replaced by fwi'''

# print(p_adj.off_x_[diff_ind_max],max_badj_SLD_TS)
# print(p_inv.off_x_[diff_ind_max],max_binv_SLD_TS)

# visualisation.overlay(tr_binv_org,tr_binv_ano55,si=p_inv.dt_,
#                       legend=["Base", "Monitor 5.5"],
#                       clist=['k', 'r'],
#                       fontsize= 26)
# plt.title('Base vs monitor QTV')
# plt.ylim(2050,0)
# plt.xlim(-0.05,0.05)

# visualisation.overlay(tr_badj_org,tr_badj_ano55,si=dt,
#                       legend=["Base", "Monitor 5.5"],
#                       clist=['k', 'r'],
#                       fontsize = 26)
# plt.title('Base vs monitor STD')
# plt.ylim(2050,0)
# plt.xlim(-0.05,0.05)

# visualisation.overlay(binv_SLD_TS,badj_SLD_TS,si=p_inv.dt_,
#                       legend=["QTV", "STD"],
#                       fontsize = 26)
# plt.title('Sliding time shift QTV vs STD \n Offset = '+str(p_inv.off_x_[diff_ind_max].astype(int))+' m')
# plt.xlim(-4,2)
# plt.ylim(2050,0)

# visualisation.overlay(binv_NRMS,badj_NRMS,si=p_inv.dt_,
#                       legend=["QTV", "STD"],
#                       fontsize = 26)
# plt.title('NRMS QTV vs STD \n Offset = '+str(p_inv.off_x_[diff_ind_max].astype(int))+' m')
# plt.xlim(-0.1,1.5)
# plt.ylim(2050,0)

# visualisation.overlay(binv_coeff_corr,badj_coeff_corr,si=p_inv.dt_,
#                       legend=["QTV", "STD"],
#                       fontsize = 26)
# plt.title('Coefficient de correlation QTV vs STD \n Offset = '+str(p_inv.off_x_[diff_ind_max].astype(int))+' m')
# plt.xlim(0.85,1)
# plt.ylim(2050,0)






# ''' Verification de l'interpolation '''
# added_int=np.zeros((nt,6))

# added_int[:,:-1] = inp3[2][:,nb_traces]
# added_int[:,-1] = finterp_trace

# test = np.zeros((nt,3))
# for k, i in enumerate([2,5,3]):
#     test[:,k] = added_int[:,i]
# plt.figure(figsize=(4,10))
# wiggle(test)
# plt.axhline(500)
# # plt.ylim(600,200)


# hmax =np.max(inp3[2])
# hmin = -hmax        
# fig = plt.figure(figsize=(10, 8), facecolor="white")
# av = plt.subplot(1, 1, 1) 
# ao = fo + np.arange(no)*do 
# hfig = av.imshow(inp3[2], extent=[ao[0], ao[-1], at[-1], at[0]],
#               vmin=hmin, vmax=hmax, aspect='auto',
#               cmap='seismic')
# plt.colorbar(hfig)
# plt.plot(finterp_trace+ao[idx_nr_off],at,'k')




#%%



""" PLOT MIGRATED IMAGE AND RAYS """

## Read migrated image

file = '../output/40_marm_ano/binv/inv_betap_x_s.dat'
file = '../input/27_marm/marm2_sm15.dat'
# file = '../output/27_marm/flat_marm/dbetap_exact.dat'
# file = '../input/vel_full.dat'

Vit_model2 = gt.readbin(file,nz,nx).T*1000

path_demig = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023'
# file_slope_adj = path_demig + '/input/042_slope_comp/slope_badj_5.csv'
# file_slope_inv = path_demig + '/input/042_slope_comp/slope_binv_5.csv'

file_slope_adj = path_demig + '/input/042_slope_comp/degrees/slope_degree_7.csv'
file_slope_inv = path_demig + '/input/042_slope_comp/degrees/slope_degree_8.csv'

# file_slope_inv = path_demig + '/output/042_slope_comp/slope_bspline/slope_binv.csv'



p_hz_adj = np.array([read_pick(file_slope_adj,0),read_pick(file_slope_adj,2)])
p_hz_inv = np.array([read_pick(file_slope_inv,0),read_pick(file_slope_inv,2)])

# file_pick_adj = '../input/40_marm_ano/badj_mig_pick_smooth.csv'
# file_pick_inv = '../input/40_marm_ano/binv_mig_pick_smooth.csv'

# pick_hz_adj = read_pick(file_pick_adj,0)   
# pick_hz_inv = read_pick(file_pick_inv,0)

plt.figure(figsize=(10,8))
plt.rcParams['font.size'] = 16
plt.plot(p_hz_adj[0], p_hz_adj[1], label='Slope 7°')
plt.plot(p_hz_inv[0], p_hz_inv[1], label='Slope 9°')
plt.scatter(p_inv.spot_x_,p_inv.spot_z_)
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
            path_ray[i] = path_demig+'/output/042_degrees_slope/' + typ +'/7/rays/opti_'+str(i)+'_ray.csv'
            
            ray_x = np.array(read_results(path_ray[i], 0))
            ray_z = np.array(read_results(path_ray[i], 2))
    
            x_disc = np.arange(p.nx_)*12.00
            z_disc = np.arange(p.nz_)*12.00
            
            fig1 = plt.figure(figsize=(18, 8), facecolor="white")
            av1 = plt.subplot(1, 1, 1)
            hmin = np.min(Vit_model2)
            # hmax = -hmin
            hmax = np.max(Vit_model2)
            
            
            hfig = av1.imshow(Vit_model2.T[:],vmin = hmin,
                        vmax = hmax,aspect = 'auto', 
                        extent=(x_disc[0],x_disc[-1],z_disc[-1],z_disc[0]),cmap='jet')
            
            plt.plot(hz[0],-hz[1],'k',linewidth=2.0)
            
            
            # plt.colorbar(hfig,format='%1.2e')
            plt.colorbar(hfig)
            
            av1.scatter(p.src_x_[i],24,c='y',marker= '*',s=200)
            av1.scatter(p.rec_x_[i],24,marker= 'v',s=200)
            av1.plot(p.spot_x_,-p.spot_z_,'ok')
            av1.scatter(ray_x,-ray_z, c="r", s=0.2)
            av1.set_title('Raytracing for '+typ+' in m/s \n ray number '+str(i)+'; offset = ' +str(int(p.off_x_[i])))
            av1.set_xlabel('Distance (m)')
            av1.set_ylabel('Depth (m)')
            av1.legend(['slope','spot','source','receiver'])
            
           # flout1 = "../png/27_marm/flat_marm/corr_az_pert/ray_sm_img_"+str(i)+"_f_p0007.png"
            # print("Export to file:", flout1)
            # fig1.savefig(flout1, bbox_inches='tight')
    
            
     
plot_rays(p_inv, p_hz_inv,diff_ind_max,'binv')     

plot_rays(p_adj, p_hz_adj,diff_ind_max,'binv')

# plot_rays(p_adj, p_hz_adj,diff_ind_max,'badj')




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
            
            path_ray_inv[i] = path_demig+'/output/042_degrees_slope/binv/7/rays/opti_'+str(i)+'_ray.csv'
            
            # path_ray_inv[i] = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/042_slope_comp/06_pick_deep_with_angle/badj/rays/opti_'+str(i)+'_ray.csv'
            
            ray_x_inv = np.array(read_results(path_ray_inv[i], 0))
            ray_z_inv = np.array(read_results(path_ray_inv[i], 2))
            
            
            path_ray_adj[i] = path_demig+'/output/042_degrees_slope/binv/9/rays/opti_'+str(i)+'_ray.csv'
            # path_ray_adj[i] = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug_050424_raytracing_solver/output/rays/ray_'+str(i)+'.csv'
            
            ray_x_adj = np.array(read_results(path_ray_adj[i], 0))
            ray_z_adj = np.array(read_results(path_ray_adj[i], 2))
            
            
            x_disc = np.arange(p1.nx_)*12.00
            z_disc = np.arange(p1.nz_)*12.00
            
            fig1 = plt.figure(figsize=(18, 8), facecolor="white")
            av1 = plt.subplot(1, 1, 1)
            hmin = np.min(Vit_model2)
            # hmax = -hmin
            hmax = np.max(Vit_model2)
            
            
            hfig = av1.imshow(Vit_model2.T[:,250:360],vmin = hmin,
                        vmax = hmax,aspect = 1, 
                        extent=(x_disc[250],x_disc[360],z_disc[-1],z_disc[0]),cmap='jet')
            
            plt.plot(hz1[0],-hz1[1],'k',linewidth=2.0)
            plt.plot(hz2[0],-hz2[1],'r',linewidth=2.0)
            
            # plt.colorbar(hfig,format='%1.2e')
            plt.colorbar(hfig)
            
            av1.scatter(p1.src_x_[i],24,c='y',marker= '*',s=200)
            av1.scatter(p1.rec_x_[i],24,c='y',marker= 'v',s=200)
            av1.plot(p1.spot_x_,-p1.spot_z_,'ok')
            av1.scatter(ray_x_inv,-ray_z_inv, c='k',s=0.1)
            av1.set_title('Raytracing for '+typ+' in m/s \n ray number '+str(i)+'; offset = ' +str(int(p1.off_x_[i])))
            
            av1.scatter(p2.src_x_[i],24,c='r',marker= '*',s=200)
            av1.scatter(p2.rec_x_[i],24,c='r',marker= 'v',s=200)
            # av1.plot(p2.spot_x_,-p2.spot_z_,'ok')
            av1.scatter(ray_x_adj,-ray_z_adj, c='r',s=0.1)
            av1.set_title('Raytracing for '+typ+' in m/s \n ray number '+str(i)+'; offset = ' +str(int(p2.off_x_[i])))
            
            av1.set_xlabel('Distance (m)')
            av1.set_ylabel('Depth (m)')
            av1.legend(['slope 1','slope 2','spot','source','receiver'])
            
           # flout1 = "../png/27_marm/flat_marm/corr_az_pert/ray_sm_img_"+str(i)+"_f_p0007.png"
            # print("Export to file:", flout1)
            # fig1.savefig(flout1, bbox_inches='tight')


plot_2rays(p_adj,p_inv,p_hz_adj,p_hz_inv,diff_ind_max,'binv')
