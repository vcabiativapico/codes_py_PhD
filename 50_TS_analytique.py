#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:16:12 2024

@author: vcabiativapico
"""



import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate import interpolate
import csv
# from wiggle.wiggle import wiggle
from spotfunk.res import procs,visualisation
import pandas as pd
import os
from shapely.geometry import Point, Polygon
import sympy as sp
import sympy.plotting as syp
import json
if __name__ == "__main__":
  
  
# Building simple vel and rho models to test modeling
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
        self.dt_ = 1.14e-3
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



def plot_model(inp,hmin,hmax):
    plt.rcParams['font.size'] = 20
    fig = plt.figure(figsize=(16,8), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect='auto'
                     )
    plt.colorbar(hfig)
    plt.xlabel('Distance (Km)')
    plt.ylabel('Profondeur (Km)')
    fig.tight_layout()
    return fig

def pick_interface_model(inp,ray_x,ray_z,nx,nz):
 # Create a plot only with integers by using the extension of the grid ax az
     
     ax = np.arange(nx)
     az = np.arange(nz)
     vbg = 1
 #### Plot initial model 
  
     %matplotlib qt5
     hmax = np.max(inp)
     hmin = -hmax
     fig = plt.figure(figsize=(25,12), facecolor = "white")
     av  = plt.subplot(1,1,1)
     print('az',az.shape)
     hfig = av.imshow(inp,vmin=hmin,vmax=hmax,extent=[ax[0], ax[-1], az[-1], az[0]],aspect='auto', cmap='seismic')
     
     plt.colorbar(hfig)
     av.scatter(ray_x/12,-ray_z/12, c='k',s=0.1)

     fig.tight_layout()

     
 # Extraction of indexes and arrangement to insert them
     pick = plt.ginput(n=-1,timeout=15)
     plt.close()
     
     pick = np.asarray(pick).astype(int)
     pickt = pick.transpose()
     pick_f = [pickt[1],pickt[0]] 
     pick_f = tuple(pick_f)   
     
     
     return  pick_f
 
def calculate_thickness(pick,plot=False):
    m = pick[1][0] - pick[1][1] 
    n = pick[0][0] - pick[0][1] 
    
    l = np.sqrt(m**2 + n**2)
    l = l*12

    return l
 

def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr

def read_index(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(int(row[srow]))
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
#%%
name = 0

'''Read the velocity model '''


if name == 0:
    '''FULL model'''
    fl1 = '../input/45_marm_ano_v3/fwi_ano_114_percent.dat'
    fl2 = '../input/45_marm_ano_v3/fwi_org.dat'
    
    gather_path_fwi_org = '../output/45_marm_ano_v3/org_1801TL'
    gather_path_fwi45 = '../output/45_marm_ano_v3/ano_114_perc_1801TL'

    picked_start_depth = 1018 # selected from the polygon plot 
    csv_out_name = '../time_shift_theorique_full.csv'
elif name==1:
    '''sm5 model'''
    gather_path_fwi_org = '../output/63_ts_sm5_fwi/org'
    gather_path_fwi45 = '../output/63_ts_sm5_fwi/ano'

    fl1 = '../input/50_ts_model/sm5_ano.dat'
    fl2 = '../input/50_ts_model/sm5_org.dat'
    
    picked_start_depth = 995 # selected from the polygon plot 
    csv_out_name = '../time_shift_theorique_sm5.csv'




inp1 = gt.readbin(fl1,nz,nx)
inp2 = gt.readbin(fl2,nz,nx)

'''Read the raypath'''
# path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/046_37_35_degrees_sm8/depth_demig_out/QTV/rays/ray_0.csv'
# path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/rays/ray_0.csv' 


# path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/063_TS_analytique/depth_demig_out/063_org_2024-08-19_12-11-20/rays/ray_0.csv'
# path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/063_TS_analytique/depth_demig_out/063_org_sm5_2024-08-20_14-33-21/rays/ray_0.csv'
path_ray_org = gen_path + '063_TS_analytique/depth_demig_out/063_org_sm5_2024-08-20_14-33-21/rays/ray_0.csv'
path_ray_ano = gen_path + '063_TS_analytique/depth_demig_out/063_ano_sm5_2024-08-20_14-33-26/rays/ray_0.csv'


ray_x = np.array(read_results(path_ray_org, 0))
ray_z = np.array(read_results(path_ray_org, 2))
tt = np.array(read_results(path_ray_org, 8))

ray_vp_org = np.array(read_results(path_ray_org, 6))
ray_vp_ano = np.array(read_results(path_ray_ano, 6))

# '''Read index of the anomaly'''
# file_index = '../input/45_marm_ano_v3/fwi_ano_114_percent.csv'
# inp_index = [np.array(read_index(file_index,0)), np.array(read_index(file_index,1))]


def find_index(inp1,inp2):
    diff = inp1-inp2

    plot_model(diff, np.max(diff), np.min(diff))

    idx_diff = np.where(diff>0.07)

    inp_diff = np.zeros_like(inp1)
    inp_diff[idx_diff] = 2
    plot_model(inp_diff, np.max(inp_diff), np.min(inp_diff))
    return inp_diff, idx_diff


inp_diff, idx_diff = find_index(inp1,inp2)




'''Create a polygon'''

def create_polygon(idx):
    sympy_idx = np.transpose(idx).tolist()
    
    sympy_idx_tuple = []
    for x in sympy_idx:
        sympy_idx_tuple.append(tuple(x))
    
    
    
    polygon = sp.Polygon(*sympy_idx_tuple)
    
    # Extract the coordinates
    x, y = zip(*polygon.vertices)
    
    # Close the polygon by repeating the first vertex at the end
    x = list(x) + [x[0]]
    y = list(y) + [y[0]]
    return x,y


x,y = create_polygon(idx_diff)


# Plot the polygon
plt.figure(figsize=(12, 12))
plt.scatter(np.array(y)*0.012, np.array(x)*0.012)
plt.scatter(ray_x/1000,-ray_z/1000, c='k',s=1.5)
plt.axhline(1.018)
plt.title("Polygon Plot")
plt.xlabel("x")
plt.ylabel("z")
plt.grid(True)
plt.ylim(1.16,0.95)
plt.xlim(3.2,3.5)


# dist = np.zeros((len(ray_x),len(x)),type(float))
# for i in range(len(ray_x)):
#     dist[i]=((ray_z[i]/1000 - np.array(x)*0.012)**2 + (ray_x[i]/1000 - np.array(y)*0.012)**2)
# dist_eu = np.sqrt(dist.astype(float))
# idx_closest_point = np.argmin(dist_eu)
# closest_point = y[idx_closest_point]*0.012


# picked_start_depth = 1018 # selected from the polygon plot 
# # picked_start_depth = 995 # selected from the polygon plot 


def attr_in_poly(ray_z,picked_start_depth): 
    idx_in_poly = []
    for i,k in enumerate(ray_z):
        if k < -picked_start_depth:
            idx_in_poly.append(i)
    ray_x_in_poly = ray_x[idx_in_poly]
    ray_z_in_poly = ray_z[idx_in_poly]
    tt_in_poly = tt[idx_in_poly]
    return tt_in_poly, idx_in_poly,ray_x_in_poly,ray_z_in_poly


tt_in_poly, idx_in_poly, ray_x_in_poly,ray_z_in_poly= attr_in_poly(ray_z,picked_start_depth)

arg_deep_point_in_poly = np.argmin(tt_in_poly)


# Plot the polygon
plt.figure(figsize=(12, 12))
plt.scatter(np.array(y)*0.012, np.array(x)*0.012)
plt.scatter(ray_x_in_poly/1000,-ray_z_in_poly/1000, c='k',s=1.5)
plt.scatter(p_inv.spot_x_[0]/1000,-p_inv.spot_z_[0]/1000)
plt.axhline(1.018)
plt.title("Polygon Plot")
plt.xlabel("x")
plt.ylabel("z")
plt.grid(True)
plt.ylim(1.16,0.95)
plt.xlim(3.2,3.5)


#%%


path_adj = gen_path + '048_sm8_correction_new_solver/STD/depth_demig_out/STD/results/depth_demig_output.csv'
path_inv = gen_path + '048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/results/depth_demig_output.csv'


path_inv = gen_path + '063_TS_analytique/depth_demig_out/063_org_sm5_2024-08-20_14-33-21/results/depth_demig_output.csv'
path_adj  = gen_path + '063_TS_analytique/depth_demig_out/063_std_org_sm5_2024-08-20_14-38-36/results/depth_demig_output.csv'


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

def modify_velocity_profile(ray_z,arg_deep_point,vp,tt,half_idx_in_poly):
    arg_deep_point = np.argmin(ray_z)
    vp_to_spot = vp[:arg_deep_point]
    tt_to_spot = tt[:arg_deep_point]
    
    
    vp_cut_ano = np.copy(vp_to_spot)
    
    vp_cut_ano[half_idx_in_poly] = vp_to_spot[half_idx_in_poly]*1.14

    plt.plot(vp_to_spot,tt_to_spot,'.')
    plt.plot(vp_cut_ano,tt_to_spot,'.')
    plt.gca().invert_yaxis()
    return vp_cut_ano, vp_to_spot

arg_deep_point = np.argmin(ray_z)
half_idx_in_poly = idx_in_poly[:len(idx_in_poly)//2-1]
vp_cut_ano, vp_cut_org = modify_velocity_profile(ray_z,arg_deep_point,ray_vp_org,tt,half_idx_in_poly)

# vel1 = procs.RMS_calculator(vp_cut_org)
# vel2 = procs.RMS_calculator(vp_cut_ano)

# vel1_ano = vp_cut_org[half_idx_in_poly]
# vel2_ano = vp_cut_org[half_idx_in_poly]*1.14

# plt.figure(figsize=(8, 8), facecolor="white")
# plt.rcParams['font.size'] = 20
# plt.title('Velocity (m/s)')
# plt.plot(vp_cut_org,ray_z[np.arange(len(vp_cut_org))])
# plt.plot(vel1_ano,ray_z[half_idx_in_poly],label='Original velocity')
# plt.plot(vel2_ano,ray_z[half_idx_in_poly],label='Modified velocity')
# plt.legend()
# plt.ylabel('Depth (m)')
# plt.ylabel('Velocity (m/s)')
# plt.ylim(-1080,-900)
# plt.xlim(2200,2900)
# pick_dist = pick_interface_model(inp_blank,ray_x,ray_z,nx,nz)
# df = pd.DataFrame(pick_dist)
# df.to_csv('../input/45_marm_ano_v3/pick_ray_into_anomaly.csv',header=False,index=False)

# %matplotlib inline


# distance = calculate_thickness(pick_dist)




tt_in_poly, idx_in_poly, ray_x_in_poly,ray_z_in_poly= attr_in_poly(ray_z,picked_start_depth)

ray_x_in_poly = ray_x[idx_in_poly]
ray_z_in_poly = ray_z[idx_in_poly]
ray_x_in_ano = ray_x[half_idx_in_poly]
ray_z_in_ano = ray_z[half_idx_in_poly]





vel_in_poly_org = ray_vp_org[half_idx_in_poly]
vel_in_poly_ano = ray_vp_ano[half_idx_in_poly]

thickness_ano =  np.sqrt((ray_z_in_ano[0]-ray_z_in_ano)**2 + (ray_x_in_ano[0]-ray_x_in_ano)**2)

# t0_org = 2 * (thickness_ano) / vel1_ano
# t0_ano = 2 * (thickness_ano) / vel2_ano

t0_org = 2 * (thickness_ano) / vel_in_poly_org*1.14
t0_ano = 2 * (thickness_ano) / vel_in_poly_org


out_idx = 0

ts0 = (t0_org - t0_ano) * 1000
t_to_add = tt_in_poly[:len(tt_in_poly)//2-1]*2


tt0_inv = (t_to_add - tt_in_poly[0]*2) + p_inv.tt_[out_idx] 

tt0_inv = t_to_add 
# tt0_adj = t_to_add + (p_adj.tt_[out_idx] - tt_in_poly[0]*2)


ts0 = np.insert(ts0,0,0)
tt0_inv = np.insert(tt0_inv,0,0)
# tt0_adj = np.insert(tt0_adj,0,0)


ts0 = np.append(ts0, ts0[-1])
tt0_inv = np.append(tt0_inv, at[-1])
# tt0_adj = np.append(tt0_adj, at[-1])


# Plot the polygon

hmin = np.min(inp1)
hmax = np.max(inp1)
plt.figure(figsize=(12, 12))
plot_model(inp1,hmin,hmax)
plt.scatter(np.array(y)*0.012, np.array(x)*0.012)
plt.plot(ray_x_in_ano/1000,-ray_z_in_ano/1000)
# plt.gca().invert_yaxis()


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



tr_binv_fwi_org = trace_from_rt(0,gather_path_fwi_org,p_inv)
tr_binv_fwi_45 = trace_from_rt(0,gather_path_fwi45,p_inv)

tr_badj_fwi_org = trace_from_rt(0,gather_path_fwi_org,p_adj)
tr_badj_fwi_45 = trace_from_rt(0,gather_path_fwi45,p_adj)

oplen = 250
binv_SLD_TS_fwi = procs.sliding_TS(tr_binv_fwi_org,tr_binv_fwi_45,oplen= oplen,si=p_inv.dt_, taper= 30)
badj_SLD_TS_fwi = procs.sliding_TS(tr_badj_fwi_org,tr_badj_fwi_45,oplen= oplen,si=p_adj.dt_, taper= 30)


out_idx = [0]
plt.figure(figsize=(6, 10), facecolor="white")
plt.title('Time-shift')
plt.plot(binv_SLD_TS_fwi,at-ft,c='tab:orange',label='TS from traces')
# plt.plot(badj_SLD_TS_fwi,at,c='tab:blue')
plt.plot(ts0,tt0_inv,'tab:green',label='Theoretical TS')
# plt.plot(ts0,tt0_adj,'y')
plt.gca().invert_yaxis()
plt.axhline(p_inv.tt_[out_idx],c='tab:orange',ls='--')
# plt.axhline(p_adj.tt_[out_idx],c='tab:blue',ls='--')
plt.legend()
plt.xlim(-0.5,7)
plt.ylim(at[-1],at[0])

df = pd.DataFrame([ts0,tt0_inv]).T
df.to_csv(csv_out_name,header=False,index=False)


#%%


