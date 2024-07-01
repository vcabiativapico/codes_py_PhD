#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 19 14:24:48 2024

@author: vcabiativapico
"""


import csv
import numpy as np
import geophy_tools as gt
import matplotlib.pyplot as plt
from spotfunk.res import procs,visualisation

from scipy.interpolate import interpolate
import sympy as sp
import tqdm

import functions_bsplines_new_kev_test_2_5D as kvbsp


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
# ## Add y dimension    
fy = -500 
ny = 21
dy = 50
ay = fy + np.arange(ny)*dy

def plot_model(inp,hmin,hmax):
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(16,4), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect=1\
                     )
    plt.colorbar(hfig)
    plt.xlabel('Distance (Km)')
    plt.ylabel('Profondeur (Km)')
    fig.tight_layout()
    return fig
    
def export_model(inp,fig,imout,flout):
    fig.savefig(imout, bbox_inches='tight')
    gt.writebin(inp,flout)  


def read_index(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(int(row[srow]))
    return attr

def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
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
    
    nb_gathers, nb_traces = create_nodes(diff_ind_max,idx_nr_off, idx_nr_src,p.nx_,p.no_)
    
    
    inp3 = read_shots_around(gather_path, nb_gathers, p)
   
    # fin_trace = interpolate_src_rec(nb_traces,nb_gathers,at,ao,inp3,-p.off_x_,p.src_x_,do,dx,diff_ind_max) 
    
    fin_trace = interpolate_src_rec(nb_traces,nb_gathers,p.at_,p.ao_,inp3,p.off_x_,p.src_x_,p.do_,p.dx_,diff_ind_max)    
    return fin_trace

def vel_in_raypath(Param,Weight,ray_x,ray_z):
    
    
    Parameters,Weights = kvbsp.load_weight_model(Param, Weight)
    
    ray_x_z = [ray_x,ray_z]
    vel = []
    
    for i in range(len(ray_x)):
        vel.append(kvbsp.Vitesse(0,ray_x[i],ray_z[i],Parameters,Weights)[0])
    
    
    vel = np.array(vel)
    return vel

def defwsrc(fmax, dt, lent,nws):
    """
    Definition of the source function
    Ricker wavelet with central frequency fmax/2.5
    Ricker = 2nd-order derivative of a Gaussian function
    """
    fc = fmax / 2.5  # Central frequency
    ns2  = nws + lent # Classical definition
    ns = int((ns2-1)/2)  # Size of the source
    wsrc = np.zeros(ns2)
    for it in range(-ns,ns+1):
        a1 = float(it) * fc * dt * np.pi
        a2 = a1 ** 2     
        wsrc[it+ns] = (1 - 2 * a2) * np.exp(-a2)
    if len(wsrc) % 2 == 0 :
        print('The wavelet is even. The wavelet must be odd. \
        There must be the same number of samples on both sides of the maximum value')
    else: 
        print('Wavelet length is odd = ' + str(len(wsrc))+ '. No correction needed' )
    return wsrc
#%%
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





'''Read the velocity model '''

fl1 = '../input/45_marm_ano_v3/fwi_org.dat'
fl2 = '../input/45_marm_ano_v3/fwi_ano_114_percent.dat'


fl1='../input/45_marm_ano_v3/fwi_sm.dat'
fl2 = '../input/50_ts_model/marmousi_ano_sm.dat'




inp_org = gt.readbin(fl1,nz,nx)
inp_ano = gt.readbin(fl2,nz,nx)


'''Read the raypath'''
# spot lancé dans le modele 3510,-1210
path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/052_TS_deep/depth_demig_out/052_TS_analytique_deep_2024-06-20_12-02-07/rays/ray_0.csv'
ray_x = np.array(read_results(path_ray, 0))
ray_z = np.array(read_results(path_ray, 2))
ray_tt = np.array(read_results(path_ray, 8))

half_idx = len(ray_x)//2
hmin = 1.5
hmax = 4.5
%matplotlib inline
# %matplotlib qt5

spot_x = 3366.6
spot_z = 1024.3

spot_x = p_inv.spot_x_[0]
spot_z = -p_inv.spot_z_[0]

plot_model(inp_org,hmin,hmax)
plt.scatter(spot_x/1000,spot_z/1000,c='w',s=1)


fig1 = plot_model(inp_ano,hmin,hmax)
plt.plot(ray_x/1000,-ray_z/1000,'-')
plt.scatter(spot_x/1000,spot_z/1000,c='w',s=1)


idx_tt_x = find_nearest(ray_x[:half_idx], spot_x)[1]
idx_tt_z = find_nearest(ray_z[:half_idx], -spot_z)[1]

idx_tt = (idx_tt_x + idx_tt_z)//2

tt_at_spot = ray_tt[idx_tt]*2



'''Read the bsplines'''
Param_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Param_marm_smooth.csv'
Weight_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Weights_marm_2p5D_smooth.csv'

Param_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Param_marm_smooth_ANO.csv'
Weight_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Weights_marm_2p5D_smooth_ANO.csv'

Param_betap_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Param_marm_smooth_org.csv'
Weight_betap_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Weights_marm_2p5D_smooth_org.csv'

Param_betap_ano = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Param_marm_smooth_ano.csv'
Weight_betap_ano = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Weights_marm_2p5D_smooth_ano.csv'


'''Extract velocity values from the full model'''
vel_ray_org = vel_in_raypath(Param_vel_org_sm, Weight_vel_org_sm, ray_x, ray_z)
vel_ray_ano = vel_in_raypath(Param_vel_ano_sm, Weight_vel_ano_sm, ray_x, ray_z)

betap_ray_org= vel_in_raypath(Param_betap_org, Weight_betap_org, ray_x, ray_z)
betap_ray_ano= vel_in_raypath(Param_betap_ano, Weight_betap_ano, ray_x, ray_z)

x_disc = np.arange(601)*12.00
z_disc = np.arange(151)*12.00


''' Creation de l'ondelette source '''
# Parameters of the analytical source wavelet
nws = 143
fmax = 25
nt2 = nt - (nws-1) / 2
nt_len = int((nt2+1) * 2)
wsrc_org = defwsrc(fmax, dt,0,nws)
nws2 = nws//2

plt.figure(figsize=(6,10))
plt.plot(wsrc_org)


# def depth_to_time(ray_z,ray_x,vel_ray):
#     '''Transform profile from depth to time using the velocity '''
#     ray_time = []
#     dz =[]
#     v0 =[]
#     time =[0]
#     dz0 =[]
#     for i in range(len(ray_x)//2-1):
#         dz = np.sqrt((ray_z[i] - ray_z[i+1])**2 + (ray_x[i] - ray_x[i+1])**2)
#         v0 = (vel_ray[i]+vel_ray[i+1])/2
#         time.append(dz/v0)
#         print('dz: ',dz,'v0: ',v0,'time: ',time[-1]*2)
#     ray_time = np.cumsum(time) 
    
#     ray_time = np.array(ray_time)*2
#     return ray_time



file = '../time_shift_theorique.csv'
ts_theorique = [np.array(read_pick(file,0)),np.array(read_pick(file,1))]


plt.figure(figsize=(6,10))
plt.plot(vel_ray_org[:half_idx],ray_z[:half_idx]/1000,'.')
plt.plot(vel_ray_ano[:half_idx],ray_z[:half_idx]/1000,'.')
plt.title('velocity or amplitude value vs depth')
# plt.ylim(-0.9,-1.1)


ray_time_org = depth_to_time(ray_z,ray_x,vel_ray_org)
ray_time_ano = depth_to_time(ray_z,ray_x,vel_ray_ano)

# ray_time_org = ray_tt[:half_idx]*2
# ray_time_ano = ray_tt[:half_idx]*2


'''Interpolation using betap'''

f = interpolate.interp1d(ray_time_org,-betap_ray_org[:half_idx],kind='cubic')
ray_time_org_int = np.arange(0, np.max(ray_time_org),dt)
betap_ray_org_int = f(ray_time_org_int)



f = interpolate.interp1d(ray_time_ano,-betap_ray_ano[:half_idx],kind='cubic')
ray_time_ano_int = np.arange(0, np.max(ray_time_ano),dt)
betap_ray_ano_int = f(ray_time_ano_int)

plt.figure(figsize=(6,10))
plt.plot(-betap_ray_org[:half_idx],ray_z[:half_idx]/1000)
plt.plot(-betap_ray_ano[:half_idx],ray_z[:half_idx]/1000)

plt.title('velocity or amplitude value vs depth')
plt.ylabel('depth (s)')

plt.figure(figsize=(6,10))
plt.plot(betap_ray_org_int,-ray_time_org_int)
plt.plot(betap_ray_ano_int,-ray_time_ano_int)
plt.axhline(-tt_at_spot,c='tab:green')
plt.title('velocity or amplitude value vs time ')
plt.ylabel('time (s)')

'''std convolution'''


convol_time_ano = np.convolve(betap_ray_ano_int, wsrc_org,mode='same')
convol_time_org = np.convolve(betap_ray_org_int, wsrc_org,mode='same')[:len(convol_time_ano)]

plt.figure(figsize=(6,12))
plt.plot(convol_time_org,-ray_time_org_int[:len(convol_time_ano)],label='org')
plt.plot(convol_time_ano,-ray_time_ano_int,label='ano')
# plt.axhline(-p_inv.tt_[0],c='tab:green')
plt.axhline(-tt_at_spot,c='tab:green')
plt.ylim(-ray_time_ano_int[-1],-ft)
plt.title('Traces from theoretical migration')
plt.ylabel('time (s)')
plt.legend()


# '''Using velocity'''
# f = interpolate.interp1d(ray_time_org,vel_ray_org[:half_idx],kind='cubic')
# ray_time_org_int = np.arange(0, np.max(ray_time_org),dt)
# vel_ray_org_int = f(ray_time_org_int)

# f = interpolate.interp1d(ray_time_org,vel_ray_ano[:half_idx],kind='cubic')
# ray_time_ano_int = np.arange(0, np.max(ray_time_org),dt)
# vel_ray_ano_int = f(ray_time_ano_int)


# plt.figure(figsize=(6,10))
# plt.plot(vel_ray_org_int,-ray_time_org_int)
# plt.plot(vel_ray_ano_int,-ray_time_ano_int)
# plt.title('velocity or amplitude value vs time')


# plt.figure(figsize=(6,10))
# plt.plot(vel_ray_org_int-vel_ray_ano_int,-ray_time_ano_int)




#%%
oplen = 500


# SLD_TS = procs.sliding_TS(convol_time_org[:-nws+1],convol_time_ano[:-nws+1],oplen= oplen,si=0.002, taper= 30)
SLD_TS = procs.sliding_TS(convol_time_org,convol_time_ano,oplen= oplen,si=dt, taper= 30)


'''Read modelled traces time-shift'''


gather_path_fwi_org = '../output/40_marm_ano/binv_ano'
gather_path_fwi45 = '../output/40_marm_ano/binv_ano'

nt = 1801
at = ft + np.arange(nt)*dt
gather_path_fwi_org = '../output/45_marm_ano_v3/org_1801TL'
gather_path_fwi45 = '../output/45_marm_ano_v3/ano_114_perc_1801TL'


idx_src = find_nearest(ax, p_inv.src_x_[0]/1000)[1]


fl_org = gather_path_fwi_org+'/t1_obs_000'+str(idx_src)+'.dat'
fl_ano = gather_path_fwi45+'/t1_obs_000'+str(idx_src)+'.dat'

tr_binv_fwi_org = -gt.readbin(fl_org, no, nt).transpose()[:,125]
tr_binv_fwi_45  = -gt.readbin(fl_ano, no, nt).transpose()[:,125]
  



diff = tr_binv_fwi_org-tr_binv_fwi_45


%matplotlib inline
# %matplotlib qt5

plt.figure(figsize=(6,12))
plt.plot(tr_binv_fwi_org,at)
plt.plot(tr_binv_fwi_45,at)
# plt.plot(diff,p_inv.at_)
plt.axhline(tt_at_spot,c='tab:green')
plt.xlim(-0.05,0.05)
plt.ylim(ft,ray_time_ano_int[-1])
plt.gca().invert_yaxis()

# badj_SLD_TS_fwi = procs.sliding_TS(tr_badj_fwi_org,tr_badj_fwi_45,oplen= oplen,si=p_adj.dt_, taper= 30)
binv_SLD_TS_fwi = procs.sliding_TS(tr_binv_fwi_org,tr_binv_fwi_45,oplen= oplen,si=dt, taper= 30)


plt.figure(figsize=(6,12))
plt.plot(SLD_TS,ray_time_ano_int,c='tab:purple')
plt.plot(ts_theorique[0],ts_theorique[1],c='tab:green')
plt.title('Sliding time-shift vs time')
plt.plot(binv_SLD_TS_fwi,at,c='tab:orange',label='QTV')
# plt.plot(badj_SLD_TS_fwi,at,c='tab:blue',label='STD')
plt.legend()
plt.gca().invert_yaxis()


#%%

# '''Read index of the anomaly'''
# file_index = '../input/45_marm_ano_v3/fwi_ano_114_percent.csv'
# inp_index = [np.array(read_index(file_index,0)), np.array(read_index(file_index,1))]


# fl1       = '../input/org_full/marm2_full.dat'
# fl2       = '../input/marm2_sm15.dat'
# inp_org2 = gt.readbin(fl1,nz,nx)
# inp_sm    = gt.readbin(fl2,nz,nx)


# inp_ano2 = np.copy(inp_org2)
# inp_ano2[inp_index] = inp_ano2[inp_index] *1.14


# betap_org = 1/inp_org2**2 - 1/inp_sm**2
# betap_ano = 1/inp_ano2**2 - 1/inp_sm**2

# hmin=np.min(betap_org)
# hmax=np.max(betap_org)
# fig=plot_model(betap_org,hmin,hmax)
# imout1 = '../png/45_marm_ano_v3/fwi_betap_org.png'
# flout1 = '../input/45_marm_ano_v3/fwi_betap_org.dat'
# export_model(betap_org,fig,imout1,flout1)



# hmin=np.min(betap_ano)
# hmax=np.max(betap_ano)
# fig=plot_model(betap_ano,hmin,hmax)
# imout1 = '../png/45_marm_ano_v3/fwi_betap_ano.png'
# flout1 = '../input/45_marm_ano_v3/fwi_betap_ano.dat'
# export_model(betap_ano,fig,imout1,flout1)