#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 10:43:55 2024

@author: vcabiativapico
"""


import csv
import numpy as np
import geophy_tools as gt
import matplotlib.pyplot as plt
from spotfunk.res import procs, visualisation
from scipy.interpolate import interpolate
import sympy as sp
import tqdm
import collections
import functions_bsplines_new_kev_test_2_5D as kvbsp


# Global parameters
labelsize = 16
nt = 1801
dt = 1.14e-3
# dt= 1.41e-3
ft = -100.3e-3
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


def plot_model(inp, hmin, hmax):
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(16, 4), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                     vmin=hmin, vmax=hmax, aspect=1
                     )
    plt.colorbar(hfig)
    plt.xlabel('Distance (Km)')
    plt.ylabel('Profondeur (Km)')
    fig.tight_layout()
    return fig


def export_model(inp, fig, imout, flout):
    fig.savefig(imout, bbox_inches='tight')
    gt.writebin(inp, flout)


def read_index(path, srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(int(row[srow]))
    return attr


def read_pick(path, srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr


# Read the results from demigration
def read_results(path, srow):
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


def extract_trace(p, idx_rt, path_shot):
    ax = p.fx_ + np.arange(p.nx_)*p.dx_
    ao = p.fo_ + np.arange(p.no_)*p.do_
    title = str(find_nearest(ax, p.src_x_[idx_rt]/1000)[1])
    title = title.zfill(3)
    print('indice source', title)
    fl = path_shot+'/t1_obs_000'+str(title)+'.dat'
    if int(title) > int(p.nx_ - p.no_//2):
        noff = p.no_ // 2 + p.nx_ - int(title) + 1
    else:
        noff = p.no_
    inp = gt.readbin(fl, noff, p.nt_).transpose()
    tr = find_nearest(ao, p.off_x_[idx_rt]/1000)[1]
    return inp, tr


def create_nodes(diff_ind_max, idx_nr_off, idx_nr_src, nx, no):
    """
    Find the indexes of the traces that will be used as nodes for the interpolation
    """
    if idx_nr_src < 2:
        nb_gathers = np.array([0, 1, 2, 3, 4])
    elif idx_nr_src > nx-3:
        nb_gathers = np.array([597, 598, 599, 600, 601])
    else:
        nb_gathers = np.arange(idx_nr_src-2, idx_nr_src+3)

    if idx_nr_off < 2:
        nb_traces = np.array([0, 1, 2, 3, 4])
    elif idx_nr_off > no-3:
        nb_traces = np.array([247, 248, 249, 250, 251])
    else:
        nb_traces = np.arange(idx_nr_off-2, idx_nr_off+3)

    return nb_gathers, nb_traces


def read_shots_around(gather_path, nb_gathers, param):
    """
    Reads the shots around the closest shot from raytracing to the numerical born modelling grid
    """

    inp3 = np.zeros((len(nb_gathers), param.nt_, param.no_))

    for k, i in enumerate(nb_gathers):
        txt = str(i)
        title = txt.zfill(3)

        tr3 = gather_path+'/t1_obs_000'+str(title)+'.dat'

        inp3[k][:, :] = -gt.readbin(tr3, param.no_, param.nt_).transpose()

    return inp3


def interpolate_src_rec(nb_traces, nb_gathers, at, ao, inp3, off_x, src_x, do, dx, diff_ind_max):
    """
    Performs interpolation between selected shots and traces
    @nb_traces : index of the reference traces to use as nodes for the interpolation
    @nb_gathers : index of the gathers to use as nodes for the interpolation
    @diff_ind_max : index of the AO result for source, receiver and spot from the raytracing file
    """
    nt = 1801
    tr_INT = np.zeros((len(nb_gathers), nt, 5))

    for k, i in enumerate(nb_gathers):

        # Interpolation on the receivers
        for j in range(len(nb_gathers)):

            f = interpolate.RegularGridInterpolator(
                (at, ao[nb_traces]), inp3[j][:, nb_traces], method='linear', bounds_error=False, fill_value=None)
            at_new = at
            ao_new = np.linspace(
                off_x[diff_ind_max]/1000-do*2, off_x[diff_ind_max]/1000+do*2, 5)
            AT, AO = np.meshgrid(at_new, ao_new, indexing='ij')
            tr_INT[j][:, :] = f((AT, AO))
            rec_int = tr_INT[:, :, 2]

        # Interpolation on the shots
        f = interpolate.RegularGridInterpolator(
            (at, nb_gathers*12), rec_int.T, method='linear', bounds_error=False, fill_value=None)
        at_new = at
        src_new = np.linspace(
            src_x[diff_ind_max] - dx*2000, src_x[diff_ind_max] + dx*2000, 5)
        AT, SRC = np.meshgrid(at_new, src_new, indexing='ij')
        src_INT = f((AT, SRC))
        interp_trace = src_INT[:, 2]
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


def trace_from_rt(diff_ind_max, gather_path, p):
    '''
    Finds the nearest trace to the source and offset given by the index
    Interpolates the traces so that the calculation is exact
    Exports the trace found by raytracing from the modelling data
    '''
    nr_src_x, idx_nr_src = find_nearest(p.ax_, p.src_x_[diff_ind_max]/1000)
    print('idx_src: ',idx_nr_src)
    nr_off_x, idx_nr_off = find_nearest(p.ao_, p.off_x_[diff_ind_max]/1000)

    nb_gathers, nb_traces = create_nodes(
        diff_ind_max, idx_nr_off, idx_nr_src, p.nx_, p.no_)

    inp3 = read_shots_around(gather_path, nb_gathers, p)

    # fin_trace = interpolate_src_rec(nb_traces,nb_gathers,at,ao,inp3,-p.off_x_,p.src_x_,do,dx,diff_ind_max)

    fin_trace = interpolate_src_rec(
        nb_traces, nb_gathers, p.at_, p.ao_, inp3, p.off_x_, p.src_x_, p.do_, p.dx_, diff_ind_max)
    return fin_trace


def vel_in_raypath(Param, Weight, ray_x, ray_z):

    Parameters, Weights = kvbsp.load_weight_model(Param, Weight)

    ray_x_z = [ray_x, ray_z]
    vel = []

    for i in range(len(ray_x)):
        vel.append(kvbsp.Vitesse(
            0, ray_x[i], ray_z[i], Parameters, Weights)[0])

    vel = np.array(vel)
    return vel


def defwsrc(fmax, dt, lent, nws):
    """
    Definition of the source function
    Ricker wavelet with central frequency fmax/2.5
    Ricker = 2nd-order derivative of a Gaussian function
    """
    fc = fmax / 2.5  # Central frequency
    ns2 = nws + lent  # Classical definition
    ns = int((ns2-1)/2)  # Size of the source
    wsrc = np.zeros(ns2)
    for it in range(-ns, ns+1):
        a1 = float(it) * fc * dt * np.pi
        a2 = a1 ** 2
        wsrc[it+ns] = (1 - 2 * a2) * np.exp(-a2)
    if len(wsrc) % 2 == 0:
        print('The wavelet is even. The wavelet must be odd. \
        There must be the same number of samples on both sides of the maximum value')
    else:
        print('Wavelet length is odd = ' +
              str(len(wsrc)) + '. No correction needed')
    return wsrc


def depth_to_time(ray_z, ray_x, vel_ray):
    '''Transform profile from depth to time using the velocity '''
    ray_time = []
    dz = []
    v0 = []
    time = [0]
    dz0 = []
    for i in range(len(ray_x)//2-1):
        dz = np.sqrt((ray_z[i] - ray_z[i+1])**2 + (ray_x[i] - ray_x[i+1])**2)
        v0 = (vel_ray[i]+vel_ray[i+1])/2
        time.append(dz/v0)
        print('dz: ', dz, 'v0: ', v0, 'time: ', time[-1]*2)
    ray_time = np.cumsum(time)

    ray_time = np.array(ray_time)*2
    return ray_time


def phase_rot(inp, fact):
    """Modification of the phase"""
    wsrcf = np.fft.rfft(inp, axis=-1)
    n = len(wsrcf)
    #     fact  = np.pi/4
    wsrcf *= np.exp(1j*fact)
    mod_ph = np.fft.irfft(wsrcf, axis=-1)
    return mod_ph



def calculate_slope(degrees, spot_x, spot_z, plot=False):
    m = np.tan(degrees * np.pi/180) 
    b = spot_z - m * spot_x
    point_x = np.array([spot_x - 150, spot_x + 150])
    point_z = point_x * m + b
    plt.figure(figsize=(10,8))
    plt.plot(point_x,point_z,'k')
    plt.scatter(spot_x,spot_z)
    plt.legend([degrees])
    p1 = np.array([point_x[0], 0 , point_z[0]])
    p2 = np.array([point_x[1], 0 , point_z[1]])
    p3 = np.array([point_x[0], 1200 , point_z[0]])
    
    if plot==True: 
        plt.figure(figsize=(10,8))
        plt.plot(point_x/1000,point_z/1000,linewidth=3)
        plt.scatter(spot_x/1000,spot_z/1000,c='k')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
    return p1, p2, p3


def t_first_non_zero(ray_time, diff, val):
    idx = np.where(diff > val)[0][0]
    t_frst_non_zero = ray_time[idx]
    return t_frst_non_zero




def find_rep_idx(array):
    repeat_val = [item for item, count in collections.Counter(array).items() if count > 1]
    # print(repeat_val)
    idx_corr = []

    for item in repeat_val: 
        # print(item) 
        idx = np.where(array == item)
        idx_corr.append(idx)
    return idx_corr

def read_rt(ind_off,mig):
    
    if mig == 0:
        p = p_adj
        title = 'adj'
        '''Read the raypath'''
        
        path_ray_deep = gen_path + \
        '056_correct_TS_deep/depth_demig_out/deeper2_39/rays/ray_'+str(ind_off)+'.csv' 
    
    elif mig == 1: 
        p = p_inv
        title = 'inv'
        '''Read the raypath'''
    
        # path_ray = gen_path + \
        #     '056_correct_TS_deep/depth_demig_out/050_TS_binv_offset2024-07-16_11-37-30/'+\
        #         'rays/ray_'+str(ind_off)+'.csv'
    
        # path_ray_deep = gen_path + \
        # '056_correct_TS_deep/depth_demig_out/deeper37_2024-07-16_15-01-32/'+\
        #     'rays/ray_'+str(ind_off)+'.csv'
          
        path_ray_deep = gen_path + \
        '056_correct_TS_deep/depth_demig_out/deeper2_37_2024-07-18_15-04-33/'+\
            'rays/ray_'+str(ind_off)+'.csv'   

        # path_ray_deep = gen_path + \
        # '048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/'+\
        #     'rays/ray_'+str(ind_off)+'.csv'   
            
    ray_x = np.array(read_results(path_ray_deep, 0))
    ray_z = np.array(read_results(path_ray_deep, 2))
    ray_tt = np.array(read_results(path_ray_deep, 8))
    return p, title, ray_x, ray_z, ray_tt, path_ray_deep

#%%

''' Creation de l'ondelette source '''
# Parameters of the analytical source wavelet
nws = 177
fmax = 25
nt2 = nt - (nws-1) / 2
nt_len = int((nt2+1) * 2)
wsrc_org = defwsrc(fmax, dt, 0, nws)
nws2 = nws//2

wsrc_rot = phase_rot(wsrc_org, np.pi/2)

plt.figure(figsize=(6, 10))
plt.plot(wsrc_org)


gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'


# Results from source to spot located after anomaly in the smooth 15 model
path_inv = gen_path + '056_correct_TS_deep/depth_demig_out/050_TS_binv_offset2024-07-16_11-37-30/results/depth_demig_output.csv'
path_adj = gen_path + '056_correct_TS_deep/depth_demig_out/050_TS_badj_offset2024-07-16_11-24-07/results/depth_demig_output.csv'

path_inv = gen_path + '056_correct_TS_deep/depth_demig_out/deeper37_2024-07-16_15-01-32/results/depth_demig_output.csv'
path_adj = gen_path + '056_correct_TS_deep/depth_demig_out/deeper39_2024-07-16_15-07-56/results/depth_demig_output.csv'

# path_inv = gen_path + '056_correct_TS_deep/depth_demig_out/deeper2_37_2024-07-18_15-04-33/results/depth_demig_output.csv'
# path_adj = gen_path + '056_correct_TS_deep/depth_demig_out/deeper2_39/results/depth_demig_output.csv'

# path_adj = gen_path + '048_sm8_correction_new_solver/STD/depth_demig_out/STD/results/depth_demig_output.csv'
# path_inv = gen_path + '048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/results/depth_demig_output.csv'


class Param_class:
    "Class for the parameters definition"
    def __init__(self, path):
        self.src_x_ = read_results(path, 1)
        self.src_y_ = read_results(path, 2)
        self.src_z_ = read_results(path, 3)
        self.rec_x_ = read_results(path, 4)
        self.rec_y_ = read_results(path, 5)
        self.rec_z_ = read_results(path, 6)
        self.spot_x_ = read_results(path, 7)
        self.spot_y_ = read_results(path, 8)
        self.spot_z_ = read_results(path, 9)
        self.off_x_ = read_results(path, 16)
        self.tt_ = read_results(path, 17)
        self.nt_ = 1801
        self.dt_ = 1.14e-3
        # self.dt_ = 1.41e-3
        self.ft_ = -100.3e-3
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


ind_off = 0

mig = 1


p, title, ray_x, ray_z, ray_tt, path_ray_deep = read_rt(ind_off,mig)

'''Read the velocity model '''

fl1 = '../input/45_marm_ano_v3/fwi_sm.dat'

fl2 = '../input/50_ts_model/marmousi_ano_sm.dat'
# fl2 = '../output/45_marm_ano_v3/mig_binv_sm8/inv_betap_x_s.dat'

inp_org = gt.readbin(fl1, nz, nx)
inp_ano = gt.readbin(fl2, nz, nx)



# ray_x_d = np.array(read_results(path_ray_deep, 0))
# ray_z_d = np.array(read_results(path_ray_deep, 2))
# ray_tt_d = np.array(read_results(path_ray_deep, 8))


# ray_x_in_poly,ray_z_in_poly = 3451, -1184
# ray_x_in_poly,ray_z_in_poly = 3369, -1071
ray_x_in_poly,ray_z_in_poly = p.spot_x_[ind_off], p.spot_z_[ind_off]

degree = 37
pt_inv1, pt_inv2, pt_inv3 = calculate_slope(degree,ray_x_in_poly,ray_z_in_poly, plot=False)

# %matplotlib inline
# %matplotlib qt5

hmin = 1.5
hmax = 4.5

# hmin = np.min(inp_ano)
# hmax = -hmin
fig1 = plot_model(inp_ano, hmin, hmax)
plt.plot(ray_x/1000, -ray_z/1000, '-')
plt.scatter(p.spot_x_/1000, -p.spot_z_/1000, c='w', s=1)
plt.plot(np.array([pt_inv1[0],pt_inv2[0]])/1000, np.array([-pt_inv1[2],-pt_inv2[2]])/1000, 'r')



half_idx = np.argmin(ray_z)

def theoretical_traces(Param_vel_sm, Weight_vel_sm, Param_betap, Weight_betap, ray_x, ray_z,ray_t,wsrc_org,off_idx):
    
    '''Creates theoretical traces from ray tracing by extracting values of the raypath on the model'''
    vel_ray = vel_in_raypath(
        Param_vel_sm, Weight_vel_sm, ray_x, ray_z)
    betap_ray = vel_in_raypath(
        Param_betap, Weight_betap, ray_x, ray_z)
         
    half_idx = np.argmin(ray_z)
    
        
    idx_corr_time = find_rep_idx(ray_t)
    ray_t[idx_corr_time[0][0][1]] = ray_t[idx_corr_time[0][0][1]] + 1e-4
    ray_t[idx_corr_time[1][0][1]] = ray_t[idx_corr_time[1][0][1]] + 1e-4
    
    '''Interpolation using betap'''

    f = interpolate.interp1d(ray_t[:half_idx], -betap_ray[:half_idx], kind='cubic')
    ray_time_int = np.arange(0, np.max(ray_t[:half_idx]), dt)
    # ray_time_int = np.arange(0, p.tt_[off_idx], dt)
  
    # print(np.shape(ray_time_int))
   
    betap_ray_int = f(ray_time_int)
 
    '''std convolution'''
    
    convol_time_int = np.convolve(betap_ray_int, wsrc_org, mode='same')
    return vel_ray, betap_ray_int, convol_time_int,ray_time_int


'''Read the bsplines'''

# Values from velocity model
Param_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Param_marm_smooth.csv'
Weight_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Weights_marm_2p5D_smooth.csv'

Param_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Param_marm_smooth_ANO.csv'
Weight_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Weights_marm_2p5D_smooth_ANO.csv'

# Values from perturbation model in the full model
Param_betap_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Param_marm_smooth_org.csv'
Weight_betap_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Weights_marm_2p5D_smooth_org.csv'

Param_betap_ano = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Param_marm_smooth_ano.csv'
Weight_betap_ano = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Weights_marm_2p5D_smooth_ano.csv'

ray_time = ray_tt*2
vel_ray_org, betap_ray_org_int, convol_time_org_int, time_ray_org = theoretical_traces(Param_vel_org_sm, Weight_vel_org_sm, Param_betap_org, Weight_betap_org, ray_x, ray_z, ray_time,wsrc_org,ind_off)
ray_time = ray_tt*2
vel_ray_ano, betap_ray_ano_int, convol_time_ano_int, time_ray_ano = theoretical_traces(Param_vel_ano_sm, Weight_vel_ano_sm, Param_betap_ano, Weight_betap_ano, ray_x, ray_z, ray_time,wsrc_org,ind_off)

ray_time_org_int = np.arange(0, np.max(ray_time[:half_idx]), dt)
ray_time_ano_int = np.arange(0, np.max(ray_time[:half_idx]), dt)


oplen = 500


'''Read modelled traces time-shift'''


# Modelled traces with Born in the full perturbation model
gather_path_fwi_org = '../output/47_marm2/org'
gather_path_fwi45 = '../output/47_marm2/ano'

# Modelled traces with FWI in the full perturbation model
gather_path_fwi_org = '../output/45_marm_ano_v3/org_1801TL'
gather_path_fwi45 = '../output/45_marm_ano_v3/ano_114_perc_1801TL'


tr_binv_fwi_org = trace_from_rt(ind_off,gather_path_fwi_org,p_inv)
tr_binv_fwi_45 = trace_from_rt(ind_off,gather_path_fwi45,p_inv)

tr_badj_fwi_org = trace_from_rt(ind_off,gather_path_fwi_org,p_adj)
tr_badj_fwi_45 = trace_from_rt(ind_off,gather_path_fwi45,p_adj)


diff_tr_mod = tr_binv_fwi_org-tr_binv_fwi_45
diff_vel_art = vel_ray_org-vel_ray_ano
diff_betap_art = betap_ray_org_int-betap_ray_ano_int
diff_convol_art = convol_time_ano_int-convol_time_org_int

f_no_tr_mod = t_first_non_zero(at, -diff_tr_mod, 5e-5)
# f_no_vel = t_first_non_zero(ray_time, diff_vel_art, 0.05)
f_no_betap = t_first_non_zero(ray_time_org_int, -diff_betap_art, 0.4)
f_no_conv = t_first_non_zero(ray_time_org_int, -diff_convol_art, 2)



# difference = f_no_conv-f_no_betap
# plt.figure(figsize=(6, 12))
# plt.plot(vel_ray_org[:half_idx], ray_time[:half_idx])
# plt.plot(vel_ray_ano[:half_idx], ray_time[:half_idx])
# plt.title('velocity vs time')
# plt.gca().invert_yaxis()
# plt.ylabel('time (s)')
# # plt.ylim(-0.9,-1.1)


plt.figure(figsize=(6, 12))
plt.plot(betap_ray_org_int, time_ray_org)
plt.plot(betap_ray_ano_int, time_ray_ano)

# plt.axhline(f_no_betap[ind_off],c='tab:green',label='RT time')
plt.title('Perturbation vs time')
plt.ylim(ft, ray_time_ano_int[-1])
plt.gca().invert_yaxis()
plt.ylabel('time (s)')
# plt.ylim(-0.9,-1.1)


plt.rcParams['font.size'] = 20
plt.figure(figsize=(6, 12))
plt.plot(convol_time_org_int, time_ray_org, label='org')
plt.plot(convol_time_ano_int, time_ray_ano, label='ano')
# plt.axhline(f_no_conv[ind_off],c='tab:purple',label='RT time')
plt.ylim(ft, ray_time_ano_int[-1])
plt.title('Traces from theoretical migration', fontsize=15)
plt.ylabel('time (s)')
plt.legend(loc='upper right', fontsize=18)
plt.gca().invert_yaxis()


# %matplotlib qt5
plt.figure(figsize=(6, 12))
plt.plot(tr_binv_fwi_org, at)
plt.plot(tr_binv_fwi_45, at)
# plt.plot(diff,p_inv.at_)
# plt.axhline(p.tt_[ind_off], c='tab:green', label='RT time')
# plt.axhline(f_no_betap,c='tab:green',label='RT time')
# plt.axhline(f_no_tr_mod, c='tab:purple', label='First 0 mod')
plt.title('Traces from modeling', fontsize=15)
plt.legend(loc='upper right', fontsize=18)
plt.xlim(-0.05, 0.05)
plt.ylim(1.5,ray_time_ano_int[-1])
plt.grid()
# plt.ylim(ft, ray_time_ano_int[-1])
plt.ylabel('time (s)')
plt.gca().invert_yaxis()




binv_SLD_TS_fwi = procs.sliding_TS(
    tr_binv_fwi_org, tr_binv_fwi_45, oplen=oplen, si=dt, taper=30)
badj_SLD_TS_fwi = procs.sliding_TS(
    tr_badj_fwi_org, tr_badj_fwi_45, oplen=oplen, si=dt, taper=30)


SLD_TS = procs.sliding_TS(
    convol_time_org_int, convol_time_ano_int, oplen=oplen, si=dt, taper=30)

a_time = np.linspace(0, p.tt_[ind_off], len(time_ray_org))



plt.figure(figsize=(7, 18))
plt.plot(binv_SLD_TS_fwi,at-ft)
plt.plot(badj_SLD_TS_fwi,at-ft)
plt.plot(SLD_TS,a_time)
plt.axhline(p_inv.tt_[ind_off],c='k')
plt.xlim(-0.5,4)
plt.gca().invert_yaxis()




#%%

oplen =300
# diff_to_correct = f_no_betap - f_no_conv


binv_SLD_TS_fwi_total = []
badj_SLD_TS_fwi_total = []
convol_time_int_total_org = []
convol_time_int_total_ano = []
SLD_theo_total = []
out_idx = []
ray_time_total = []
ray_time_corr = []
betap_ray_org_int_total = []
betap_ray_ano_int_total = []
result = []
f_no_betap = []
f_no_conv = []
diff_to_corr = []
tr_binv_fwi_org_total= []
tr_binv_fwi_45_total= []

for i in tqdm.tqdm(range(0,100,8)):
    if i > 99: break
    tr_binv_fwi_org = trace_from_rt(i,gather_path_fwi_org,p_inv)
    tr_binv_fwi_45 = trace_from_rt(i,gather_path_fwi45,p_inv)
    
    tr_badj_fwi_org = trace_from_rt(i,gather_path_fwi_org,p_adj)
    tr_badj_fwi_45 = trace_from_rt(i,gather_path_fwi45,p_adj)
    
    tr_binv_fwi_org_total.append(tr_binv_fwi_org)
    tr_binv_fwi_45_total.append(tr_binv_fwi_45)
    
    
    binv_SLD_TS_fwi = procs.sliding_TS(
    tr_binv_fwi_org, tr_binv_fwi_45, oplen=oplen, si=dt, taper=30)
    badj_SLD_TS_fwi = procs.sliding_TS(
    tr_badj_fwi_org, tr_badj_fwi_45, oplen=oplen, si=dt, taper=30)
    
    binv_SLD_TS_fwi_total.append(binv_SLD_TS_fwi.T)
    badj_SLD_TS_fwi_total.append(badj_SLD_TS_fwi.T)
        
    p, title, ray_x, ray_z, ray_tt, path_ray_deep = read_rt(i,mig)
    # print(title)
    result_org = theoretical_traces(Param_vel_org_sm, Weight_vel_org_sm, Param_betap_org, Weight_betap_org, ray_x, ray_z, ray_tt*2,wsrc_org,i)
    result_ano = theoretical_traces(Param_vel_ano_sm, Weight_vel_ano_sm, Param_betap_ano, Weight_betap_ano, ray_x, ray_z, ray_tt*2,wsrc_org,i)
    betap_ray_org_int_total.append(result_org[1])
    betap_ray_ano_int_total.append(result_ano[1])
    convol_time_int_total_org.append(result_org[2])
    convol_time_int_total_ano.append(result_ano[2])
    ray_time_total.append(result_org[3])
    
    SLD_theo_total.append(procs.sliding_TS(convol_time_int_total_org[-1], convol_time_int_total_ano[-1], oplen=oplen, si=dt, taper=30))
    out_idx.append(i)
    ray_time_corr.append(np.linspace(0, p.tt_[i], len(ray_time_total[-1])))
    
    diff_betap_art = betap_ray_org_int_total[-1] - betap_ray_ano_int_total[-1]
    diff_convol_art = convol_time_int_total_ano[-1] - convol_time_int_total_org[-1]
    f_no_betap.append(t_first_non_zero(ray_time_total[-1], -diff_betap_art, 1))
    f_no_conv.append(t_first_non_zero(ray_time_total[-1], -diff_convol_art, 2))
    
diff = np.array(f_no_betap) - np.array(f_no_conv)

# out_idx = np.arange(26)
%matplotlib inline
# %matplotlib qt5

#%%
'''Plot Time-shifts panel'''

def plot_panel_att(attr_binv,attr_badj,attr_theo,ray_time_corr,p,out_idx,xmin,xmax,title):
    plt.rcParams['font.size'] = 20
    ncols = int(np.shape(attr_binv)[0])
    axi = np.zeros(ncols)
    fig, (axi) = plt.subplots(nrows=1, ncols=ncols,
                              sharey=True,
                              figsize=(27, 8),
                              facecolor="white")
    
    axi[0].set_ylabel('Time (s)')
    axi[0].set_xlabel('TS (ms)')
    
    for i,k in enumerate(range(0,len(attr_binv))):
            if i>= ncols: break
            axi[i].plot(attr_binv[k],at-ft,c='tab:blue')
            axi[i].plot(attr_badj[k],at-ft,c='tab:orange')
            axi[i].plot(attr_theo[k],ray_time_corr[k]+diff[k],c='tab:purple')
            axi[i].set_title(str(int(p.off_x_[out_idx[k]])))
            axi[i].set_xlim(xmin,xmax)
            axi[i].set_ylim(at[-1],at[0])
            # axi[i].axhline(p.tt_[out_idx[k]],c='k')

    axi[i-1].legend(['QTV','STD','THEO'],loc='upper right') 
    fig.suptitle(title)
    return fig

fig = plot_panel_att(binv_SLD_TS_fwi_total,badj_SLD_TS_fwi_total,SLD_theo_total,ray_time_corr,p,out_idx,-0.5,4,'TS')



# ray_time_corr = np.linspace(0, p.tt_[-1], len(ray_time_total[-1]))
file = '../time_shift_theorique.csv'
ts_theorique = [np.array(read_pick(file, 0)), np.array(read_pick(file, 1))]


id_nb = 0




plt.figure(figsize=(7, 18))
plt.rcParams['font.size'] = 20
plt.plot(binv_SLD_TS_fwi_total[id_nb],at)
plt.plot(badj_SLD_TS_fwi_total[id_nb],at)
plt.plot(ts_theorique[0],ts_theorique[1]+ft,c='tab:green')
plt.plot(SLD_theo_total[id_nb],ray_time_corr[id_nb]+diff[id_nb])
plt.axhline(p.tt_[id_nb],c='k')
plt.gca().invert_yaxis()

plt.figure(figsize=(6, 12))
plt.plot(betap_ray_org_int_total[id_nb], ray_time_total[id_nb])
plt.plot(betap_ray_ano_int_total[id_nb], ray_time_total[id_nb])
plt.axhline(f_no_betap[id_nb],c='tab:green',label='RT time')
plt.title('Perturbation vs time')
plt.ylim(ft, ray_time_total[id_nb][-1])
plt.gca().invert_yaxis()
plt.ylabel('time (s)')
# plt.ylim(-0.9,-1.1)

plt.figure(figsize=(6, 12))
plt.plot(convol_time_int_total_org[id_nb], ray_time_total[id_nb])
plt.plot(convol_time_int_total_ano[id_nb], ray_time_total[id_nb])
plt.axhline(f_no_conv[id_nb],c='tab:green',label='RT time')
plt.title('Perturbation vs time')
plt.ylim(ft, ray_time_total[id_nb][-1])
plt.axhline(p.tt_[id_nb],c='k')
plt.gca().invert_yaxis()
plt.ylabel('time (s)')
# plt.ylim(-0.9,-1.1)

plt.figure(figsize=(6, 12))
plt.rcParams['font.size'] = 20
plt.plot(tr_binv_fwi_org_total[id_nb],at)
plt.plot(tr_binv_fwi_45_total[id_nb],at)
plt.axhline(p.tt_[id_nb],c='k')
plt.axhline(f_no_tr_mod,c='k')
plt.ylabel('time (s)')
plt.xlim(-0.05,0.05)
plt.ylim(ft, ray_time_total[id_nb][-1])
plt.gca().invert_yaxis()

#%%


flout = '../png/62_TS_analytique_offsets/Panel_TS_off_at_spot_250ms.png'
fig.savefig(flout, bbox_inches='tight')