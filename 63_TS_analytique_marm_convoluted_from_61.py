#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 16:24:57 2024

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

import functions_bsplines_new_kev_test_2_5D as kvbsp


# Global parameters
labelsize = 16
nt = 1801
dt = 1.14e-3
ft = -100.32e-3
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
    plt.rcParams['font.size'] = 20
    fig = plt.figure(figsize=(16, 8), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                     vmin=hmin, vmax=hmax, aspect='auto'
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
        # header = next(spamreader)
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
    nr_off_x, idx_nr_off = find_nearest(p.ao_, p.off_x_[diff_ind_max]/1000)
    print('src_x ', idx_nr_src)
    print('off_x ', idx_nr_off)
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


# %%
gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'

# path_inv = gen_path + '064_TS_analytique_deep/depth_demig_out/063_org_deep_2024-08-19_11-00-05/results/depth_demig_output.csv'
# path_adj  = gen_path +'061_flat_taper_const/depth_demig_out/061_TS_flat_deeper_ano_1750/results/depth_demig_output.csv'

# path_inv = gen_path + '064_TS_analytique_deep/depth_demig_out/deeper2_org_40_2024-08-19_11-49-02/results/depth_demig_output.csv'

# Model with smoothing 15 depth to the spot
path_inv = gen_path + \
    '063_TS_analytique/depth_demig_out/063_org_2024-08-19_12-11-20/results/depth_demig_output.csv'
path_adj = gen_path + \
    '063_TS_analytique/depth_demig_out/063_std_org_2024-08-19_12-11-27/results/depth_demig_output.csv'

# Model with smoothing 5 depth to the spot
path_inv = gen_path + \
    '063_TS_analytique/depth_demig_out/063_org_sm5_2024-08-20_14-33-21/results/depth_demig_output.csv'
path_adj = gen_path + \
    '063_TS_analytique/depth_demig_out/063_std_org_sm5_2024-08-20_14-38-36/results/depth_demig_output.csv'

# Model with smoothing 3 depth to the spot
# path_inv = gen_path + '063_TS_analytique/depth_demig_out/063_org_sm3_2024-08-21_11-07-31/results/depth_demig_output.csv'
# path_adj  = gen_path + '063_TS_analytique/depth_demig_out/063_std_org_sm3_2024-08-21_11-11-54/results/depth_demig_output.csv'


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


p_adj = Param_class(path_adj)
p_inv = Param_class(path_inv)


'''Read the velocity model '''

fl1 = '../input/45_marm_ano_v3/fwi_org.dat'
fl2 = '../input/50_ts_model/full_mod.dat'
fl1 = '../input/50_ts_model/sm5_ano.dat'
fl2 = '../input/50_ts_model/sm5_org.dat'

inp_org = gt.readbin(fl1, nz, nx)
inp_ano = gt.readbin(fl2, nz, nx)


'''Read the raypath'''


# Model with smoothing 5 deeper
path_ray = gen_path + \
    '064_TS_analytique_deep/depth_demig_out/deeper_sm5_org_40_2024-08-20_16-00-33/rays/ray_0.csv'
path_ray2 = gen_path + \
    '064_TS_analytique_deep/depth_demig_out/deeper_sm5_ano_40_2024-08-20_16-00-27/rays/ray_0.csv'

# Model with smoothing 15 depth to the spot
path_ray = gen_path + \
    '063_TS_analytique/depth_demig_out/063_org_2024-08-19_12-11-20/rays/ray_0.csv'
path_ray2 = gen_path + \
    '063_TS_analytique/depth_demig_out/063_ano_2024-08-19_12-11-16/rays/ray_0.csv'

# Model with smoothing 5 depth to the spot
path_ray = gen_path + \
    '063_TS_analytique/depth_demig_out/063_org_sm5_2024-08-20_14-33-21/rays/ray_0.csv'
path_ray2 = gen_path + \
    '063_TS_analytique/depth_demig_out/063_ano_sm5_2024-08-20_14-33-26/rays/ray_0.csv'

# Model with smoothing 3 depth to the spot
# path_ray = gen_path + '063_TS_analytique/depth_demig_out/063_org_sm3_2024-08-21_11-07-31/rays/ray_0.csv'
# path_ray2 = gen_path + '063_TS_analytique/depth_demig_out/063_ano_sm3_2024-08-21_11-07-00/rays/ray_0.csv'


ray_x = np.array(read_results(path_ray, 0))
ray_z = np.array(read_results(path_ray, 2))
ray_tt = np.array(read_results(path_ray, 8))

ray_x2 = np.array(read_results(path_ray2, 0))
ray_z2 = np.array(read_results(path_ray2, 2))
ray_tt2 = np.array(read_results(path_ray2, 8))


half_idx = len(ray_tt)//2
half_idx2 = len(ray_tt2)//2


hmin = 1.5
hmax = 4.0

spot_x = p_inv.spot_x_[0]
spot_z = -p_inv.spot_z_[0]


plt.figure()
plot_model(inp_org, hmin, hmax)
plt.scatter(spot_x/1000, spot_z/1000, c='r', s=5)
plt.plot(ray_x/1000, -ray_z/1000, 'w')

plt.figure()
plot_model(inp_ano, hmin, hmax)
plt.scatter(spot_x/1000, spot_z/1000, c='r', s=5)
plt.plot(ray_x2/1000, -ray_z2/1000, 'w')


'''Read the bsplines'''


# # Values from velocity model
# Param_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Param_marm_smooth.csv'
# Weight_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Weights_marm_2p5D_smooth.csv'

# Param_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Param_marm_smooth_ANO.csv'
# Weight_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_full_marm_Weights_marm_2p5D_smooth_ANO.csv'

# # Values from perturbation model in the full model
# Param_betap_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Param_marm_smooth_org.csv'
# Weight_betap_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Weights_marm_2p5D_smooth_org.csv'

# Param_betap_ano = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Param_marm_smooth_ano.csv'
# Weight_betap_ano = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/051_betap_marm_Weights_marm_2p5D_smooth_ano.csv'


# Param_vel_org_smooth = '../../../../Demigration_SpotLight_Septembre2023/063_TS_analytique/060_Param_marm_sm_org.csv'
# Weight_vel_org_smooth = '../../../../Demigration_SpotLight_Septembre2023/063_TS_analytique/060_Weights_marm_sm_org.csv'

# Param_vel_ano_smooth = '../../../../Demigration_SpotLight_Septembre2023/063_TS_analytique/060_Param_marm_sm_ano.csv'
# Weight_vel_ano_smooth = '../../../../Demigration_SpotLight_Septembre2023/063_TS_analytique/060_Weights_marm_sm_ano.csv'

# value_z = np.arange(-88*12,0,1)
# value_x = np.zeros_like(value_z)+3350
# vel_sm_org = vel_in_raypath(Param_vel_ano_smooth, Weight_vel_ano_smooth,value_x,value_z)
# vel_sm_ano = vel_in_raypath(Param_vel_org_smooth, Weight_vel_org_smooth,value_x,value_z)
# plt.figure(figsize=(6,8))
# plt.plot(vel_sm_org,value_z,c='b')
# plt.plot(vel_sm_ano,value_z,c='k')


# '''Extract velocity values from the full model'''
# # vel_ray_org = vel_in_raypath(Param_vel_org_sm, Weight_vel_org_sm, ray_x, ray_z)
# # vel_ray_ano = vel_in_raypath(
# #     Param_vel_ano_sm, Weight_vel_ano_sm, ray_x2, ray_z2)

# # betap_ray_org = vel_in_raypath(Param_betap_org, Weight_betap_org, ray_x, ray_z)
# # betap_ray_ano = vel_in_raypath(
# #     Param_betap_ano, Weight_betap_ano, ray_x2, ray_z2)


# redef_betap_org = betap_ray_org
# redef_betap_ano = betap_ray_ano


# x_disc = np.arange(601)*12.00
# z_disc = np.arange(151)*12.00


# ''' Creation de l'ondelette source '''
# # Parameters of the analytical source wavelet
# nws = 177
# fmax = 25
# nt2 = nt - (nws-1) / 2
# nt_len = int((nt2+1) * 2)
# wsrc_org = defwsrc(fmax, dt, 0, nws)
# nws2 = nws//2

# # wsrc_rot = phase_rot(wsrc_org,np.pi/2)
# time_wsrc = np.arange(ft, -ft, dt)

# plt.figure(figsize=(6, 10))
# plt.plot(wsrc_org, time_wsrc)
# plt.title('Wavelet at 25hz')

# # plt.figure(figsize=(6,10))
# # # wsrc_rot_roll = np.roll(wsrc_rot)
# # plt.plot(wsrc_rot)


# # plt.figure(figsize=(6,10))
# # plt.plot(vel_ray_org[:half_idx],ray_z[:half_idx]/1000)
# # plt.plot(vel_ray_ano[:half_idx],ray_z[:half_idx]/1000)
# # plt.axhline(-tt_at_spot,c='tab:green')
# # plt.title('velocity or amplitude value vs depth')
# # # plt.ylim(-0.9,-1.1)


# # ray_time = ray_tt[:half_idx]*2
# ray_time_org = ray_tt[:half_idx]*2
# ray_time_ano = ray_tt2[:half_idx2]*2


# plt.figure(figsize=(6, 10))
# plt.plot(redef_betap_org[:half_idx], ray_time_org[:half_idx])
# plt.plot(redef_betap_ano[:half_idx2], ray_time_ano[:half_idx2])
# # plt.axhline(tt_at_spot,c='tab:green')
# plt.title('Perturbation vs time')
# plt.gca().invert_yaxis()
# plt.ylabel('time (s)')
# # plt.ylim(-0.9,-1.1)

# plt.figure(figsize=(6, 10))
# plt.plot(vel_ray_org[:half_idx], ray_time_org[:half_idx])
# plt.plot(vel_ray_ano[:half_idx2], ray_time_ano[:half_idx2])
# # plt.axhline(tt_at_spot,c='tab:green')
# plt.title('velocity vs time')
# plt.gca().invert_yaxis()
# plt.ylabel('time (s)')
# # plt.ylim(-0.9,-1.1)


# # '''Interpolation using betap'''

# f = interpolate.interp1d(
#     ray_time_ano[:half_idx2], -betap_ray_ano[:half_idx2], kind='cubic')
# ray_time_int = np.linspace(0, np.max(ray_time_ano), 2000)
# betap_ray_ano_int = f(ray_time_int)

# f = interpolate.interp1d(
#     ray_time_org[:half_idx], -betap_ray_org[:half_idx], kind='cubic')
# betap_ray_org_int = f(ray_time_int)

# convol_time_ano_int = np.convolve(betap_ray_ano_int, wsrc_org, mode='same')
# convol_time_org_int = np.convolve(betap_ray_org_int, wsrc_org, mode='same')

# plt.figure(figsize=(6, 10))
# plt.plot(betap_ray_org_int, ray_time_int, '-', label='org')
# plt.plot(betap_ray_ano_int, ray_time_int, '-', label='ano')
# plt.title('Perturbation vs time')
# plt.legend()
# plt.gca().invert_yaxis()
# plt.ylabel('time (s)')

# plt.figure(figsize=(6, 10))
# plt.plot(convol_time_org_int, ray_time_int, label='org')
# plt.plot(convol_time_ano_int, ray_time_int, label='ano')
# plt.title('Perturbation vs time')
# plt.legend()
# plt.gca().invert_yaxis()
# plt.ylabel('time (s)')


# new_dt = ray_time_int[51]-ray_time_int[50]

# %%
oplen = 400
new_dt = 0.001

# SLD_TS = procs.sliding_TS(
#     convol_time_org_int, convol_time_ano_int, oplen=oplen, si=new_dt, taper=30)

# plt.figure(figsize=(6, 10))
# plt.plot(SLD_TS, ray_time_int, '-')
# plt.title('time-shift vs time ')
# plt.ylabel('time (s)')
# plt.gca().invert_yaxis()


def t_first_non_zero(ray_time, tr1, tr2):
    diff_t = tr1-tr2
    idx = np.where(diff_t > 0.01)[0][0]
    t_frst_non_zero = ray_time[idx]
    return t_frst_non_zero


# oplen = 350
'''Read modelled traces time-shift'''


nt = 1801
at = ft + np.arange(nt)*dt

# Traces modelled with the full model
gather_path_fwi_org = '../output/45_marm_ano_v3/org_1801TL'
gather_path_fwi45 = '../output/45_marm_ano_v3/ano_114_perc_1801TL'


# Traces modelled with the sm5 model
# gather_path_fwi_org = '../output/63_ts_sm5_fwi/org'
# gather_path_fwi45 = '../output/63_ts_sm5_fwi/ano'


# Traces modelled with the sm3 model. The result is not coherent in the adj case
# gather_path_fwi_org = '../output/63_ts_sm3_fwi/org'
# gather_path_fwi45 = '../output/63_ts_sm3_fwi/ano'


print('inv')
tr_binv_fwi_org = trace_from_rt(0, gather_path_fwi_org, p_inv)
tr_binv_fwi_45 = trace_from_rt(0, gather_path_fwi45, p_inv)

print('adj')
tr_badj_fwi_org = trace_from_rt(0, gather_path_fwi_org, p_adj)
tr_badj_fwi_45 = trace_from_rt(0, gather_path_fwi45, p_adj)


# idx_src = find_nearest(p_inv.ax_, p_inv.src_x_[0]/1000)[1]
# idx_src_adj = find_nearest(p_adj.ax_, p_adj.src_x_[0]/1000)[1]
# idx_src =  301

# fl_org = gather_path_fwi_org+'/t1_obs_000'+str(idx_src)+'.dat'
# fl_ano = gather_path_fwi45+'/t1_obs_000'+str(idx_src)+'.dat'

# fl_org_adj = gather_path_fwi_org+'/t1_obs_000'+str(idx_src_adj)+'.dat'
# fl_ano_adj = gather_path_fwi45+'/t1_obs_000'+str(idx_src_adj)+'.dat'


# tr_binv_fwi_45  = -gt.readbin(fl_ano, no, nt).transpose()[:,125]
# tr_binv_fwi_org  = -gt.readbin(fl_org, no, nt).transpose()[:,125]


binv_SLD_TS_fwi = procs.sliding_TS(
    tr_binv_fwi_org, tr_binv_fwi_45, oplen=oplen, si=dt, taper=30)
badj_SLD_TS_fwi = procs.sliding_TS(
    tr_badj_fwi_org, tr_badj_fwi_45, oplen=oplen, si=dt, taper=30)


# conv_t_org_norm = convol_time_org_int/np.min(convol_time_org_int[500:1400])
# conv_t_ano_norm = convol_time_ano_int/np.min(convol_time_ano_int[500:1400])

tr_binv_fwi_org_norm = tr_binv_fwi_org/np.min(tr_binv_fwi_org[500:])
tr_binv_fwi_45_norm = tr_binv_fwi_45/np.min(tr_binv_fwi_45[500:])



# plt.rcParams['font.size'] = 20
# plt.figure(figsize=(7, 12))
# plt.plot(conv_t_org_norm, ray_time_int, '-', label='org')
# plt.plot(conv_t_ano_norm, ray_time_int, '-', label='ano')
# plt.legend()
# plt.title('raytracing traces')
# # plt.gca().invert_yaxis()
# # plt.grid()
# # plt.axhline(-1.19736)
# plt.ylim(1.5, ft)
# plt.xlim(-1, 1)

plt.figure(figsize=(7, 12))
plt.plot(-tr_binv_fwi_org_norm, at, label='org')
plt.plot(-tr_binv_fwi_45_norm, at, label='ano')
plt.title('modelled traces')
plt.legend()
plt.ylim(1.5, ft)
# plt.gca().invert_yaxis()
plt.xlim(-1.4, 1.4)

differe = tr_badj_fwi_org-tr_badj_fwi_45
plt.plot(differe)

plt.figure(figsize=(6, 12))
plt.plot(-tr_badj_fwi_org, at, label='org')
plt.plot(-tr_badj_fwi_45, at, label='ano')
plt.title('modelled traces')
plt.legend()
plt.gca().invert_yaxis()
plt.xlim(-0.1, 0.1)

file = '../time_shift_theorique_full.csv'
# file = '../time_shift_theorique_sm5.csv'


ts_theo = np.array(read_pick(file, 0))
ax_theo = np.array(read_pick(file, 1))

ax_theo_corr = ax_theo - ax_theo[1] + p_inv.tt_ - (ax_theo[-2] - ax_theo[1])

plt.figure(figsize=(6, 12))
plt.plot(binv_SLD_TS_fwi, at, c='tab:purple', label='QTV')
plt.plot(badj_SLD_TS_fwi, at, c='tab:orange', label='STD')
# plt.plot(SLD_TS, ray_time_int, '-', label='RT')
plt.plot(ts_theo, ax_theo_corr, c='tab:green', label='theo')
# plt.axhline(p_inv.tt_)
plt.xlim(-0.5, 5.5)
plt.title('Sliding time-shift vs time')
plt.legend()
plt.gca().invert_yaxis()
