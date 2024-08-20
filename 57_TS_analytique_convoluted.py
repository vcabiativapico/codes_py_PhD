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
from spotfunk.res import procs, visualisation

from scipy.interpolate import interpolate
import sympy as sp
import tqdm

import functions_bsplines_new_kev_test_2_5D as kvbsp


# Global parameters
labelsize = 16
nt = 1801
dt = 1.14e-3
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


# %%
gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'


# path_adj = gen_path + '048_sm8_correction_new_solver/STD/depth_demig_out/STD/results/depth_demig_output.csv'
# path_inv = gen_path + '048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/results/depth_demig_output.csv'
# path_inv = gen_path + '055_sm15_marm/depth_demig_out/QTV/results/depth_demig_output.csv'

# Results from source to spot located after anomaly in the smooth 15 model
path_inv = gen_path + '056_correct_TS_deep/depth_demig_out/050_TS_analytiquedeep_2024-07-05_11-53-25/results/depth_demig_output.csv'
path_adj = gen_path + '056_correct_TS_deep/depth_demig_out/050_TS_analytiquedeep_badj_2024-07-12_16-57-27/results/depth_demig_output.csv'

# Results from source to spot located after anomaly in the constant 2000m/s model with slope 37
# path_inv = gen_path + '057_const_TS_deep/depth_demig_out/050_TS_constant_2024-07-08_10-10-47/results/depth_demig_output.csv'

# Results from the source to spot located after anomaly in the constant model with slope 40
# path_inv = gen_path + '062_panneau/depth_demig_out/panneau42_2024-07-09_15-15-27/results/depth_demig_output.csv'


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


'''Read the velocity model '''

# fl1 = '../input/45_marm_ano_v3/fwi_org.dat'
# fl2 = '../input/45_marm_ano_v3/fwi_ano_114_percent.dat'


fl1 = '../input/45_marm_ano_v3/fwi_sm.dat'
fl2 = '../input/50_ts_model/marmousi_ano_sm.dat'


inp_org = gt.readbin(fl1, nz, nx)
inp_ano = gt.readbin(fl2, nz, nx)


# inp_

'''Read the raypath'''
# spot lancé dans le modele 3510,-1210

# path_ray2 = gen_path + '056_correct_TS_deep/depth_demig_out/050_TS_analytiquedeep_2024-07-05_11-53-25/rays/ray_0.csv'
# path_ray = gen_path + '054_TS_deeper/depth_demig_out/050_TS_analytiquedeep_2024-07-02_15-30-30/rays/ray_0.csv'

# path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/052_TS_deep/depth_demig_out/052_TS_analytique_deep_2024-06-20_12-02-07/rays/ray_0.csv'

path_ray2 = gen_path + \
    '057_const_TS_deep/depth_demig_out/050_TS_constant_2024-07-08_10-10-47/rays/ray_0.csv'


path_ray = gen_path + \
    '057_const_TS_deeper/depth_demig_out/050_TS_const_deeper_2024-07-08_11-08-01/rays/ray_0.csv'


# path_ray2 = gen_path + '062_panneau/depth_demig_out/panneau42_2024-07-09_15-15-27/rays/ray_0.csv'

ray_x = np.array(read_results(path_ray, 0))
ray_z = np.array(read_results(path_ray, 2))
ray_tt = np.array(read_results(path_ray, 8))

ray_x2 = np.array(read_results(path_ray2, 0))
ray_z2 = np.array(read_results(path_ray2, 2))
ray_tt2 = np.array(read_results(path_ray2, 8))


half_idx = len(ray_x)//2
half_idx2 = len(ray_x2)//2

hmin = 1.5
hmax = 4.5
%matplotlib inline
# %matplotlib qt5

spot_x = 3366.6
spot_z = 1024.3

spot_x = p_inv.spot_x_[0]
spot_z = -p_inv.spot_z_[0]

plot_model(inp_org, hmin, hmax)
plt.scatter(spot_x/1000, spot_z/1000, c='w', s=1)


idx_tt_x = find_nearest(ray_x[:half_idx], spot_x)[1]
idx_tt_z = find_nearest(ray_z[:half_idx], -spot_z)[1]
idx_tt = (idx_tt_x + idx_tt_z)//2


nb_to_idx = 45
# nb_to_idx = 25

tt_at_spot = ray_tt2[idx_tt+nb_to_idx]*2


ray_x_corr = np.append(ray_x2[:half_idx2], ray_x[idx_tt+nb_to_idx:half_idx])
ray_z_corr = np.append(ray_z2[:half_idx2], ray_z[idx_tt+nb_to_idx:half_idx])
ray_tt_corr = np.append(ray_tt2[:half_idx2], ray_tt[idx_tt+nb_to_idx:half_idx])

fig1 = plot_model(inp_ano, hmin, hmax)
plt.plot(ray_x[idx_tt:half_idx]/1000, -ray_z[idx_tt:half_idx]/1000, '-')
plt.scatter(spot_x/1000, spot_z/1000, c='w', s=1)


plt.plot(ray_x[idx_tt+nb_to_idx:half_idx], ray_z[idx_tt+nb_to_idx:half_idx])
plt.plot(ray_x2[:half_idx2], ray_z2[:half_idx2])

fig1 = plot_model(inp_ano, hmin, hmax)
plt.plot(ray_x_corr/1000, -ray_z_corr/1000, '-')
plt.scatter(spot_x/1000, spot_z/1000, c='w', s=1)

plt.figure(figsize=(10, 8))
plt.plot(ray_tt_corr, '.')
plt.plot(ray_tt2[:half_idx2], '.')
plt.plot(ray_tt[idx_tt+nb_to_idx:half_idx], '.')
plt.ylim(0.65, 0.75)

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


# Values from velocity model in the constant model

# Param_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/057_const_TS_deep/056_Param_marm_org.csv'
# Weight_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/057_const_TS_deep/056_Weights_marm_org.csv'

# Param_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/057_const_TS_deep/056_Param_marm_ano.csv'
# Weight_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/057_const_TS_deep/056_Weights_marm_ano.csv'

# ## Values from velocity model in 2000 m/s model

# # Param_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/060_flat_TS/057_Param_marm_const_org.csv'
# # Weight_vel_org_sm = '../../../../Demigration_SpotLight_Septembre2023/060_flat_TS/057_Weights_marm_const_org.csv'

# # Param_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/060_flat_TS/057_Param_marm_const_ano.csv'
# # Weight_vel_ano_sm = '../../../../Demigration_SpotLight_Septembre2023/060_flat_TS/057_Weights_marm_const_ano.csv'

# ## Values from pertubation model in 2000 m/s model

# Param_betap_org = '../../../../Demigration_SpotLight_Septembre2023/057_const_TS_deep/056_Param_marm_betap_org.csv'
# Weight_betap_org = '../../../../Demigration_SpotLight_Septembre2023/057_const_TS_deep/056_Weights_marm_betap_org.csv'

# Param_betap_ano = '../../../../Demigration_SpotLight_Septembre2023/057_const_TS_deep/056_Param_marm_betap_ano.csv'
# Weight_betap_ano = '../../../../Demigration_SpotLight_Septembre2023/057_const_TS_deep/056_Weights_marm_betap_ano.csv'


'''Extract velocity values from the full model'''
vel_ray_org = vel_in_raypath(
    Param_vel_org_sm, Weight_vel_org_sm, ray_x_corr, ray_z_corr)
vel_ray_ano = vel_in_raypath(
    Param_vel_ano_sm, Weight_vel_ano_sm, ray_x_corr, ray_z_corr)

betap_ray_org = vel_in_raypath(
    Param_betap_org, Weight_betap_org, ray_x_corr, ray_z_corr)
betap_ray_ano = vel_in_raypath(
    Param_betap_ano, Weight_betap_ano, ray_x_corr, ray_z_corr)

x_disc = np.arange(601)*12.00
z_disc = np.arange(151)*12.00


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

plt.figure(figsize=(6, 10))
# wsrc_rot_roll = np.roll(wsrc_rot)
plt.plot(wsrc_rot)


file = '../time_shift_theorique.csv'
ts_theorique = [np.array(read_pick(file, 0)), np.array(read_pick(file, 1))]


# plt.figure(figsize=(6,10))
# plt.plot(vel_ray_org[:half_idx],ray_z[:half_idx]/1000,'.')
# plt.plot(vel_ray_ano[:half_idx],ray_z[:half_idx]/1000,'.')
# plt.axhline(-tt_at_spot,c='tab:green')
# plt.title('velocity or amplitude value vs depth')
# # plt.ylim(-0.9,-1.1)


# ray_time = ray_tt[:half_idx]*2
ray_time = ray_tt_corr*2

plt.figure(figsize=(6, 10))
plt.plot(betap_ray_org[:half_idx], ray_time[:half_idx])
plt.plot(betap_ray_ano[:half_idx], ray_time[:half_idx])
plt.axhline(tt_at_spot, c='tab:green')
plt.title('Perturbation vs time')
plt.gca().invert_yaxis()
plt.ylabel('time (s)')
# plt.ylim(-0.9,-1.1)

plt.figure(figsize=(6, 10))
plt.plot(vel_ray_org[:half_idx], ray_time[:half_idx])
plt.plot(vel_ray_ano[:half_idx], ray_time[:half_idx])
plt.axhline(tt_at_spot, c='tab:green')
plt.title('velocity vs time')
plt.gca().invert_yaxis()
plt.ylabel('time (s)')
# plt.ylim(-0.9,-1.1)

'''Interpolation using betap'''

f = interpolate.interp1d(
    ray_time[:half_idx], -betap_ray_org[:half_idx], kind='cubic')
ray_time_org_int = np.arange(0, np.max(ray_time[:half_idx]), dt)
betap_ray_org_int = f(ray_time_org_int)

f = interpolate.interp1d(
    ray_time[:half_idx], -betap_ray_ano[:half_idx], kind='cubic')
ray_time_ano_int = np.arange(0, np.max(ray_time[:half_idx]), dt)
betap_ray_ano_int = f(ray_time_ano_int)


plt.figure(figsize=(6, 10))
plt.plot(betap_ray_org_int, -ray_time_org_int)
plt.plot(betap_ray_ano_int, -ray_time_ano_int)
plt.axhline(-tt_at_spot, c='tab:green')
plt.title('perturbation interpolated vs time ')
plt.ylabel('time (s)')


'''std convolution'''


convol_time_ano_int = np.convolve(betap_ray_ano_int, wsrc_org, mode='same')
convol_time_org_int = np.convolve(betap_ray_org_int, wsrc_org, mode='same')[
    :len(convol_time_ano_int)]


def t_first_non_zero(ray_time, diff, val):
    idx = np.where(diff > val)[0][0]
    t_frst_non_zero = ray_time[idx]
    return t_frst_non_zero


# %%
oplen = 500


'''Read modelled traces time-shift'''


nt = 1801
at = ft + np.arange(nt)*dt


# Modelled traces with Born in the full perturbation model
# gather_path_fwi_org = '../output/47_marm2/org'
# gather_path_fwi45 = '../output/47_marm2/ano'

# Modelled traces with FWI in the full perturbation model
gather_path_fwi_org = '../output/45_marm_ano_v3/org_1801TL'
gather_path_fwi45 = '../output/45_marm_ano_v3/ano_114_perc_1801TL'

# Modelled traces with FWI in the constant model
# gather_path_fwi_org = '../output/48_const_2000_ano/org'
# gather_path_fwi45 = '../output/48_const_2000_ano/ano'

idx_src_inv = find_nearest(p_inv.ax_, p_inv.src_x_[0]/1000)[1]
idx_src_adj = find_nearest(p_adj.ax_, p_adj.src_x_[0]/1000)[1]
# idx_src = 232

fl_org = gather_path_fwi_org+'/t1_obs_000'+str(idx_src_inv)+'.dat'
fl_ano = gather_path_fwi45+'/t1_obs_000'+str(idx_src_inv)+'.dat'

fl_org_adj = gather_path_fwi_org+'/t1_obs_000'+str(idx_src_adj)+'.dat'
fl_ano_adj = gather_path_fwi45+'/t1_obs_000'+str(idx_src_adj)+'.dat'


tr_binv_fwi_org = -gt.readbin(fl_org, no, nt).transpose()[:, 125]
tr_binv_fwi_45 = -gt.readbin(fl_ano, no, nt).transpose()[:, 125]

tr_badj_fwi_org = -gt.readbin(fl_org_adj, no, nt).transpose()[:, 125]
tr_badj_fwi_45 = -gt.readbin(fl_ano_adj, no, nt).transpose()[:, 125]


diff_tr_mod = tr_binv_fwi_org-tr_binv_fwi_45
diff_vel_art = vel_ray_org-vel_ray_ano
diff_betap_art = betap_ray_org-betap_ray_ano
diff_convol_art = convol_time_ano_int-convol_time_org_int

f_no_tr_mod = t_first_non_zero(at, -diff_tr_mod, 5e-5)
f_no_vel = t_first_non_zero(ray_time, diff_vel_art, 0.05)
f_no_betap = t_first_non_zero(ray_time, diff_betap_art, 0.05)
f_no_conv = t_first_non_zero(ray_time_org_int, -diff_convol_art, 5)


binv_SLD_TS_fwi = procs.sliding_TS(
    tr_binv_fwi_org, tr_binv_fwi_45, oplen=oplen, si=dt, taper=30)
badj_SLD_TS_fwi = procs.sliding_TS(
    tr_badj_fwi_org, tr_badj_fwi_45, oplen=oplen, si=dt, taper=30)


SLD_TS = procs.sliding_TS(
    convol_time_org_int, convol_time_ano_int, oplen=oplen, si=dt, taper=30)


diff_to_correct = f_no_betap - f_no_conv


plt.figure(figsize=(6, 12))
plt.plot(diff_convol_art, ray_time_org_int[:len(convol_time_ano_int)])
plt.title('Difference modelled traces', fontsize=15)
# plt.axhline(p_inv.tt_,c='tab:green')
plt.axhline(f_no_betap, c='tab:orange')
plt.axhline(f_no_tr_mod, c='tab:purple')
plt.ylim(ft, ray_time_ano_int[-1])
plt.gca().invert_yaxis()


plt.figure(figsize=(6, 12))
plt.plot(diff_tr_mod*1000, at)
plt.title('Difference modelled traces', fontsize=15)
# plt.axhline(p_inv.tt_,c='tab:green')
plt.axhline(f_no_betap, c='tab:orange')
plt.axhline(f_no_tr_mod, c='tab:purple')
plt.ylim(ft, ray_time_ano_int[-1])
plt.gca().invert_yaxis()


plt.rcParams['font.size'] = 20
plt.figure(figsize=(6, 12))
plt.plot(convol_time_org_int, ray_time_org_int[:len(
    convol_time_ano_int)], label='org')
plt.plot(convol_time_ano_int, ray_time_ano_int, label='ano')
plt.axhline(p_inv.tt_[0], c='tab:green')
plt.ylim(ft, ray_time_ano_int[-1])
plt.title('Traces from theoretical migration', fontsize=15)
plt.ylabel('time (s)')
plt.legend(loc='upper right', fontsize=18)
plt.gca().invert_yaxis()


plt.figure(figsize=(6, 12))
plt.plot(tr_binv_fwi_org, at-ft)
plt.plot(tr_binv_fwi_45, at-ft)
# plt.plot(diff,p_inv.at_)
plt.axhline(p_inv.tt_, c='tab:green', label='RT time')
# plt.axhline(f_no_betap,c='tab:green',label='RT time')
plt.axhline(f_no_tr_mod, c='tab:purple', label='First 0 mod')
plt.title('Traces from modeling', fontsize=15)
plt.legend(loc='upper right', fontsize=18)
plt.xlim(-0.05, 0.05)
plt.ylim(ft, ray_time_ano_int[-1])
plt.ylabel('time (s)')
plt.gca().invert_yaxis()


plt.figure(figsize=(6, 12))
plt.plot(SLD_TS, ray_time_org_int+diff_to_correct, c='tab:purple')
plt.plot(binv_SLD_TS_fwi, at, c='tab:orange', label='QTV')
plt.plot(badj_SLD_TS_fwi, at, c='tab:blue', label='STD')
# plt.plot(ts_theorique[0],ts_theorique[1],c='tab:green')
plt.title('Sliding time-shift vs time')
plt.legend()
plt.gca().invert_yaxis()


# %%

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
