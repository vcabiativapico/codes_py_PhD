#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 10:23:39 2024
@author: vcabiativapico
"""
import numpy as np
from math import sin, cos, pi, sqrt, log, atan, log10, exp
import matplotlib.pyplot as plt
# from matplotlib import rc
# import geophy_tools as gt
from scipy.special import hankel1, hankel2
from tqdm import tqdm
from scipy.signal import hilbert
# from wave2d_ana import defwsrc, ana2d
""" Parameters [km, s]"""

dz = 12. / 1000
dx = 12. / 1000
dt = 1.41e-3

dz = 3. / 1000
dx = 3. / 1000
dt = 0.35e-3
do = dx
nx = 601
nz = 151
no = 251
nt = 1501
fx = 0.0
fz = 0.0
fo = -(no-1) / 2 * do
ft = -100.11e-3
az = fz + np.arange(nz) * dz
ax = fx + np.arange(nx) * dx
ao = fo + np.arange(no) * do
at = ft + np.arange(nt) * dt

src_x = 301 * dx
rec_x = 301 * dx
src_z = 2 * dz
rec_z = 2 * dz



""" Reads the velocity model """
def readbin(flnam,nz,nx):
    # Read binary file (32 bits)
    with open(flnam,"rb") as fl:
        im = np.fromfile(fl, dtype=np.float32)
    im = im.reshape(nz,nx,order='F')
    return im
fl1 = '../../input/31_const_flat_tap/inp_flat_2050_const.dat'
inp1 = readbin(fl1, nz, nx)

fl2 = '../../input/30_marm_flat/2_0_sm_constant.dat'
inp2 = readbin(fl2, nz, nx)
# Calculates delta_v
inp_flat = inp1 - inp2

fl3 = '../../output/32_2050_flat_tap/t1_obs_000301.dat'
inp3 = readbin(fl3, no, nt)

fl4 = '../../output/t1_obs_000301.dat'
inp4 = readbin(fl4, no, nt)

plt.figure(figsize=(10,8))
plt.plot(inp3[:,125])

plt.show()
""" Definition of the source wavelet """
def defwsrc(fmax, dt):
    """
    Definition of the source function
    Ricker wavelet with central frequency fmax/2.5
    Ricker = 2nd-order derivative of a Gaussian function
    """
    fc = fmax / 2.5  # Central frequency
    ns2 = int(2 / fc / dt) + 1500  # Classical definition
    # ns2 = int(7.5/fmax/dt+0.5) # Large source, but better for inverse WWW
    # ns2 = 4 * int(3.0/2.5/fc/dt + 0.49)
    ns = 1 + 2 * ns2  # Size of the source

    wsrc = np.zeros(ns)
    aw = np.zeros(ns)  # Time axis
    for it in range(ns):
        a1 = float(it - ns2) * fc * dt * pi
        a2 = a1 ** 2
        wsrc[it] = (1 - 2 * a2) * exp(-a2)
        aw[it] = float(it - ns2) * dt
    return wsrc, aw

# Parameters of the source
wsrc = defwsrc(25, dt)[0]
nws = len(wsrc)
nf = nws
aw2 = 2. * pi * np.fft.fftfreq(nf, dt)
wsrcf = np.fft.fft(wsrc)

plt.plot(wsrc)
plt.show()
""" Build the absolute distance r table """
# Distance is calculated through Pythagoras theorem
# The calculation is done according to the source and receiver position
def distance_table(pos_x,pos_z):
    r = np.zeros((nx, nz))
    for i in range(nx):
        for k in range(nz):
            r[i, k] = np.sqrt((pos_x - ax[i]) ** 2 + (pos_z - az[k]) ** 2)
    return r

s_p = distance_table(src_x,src_z)
p_r = distance_table(rec_x,rec_z)


def plot_table(input):
    plt.figure(figsize=(10,8))
    plt.imshow(input.T, extent=[ax[0], ax[-1], az[-1], az[0]])
    plt.title('Distance')
    # plt.xlabel('')
    plt.show()


plot_table(p_r)
plot_table(s_p)



""" Normalization of the velocity with v0 """
t = np.arange(len(wsrc)) * dt
v0 = 2.000
delta_v_n = (2 * inp_flat.T) / v0 ** 3

# #%%
hk = np.zeros((nf),dtype='complex')
#
hk[0] = 0
for ix in tqdm(range(0,nx)):
    for iz in range(51,99):
        hk += (1j / 4. * hankel1(0., s_p[ix, iz] * aw2 / v0)) * delta_v_n[ix, iz]\
                * (1j / 4. * hankel1(0., p_r[ix, iz] * aw2 / v0))

#
#
#
hk[0] = 0
wsrcf = np.fft.fft(wsrc)

dp_f = hk * wsrcf * -(aw2**2)

dp_t = np.real(np.fft.ifft(dp_f))

# #
# # plt.figure()
# # plt.plot(np.real(dp_f))
# # plt.title('Hankel summed before')
# # plt.show()
#
#
#
# idx_max_wsrc = np.where(wsrc == np.max(wsrc))
# idx_max_dp_t = np.where(dp_t == np.max(dp_t))
#
# shift = idx_max_wsrc[0] - idx_max_dp_t
# print(shift)
# shift = int(np.mean(shift))
# print(shift)
#
# roll = idx_max_wsrc[0]
# # roll_wsrc = idx_max_wsrc[0]
# dp_t_norm = dp_t/np.max(abs(dp_t))
# dp_t_norm_inv = dp_t_norm[::-1]
#
# corr_wsrc = np.roll(wsrc,-roll)
# corr_dp_t_norm = np.roll(dp_t_norm_inv,-roll)
#
# trace = inp3[125]/np.max(abs(inp3[125]))
#
# plt.figure(figsize=(16, 10))
# plt.plot(t,dp_t_norm)
# plt.plot(t,wsrc)
# # plt.scatter(idx_max_wsrc[0],0)
# plt.title('dp in frequency domain')
# plt.show()
#
# plt.figure(figsize=(16, 10))
# plt.plot(corr_dp_t_norm)
# plt.plot(corr_wsrc)
# plt.show()
#
# plt.figure(figsize=(16, 10))
# plt.plot(t,corr_dp_t_norm)
# plt.plot(at,-trace)
# plt.show()



#%%

""" Calculate the hankel function for the source and receiver term """
hk_s = np.zeros((nf, nx, nz),dtype='complex')
hk_r = np.zeros((nf, nx, nz),dtype='complex')
for i in tqdm(range(nx)):
    for k in range(nz):
        hk_s[:, i, k] = 1j / 4. * hankel1(0., s_p[i, k] * aw2 / v0)
        hk_r[:, i, k] = 1j / 4. * hankel1(0., p_r[i, k] * aw2 / v0)

hk_v = hk_s * delta_v_n * hk_r
hk_v0 = np.nan_to_num(hk_v)

in_gral_sum = np.sum(np.sum(hk_v0, axis=1),axis=1)

dp_f_v = in_gral_sum * wsrcf * -(aw2**2)

dp_t_v_new = np.real(np.fft.ifft(dp_f_v))





idx_max_wsrc = np.where(wsrc == np.max(wsrc))
idx_max_dp_t = np.where(dp_t_v == np.max(dp_t_v))

shift = idx_max_wsrc[0] - idx_max_dp_t
print(shift)
shift = int(np.mean(shift))
print(shift)

roll = idx_max_wsrc[0]
# roll_wsrc = idx_max_wsrc[0]
dp_t_v_norm = dp_t_v/np.max(abs(dp_t_v))
dp_t_v_norm_inv = dp_t_v_norm[::-1]

corr_wsrc = np.roll(wsrc,-roll)
corr_dp_t_norm = np.roll(dp_t_v_norm_inv,-roll)

trace = inp3[125]/np.max(abs(inp3[125]))
trace_new = inp4[125]/np.max(abs(inp4[125]))

def modifsrc(wsrc,fact):
    """Modification of the source"""
    wsrcf = np.fft.rfft(wsrc,axis=-1)
    n     = len(wsrcf)
    # fact  = np.pi/4
    wsrcf *= np.exp(1j*fact)
    wsrc2 = np.fft.irfft(wsrcf,axis=-1)
    return wsrc2

fact =np.pi/2
corr_dp_t_norm_mod = modifsrc(corr_dp_t_norm,fact )
corr_dp_t_norm_mod = corr_dp_t_norm_mod / np.max(abs(corr_dp_t_norm_mod ))


hilb_new = hilbert(corr_dp_t_norm)
hilb_new = hilb_new.imag

plt.figure(figsize=(16, 10))
plt.plot(t,dp_t_v_norm)
plt.plot(t,wsrc)
# plt.scatter(idx_max_wsrc[0],0)
plt.title('dp in frequency domain')
plt.show()

plt.figure(figsize=(16, 10))
plt.plot(t,corr_dp_t_norm)
plt.plot(t,corr_wsrc)
plt.show()

plt.figure(figsize=(16, 10))
plt.plot(t, corr_dp_t_norm,label='hankel')
plt.plot(at,-trace,label='modelisation')
# plt.plot(t,hilb_new)
plt.xlim(0.3,1.5)
plt.legend()
plt.show()


plt.figure(figsize=(16, 10))
plt.plot(dp_t_v_norm,label='hankel')
plt.plot(-trace,label='modelisation')
plt.plot(-trace_new,label='modelisation')
# plt.plot(t,hilb_new)
# plt.xlim(0.3,1.5)
plt.legend()
plt.show()


""""""
idx = np.argmax(abs(trace)) - np.argmax(abs(dp_t_v_norm))
dp_test = np.roll(dp_t_v_norm, idx)
plt.plot(dp_test, label='hankel')
plt.plot(-trace, label='modelisation')

plt.xlim(400, 600)
plt.legend()
plt.show()








dp_t_v_norm_new = dp_t_v_new / np.max(abs(dp_t_v_new ))
idx_new = np.argmax(abs(trace_new)) - np.argmax(abs(dp_t_v_norm_new))
dp_test_new = np.roll(dp_t_v_norm_new, idx_new)
plt.plot(dp_test_new, label='hankel')
plt.plot(-trace_new, label='modelisation')
# plt.xlim(0, 200)
plt.legend()
plt.show()


# dp_t_v_new_norm = dp_t_v/np.max(abs(dp_t_v))
# idx = np.argmax(abs(trace_new)) - np.argmax(abs(dp_t_v_norm))
#
# dp_test = np.roll(dp_t_v_norm, idx)
#
# plt.plot(dp_test, label='hankel')
# plt.plot(trace_new, label='modelisation')
# plt.xlim(0, 250)
# plt.legend()
# plt.show()









#%%
# in_hk_sp = np.zeros((nf,nx,nz),dtype='complex')
# in_hk_pr = np.zeros((nf,nx,nz),dtype='complex')
#
# for i in tqdm(range(1, nx)):
#     for k in range(1, nz):
#         in_hk_sp[:, i, k] = 1j / 4. * s_p[i, k] * aw2 / v0
#         in_hk_pr[:, i, k] = 1j / 4. * p_r[i, k] * aw2 / v0
#
# out_hk_sp = 1j / 4. * hankel1(0., in_hk_sp)
# out_hk_pr = 1j / 4. * hankel1(0., in_hk_pr)
#
#
# out_hk_sp0 = np.nan_to_num(out_hk_sp)
# out_hk_pr0 = np.nan_to_num(out_hk_pr)
#
# mat_in_gral = np.zeros((nf, nx, nz), dtype='complex')
# for i in range(nf):
#     mat_in_gral[i, :, :] = out_hk_sp0[i] * delta_v_n * out_hk_pr0[i]
#
#
# hk_vec_x = np.sum(out_hk_sp0[:, :, :]* out_hk_pr0[:, :, :]*delta_v_n[np.newaxis, :, :], axis=1)
# hk_vec_x_z = np.sum(hk_vec_x, axis=1)
#
# hk_vec = np.sum(np.sum(out_hk_sp0[:, :, :]* out_hk_pr0[:, :, :]*delta_v_n[np.newaxis, :, :],\
#                        axis=1),axis=1)
#
# wsrcf = np.fft.fft(wsrc)
#
# dp_f_vec = hk_vec * wsrcf * -(aw2**2)
#
# dp_t_vec = np.real(np.fft.ifft(dp_f_vec))
#
#
#
# plt.figure(figsize=(16, 10))
# plt.plot(dp_t_vec/np.max(dp_t_vec))
# plt.plot(wsrc)
# plt.title('dp in frequency domain')
# plt.show()
#
#
# #%%
# """ Save the hankel function result """
# # def write_result(inp,flnam):
# #     # Write binary fila on disk (32 bits)
# #     with open(flnam,"wb") as fl:
# #         inp.astype('float32').tofile(fl)
# #
# # #
# # write_result(hk_r,'./results/hk_r.dat')
# # write_result(hk_s,'./results/hk_s.dat')
# #
# # #%%
# # """ Save the hankel function result """
# # def read_hankel_result(flnam, nf, nx, nz):
# #     with open(flnam, "rb") as fl:
# #         im = np.fromfile(fl, dtype=np.float32)
# #         im = im.reshape(nf,nx,nz,order='C')
# #     return im
# #
# # # hk_r_in = read_hankel_result('./results/hk_r.dat',nf,nx,nz)
# # # hk_s_in = read_hankel_result('./results/hk_s.dat',nf,nx,nz)
# #
# # hk_r_in = read_hankel_result('./results/hk_r.dat',nf,nx,nz)
# # hk_s_in = read_hankel_result('./results/hk_s.dat',nf,nx,nz)
#
#
# """ Correction of the nan values to zeros """
#
# # hk_s[0] = 0
# # hk_r[0] = 0
# hk_s0 = np.nan_to_num(hk_s)
# hk_r0 = np.nan_to_num(hk_r)


""" Integral calculation """
# Calculation of the values inside the integral
# mat_in_gral = np.zeros((nf, nx, nz),dtype='complex')
#
# for i in range(nf):
#     mat_in_gral[i, :, :] = hk_s0[i] * delta_v_n * hk_r0[i]
#
#
#
# # Integral sum over the x values
# gral_x = np.zeros((nf, nz),dtype='complex')
# for i in range(nz):
#     gral_x[:,i] = np.sum(mat_in_gral[:, i, :])
#
# # Integral sum over the x and z values
# gral_x_z = np.zeros(nf,dtype='complex')
# for i in range(nf):
#     gral_x_z[i] = np.sum(gral_x[i])
#
# wsrcf = np.fft.fft(wsrc)
# dp_f = gral_x_z * wsrcf * -(aw2**2)
#
# dp_t = np.real(np.fft.ifft(dp_f))

#
# dp_ana_test = np.real(np.fft.ifft(mat_in_gral))
# plt.figure(figsize=(16, 10))
# idx = 1
# hfig = plt.imshow(dp_ana_test[idx, :, :].T, extent=[ax[0], ax[-1], az[-1], az[0]])
# plt.title('Before integration TIME = '+str(aw2[idx]/(2*pi)))
# plt.colorbar(hfig)
# plt.show()
#
#
# plt.figure(figsize=(16, 10))
# hfig = plt.imshow(s_p.T, extent=[ax[0], ax[-1], az[-1], az[0]])
# plt.title('Distance to the source')
# plt.colorbar(hfig)
# plt.show()
#
# plt.figure(figsize=(16, 10))
# hfig = plt.imshow(p_r.T, extent=[ax[0], ax[-1], az[-1], az[0]])
# plt.title('Distance to the receiver')
# plt.colorbar(hfig)
# plt.show()
#
#
# plt.figure()
# plt.imshow(np.real(hk_r[1,:,:].T * hk_s[1,:,:].T), extent=[ax[0], ax[-1], az[-1], az[0]])
# plt.title('Hankel for the receiver')
# plt.show()
#
# plt.figure()
# plt.imshow(np.real(hk_r[1,:,:].T), extent=[ax[0], ax[-1], az[-1], az[0]])
# plt.title('Hankel for the receiver')
# plt.show()
#
# plt.figure(figsize=(16, 10))
# idx = 5
# hfig = plt.imshow(np.real(mat_in_gral[idx, :, :].T), extent=[ax[0], ax[-1], az[-1], az[0]])
# plt.title('Before integration FREQ = '+str(aw2[idx]/(2*pi)))
# plt.colorbar(hfig)
# plt.show()
#
# plt.figure(figsize=(16, 10))
# plt.plot(aw2/(2*pi), abs(dp_f))
# # plt.scatter(main_f,np.max(dp_f),c='r')
# plt.title('dp in frequency domain')
# plt.show()
#
# plt.figure(figsize=(16, 10))
# plt.plot(dp_t/np.max(dp_t))
# plt.plot(wsrc)
# plt.title('dp in frequency domain')
# plt.show()
#

#%% Tests

# idx_main_f = np.where(dp_f == np.max(dp_f))
# main_f = aw2[idx_main_f] / (2*pi)