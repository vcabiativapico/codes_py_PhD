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
# from wave2d_ana import defwsrc, ana2d
""" Parameters [km, s]"""

dz = 12. / 1000
dx = 12. / 1000
dt = 1.41e-3
nx = 601
nz = 151
fx = 0.0
fz = 0.0
az = fz + np.arange(nz) * dz
ax = fx + np.arange(nx) * dx
src_x = 200 * dx
rec_x = 400 * dx
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

""" Definition of the source wavelet """
def defwsrc(fmax, dt):
    """
    Definition of the source function
    Ricker wavelet with central frequency fmax/2.5
    Ricker = 2nd-order derivative of a Gaussian function
    """
    fc = fmax / 2.5  # Central frequency
    ns2 = int(2 / fc / dt) + 200 # Classical definition
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
fmax = 1. / 2. / dt
fmin = -fmax
wsrc = defwsrc(25, dt)[0]
nws = len(wsrc)
ns = (nws - 1) // 2
nf = nws
aw2 = 2. * pi * np.fft.fftfreq(nf, dt)
s_p = np.zeros((nx, nz))
p_r = np.zeros((nx, nz))
wsrcf = np.fft.fft(wsrc)

""" Build the absolute distance r table """
# Distance is calculated through Pythagoras theorem
# The calculation is done according to the source and receiver position

for i in range(nx):
    for k in range(nz):
        s_p[i, k] = np.sqrt((src_x - ax[i]) ** 2 + (src_z - az[k]) ** 2)

for i in range(nx):
    for k in range(nz):
        p_r[i, k] = np.sqrt((rec_x - ax[i]) ** 2 + (rec_z - az[k]) ** 2)

# plt.figure(figsize=(16, 10))
# hfig = plt.imshow(s_p.T, extent=[ax[0], ax[-1], az[-1], az[0]])
# plt.title('Distance to the source')
# plt.colorbar(hfig)
#
# plt.figure(figsize=(16, 10))
# hfig = plt.imshow(p_r.T, extent=[ax[0], ax[-1], az[-1], az[0]])
# plt.title('Distance to the receiver')
# plt.colorbar(hfig)

""" Normalization of the velocity with v0 """
v0 = 2.000
delta_v_n = (2 * inp_flat.T) / v0 ** 3


r = 0.2

""" Calculate the hankel function for the source and receiver term """
hk_s = np.zeros((nf))
hk_r = np.zeros((nf))

hk_s = 1j / 4. * hankel1(0., r * aw2 / v0)


# """ Save the hankel function result """
# def write_result(inp,flnam):
#     # Write binary fila on disk (32 bits)
#     with open(flnam,"wb") as fl:
#         inp.astype('float32').tofile(fl)
#
# #
# write_result(hk_r,'./results/hk_r_direct.dat')
# write_result(hk_s,'./results/hk_s_direct.dat')
#
# #%%
# """ Save the hankel function result """
# def read_hankel_result(flnam, nf, nx, nz):
#     with open(flnam, "rb") as fl:
#         im = np.fromfile(fl, dtype=np.float32)
#         im = im.reshape(nf,nx,nz,order='C')
#     return im
#
# # hk_r_in = read_hankel_result('./results/hk_r.dat',nf,nx,nz)
# # hk_s_in = read_hankel_result('./results/hk_s.dat',nf,nx,nz)
#
# hk_r_in = read_hankel_result('./results/hk_r_direct.dat',nf,nx,nz)
# hk_s_in = read_hankel_result('./results/hk_s_direct.dat',nf,nx,nz)

""" Normalization of the velocity with v0 """
v0 = 2.000
delta_v_n = (2 * inp_flat.T) / v0 ** 3

""" Correction of the nan values to zeros """
hk_s[0] = 0
# hk_s0 = np.nan_to_num(hk_s)

# hk_r0 = np.nan_to_num(hk_r_in)
#
# """Test"""
# P = np.zeros((nx, nz, nt))
# nt = 100
# for it in range(1, nt):
#     for ix in range(1, nx - 1):
#         for iz in range(1, nz - 1):
#             P[ix, iz, it + 1] = 2 * P[ix, iz, it] - P[ix, iz, it - 1] + (v0 * dt / dx)**2 * hankel_function * (
#             P[ix + 1, iz, it] - 2 * P[ix, iz, it] + P[ix - 1, iz, it]) + (v0 * dt / dz)**2 * hankel_function * (
#             P[ix, iz + 1, it] - 2 * P[ix, iz, it] + P[ix, iz - 1, it])
#

# """ Integral calculation """
# # Calculation of the values inside the integral
# mat_in_gral = np.zeros((nf, nx, nz))
# for i in range(nf):
#     mat_in_gral[i, :, :] = hk_s0[i] * delta_v_n * hk_r0[i]

#
#
# # Integral sum over the x values
# gral_x = np.zeros((nf, nz))
# for i in range(nz):
#     gral_x[:,i] = np.sum(mat_in_gral[:, i, :])
#
# # Integral sum over the x and z values
# gral_x_z = np.zeros(nf)
# for i in range(nf):
#     gral_x_z[i] = np.sum(gral_x[i])

wsrcf = np.fft.fft(wsrc)
dp_f = hk_s * wsrcf * -(aw2**2)

dp_t = np.real(np.fft.ifft(dp_f))




dp_ana_test = np.real(np.fft.ifft(mat_in_gral))
plt.figure(figsize=(16, 10))
idx = 1
hfig = plt.imshow(dp_ana_test[idx, :, :].T, extent=[ax[0], ax[-1], az[-1], az[0]])
plt.title('Before integration TIME = '+str(aw2[idx]/(2*pi)))
plt.colorbar(hfig)
plt.show()


plt.figure(figsize=(16, 10))
hfig = plt.imshow(s_p.T, extent=[ax[0], ax[-1], az[-1], az[0]])
plt.title('Distance to the source')
plt.colorbar(hfig)
plt.show()

plt.figure(figsize=(16, 10))
hfig = plt.imshow(p_r.T, extent=[ax[0], ax[-1], az[-1], az[0]])
plt.title('Distance to the receiver')
plt.colorbar(hfig)
plt.show()


plt.figure()
plt.imshow(hk_r_in[1,:,:].T*hk_s_in[1,:,:].T, extent=[ax[0], ax[-1], az[-1], az[0]])
plt.title('Hankel for the receiver')
plt.show()

plt.figure()
plt.imshow(hk_r_in[1,:,:].T, extent=[ax[0], ax[-1], az[-1], az[0]])
plt.title('Hankel for the receiver')
plt.show()

plt.figure(figsize=(16, 10))
idx = 1
hfig = plt.imshow(mat_in_gral[idx, :, :].T, extent=[ax[0], ax[-1], az[-1], az[0]])
plt.title('Before integration FREQ = '+str(aw2[idx]/(2*pi)))
plt.colorbar(hfig)
plt.show()

plt.figure(figsize=(16, 10))
plt.plot(aw2/(2*pi), abs(dp_f))
# plt.scatter(main_f,np.max(dp_f),c='r')
plt.title('dp in frequency domain')
plt.show()

plt.figure(figsize=(16, 10))
plt.plot(dp_t/np.max(dp_t))
plt.plot(wsrc)
plt.title('dp in frequency domain')
plt.show()


#%% Tests
