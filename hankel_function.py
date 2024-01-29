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

def readbin(flnam,nz,nx):
    # Read binary file (32 bits)
    with open(flnam,"rb") as fl:
        im = np.fromfile(fl, dtype=np.float32)
    im = im.reshape(nz,nx,order='F')
    return im




fl1 = '../input/31_const_flat_tap/inp_flat_2050_const.dat'
inp1 = readbin(fl1, nz, nx)
fl2 = '../input/30_marm_flat/2_0_sm_constant.dat'
inp2 = readbin(fl2, nz, nx)
inp_flat = inp1 - inp2

def defwsrc(fmax, dt):
    """
    Definition of the source function
    Ricker wavelet with central frequency fmax/2.5
    Ricker = 2nd-order derivative of a Gaussian function
    """
    fc = fmax / 2.5  # Central frequency
    ns2 = int(2 / fc / dt)  # Classical definition
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

for i in range(nx):
    for k in range(nz):
        s_p[i, k] = np.sqrt((src_x - ax[i]) ** 2 + (src_z - az[k]) ** 2)

for i in range(nx):
    for k in range(nz):
        p_r[i, k] = np.sqrt((rec_x - ax[i]) ** 2 + (rec_z - az[k]) ** 2)

plt.figure(figsize=(16, 10))
hfig = plt.imshow(s_p.T, extent=[ax[0], ax[-1], az[-1], az[0]])
plt.title('Distance to the source')
plt.colorbar(hfig)

plt.figure(figsize=(16, 10))
hfig = plt.imshow(p_r.T, extent=[ax[0], ax[-1], az[-1], az[0]])
plt.title('Distance to the receiver')
plt.colorbar(hfig)

v0 = 2.000
delta_v_n = (2 * inp_flat.T) / v0 ** 3
hk_s = np.zeros((nf, nx, nz))
hk_r = np.zeros((nf, nx, nz))
for i in tqdm(range(nx)):
    for k in range(nz):
        hk_s[:, i, k] = hankel2(0., s_p[i, k] * aw2 / v0)
        hk_r[:, i, k] = hankel2(0., p_r[i, k] * aw2 / v0)

def write_result(inp,flnam):
    # Write binary fila on disk (32 bits)
    with open(flnam,"wb") as fl:
        inp.astype('float32').tofile(fl)


write_result(hk_r,'./results/hk2_r.dat')
write_result(hk_s,'./results/hk2_s.dat')

#%%
def read_hankel_result(flnam, nf, nx, nz):
    with open(flnam, "rb") as fl:
        im = np.fromfile(fl, dtype=np.float32)
        im = im.reshape(nf,nx,nz,order='C')
    return im

hk_r_in = read_hankel_result('./results/hk_r.dat',nf,nx,nz)
hk_s_in = read_hankel_result('./results/hk_s.dat',nf,nx,nz)


plt.figure()
plt.imshow(hk_r[1,:,:].T)
plt.show()

plt.figure()
plt.imshow(hk_r_in[1,:,:].T)
plt.show()

np.array(hk_s) == np.array(hk_s_in)

hk_s0 = np.nan_to_num(hk_s_in)
hk_r0 = np.nan_to_num(hk_r_in)

v0 = 2.0

delta_v_n = 2 * inp_flat.T / v0 ** 3


mat_in_gral = np.zeros((nf, nx, nz))
for i in range(nf):
    mat_in_gral[i, :, :] = hk_s0[i] * delta_v_n * hk_r0[i]

gral_x = np.zeros((nf, nz))
for i in range(nz):
    gral_x[:,i] = np.sum(mat_in_gral[:, i, :])

gral_x_z = np.zeros(nf)
for i in range(nf):
    gral_x_z[i] = np.sum(gral_x[i])

wsrcf = np.fft.fft(wsrc)
dp_f = gral_x_z * wsrcf * -(aw2**2)

dp_t = np.real(np.fft.ifft(dp_f))



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

plt.figure(figsize=(16, 10))
idx = 2
hfig = plt.imshow(mat_in_gral[idx, :, :].T, extent=[ax[0], ax[-1], az[-1], az[0]])
plt.title('Before integration FREQ = '+str(aw2[idx]/(2*pi)))
plt.colorbar(hfig)
plt.show()

plt.figure(figsize=(16, 10))
plt.plot(aw2/(2*pi), dp_f)
plt.title('dp in frequency domain')
plt.show()

at = np.arange(nf)*dt
plt.figure(figsize=(16, 10))
plt.plot(dp_t)
plt.title('dp in time domain')
plt.show()