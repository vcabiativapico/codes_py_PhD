# -*- coding: utf-8 -*-

"""
Diffusion equation
FE in 2d
Backward Euler Method in time
"""

import numpy as np
from math import sin, cos, pi, sqrt, log, atan, log10, exp
import matplotlib.pyplot as plt
from matplotlib import rc
import geophy_tools as gt
from scipy.special import hankel1


def defwsrc(fmax, dt):
    """
    Definition of the source function
    Ricker wavelet with central frequency fmax/2.5
    Ricker = 2nd-order derivative of a Gaussian function
    """
    fc = fmax / 2.5  # Central frequency
    # ns2  = int(2/fc/dt) # Classical definition
    # ns2 = int(7.5/fmax/dt+0.5) # Large source, but better for inverse WWW
    ns2 = 4 * int(3.0 / 2.5 / fc / dt + 0.49)
    ns = 1 + 2 * ns2  # Size of the source
    wsrc = np.zeros(ns)
    aw = np.zeros(ns)  # Time axis
    for it in range(ns):
        a1 = float(it - ns2) * fc * dt * pi
        a2 = a1 ** 2
        wsrc[it] = (1 - 2 * a2) * exp(-a2)
        aw[it] = float(it - ns2) * dt
    return wsrc, aw


def ana2d(wsrc, dt, r, v0):
    """Analytic 2d solution"""
    # Fourier transform
    wsrcf = np.fft.fft(wsrc)
    nws = len(wsrc)
    ns = (nws - 1) // 2
    fmax = 1. / 2. / dt
    fmin = -fmax
    nf = nws
    aw = 2. * pi * np.linspace(fmin, fmax, nf)
    aw = -np.roll(aw, ns + 1)

    # Distance and travel time
    t0 = r / v0
    # Hankel function of the first kind
    hk = 1j / 4. * hankel1(0., r / v0 * aw)
    # (1j*aw)**2 = -aw**2 := (i \omega)^2
    anaf = hk * wsrcf
    anaf[0] = 0  # no zero-frequency
    ana = np.real(np.fft.ifft(anaf))
    return ana


if __name__ == "__main__":
    print("Start...")

    # Only works for symmetric input wavelet???

    # Read the source function
    fl = 'input/wsrc.dat'
    nws = 143
    wsrc = gt.readbin(fl, nws, 1)[:, 0]
    ns = (nws - 1) // 2

    # The snapshot
    fl = 'input/p2d_lsm_000001.dat'
    green2d = gt.readbin(fl,nws,1)[:,0]

    # Analytic solution
    fmax = 25.
    dt = 0.001
    wsrc, aw = defwsrc(fmax, dt)

    # Analytic solution
    r0 = 180.  # distance in m
    # r0  = 0.0001
    v0 = 2500.  # homogeneous velocity
    ana = ana2d(wsrc, dt, r0, v0)

    # --------------------------------
    # Display the result
    fig = plt.figure(figsize=(12, 10), facecolor="white")
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)

    ax = plt.subplot(2, 1, 1)
    plt.plot(wsrc, 'b')
    plt.xlabel("Position x")
    plt.ylabel("Position z")
    plt.title("Geometry")

    ax = plt.subplot(2, 1, 2)
    # plt.plot(green2d,'b')
    plt.plot(ana, 'r')
    # plt.plot(green2d-ana,'k')
    plt.xlabel("Position x")
    plt.ylabel("Position z")
    plt.title("Solutions")
    plt.show()
    fig.savefig("./ana2d_new.png", bbox_inches='tight')
