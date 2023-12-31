#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 17:00:41 2023

@author: vcabiativapico
"""

import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from spotfunk.res import input, GUI_picking_horizon
from spotfunk.res.visualisation import *
from spotfunk.res.procs import *
import matplotlib
from matplotlib.backend_bases import MouseButton
import numpy as np
import os
import pandas as pd
from math import log, sqrt, log10, pi, cos, sin, atan
from scipy import interpolate
from scipy.interpolate import splrep, BSpline
import matplotlib.pyplot as plt
from matplotlib import use
from mpl_toolkits.axes_grid1 import make_axes_locatable
import geophy_tools as gt
from scipy.ndimage import gaussian_filter, sobel
import pickle as pk
from scipy.interpolate import splrep, BSpline

if __name__ == "__main__":
  
  
## Building simple vel and rho models to test modeling
    labelsize = 16
    nt        = 1001
    dt        = 2.08e-3
    ft        = -99.84e-3
    nz        = 151
    fz        = 0.0
    dz        = 12.00/1000.
    nx        = 601
    fx        = 0.0
    dx        = 12.00/1000.
    no        = 251
    do        = dx
    fo        = -(no-1)/2*do
    ao        = fo + np.arange(no)*do
    at        = ft + np.arange(nt)*dt
    az        = fz + np.arange(nz)*dz
    ax        = fx + np.arange(nx)*dx

si = dz


# fl3 = '../output/23_mig/org/nh10_is4/dens_corr/inv_betap_x.dat'
fl3 = '../output/26_mig_4_interfaces/badj_rc_norm/inv_betap_x_s.dat'
inp3 = gt.readbin(fl3,nz,nx).T


%matplotlib qt5

app = QApplication(sys.argv)

window = GUI_picking_horizon.MainWindow(inp3, si=si, win1=0, win2=int(1800),gain=3,trace=False,dpix=500,dpiy=500)

window.show()

app.exec_()

%matplotlib inline

pointe_mute_base = window.output() #pointe_mute_base to be changed


#%% SMOOTH
pointe_smooth = gaussian_filter(pointe_mute_base[0], 20)

plt.figure(figsize=(16,8))
plt.plot(pointe_smooth)
plt.plot(pointe_mute_base[0],'-r')
plt.ylim(1000,1800)
plt.gca().invert_yaxis()


#%% EXPORT
df = pd.DataFrame(pointe_mute_base[0])
df.to_csv('../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/horizon_badj_rc_norm_12.csv',header=False,index=False)

df = pd.DataFrame(pointe_smooth)
df.to_csv('../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/horizon_badj_smooth_rc_norm_12.csv',header=False,index=False)