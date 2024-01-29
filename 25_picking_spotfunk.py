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


if __name__ == "__main__":
  
  
## Building simple vel and rho models to test modeling
     # Global parameters
     labelsize = 16
     nt = 1501
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


si = dz

# mig = 'adj'
# fl3 = '../output/23_mig/org/nh10_is4/dens_corr/inv_betap_x.dat'
# fl3 = '../output/27_marm/b'+str(mig)+'/inv_betap_x_s.dat'
# fl3 = '../output/27_marm/flat_marm/inv_betap_x_s.dat'
fl3 = '../input/27_marm/csg_raytracing_modeling.dat'
fl3 = '../input/27_marm/csg_raytracing_modeling_2_0.dat'
# inp3 = gt.readbin(fl3,nz,nx).T
inp3 = gt.readbin(fl3,1501,101).T


%matplotlib qt5

app = QApplication(sys.argv)

window = GUI_picking_horizon.MainWindow(inp3, win1= 0, win2=int(at[-1]*1000+100.11) ,si=dt,gain=3,trace=False,dpix=500,dpiy=500)

window.show()

app.exec_()

%matplotlib inline

pointe_mute_base = window.output() #pointe_mute_base to be changed


#%% SMOOTH
sigma = 3
pointe_smooth = gaussian_filter(pointe_mute_base[0], sigma)

plt.figure(figsize=(16,8))
plt.plot(pointe_smooth)
plt.ylim(500,1800)
plt.gca().invert_yaxis()
plt.figure(figsize=(16,8))
plt.plot(pointe_mute_base[0],'-r')
plt.ylim(500,1800)
plt.gca().invert_yaxis()


#%% EXPORT

# df = pd.DataFrame(pointe_mute_base[0])
# df.to_csv('../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_marm_'+str(mig)+'_02.csv',header=False,index=False)

# df = pd.DataFrame(pointe_smooth)
# df.to_csv('../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_sm'+str(sigma)+'_marm_'+str(mig)+'_02.csv',header=False,index=False)

# df = pd.DataFrame(pointe_mute_base[0])
# df.to_csv('../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_flat_marm.csv',header=False,index=False)

# df = pd.DataFrame(pointe_smooth)
# df.to_csv('../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_sm_flat_marm.csv',header=False,index=False)

df = pd.DataFrame(pointe_mute_base[0])
df.to_csv('../input/27_marm/29_pick_csg_flat.csv',header=['pick'],index=False)
