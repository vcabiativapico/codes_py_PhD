#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 10 12:02:32 2025

@author: vcabiativapico
"""


import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate import interpolate
import csv
from wiggle.wiggle import wiggle
from spotfunk.res import procs, visualisation, input
import segyio
from PIL import Image 
from scipy.ndimage import gaussian_filter

path_ref = '/home/vcabiativapico/local/src/victor/out2dcourse/input/87_complex_model_bp/vel_z6_exact.segy'
# path_ref = '/home/vcabiativapico/local/src/victor/out2dcourse/input/87_complex_model_bp/vel_z6_lw.segy'


vel_mig_model = input.segy_reader(path_ref).dataset.T


plt.figure(figsize=(53,19))
plt.imshow(vel_mig_model,
           vmin=np.min(vel_mig_model),vmax=np.max(vel_mig_model),
           cmap='seismic')