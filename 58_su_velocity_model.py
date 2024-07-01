#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 15:49:56 2024

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
import pandas as pd

nx = 601
nz = 151
no = 251
nt = 1501

in_path = '/home/vcabiativapico/local/src/victor/out2dcourse/input/45_marm_ano_v3/fwi_sm.dat'
inp = gt.readbin(in_path, nz, nx)


out_path2 = '/home/vcabiativapico/local/src/victor/seisunix/velocity_model/fwi_sm.txt'
out_path3 = '/home/vcabiativapico/local/src/victor/seisunix/velocity_model/fwi_sm_line_90.txt'


vel_list = []
cdp_list = []


for i in range(601): 
    vel_list=np.append(vel_list,inp.T[i])

vel_list = vel_list*0.9

cdp_list = [i for i in range(1, 602) for _ in range(151)]


base_trace_list = list(range(1, 152))
trace_list = base_trace_list * 601

df2 = pd.DataFrame([cdp_list,trace_list,vel_list]).T
df2.to_csv(out_path2,sep=' ',header=False,index=False)
    


df3 = pd.DataFrame(vel_list)
df3.to_csv(out_path3,sep=' ',header=False,index=False)
    