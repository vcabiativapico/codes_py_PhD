#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 15:05:33 2024

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

# path = ''
# seismic = input.segy_reader(path)

# data = seismic.dataset


nx = 601
nz = 151
no = 251
nt = 1501

for i in range(1,602):
    txt = str(i)
    title = txt.zfill(3)
    tr5 = '/home/vcabiativapico/local/src/victor/out2dcourse/output/45_marm_ano_v3/mig_binv_sm8/t1_obs_000'+str(title)+'.dat'
    if i >= 126 and i <= 476:
        no = 251 
    elif i > 0 and i < 126:
        no = 251//2 +i
    else: 
        no = 477-251-i-1
        
    inp5 = gt.readbin(tr5, no, nt).transpose()
    path =  '/home/vcabiativapico/local/src/victor/seisunix/shots_born/'+str(i)+'.sgy'
    segyio.tools.from_array2D(path, inp5.T, iline=i, dt=1410)
    
    # with segyio.open(path, 'r') as f:
    #     tr = f.trace[20]

#%%

in_path = '/home/vcabiativapico/local/src/victor/out2dcourse/input/45_marm_ano_v3/fwi_sm.dat'
inp = gt.readbin(in_path, nz, nx)


out_path2 = '/home/vcabiativapico/local/src/victor/seisunix/velocity_model/fwi_sm.txt'

vel_list = []
cdp_list = []


for i in range(601): 
    vel_list=np.append(vel_list,inp.T[i])


vel_list = list(vel_list)
cdp_list = [i for i in range(1, 602) for _ in range(151)]

base_trace_list = list(range(1, 152))
trace_list = base_trace_list * 601

df2 = pd.DataFrame([cdp_list,trace_list,vel_list]).T
df2.to_csv(out_path2,header=False,index=False)
    


# out_path = '/home/vcabiativapico/local/src/victor/seisunix/velocity_model/fwi_sm.sgy'

# segyio.tools.from_array2D(out_path, inp.T, dt=1)


# plt.imshow(inp5)

# spec = segyio.spec()
# spec.ilines  = [1]
# spec.xlines  = [1]
# spec.offsets = list(range(no))
# spec.samples = list(range(nt))
# spec.sorting = 2
# spec.format  = 1



    
    