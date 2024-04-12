#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  4 09:59:38 2024

@author: vcabiativapico
"""



import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate import interpolate
import csv
from wiggle.wiggle import wiggle
from spotfunk.res import procs,visualisation
import pandas as pd
if __name__ == "__main__":
  
  
## Building simple vel and rho models to test modeling
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
## Add y dimension    
    fy = -500 
    ny = 21
    dy = 50
    ay = fy + np.arange(ny)*dy




def read_results(path,srow):
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



gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/042_slope_comp/05_pick_deep_flat_event/'


path_adj = gen_path + 'tolerance_badj/042_rt_badj_marm_slope_function.csv'
path_inv = gen_path + 'tolerance_binv/042_rt_binv_marm_slope_function.csv'



pick1 = '../input/40_marm_ano/badj_mig_pick_smooth.csv'
pick2 = '../input/40_marm_ano/binv_mig_pick_smooth.csv'

class Param_class:
    
    def __init__(self,path):
        self.src_x_ = read_results(path,1)
        self.src_y_ = read_results(path,2)
        self.src_z_ = read_results(path,3)
        self.rec_x_ = read_results(path,4)
        self.rec_y_ = read_results(path,5)
        self.rec_z_ = read_results(path,6)
        self.spot_x_ = read_results(path,7)
        self.spot_y_ = read_results(path,8)
        self.spot_z_ = read_results(path,9)
        self.off_x_ = read_results(path,16)
        self.tt_inv_ = read_results(path,17)
        self.nt_ = 1501
        self.dt_ = 1.41e-3
        self.ft_ = -100.11e-3
        self.nz_ = 151
        self.fz_ = 0.0
        self.dz_ = 12.0/1000.
        self.nx_ = 601
        self.fx_ = 0.0
        self.dx_ = 12.0/1000.
        self.no_ = 251
        self.do_ = dx
        self.fo_ = -(no-1)/2*do
        self.ao_ = fo + np.arange(no)*do
        self.at_ = ft + np.arange(nt)*dt
        self.az_ = fz + np.arange(nz)*dz
        self.ax_ = fx + np.arange(nx)*dx
   
        
p_adj = Param_class(path_adj)
p_inv = Param_class(path_inv)




path_demig = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/input/042_slope_comp'


# df = pd.read_csv(path_demig+'/results_both_tol05.csv')

df = pd.read_csv(path_demig+'/result_ts_off/results_both_angle_7_9_2.csv')



for column in df.columns:
    globals()[column] = np.array(df[column].values.tolist())



plt.figure(figsize=(10,8))
plt.title('Maximum Timeshift vs Offset for Angle 7° and 9°')
plt.plot(offset_7,max_7,'-')


plt.plot(offset_9,max_9,'-')
plt.ylabel('Maximum Timeshift')
plt.xlabel('Offset')
plt.legend(['7°','9°'])
# plt.ylim(2.9,6.1)
# plt.xlim(580,1210)

plt.figure(figsize=(10,8))
plt.plot(offset_9,-rec_x_9 + rec_x_7,'-')
plt.plot(offset_9,-src_x_9 + src_x_7,'-')
plt.ylabel('diff rec x (m)')
plt.xlabel('Offset')
plt.legend(['rec','src'])
plt.ylim(35,50)


plt.figure(figsize=(10,8))
plt.plot(offset_7,max_7,'.')
plt.plot(offset_9,max_9,'.')
plt.plot(offset_9,-rec_x_9 + rec_x_7,'-')
plt.plot(offset_9,-src_x_9 + src_x_7,'-  ')
plt.ylabel('diff rec x (m)')
plt.xlabel('Offset')
plt.ylim(0,50)