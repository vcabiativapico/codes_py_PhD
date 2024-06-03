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



gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/Debug_110424_raytracing_new_solver/'


path_adj = gen_path + 'depth_demig_out/35_degrees/results/depth_demig_output.csv'
path_inv = gen_path + 'depth_demig_out/37_degrees/results/depth_demig_output.csv'



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




path_demig = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/'



df = pd.read_csv(path_demig+'046_37_35_degrees_sm8/result_ts_off/046_sm8_37_35_degrees.csv')

# df = pd.read_csv(path_demig+'044_fault_slope/result_ts_off/results_both_angles_fault_114perc.csv')


for column in df.columns:
    globals()[column] = np.array(df[column].values.tolist())


ratio_mean_TS = mean_TS_inv/mean_TS_adj
ratio_ext_TS = ext_binv_SLD_TS/ext_badj_SLD_TS


plt.figure(figsize=(10,8))
plt.title('Mean Timeshift vs Offset for QTV and STD')
plt.plot(offset_inv,mean_TS_inv,'.')
plt.plot(offset_adj,mean_TS_adj,'.')
plt.ylabel('Mean Timeshift (ms)')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])
# plt.ylim(2.9,6.1)
# plt.xlim(580,1210)

plt.figure(figsize=(10,8))
plt.title('Extremum Timeshift vs Offset')
plt.plot(offset_inv,ext_binv_SLD_TS,'.')
plt.plot(offset_adj,ext_badj_SLD_TS,'.')
plt.ylabel('Extemum Timeshift (ms)')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])
# plt.ylim(1,10)
# plt.xlim(580,1210)


#%%
plt.rcParams['font.size'] = 16
plt.figure(figsize=(10,8))
plt.title('Ratio of the mean Timeshift')
plt.plot(offset_inv,ratio_mean_TS,'.')
plt.ylabel(r'$\frac{MEAN \ TS \ QTV}{MEAN \ TS \ STD} $ ')
plt.xlabel('Offset')
# plt.ylim(0,4)



plt.figure(figsize=(10,8))
plt.title('Ratio of the extremum TS QTV and STD')
plt.plot(offset_inv,ratio_ext_TS,'.')
plt.ylabel(r'$\frac{MAX \ TS \ QTV}{MAX \ TS \ STD}$ ')
plt.xlabel('Offset')





plt.figure(figsize=(10,8))
plt.title('Minimum CC vs Offset for Angle QTV and STD')
plt.plot(offset_inv,min_CC_inv,'.')
plt.plot(offset_adj,min_CC_adj,'.')
plt.ylabel('Extremum CC')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])

plt.figure(figsize=(10,8))
plt.title('Mean CC vs Offset for Angle QTV and STD')
plt.plot(offset_inv,mean_CC_inv,'.')
plt.plot(offset_adj,mean_CC_adj,'.')
plt.ylabel('Mean CC')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])
# plt.ylim(2.9,6.1)
# plt.xlim(580,1210)

ratio_mean_CC = mean_CC_inv/mean_CC_adj
ratio_ext_CC = min_CC_inv/min_CC_adj

plt.figure(figsize=(10,8))
plt.title('Ratio of the cross-correlation')
plt.plot(offset_inv,ratio_mean_CC,'.')
plt.ylabel(r'$\frac{MEAN \ TS \ QTV}{MEAN \ TS \ STD} $ ')
plt.xlabel('Offset')
# plt.ylim(0,4)



plt.figure(figsize=(10,8))
plt.title('Ratio of the minimum cross-correlation QTV and STD')
plt.plot(offset_inv,ratio_ext_CC,'.')
plt.ylabel(r'$\frac{MIN \ CC \ QTV}{MIN \ CC \ STD}$ ')
plt.xlabel('Offset')




plt.figure(figsize=(10,8))
plt.title('Mean NRMS vs Offset for Angle QTV and STD')
plt.plot(offset_inv,mean_NRMS_inv,'.')
plt.plot(offset_adj,mean_NRMS_adj,'.')
plt.ylabel('Mean NRMS')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])
# plt.ylim(2.9,6.1)
# plt.xlim(580,1210)


plt.figure(figsize=(10,8))
plt.title('Extremum NRMS vs Offset for Angle QTV and STD')
plt.plot(offset_inv,ext_NRMS_binv,'.')
plt.plot(offset_adj,ext_NRMS_badj,'.')
plt.ylabel('Mean NRMS')
plt.xlabel('Offset')
plt.legend(['QTV','STD'])
# plt.ylim(2.9,6.1)
# plt.xlim(580,1210)

ratio_mean_NRMS = mean_NRMS_inv/mean_NRMS_adj
ratio_ext_NRMS = ext_NRMS_binv/ext_NRMS_badj


plt.figure(figsize=(10,8))
plt.title('Ratio of the mean NRMS QTV and STD')
plt.plot(offset_inv,ratio_mean_NRMS,'.')
plt.ylabel(r'$\frac{Mean \ NRMS \ QTV}{Mean \ NRMS \ STD}$ ')
plt.xlabel('Offset')

plt.figure(figsize=(10,8))
plt.title('Ratio of the extremum NRMS QTV and STD')
plt.plot(offset_inv,ratio_ext_NRMS,'.')
plt.ylabel(r'$\frac{EXT \ NRMS \ QTV}{EXT \ NRMS \ STD}$ ')
plt.xlabel('Offset')




plt.figure(figsize=(10,8))
plt.plot(offset_adj, rec_x_adj - rec_x_inv,'-')
plt.plot(offset_adj, src_x_adj - src_x_inv,'-')
plt.ylabel('diff rec x (m)')
plt.xlabel('Offset')
plt.legend(['REC','SRC'])
# plt.ylim(35,50)


plt.figure(figsize=(10,8))
plt.plot(offset_inv,max_TS_inv,'.')
plt.plot(offset_adj,max_TS_adj,'.')
plt.plot(offset_adj,rec_x_adj - rec_x_inv,'-')
plt.plot(offset_adj,src_x_adj - src_x_inv,'-')
plt.ylabel('diff rec x (m)')
plt.xlabel('Offset')
# plt.ylim(0,50)