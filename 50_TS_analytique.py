#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 22 12:16:12 2024

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
import os
from shapely.geometry import Point, Polygon
import sympy as sp
import sympy.plotting as syp

if __name__ == "__main__":
  
  
# Building simple vel and rho models to test modeling
    labelsize = 16
    nt = 1801
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
# ## Add y dimension    
    fy = -500 
    ny = 21
    dy = 50
    ay = fy + np.arange(ny)*dy


def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr

## Read the results from demigration
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

def norm(inp):
    """
    Normalize the data according to its maximum
    Normalization is no longer necessary due to the correction of amplitude on the formula
    """
    norm_inp = inp/np.max(abs(inp))
#     return inp
    return norm_inp


gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'



path_adj = gen_path + '048_sm8_correction_new_solver/STD/depth_demig_out/STD/results/depth_demig_output.csv'
path_inv = gen_path + '048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/results/depth_demig_output.csv'

class Param_class:
    "Class for the parameters definition"
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
        self.tt_ = read_results(path,17)
        self.nt_ = 1801
        self.dt_ = 1.41e-3
        self.ft_ = -100.11e-3
        self.nz_ = 151
        self.fz_ = 0.0
        self.dz_ = 12.0/1000.
        self.nx_ = 601
        self.fx_ = 0.0
        self.dx_ = 12.0/1000.
        self.no_ = 251
        self.do_ = self.dx_
        self.fo_ = -(self.no_-1)/2*self.do_
        self.ao_ = self.fo_ + np.arange(self.no_)*self.do_
        self.at_ = self.ft_ + np.arange(self.nt_)*self.dt_
        self.az_ = self.fz_ + np.arange(self.nz_)*self.dz_
        self.ax_ = self.fx_ + np.arange(self.nx_)*self.dx_
   
        
p_adj = Param_class(path_adj)
p_inv = Param_class(path_inv)



def plot_model(inp):
    plt.rcParams['font.size'] = 20
    hmax = 2.05
    hmin = 2.0
    hmin = np.min(inp)

    hmax = np.max(inp)

    
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp[:], extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
 
    plt.colorbar(hfig1, format='%1.1f',label='m/s')
  
    fig.tight_layout()

def pick_interface_model(inp,ray_x,ray_z,hz,nx,nz):
 # Create a plot only with integers by using the extension of the grid ax az
     
     ax = np.arange(nx)
     az = np.arange(nz)
     vbg = 1
 #### Plot initial model 
  
     %matplotlib qt5
     hmax = np.max(inp)
     hmin = -hmax
     fig = plt.figure(figsize=(25,12), facecolor = "white")
     av  = plt.subplot(1,1,1)
     print('az',az.shape)
     hfig = av.imshow(inp,vmin=hmin,vmax=hmax,extent=[ax[0], ax[-1], az[-1], az[0]],aspect='auto', cmap='seismic')
     # plt.plot(ax,hz*1000/102)
     plt.colorbar(hfig)
     av.scatter(ray_x/12,-ray_z/12, c='k',s=0.1)

     fig.tight_layout()

     
 # Extraction of indexes and arrangement to insert them
     pick = plt.ginput(n=-1,timeout=15)
     plt.close()
     
     pick = np.asarray(pick).astype(int)
     pickt = pick.transpose()
     pick_f = [pickt[1],pickt[0]] 
     pick_f = tuple(pick_f)   
     
     
     return  pick_f
 
def calculate_thickness(pick,plot=False):
    m = pick[1][0] - pick[1][1] 
    n = pick[0][0] - pick[0][1] 
    
    l = np.sqrt(m**2 + n**2)
    l = l*12

    return l
 

def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr

def read_index(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(int(row[srow]))
    return attr


'''Read the velocity model '''
fl1 = '../input/45_marm_ano_v3/fwi_ano_114_percent.dat'
fl2 = '../input/45_marm_ano_v3/fwi_org.dat'
inp1 = gt.readbin(fl1,nz,nx)
inp2 = gt.readbin(fl2,nz,nx)

'''Read the raypath'''
path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/046_37_35_degrees_sm8/depth_demig_out/QTV/rays/ray_0.csv'
ray_x = np.array(read_results(path_ray, 0))
ray_z = np.array(read_results(path_ray, 2))
vp =  np.array(read_results(path_ray, 6))
tt = np.array(read_results(path_ray, 8))

'''Read index of the anomaly'''
file_index = '../input/45_marm_ano_v3/fwi_ano_114_percent.csv'
inp_index = [np.array(read_index(file_index,0)), np.array(read_index(file_index,1))]

'''Compute the values of the anomaly only'''
inp_blank = np.zeros_like(inp1)


inp_blank[inp_index] = 2
inp_blank[:,0:270] = 0

new_idx = np.where(inp_blank>1)
plot_model(inp_blank)
plt.scatter(ray_x/1000,-ray_z/1000, c='k',s=0.1)

'''Create a polygon'''
sympy_idx = np.transpose(new_idx).tolist()

sympy_idx_tuple = []
for x in sympy_idx:
    sympy_idx_tuple.append(tuple(x))



polygon = sp.Polygon(*sympy_idx_tuple)

# Extract the coordinates
x, y = zip(*polygon.vertices)

# Close the polygon by repeating the first vertex at the end
x = list(x) + [x[0]]
y = list(y) + [y[0]]

# Plot the polygon
plt.figure(figsize=(12, 12))
plt.scatter(np.array(y)*0.012, np.array(x)*0.012)
plt.scatter(ray_x/1000,-ray_z/1000, c='k',s=1.5)
plt.axhline(1.018)
plt.title("Polygon Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.ylim(1.16,0.95)
plt.xlim(3.2,3.5)


idx_in_poly = []

for i,k in enumerate(ray_z):
    if k < -1018:
        idx_in_poly.append(i)
ray_x_in_poly = ray_x[idx_in_poly]
ray_z_in_poly = ray_z[idx_in_poly]
tt_in_poly = tt[idx_in_poly]


# Plot the polygon
plt.figure(figsize=(12, 12))
plt.scatter(np.array(y)*0.012, np.array(x)*0.012)
plt.scatter(ray_x_in_poly/1000,-ray_z_in_poly/1000, c='k',s=1.5)
plt.axhline(1.018)
plt.title("Polygon Plot")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.ylim(1.16,0.95)
plt.xlim(3.2,3.5)



'''Pick the anomaly values '''
pick_adj = pick_interface_model(inp_blank,ray_x,ray_z,pick_hz_badj,nx,nz)
%matplotlib inline
plt.plot(pick_adj[1]*12,pick_adj[0]*12,'.')

thickness = calculate_thickness(pick_adj)

plot_model(inp1-inp2)
plt.scatter(ray_x/1000,-ray_z/1000, c='k',s=0.1)


vel1 = np.mean(inp1[inp_index])*1000
vel2 = np.mean(inp2[inp_index])*1000

diff_vel = vel1 - vel2

ts0 = 2 * (thickness/2) / diff_vel
ts1 = 2 * thickness / diff_vel


p_adj.tt_[0] 
