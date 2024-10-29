#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 17 11:52:55 2024

@author: vcabiativapico
"""

import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from scipy.interpolate import interpolate
import csv
# from wiggle.wiggle import wiggle
from spotfunk.res import procs,visualisation
import pandas as pd
import os
# from shapely.geometry import Point, Polygon
import sympy as sp
import sympy.plotting as syp
import json



# # df = pd.DataFrame(table_spot_pos)
# # df.to_csv('/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/Demigration_Victor/042_auto_table_input.csv',header=False,index=False)

# table_spot_pos = np.array(table_spot_pos)


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
def calculate_slope(degrees, spot_x, spot_z, plot=False):
    m = np.tan(degrees * np.pi/180) 
    b = spot_z - m * spot_x
    point_x = np.array([spot_x - 150, spot_x + 150])
    point_z = point_x * m + b
    plt.figure(figsize=(10,8))
    plt.plot(point_x,point_z,'k')
    plt.scatter(spot_x,spot_z)
    plt.legend([degrees])
    p1 = np.array([point_x[0], 0 , point_z[0]])
    p2 = np.array([point_x[1], 0 , point_z[1]])
    p3 = np.array([point_x[0], 1200 , point_z[0]])
    
    if plot==True: 
        plt.figure(figsize=(10,8))
        plt.plot(point_x/1000,point_z/1000,linewidth=3)
        plt.scatter(spot_x/1000,spot_z/1000,c='k')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
    return p1, p2, p3


def plot_plane_from_points(p1,p2,p3, plot=False):
    ''' 
    a plane is a*x+b*y+c*z+d=0 is calculated
    [a,b,c] is calculated it is the normal. 
    d is also obtained'''
    u = p2 - p1
    v = p3 - p1
    
    normal = np.cross(u,v)
    
    d = -p1.dot(normal)
    
    a, b, c = normal
    
    x_min, x_max = min(p1[0], p2[0], p3[0]), max(p1[0], p2[0], p3[0])
    y_min, y_max = min(p1[1], p2[1], p3[1]), max(p1[1], p2[1], p3[1])
    
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 10), np.linspace(y_min, y_max, 10))
   
    # Calculate z
    # z = (-a*xx - d) / c # Simplified formula because y is zero
    z = (-a*xx - b*yy - d) / c

       
    
    
    if plot== True :
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(projection='3d')
        ax.plot_surface(xx, yy, z, alpha=0.2)
        ax.scatter(p1[0] , p1[1] , p1[2],  color='green') 
        ax.scatter(p2[0] , p2[1] , p2[2],  color='green')
        ax.scatter(p3[0] , p3[1] , p3[2],  color='green')
    
     
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
    
    return normal,d

def print_input_values_rt(point1,point2,norm,d):
    input_values = {
      "spot_x_input" : abs( ( point1[0] + point2[0] ) // 2),
      "a": norm[0],
      "b": norm[1],
      "c": norm[2],
      "d": d,
    }
    for x in input_values: 
        print(x,' : ',input_values[x])
    return input_values



def write_json_file_flat(json_file, out_path, input_table,z_value,thickness,parameter_path,weights_path):
    # Step 1: Read the JSON file
    with open(json_file+'.json', 'r') as file:
        data = json.load(file)
    
    data['paths_config']['input_file']= input_table
    data['raytracing_config']['interfaces_data'][0]['depth'] = z_value
    # Step 2: Modify the dictionary
    data['raytracing_config']['interfaces_data'][1]['depth'] = -thickness
    data['raytracing_config']['velocity_model_data'][0]['parameter_path'] = parameter_path
    data['raytracing_config']['velocity_model_data'][0]['weights_path'] = weights_path
    # Step 3: Write the modified dictionary back to the JSON file
    with open(out_path+'.json', 'w') as file:
        json.dump(data, file, indent=4)
    return data


def plot_mig_image(inp,ax,az):
    hmax = np.max(inp)
    hmin = np.min(inp)
    fig = plt.figure(figsize=(15,7), facecolor = "white")
    av  = plt.subplot(1,1,1)
    # hfig = av.imshow(inp[50:100,200:350], vmin=hmin,vmax=hmax,extent=[ax[200], ax[350], az[100], az[50]],aspect='auto')
    hfig = av.imshow(inp, vmin=hmin,vmax=hmax,extent=[ax[0], ax[-1], az[-1], az[0]],aspect='auto')
    plt.colorbar(hfig)
#%%   


# Global parameters
labelsize = 16
nt = 1801
dt = 1.14e-3
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

base_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/067_TS_graded_flat/'

json_file = base_path+'input'
input_table =  base_path+'060_TS_flat_table.csv'


z_value = -12

for i in range(12,600,48):
    parameter_path = "067_Param_vel_graded_"+str(i)+"_ano.csv"
    weights_path = "067_Weights_vel_graded_"+str(i)+"_ano.csv"
    out_path = base_path+ '_thickness_ano_'+str(i)
    write_json_file_flat(json_file, out_path, input_table,z_value,-1750,parameter_path,weights_path)


# fl1='../input/45_marm_ano_v3/fwi_ano.dat'

# inp_org = gt.readbin(fl1,nz,nx)

# plt.figure()
# plot_mig_image(inp_org,ax,az)
# # plt.plot(ax,bspline_inv_hz[0:601]/1000)
# plt.scatter(ray_x_in_poly/1000,-ray_z_in_poly/1000)
# plt.plot(np.array([pt_inv1[0],pt_inv2[0]])/1000, np.array([-pt_inv1[2],-pt_inv2[2]])/1000, 'r')
# plt.legend()
# plt.gca().invert_yaxis()