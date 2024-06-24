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
from wiggle.wiggle import wiggle
from spotfunk.res import procs,visualisation
import pandas as pd
import os
from shapely.geometry import Point, Polygon
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



def write_json_file(json_file,out_path, input_val,z_value,table):
    # Step 1: Read the JSON file
    with open(json_file+'.json', 'r') as file:
        data = json.load(file)
    
    data['paths_config']['input_file']= table
    data['raytracing_config']['interfaces_data'][0]['depth'] = z_value
    # Step 2: Modify the dictionary
    data['raytracing_config']['interfaces_data'][1]['a'] = input_val['a']
    data['raytracing_config']['interfaces_data'][1]['b'] = input_val['b']
    data['raytracing_config']['interfaces_data'][1]['c'] = input_val['c']
    data['raytracing_config']['interfaces_data'][1]['d'] = input_val['d']
    
    # Step 3: Write the modified dictionary back to the JSON file
    with open(out_path+'.json', 'w') as file:
        json.dump(data, file, indent=4)
    return data

path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'
# json_file = path+'050_TS_analytique/to_input'
# out_path= path+'050_TS_analytique/050_TS_analytique'

json_file = path+'050_TS_analytique_ano/to_input'
out_path= path+'050_TS_analytique_ano/050_TS_analytique_ano'

#%%
'''Read the raypath'''
path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/046_37_35_degrees_sm8/depth_demig_out/QTV/rays/ray_0.csv'
ray_x = np.array(read_results(path_ray, 0))
ray_z = np.array(read_results(path_ray, 2))
vp =  np.array(read_results(path_ray, 6))
tt = np.array(read_results(path_ray, 8))

idx_in_poly = []

for i,k in enumerate(ray_z):
    if k < -1018:
        idx_in_poly.append(i)
ray_x_in_poly = ray_x[idx_in_poly]
ray_z_in_poly = ray_z[idx_in_poly]
tt_in_poly = tt[idx_in_poly]


normales = []
d_values = []
dict_input = []
spot_x = []
spot_z = []
# for i in range(4,len(ray_z_in_poly[:5])):
for i in range(1,17):
    
    pt_inv1, pt_inv2, pt_inv3 = calculate_slope(39, ray_x_in_poly[i],ray_z_in_poly[i], plot=True)
    d_val = plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[1]
    norm = plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3,plot=True)[0]
    normales.append(norm)
    d_values.append(d_val)
    dict_input.append(print_input_values_rt(pt_inv1,pt_inv2,norm,d_val))
    
    table_input = [ray_x_in_poly[i],0,0,0.01]
    spot_x.append(-(dict_input[-1]['c']*ray_z_in_poly[i]+dict_input[-1]['d'])/dict_input[-1]['a'])
    spot_z.append(-(dict_input[-1]['a']*ray_x_in_poly[i]+dict_input[-1]['d'])/dict_input[-1]['c'])
    print(ray_z_in_poly[i],spot_z[-1])
    # df = pd.DataFrame(table_input).T
    # df.to_csv(out_path+str(i)+'_table.csv',header=False,index=False)
    
    # write_json_file(json_file, out_path+str(i),dict_input[-1],ray_z_in_poly[0],'050_TS_analytique_ano'+str(i)+'_table.csv')



ray_x_in_poly,ray_z_in_poly = 3510,-1210

pt_inv1, pt_inv2, pt_inv3 = calculate_slope(39,ray_x_in_poly,ray_z_in_poly, plot=True)
d_val = plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[1]
norm = plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3,plot=True)[0]
normales=norm
d_values=d_val
dict_input=print_input_values_rt(pt_inv1,pt_inv2,norm,d_val)

table_input = [ray_x_in_poly,0,0,0.01]
spot_x=(-(dict_input['c']*ray_z_in_poly+dict_input['d'])/dict_input['a'])
spot_z=(-(dict_input['a']*ray_x_in_poly+dict_input['d'])/dict_input['c'])
print(ray_z_in_poly,spot_z)
df = pd.DataFrame(table_input).T
df.to_csv(out_path+'deep_table.csv',header=False,index=False)
 
write_json_file(json_file, out_path+'deep',dict_input,-1018.245073,'050_TS_analytique_ano_deep_table.csv')
