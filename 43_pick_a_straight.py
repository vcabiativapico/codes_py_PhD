#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 15:30:57 2023

@author: vcabiativapico
"""

"""
Display the results
"""

#%matplotlib inline
import os
import numpy as np
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
import csv 
import json

if __name__ == "__main__":
  
  
## Building simple vel and rho models to test modeling
    labelsize = 16
    nt        = 1001
    dt        = 2.08e-3
    ft        = -100.11e-3
    nz        = 151
    fz        = 0.0
    dz        = 12/1000.
    nx        = 601
    fx        = 0.0
    dx        = 12/1000.
    no        = 251
    do        = dx
    fo        = -(no-1)/2*do
    ao        = fo + np.arange(no)*do
    at        = ft + np.arange(nt)*dt
    az        = fz + np.arange(nz)*dz
    ax        = fx + np.arange(nx)*dx

    
#%%
def plot_mig_image(inp,ax,az):
    hmax = np.max(inp)
    hmin = -hmax
    fig = plt.figure(figsize=(15,7), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp[50:100,200:350], vmin=hmin,vmax=hmax,extent=[ax[200], ax[350], az[100], az[50]],aspect='auto', cmap='seismic')
    print('az',az.shape)
   
    
def pick_interface_model(inp,hz,nx,nz):
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
     
     fig.tight_layout()

     
 # Extraction of indexes and arrangement to insert them
     pick = plt.ginput(n=-1,timeout=15)
     plt.close()
     
     pick = np.asarray(pick).astype(int)
     pickt = pick.transpose()
     pick_f = [pickt[1],pickt[0]] 
     pick_f = tuple(pick_f)   
     
     
     return  pick_f
    
def plot_pick(inp,pick_x,pick_y,xnew,ynew):
    hmax = np.max(inp)
    hmin = -hmax
    fig = plt.figure(figsize=(15,8), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp,vmin=hmin,vmax=hmax,aspect='auto', alpha =0.7,\
                      cmap='seismic')
    plt.colorbar(hfig)
    
    plt.plot(xnew,ynew,'k')
    plt.plot(pick_x,pick_y,'r*')
    fig.tight_layout()
    
    plt.ylim(151,0) 
    flout2 = '../png/24_for_ray_tracing/picking_model.png'
    print("Export to file:",flout2)
    fig.savefig(flout2, bbox_inches='tight')



def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr

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
    
def horizon_for_bspline(pick,dz):
    """ 
    Produces a horizon with the slope given by the picking points. 
    The horizon must be 601 points length, so it is filled with horizontal lines on both sides of the slope
    """
    ind_z = (-pick[0]/dz).astype(int)
    ind_x = (pick[1]/dz).astype(int)
    
    
    
    m = (ind_z[1] - ind_z[0])/(ind_x[1] - ind_x[0])
    b = ind_z[0] - ind_x[0] * m
    print(ind_x)
    x = np.arange(ind_x[0],ind_x[1])
    print(x)
    horizon = np.zeros(nx)
    
    
    print(horizon.shape)
    
    horizon[ind_x[0]:ind_x[1]] = x * m + b
    horizon[0:ind_x[0]] = x[0] * m + b
    horizon[ind_x[1]:] = x[-1] * m + b
    horizon = horizon*dz*1000
    
    plt.figure()
    plt.plot(horizon)
    plt.ylim(-1800,0)
    return horizon


#%%


file_pick_badj = '../input/40_marm_ano/badj_mig_pick_smooth.csv'
pick_hz_badj = np.array(read_pick(file_pick_badj,0))/1000

  
file_pick_binv = '../input/40_marm_ano/binv_mig_pick_smooth.csv'
pick_hz_binv = np.array(read_pick(file_pick_binv,0))/1000


# fl3_adj = '../output/40_marm_ano/badj/inv_betap_x.dat'
fl3_adj = '../output/45_marm_ano_v3/mig_badj_sm8/inv_betap_x_s.dat'
inp3_adj = gt.readbin(fl3_adj,nz,nx) 
pick_adj = pick_interface_model(inp3_adj,pick_hz_badj,nx,nz)
# print('pick_adj',pick_adj)
pick_adj = np.array(pick_adj) * dx * 1000
print('pick_adj',pick_adj)


# fl3_inv = '../output/40_marm_ano/binv/inv_betap_x.dat'
fl3_inv = '../output/45_marm_ano_v3/mig_binv_sm8/inv_betap_x_s.dat'

inp3_inv = gt.readbin(fl3_inv,nz,nx) 
pick_inv = pick_interface_model(inp3_inv,pick_hz_binv,nx,nz)
# print('pick_adj',pick_inv)
pick_inv = np.array(pick_inv) * dx * 1000

# hz_inv = horizon_for_bspline(pick_inv,dz)


# print('pick_inv',pick_inv)
# # 




out_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/input/046_37_35_degrees_sm8/'

df = pd.DataFrame(pick_inv)
df.to_csv(out_path+'pick_inv_sm8.csv', header=False, index=False)

df = pd.DataFrame(pick_adj)
df.to_csv(out_path+'pick_adj_sm8.csv', header=False, index=False)


#%%


# path_demig = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/input/042_slope_comp'
# path_slope = '/slope_binv_fault.csv'
# file = path_demig + path_slope
# point_x_inv = np.array(read_pick(file,0))
# point_z_inv = -np.array(read_pick(file,2))
# pick_inv = np.array([[point_z_inv[0],point_z_inv[1]],[point_x_inv[0],point_x_inv[1]]])


# path_slope = '/slope_badj_fault.csv'
# file = path_demig + path_slope
# point_x_adj = np.array(read_pick(file,0))
# point_z_adj = -np.array(read_pick(file,2))
# pick_adj = np.array([[point_z_adj[0],point_z_adj[1]],[point_x_adj[0],point_x_adj[1]]])



%matplotlib inline


point1_inv  = np.array([pick_inv[1,0], 0, -pick_inv[0,0]])
point2_inv = np.array([pick_inv[1,1], 0, -pick_inv[0,1]])
point3_inv = np.array([pick_inv[1,0], 1200, -pick_inv[0,0]])

point1_adj  = np.array([pick_adj[1,0], 0, -pick_adj[0,0]])
point2_adj = np.array([pick_adj[1,1], 0, -pick_adj[0,1]])
point3_adj= np.array([pick_adj[1,0], 1200, -pick_adj[0,0]])


# plot the surface

norm_inv, d_inv = plot_plane_from_points(point1_inv, point2_inv, point3_inv)
norm_adj, d_adj = plot_plane_from_points(point1_adj, point2_adj, point3_adj)



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



# plt.gca().invert_yaxis()


print('\nValues for inv (quantitative) : ')
input_val_inv = print_input_values_rt(point1_inv,point2_inv, norm_inv, d_inv)

print('\nValues for adj (standard) :')
input_val_adj = print_input_values_rt(point1_adj,point2_adj, norm_adj, d_adj)


spot_z_inv = (input_val_inv['a']*input_val_inv['spot_x_input']+input_val_inv['d'])/-input_val_inv['c']
spot_z_adj = (input_val_adj['a']*input_val_inv['spot_x_input']+input_val_adj['d'])/-input_val_adj['c']

plt.figure(figsize=(10,8))
plt.rcParams['font.size'] = 16
plt.plot(pick_adj[1], -pick_adj[0], label='Slope STD')
plt.plot(pick_inv[1], -pick_inv[0], label='Slope QTV')
plt.scatter(input_val_inv['spot_x_input'],spot_z_adj, label='Spot STD')
plt.scatter(input_val_inv['spot_x_input'],spot_z_inv, label='Spot QTV')
plt.title('Slopes and spots picked in the Standard and Quantitative images')
plt.legend()
plt.xlabel('Distance (m)')
plt.ylabel('Depth (m)')


demig_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/input/046_37_35_degrees_sm8'

# df = pd.DataFrame(np.append(norm_inv,d_inv))
# df.to_csv(demig_path+'/norm_d_binv_sm8.csv',header=False,index=False)

# df = pd.DataFrame(np.append(norm_adj,d_adj))
# df.to_csv(demig_path+'/norm_d_badj_sm8.csv',header=False,index=False)


# df = pd.DataFrame([point1_adj,point2_adj])
# df.to_csv(demig_path+'/slope_badj_sm8.csv',header=False,index=False)

# df = pd.DataFrame([point1_inv,point2_inv])
# df.to_csv(demig_path+'/slope_binv_sm8.csv',header=False,index=False)




# df = pd.DataFrame(z_inv)
# df.to_csv(demig_path+'z_binv_2.csv',header=False,index=False)

# pt1 = np.array([1,0,0])
# pt2 =np.array([0,2,0])
# pt3 =np.array([0,0,3])
# p_test = plot_plane_from_points(pt1, pt2, pt3)
'''Create input table for raytracing'''
table_input = []
for i in range(1,101):
    table_input.append([input_val_inv['spot_x_input'],0,0,12*i])
df = pd.DataFrame(table_input)
# df.to_csv('/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/Demigration_Victor/042_auto_table_input.csv',header=False,index=False)



def horizon_for_bspline(pick,dz):
    """ 
    Produces a horizon with the slope given by the picking points. 
    The horizon must be 601 points length, so it is filled with horizontal lines on both sides of the slope
    """
    ind_z = (-pick[0]/dz/1000).astype(int)
    ind_x = (pick[1]/dz/1000).astype(int) 
    
    m = (ind_z[1] - ind_z[0])/(ind_x[1] - ind_x[0])
    b = ind_z[0] - ind_x[0] * m
    
    x = np.arange(np.min(ind_x),np.max(ind_x))
    
    horizon = np.zeros(nx)
    
    horizon[np.min(ind_x):np.max(ind_x)] = x * m + b
    horizon[0:np.min(ind_x)] = x[0] * m + b
    horizon[np.max(ind_x):] = x[-1] * m + b
    horizon = horizon*dz*1000
    
    plt.figure()
    plt.plot(ax*1000,horizon)
    plt.ylim(-1800,0)
    return horizon

hz_inv = horizon_for_bspline(pick_inv,dz)
plt.title('bspline inv')

hz_adj = horizon_for_bspline(pick_adj,dz)
plt.title('bspline adj')


#%%



import spotbsplines
from spotbsplines import interp1d       


class Param_Input_class:
    
    def __init__(self, Param):
        self.start_x_ = Param[0]
        self.start_y_ = Param[1]
        self.start_z_ = Param[2]
        self.delta_x_ = Param[3]
        self.delta_y_ = Param[4]
        self.delta_z_ = Param[5]
        self.INL_step_ = Param[6]
        self.XL_step_ = Param[7]
        self.azimuth_ = Param[8]
        self.I_ = Param[9]
        self.J_ = Param[10]
        self.K_ = Param[11]
        self.X_or_ = Param[12]
        self.Y_or_ = Param[13]

        self.end_x_ = self.start_x_ + (self.I_ - 1) * self.delta_x_
        self.end_y_ = self.start_y_ + (self.J_ - 1) * self.delta_y_
        self.end_z_ = self.start_z_ + (self.K_ - 1) * self.delta_z_
        
        self.tx_ = np.arange(self.start_x_ - 2. * self.delta_x_, self.end_x_ + self.delta_x_ + 0.01, self.delta_x_)
        self.ty_ = np.arange(self.start_y_ - 2. * self.delta_y_, self.end_y_ + self.delta_y_ + 0.01, self.delta_y_)
        self.tz_ = np.arange(self.start_z_ - 2. * self.delta_z_, self.end_z_ + self.delta_z_ + 0.01, self.delta_z_)

def bspline_1_5D(hz,fl):
    """Generates bsplines weights in 1.5D from a horizon in 1D"""
    INL_step = 200 
    XL_step = 12.00   
    azimuth1 = 90
    azimuth = azimuth1*2*np.pi/360
    X_or = 0
    Y_or = 0
    
    I = 5
    J = 601
    K = 151
    
    M = I+3
    N = J+3
    L = K+3
    
    start_x = -2
    start_y = 0
    start_z = 0
    
    delta_x = 1
    delta_y = 1
    delta_z = 12.00
                
    Param_Input1 = [start_x,start_y,start_z,
                  delta_x,delta_y,delta_z,
                  INL_step,XL_step,azimuth,
                  I,J,K,X_or,Y_or]
    
    x_disc = np.arange(601)*12.00
    z_disc = np.arange(151)*12.00
    
    Param_Input = Param_Input_class(Param_Input1)
    
    Param_Input = [start_y,start_z,
                  delta_y,delta_z,
                  XL_step,1,
                  J,K]

    
    Weights_hz = interp1d(hz, Param_Input)
    
    Weights_2D_mat_hz = Weights_hz[np.newaxis,:]*np.ones(M)[:,np.newaxis]*2/3
    
    Weight_2D_inline = Weights_2D_mat_hz.reshape(M*N)
    
    
    
    np.savetxt('../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/'+fl,Weight_2D_inline,fmt='%f',delimiter=',') 
    return Weight_2D_inline


bspline_inv_hz = bspline_1_5D(hz_inv,'042_weights_hz_pick_inv_slope6.csv')
bspline_adj_hz = bspline_1_5D(hz_adj,'042_weights_hz_pick_adj_slope6.csv')


fl3_inv = '../input/45_marm_ano_v3/fwi_ano_45.dat'
inp3_adj = gt.readbin(fl3_inv,nz,nx) 
inp3_inv = gt.readbin(fl3_inv,nz,nx) 

plt.figure()
plot_mig_image(inp3_inv, ax, -az)
# plt.plot(ax,bspline_inv_hz[0:601]/1000)
plt.plot(ax,hz_inv/1000)
plt.plot(pick_inv[1]/1000, -pick_inv[0]/1000, label='QTV')
plt.legend()
# plt.gca().invert_yaxis()



plt.figure()
plot_mig_image(inp3_adj, ax, -az)
# plt.plot(ax,bspline_adj_hz[0:601]/1000)
plt.plot(ax,hz_adj/1000)
plt.plot(pick_adj[1]/1000, -pick_adj[0]/1000, label='QTV')
plt.legend()
# plt.plot(pick_inv[1], pick_inv[0])


plt.figure()
plt.plot(pick_adj[1], -pick_adj[0], label='QTV')
plt.plot(pick_inv[1], -pick_inv[0], label='QTV')

#%%
def calculate_angle(pick):
    angle = np.arctan( (pick[0][0] -pick[0][1]) / (pick[1][0] - pick[1][1] )) * 180 / np.pi
    percentage = (pick[0][0] -pick[0][1]) / (pick[1][0] - pick[1][1] ) * 100
    print('Slope angle is :', angle)
    print('Slope percentage is :', percentage)
    return angle, percentage

calculate_angle(pick_inv)
calculate_angle(pick_adj)



#%%

gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/042_degrees_slope/'

path_inv = gen_path + 'binv/8/042_rt_binv_marm_slope_function_deg.csv'

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
        self.tt_i_TL1801nv_ = read_results(path,17)
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
   
        

p_inv = Param_class(path_inv)



def calculate_slope(degrees, spot_x, spot_z,inp=inp3_adj, plot=False):
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
        plot_mig_image(inp, ax, -az)
        plt.plot(point_x/1000,point_z/1000,linewidth=3)
        plt.scatter(spot_x/1000,spot_z/1000,c='k')
        plt.xlabel('Distance (km)')
        plt.ylabel('Depth (km)')
    return p1, p2, p3



def write_json_file(json_file, input_val,degree):
    # Step 1: Read the JSON file
    with open(json_file+'.json', 'r') as file:
        data = json.load(file)
    
    # Step 2: Modify the dictionary
    data['raytracing_config']['interfaces_data'][1]['a'] = input_val['a']
    data['raytracing_config']['interfaces_data'][1]['b'] = input_val['b']
    data['raytracing_config']['interfaces_data'][1]['c'] = input_val['c']
    data['raytracing_config']['interfaces_data'][1]['d'] = input_val['d']
    
    # Step 3: Write the modified dictionary back to the JSON file
    with open(json_file+degree+'.json', 'w') as file:
        json.dump(data, file, indent=4)
    return data




# normales = []
# d_values = []
# dict_input = []
# for i in range(7,12,1):
#     pt_inv1, pt_inv2, pt_inv3 = calculate_slope(i, p_inv.spot_x_[0], p_inv.spot_z_[0])     
#     normales.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[0])
#     d_values.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[1])
#     calculate_angle([[-pt_inv1[2],-pt_inv2[2]],[pt_inv1[0],pt_inv2[0]]])
#     dict_input.append(print_input_values_rt(pt_inv1,pt_inv2,normales[-1],d_values[-1]))
    
#     df = pd.DataFrame([pt_inv1,pt_inv2])
#     df.to_csv(demig_path+'/degrees/slope_degree_'+str(i)+'.csv',header=False,index=False)
#     write_json_file(json_file, dict_input[-1],'degree_'+str(i))


 # df = pd.DataFrame([point1_inv,point2_inv])
 # df.to_csv(demig_path+'/slope_binv_6.csv',header=False,index=False)
   

fl3_adj = '../output/40_marm_ano/badj/inv_betap_x.dat'
fl3_inv = '../output/40_marm_ano/binv/inv_betap_x.dat'
inp3_adj = gt.readbin(fl3_adj,nz,nx) 
inp3_inv = gt.readbin(fl3_inv,nz,nx) 

p_inv.spot_x_[0] = 3360.
p_inv.spot_z_[0] = -1080.
pt_inv1, pt_inv2, pt_inv3 = calculate_slope(35, p_inv.spot_x_[0], p_inv.spot_z_[0],inp=inp3_inv, plot=True)
normales.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[0])
d_values.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[1])
dict_input.append(print_input_values_rt(pt_inv1,pt_inv2,normales[-1],d_values[-1]))
# write_json_file(json_file, dict_input[-1],'steep_35_degree_'+str(i))


base_path = '/home/vcabiativapico/local/src/victor/out2dcourse/input/45_marm_ano_v3'

df = pd.DataFrame([pt_inv1, pt_inv2])
df.to_csv(base_path+'/slope_binv_35.csv',header=False,index=False)


p_inv.spot_x_[0] = 3360.
p_inv.spot_z_[0] = -1094.
pt_inv1, pt_inv2, pt_inv3 = calculate_slope(37, p_inv.spot_x_[0], p_inv.spot_z_[0],inp=inp3_adj, plot=True)
normales.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[0])
d_values.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[1])
# dict_input.append(print_input_values_rt(pt_inv1,pt_inv2,normales[-1],d_values[-1]))
# write_json_file(json_file, dict_input[-1],'steep_37_degree_'+str(i))

# base_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/input/45_marm_ano_v3/'


df = pd.DataFrame([pt_inv1, pt_inv2])
df.to_csv(base_path+'/slope_binv_37.csv',header=False,index=False)

#%%

# fl3_adj = '../output/40_marm_ano/badj/inv_betap_x.dat'


# fl3_inv = '../output/40_marm_ano/binv/inv_betap_x.dat'

json_file = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/048_sm8_correction_new_solver/046_sm8_QTV_degree_37'

fl3_inv = '../output/45_marm_ano_v3/mig_binv_sm8_TL1801/inv_betap_x_s.dat'
fl3_adj = '../output/45_marm_ano_v3/mig_badj_sm8_TL1801/inv_betap_x_s.dat'

inp3_adj = gt.readbin(fl3_adj,nz,nx) 
inp3_inv = gt.readbin(fl3_inv,nz,nx) 

normales = []
d_values = []
dict_input = []
p_inv.spot_x_[0] = 3396.
p_inv.spot_z_[0] = -1058.
pt_inv1, pt_inv2, pt_inv3 = calculate_slope(37, p_inv.spot_x_[0], p_inv.spot_z_[0],inp=inp3_inv, plot=True)
normales.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[0])
d_values.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[1])
dict_input.append(print_input_values_rt(pt_inv1,pt_inv2,normales[-1],d_values[-1]))
# write_json_file(json_file, dict_input[-1],'045_sm8_QTV_degree_'+str(i))

base_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/048_sm8_correction_new_solver'


df = pd.DataFrame([pt_inv1, pt_inv2])
df.to_csv(base_path+'/slope_QTV_sm8.csv',header=False,index=False)


normales = []
d_values = []
dict_input = []
p_inv.spot_x_[0] = 3396.
p_inv.spot_z_[0] = -1065.
pt_inv1, pt_inv2, pt_inv3 = calculate_slope(39, p_inv.spot_x_[0], p_inv.spot_z_[0],inp=inp3_adj, plot=True)
normales.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[0])
d_values.append(plot_plane_from_points(pt_inv1,pt_inv2,pt_inv3)[1])
dict_input.append(print_input_values_rt(pt_inv1,pt_inv2,normales[-1],d_values[-1]))
# write_json_file(json_file, dict_input[-1],'045_sm8_STD_degree_'+str(i))

base_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/048_sm8_correction_new_solver'


df = pd.DataFrame([pt_inv1, pt_inv2])
df.to_csv(base_path+'/slope_STD_sm8.csv',header=False,index=False)

