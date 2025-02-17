#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 15:42:54 2024

@author: vcabiativapico
"""

import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from spotfunk.res import procs
import csv
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from matplotlib.gridspec import GridSpec
from tqdm import tqdm
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

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
no =251
do = dx
fo = -(no-1)/2*do
ao = fo + np.arange(no)*do
at = ft + np.arange(nt)*dt
az = fz + np.arange(nz)*dz
ax = fx + np.arange(nx)*dx


def plot_mig(inp, flout):
    plt.rcParams['font.size'] = 25

    hmin = np.min(inp)
    hmax = -hmin
    # hmax = np.max(inp)
    hmax = 3.5
    
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', cmap='viridis')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig1, format='%1.1f',label='km/s')
    # fig.tight_layout()

    # print("Export to file:", flout)
    # fig.savefig(flout, bbox_inches='tight')
    return inp, fig

def read_results(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
        # attr = [x for x in attr if str(x) != 'nan']
    return attr


def calculate_slope(degrees, spot_x, spot_z, plot=False):
    m = np.tan(degrees * np.pi/180) 
    b = spot_z - m * spot_x
    point_x = np.array([spot_x - 150, spot_x + 150])
    point_z = point_x * m + b
    plt.figure(figsize=(10,8))
    # plt.plot(point_x,point_z,'k')
    # plt.scatter(spot_x,spot_z)
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


def plot_mig_min_max(inp, hmin,hmax):

    plt.rcParams['font.size'] = 25
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', cmap='seismic')
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')

    plt.colorbar(hfig1, format='%1.1f',label='m/s')

    return inp, fig


def convert_slowness_to_vel(inp):
    inp = inp.reshape(nz*nx)
    inp_corr_amp = [0]*(nz*nx) 
    for i,x in enumerate(inp):
        inp_corr_amp[i] = 1/np.sqrt(inp[i])
    inp_corr_amp = np.reshape(inp_corr_amp,(nz,nx))
    return inp_corr_amp




fl_mig = '../output/78_marm_sm8_thick_sum_pert/ano/inv_betap_x_s.dat'
inp_mig = gt.readbin(fl_mig,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig_min_max(inp_mig,np.min(inp_mig), -np.min(inp_mig))



fl_sm = '../input/68_thick_marm_ano/marm_thick_org_sm8.dat'
inp_sm = gt.readbin(fl_sm,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_sm,flout)

flnm_org = '../input/78_marm_sm8_thick_sum_pert/full_org.dat'
inp_org = gt.readbin(flnm_org,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_org,flout)


flnm_ano = '../input/78_marm_sm8_thick_sum_pert/full_ano.dat'
inp_ano = gt.readbin(flnm_ano,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano,flout)



flnm_ano_old = '../input/68_thick_marm_ano/marm_thick_org.dat'
# flnm_ano = '../input/vel_full.dat'
inp_ano_old = gt.readbin(flnm_ano_old,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano_old,flout)




gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'

path1 = gen_path + '074_thick_ano_pert_sm8/depth_demig_out/074_thick_marm_org_sm_2024-11-04_14-09-32/results/depth_demig_output.csv'

src_x = np.array(read_results(path1,1))
src_y = np.array(read_results(path1,2))
src_z = np.array(read_results(path1,3))    
rec_x = np.array(read_results(path1,4))  
rec_y = np.array(read_results(path1,5))    
rec_z = np.array(read_results(path1,6))
spot_x = np.array(read_results(path1,7)) 
spot_y = np.array(read_results(path1,8))
spot_z= np.array(read_results(path1,9))
off_x  = np.array(read_results(path1,16))
tt_inv = np.array(read_results(path1,17))
   
path_ray_org = gen_path + '074_thick_ano_pert_sm8/depth_demig_out/074_thick_marm_org_sm_2024-11-04_14-09-32/rays/ray_99.csv'
    
ray_x  = np.array(read_results(path_ray_org, 0))
ray_z  = np.array(read_results(path_ray_org, 2))
ray_tt = np.array(read_results(path_ray_org, 8))

fig = plt.plot(ray_x/1000,-ray_z/1000,'w')


fl1 = '../input/vel_full.dat'

inp1 = gt.readbin(fl1, nz, nx)  # model


fl1   = '../output/68_thick_marm_ano/org_thick/p2d_fwi_000219.dat'
fl2   = '../output/68_thick_marm_ano/ano_thick/p2d_fwi_000219.dat'

fl1   = '../output/80_smooth_ano_sum_pert/sim_org/p2d_fwi_000291.dat'
fl2   = '../output/80_smooth_ano_sum_pert/sim_ano/p2d_fwi_000291.dat'

fl1   = '../output/80_smooth_ano_sum_pert/sim_org/p2d_fwi_000273.dat'
fl2   = '../output/80_smooth_ano_sum_pert/sim_ano/p2d_fwi_000273.dat'

inp_mig = gt.readbin(fl_mig,nz,nx)

    
flout = '../png/inv_betap_x_s.png'
plot_mig_min_max(inp_mig,np.min(inp_mig), -np.min(inp_mig))


fl_sm = '../input/68_thick_marm_ano/marm_thick_org_sm8.dat'
inp_sm = gt.readbin(fl_sm,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_sm,flout)

flnm_org = '../input/78_marm_sm8_thick_sum_pert/full_org.dat'
inp_org = gt.readbin(flnm_org,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_org,flout)


flnm_ano = '../input/78_marm_sm8_thick_sum_pert/full_ano.dat'
inp_ano = gt.readbin(flnm_ano,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano,flout)


flnm_ano_new = '../input/80_smooth_ano_sum_pert/full_ano_mod_5p.dat'
inp_ano_new = gt.readbin(flnm_ano_new,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano_new,flout)

test = inp_org-inp_ano_new
plot_mig(test*10,flout)


inp_mig_vel = convert_slowness_to_vel(inp_mig) 
    




nt    = 1801

nxl   = 291
h_nxl = int((nxl-1)/2)

org =  -gt.readbin(fl1, nz, nxl*nt) 
ano  = -gt.readbin(fl2, nz, nxl*nt)   #


# left_p  = 300-75 - h_nxl  # left point
# right_p = 300-75 + h_nxl  # right point

left_p  = 300-9 - h_nxl  # left point
right_p = 300-9 + h_nxl  # right point

left_p  = 300-27 - h_nxl  # left point
right_p = 300-27 + h_nxl  # right point


org = np.reshape(org, (nz, nt, nxl))
ano = np.reshape(ano, (nz, nt, nxl))   

idx_min_z =np.argmin(ray_z)

ray_x1 = ray_x[:idx_min_z][::-1]
ray_x2 = ray_x[idx_min_z:]

ray_z1 = ray_z[:idx_min_z][::-1]
ray_z2 = ray_z[idx_min_z:]




degree = 37
pt_inv1, pt_inv2, pt_inv3 = calculate_slope(degree,spot_x[99],spot_z[99])


def plot_sim_rt(bg):
    hmax = np.max(inp1)
    print('hmax: ', hmax)
    hmax = 1
    hmin = -hmax
    for i in range(0,len(ray_x2),80):
        plt.rcParams['font.size'] = 18
        fig = plt.figure(figsize=(10, 10), facecolor="white")
        av = plt.subplot(1, 1, 1)
     
        hfig1 = av.imshow(bg[:, left_p:right_p], extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                          aspect=1, alpha=1,
                          cmap='viridis')
        av.set_title('Raytracing propagation\n t = '+str(int(i*dt*1000)*2)+' s')
        av.scatter(ray_x1[:i*2]/1000,-ray_z1[:i*2]/1000,c='w',s=0.1)
        av.scatter(ray_x2[:i]/1000,-ray_z2[:i]/1000,c='w',s=0.1)
        if i > 80*7-6:
            av.scatter(rec_x[99]/1000,0.04,marker='*',s=100,c='r')
            av.scatter(src_x[99]/1000,0.04,marker='v',s=100,c='g')
        av.set_xlabel('Distance (km)')
        av.set_ylabel('Depth (km)')
        av.scatter(spot_x/1000,-spot_z/1000,c='r',s=1.5)
        divider = make_axes_locatable(av)
        cax = divider.append_axes("right", size="4%", pad=0.25)
        plt.colorbar(hfig1, cax=cax)
        av.plot(np.array([pt_inv1[0],pt_inv2[0]])/1000, np.array([-pt_inv1[2],-pt_inv2[2]])/1000, 'r')
        fig.tight_layout()
    
        flout2 = '../png/80_displays/rt/org_sim_rt_'+str(i)+'.png'
        print("Export to file:", flout2)
        fig.savefig(flout2, bbox_inches='tight')
    

# plot_sim_rt(inp1)

def plot_sim_wf(bg, inp1):
    hmax = np.max(inp1)
    print('hmax: ', hmax)
    hmax = 1
    hmin = -hmax
    plt.rcParams['font.size'] = 20
    for i in range(800, 1400, 100):
        fig = plt.figure(figsize=(10, 10), facecolor="white")
        av = plt.subplot(1, 1, 1)
        palette = sns.color_palette("coolwarm",as_cmap=True)
        palette2 = sns.color_palette("Greys", as_cmap=True)
        
        hfig = av.imshow(inp1[:, i, :]*115, extent=[ax[left_p], ax[right_p], az[-1], az[0]],
                         vmin=hmin, vmax=hmax, aspect=1, alpha=1,
                         cmap=palette)
        hfig1 = av.imshow(bg[:,left_p:right_p],extent=[ax[left_p],ax[right_p],az[-1],az[0]],
                          aspect=1, alpha= 0.4,cmap='Greys')
   



        # plt.plot(ray_x/1000,-ray_z/1000,'w')
        
        av.set_title('Wavefield propagation \n t = '+str(int(i*dt*1000))+' ms')
        av.set_xlabel('Distance (km)')
        av.set_ylabel('Depth (km)')
        divider = make_axes_locatable(av)
        cax = divider.append_axes("bottom", size="5%", pad=0.8)
        plt.colorbar(hfig, cax=cax, orientation='horizontal',label='amplitude')
        fig.tight_layout()
        flout2 = '../png/80_displays/fwi/sim_fwi_'+str(i)+'.png'
        print("Export to file:", flout2)
        fig.savefig(flout2, bbox_inches='tight')
        
    print(np.shape(bg))
    print(np.shape(inp1))

plot_sim_wf(inp_ano, org-ano)

#%%
import cv2
import os


# image_folder = '/home/vcabiativapico/local/src/victor/out2dcourse/png/78_marm_sm8_thick_sum_pert/shots/full'
# video_name = '/home/vcabiativapico/local/src/victor/out2dcourse/png/78_marm_sm8_thick_sum_pert/shots/full/video.avi'
# video_name_mp4 = '/home/vcabiativapico/local/src/victor/out2dcourse/png/78_marm_sm8_thick_sum_pert/shots/full/video'


# image_folder = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/marmousi'
# video_name = '/home/vcabiativapico/local/
flnm_ano_old = '../input/80_smooth_ano_sum_pert/full_ano_mod.dat'
# flnm_ano = '../input/vel_full.dat'
inp_ano_old = gt.readbin(flnm_ano_old,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano_old,flout)

# video_name_mp4 = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/marmousi/video'

# image_folder = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/maps/amplitude'
# video_name = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/maps/amplitude/video.avi'
# video_name_mp4 = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/maps/amplitude/video'

image_folder = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/rays'
video_name = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/rays/video.avi'
video_name_mp4 = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/rays/video'

# image_folder = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/rays_points'
# video_name = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/rays_points/video.avi'
# video_name_mp4 = '/home/vcabiativapico/local/src/victor/out2dcourse/png/80_displays/rays_points/video'



images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
frame = cv2.imread(os.path.join(image_folder, images[0]))
height, width, layers = frame.shape

video = cv2.VideoWriter(video_name, 0, 10, (width,height))

images2  = images.sort(reverse=False)
for image in images:
flnm_ano_old = '../input/80_smooth_ano_sum_pert/full_ano_mod.dat'
# flnm_ano = '../input/vel_full.dat'
inp_ano_old = gt.readbin(flnm_ano_old,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano_old,flout)


    video.write(cv2.imread(os.path.join(image_folder, image)))

cv2.destroyAllWindows()
video.release()

def convert_avi_to_mp4(avi_file_path, output_name):
    os.popen("ffmpeg -i '{input}' -ac 2 -b:v 2000k -c:a aac -c:v libx264 -b:a 160k -vprofile high -bf 0 -strict experimental -f mp4 '{output}.mp4'".format(input = avi_file_path, output = output_name))
    return True

convert_avi_to_mp4(video_name, video_name_mp4)


#%%
shot_nb = np.arange(152,350)
plt.rcParams['font.size'] = 20
hmin = np.min(inp_org)
# hmax = -hmin
hmax = 3.5




for i in range(len(shot_nb)):
    fig = plt.figure(figsize=(14, 7), facecolor="white")
    av = plt.subplot(1, 1, 1)
    hfig1 = av.imshow(inp_org, extent=[ax[0], ax[-1], az[-1], az[0]],
                      vmin=hmin, vmax=hmax, aspect='auto', cmap='viridis')

    
    
    for j in range(-no//2+1,no//2+1,25):
        if j != 0:
            plt.scatter((shot_nb[i]+j)*dx,0.05, marker='v',c='w',s=150)
    plt.scatter(shot_nb[i]*dx,0.05, marker='*',c='r',s=250)
    # plt.scatter(recs2[i]*dx,0.05, marker='v',c='w',s=250)
    plt.xlabel('Distance (km)')
    plt.ylabel('Depth (km)')
    plt.colorbar(hfig1, format='%1.1f',label='km/s')
    fig.tight_layout()
    flout2 = '../png/80_displays/marmousi/'+str(i).zfill(3)+'.png'
    print("Export to file:", flout2)
    fig.savefig(flout2, bbox_inches='tight')
    
#%%

''' Ray-tracing rays'''

for i in range(0,125,4):
    path_ray_org = gen_path + '074_thick_ano_pert_sm8/depth_demig_out/074_thick_marm_org_sm_2024-11-04_14-09-32/rays/ray_'+str(i)+'.csv'
    
    ray_x  = np.array(read_results(path_ray_org, 0))
    ray_z  = np.array(read_results(path_ray_org, 2))
    ray_tt = np.array(read_results(path_ray_org, 8))   
    
    
    plt.rcParams['font.size'] = 18
    fig = plt.figure(figsize=(10, 10), facecolor="white")
    av = plt.subplot(1, 1, 1)
     
    hfig1 = av.imshow(inp_sm, extent=[ax[0]
flnm_ano_old = '../input/80_smooth_ano_sum_pert/full_ano_mod.dat'
# flnm_ano = '../input/vel_full.dat'
inp_ano_old = gt.readbin(flnm_ano_old,nz,nx)
flout = '../png/inv_betap_x_s.png'
plot_mig(inp_ano_old,flout)

, ax[-1], az[-1], az[0]],
                      aspect=1, alpha=1,
                      cmap='viridis')
    av.set_title('Raytracing propagation\n offset = '+str(i*12)+' m')
    av.scatter(ray_x/1000,-ray_z/1000,c='w',s=0.1)
    av.scatter(rec_x[i]/1000,0.04,marker='*',s=100,c='r')
    av.scatter(src_x[i]/1000,0.04,marker='v',s=100,c='g')
    av.set_xlabel('Distance (km)')
    av.set_ylabel('Depth (km)')
    av.scatter(spot_x/1000,-spot_z/1000,c='r',s=1.5)
    divider = make_axes_locatable(av)
    cax = divider.append_axes("right", size="4%", pad=0.25)
    plt.colorbar(hfig1, cax=cax)
    av.plot(np.array([pt_inv1[0],pt_inv2[0]])/1000, np.array([-pt_inv1[2],-pt_inv2[2]])/1000, 'r')
    fig.tight_layout()
    
    flout2 = '../png/80_displays/rays/'+str(125-i).zfill(3)+'rev.png'
    print("Export to file:", flout2)
    fig.savefig(flout2, bbox_inches='tight')
            
#%%

''' '''

for i in range(0,125,4):
    plt.rcParams['font.size'] = 22
    fig, (ax0) = plt.subplots(figsize=(10,10),nrows=1)
    ax0.scatter(src_x[-1],off_x[-1],marker='.', c='w',label='RT')
    ax0.scatter(src_x[0],off_x[-1],marker='o', c='w',label='RT')
    ax0.scatter(rec_x[-1],-off_x[-1],marker='.', c='w',label='RT')
    ax0.scatter(src_x[::-1][:i],off_x[::-1][:i],marker='.', c='k',label='RT')
    
    ax0.set_title('Source x Offset map in raytracing')
    ax0.set_xlabel('Source x')
    ax0.set_ylabel('Offset x')
    flout2 = '../png/80_displays/rays_points/'+str(125-i).zfill(3)+'.png'
    print("Export to file:", flout2)
    fig.savefig(flout2, bbox_inches='tight')
    
    
plt.rcParams['font.size'] = 22
fig, (ax0) = plt.subplots(figsize=(10,10),nrows=1)
ax0.scatter(src_x[-1]/1000,off_x[-1]/1000,marker='.', c='w',label='RT')
ax0.scatter(src_x[0]/1000,off_x[-1]/1000,marker='o', c='w',label='RT')
ax0.scatter(rec_x[-1]/1000,-off_x[-1]/1000,marker='o', c='w',label='RT')
ax0.scatter(src_x/1000,off_x/1000,marker='.', c='k',label='RT')
ax0.scatter(rec_x/1000,-off_x/1000,marker='.', c='k')
ax0.set_title('Source x Offset map in raytracing')
ax0.set_xlabel('Source x')
ax0.set_ylabel('Offset x')
flout2 = '../png/80_displays/rays_points/'+str(125-i).zfill(3)+'.png'
print("Export to file:", flout2)
fig.savefig(flout2, bbox_inches='tight')





