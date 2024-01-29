#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 18:07:26 2023

@author: vcabiativapico
"""

import numpy as np
import matplotlib.pyplot as plt
from math import cos, sin
from numba import jit
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import lsqr
import time
import csv
import sys
from spotfunk.res.input import segy_reader
from spotfunk.res import bspline
import geophy_tools as gt


nt = 1501
dt = 1.41e-3
ft = -100.11e-3
nz = 151
fz = 0.0
dz = 12.0/1000.
nx = 601
fx = 0.0
dx = 12.0/1000.
no =251
# no        = 2002
do = dx
fo = -(no-1)/2*do
ao = fo + np.arange(no)*do
at = ft + np.arange(nt)*dt
az = fz + np.arange(nz)*dz
ax = fx + np.arange(nx)*dx

## Read the original smoothed horizon
def read_pick(path,srow):
    rec_x = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            rec_x.append(float(row[srow]))
    return rec_x

## Read the results from demigration
def read_results(path,srow):
    rec_x = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            rec_x.append(float(row[srow]))
        rec_x = [x for x in rec_x if str(x) != 'nan']
    return rec_x

#%%

nt = 1501
dt = 1.41e-3
ft = -100.11e-3
nz = 151
fz = 0.0
dz = 12.0/1000.
nx = 601
fx = 0.0
dx = 12.0/1000.
no =251
# no        = 2002
do = dx
fo = -(no-1)/2*do
ao = fo + np.arange(no)*do
at = ft + np.arange(nt)*dt
az = fz + np.arange(nz)*dz
ax = fx + np.arange(nx)*dx


name = 'adj'
path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/011_marm_sm_b'+str(name)+'_PP21_P021_hz02.csv'


hz_inv_marm_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_sm3_marm_'+str(name)+'_02.csv'
hz_inv_marm_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_marm_'+str(name)+'_02.csv'
hz_inv_marm_sm = read_pick(hz_inv_marm_sm,0)   
hz_inv_marm_org = read_pick(hz_inv_marm_org,0) 

fl1 = '../input/27_marm/marm2_sm15.dat'
inp1 = gt.readbin(fl1,nz,nx)

 
idx = read_results(path,0)  
src_x = read_results(path,1)  
src_y = read_results(path,2)
src_z = read_results(path,3)    
rec_x = read_results(path,4)  
rec_y = read_results(path,5)    
rec_z = read_results(path,6)
spot_x = read_results(path,7) 
spot_y = read_results(path,8)
spot_z= read_results(path,9)

offset = read_results(path,16)

line = np.zeros_like((src_x))

x_disc = np.arange(601)*12.00

colors = np.random.rand(line.size)
colors = offset

plt.figure(figsize= (16,12))
plt.scatter(src_x,src_y,c=colors,marker='*',cmap='jet')
# plt.figure(figsize= (16,12))
plt.scatter(rec_x,rec_y,c=colors,marker='v',cmap='jet')
plt.scatter(spot_x,spot_y,c='r')


plt.figure(figsize= (10,10))
plt.scatter(src_x,line,c=colors,marker='*',cmap='jet')
# plt.figure(figsize= (16,12))
hmin,hmax = 1.5,4.5
plt.imshow(inp1, vmin=hmin, vmax=hmax, 
                 extent=[ax[0]*1000, ax[-1]*1000, -at[-1]*1000, -at[0]*1000],
                 aspect='auto',cmap='seismic')
plt.scatter(rec_x,line,c=colors,marker='v',cmap='jet')
plt.plot(spot_x,spot_z,'r.')
plt.plot(x_disc,-np.array(hz_inv_marm_sm),'k')
plt.xlim(2000,4800)
plt.colorbar()

#%%
import matplotlib.pyplot as plt
from kneed import KneeLocator
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


name = 'inv'
path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/output/011_marm_sm_b'+str(name)+'_PP21_P021_hz02.csv'


hz_inv_marm_sm = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_sm3_marm_'+str(name)+'_02.csv'
hz_inv_marm_org = '../../../../Demigration_SpotLight_Septembre2023/Demigration_Victor/pick/27_hz_marm_'+str(name)+'_02.csv'
hz_inv_marm_sm = read_pick(hz_inv_marm_sm,0)   
hz_inv_marm_org = read_pick(hz_inv_marm_org,0) 

idx = read_results(path,0)  
src_x = read_results(path,1)  
src_y = read_results(path,2)
src_z = read_results(path,3)    
rec_x = read_results(path,4)  
rec_y = read_results(path,5)    
rec_z = read_results(path,6)
spot_x = read_results(path,7) 
spot_y = read_results(path,8)
spot_z= read_results(path,9)
offset = read_results(path,16)


line = np.zeros_like((src_x))

x_disc = np.arange(601)*12.00

colors = np.random.rand(line.size)
colors = offset

src_pos = np.zeros((np.size(src_x),2))
for i in range(np.size(src_x)):
    src_pos[i] = [src_x[i],rec_x[i]]
    

scaler = StandardScaler()
scaled_features = scaler.fit_transform(src_pos)



kmeans = KMeans(
    init="random",
    n_clusters=3,
    n_init=10,
    max_iter=300,
    random_state=42
)

kmeans.fit(scaled_features)

# The lowest SSE value
kmeans.inertia_

# Final locations of the centroid
kmeans.cluster_centers_

# The number of iterations required to converge
kmeans.n_iter_

fig, (ax1) = plt.subplots(1, 1, figsize=(16, 8))
fig.suptitle(f"Clustering ", fontsize=16)
fte_colors = {0: "r", 1: "b",2: "k"}
# The k-means plot
km_colors = [fte_colors[label] for label in kmeans.labels_]
ax1.scatter(src_x, src_y, c=km_colors)
ax1.scatter(rec_x,rec_y, c=km_colors)
ax1.scatter(spot_x,spot_y, c='green',marker='*')
# ax1.scatter(scaled_features[:, 0], scaled_features[:, 1], c=km_colors)
ax2 = ax1.twiny()  
ax2.scatter(offset,np.array(src_y)*2000)
ax2.spines.top.set_position(("axes", 0))
ax1.set_xlabel('Distance (m)', loc='left')
ax2.set_xlabel(' Offset (m)',loc='left')
plt.ylim(-2.9,2.9)


fig = plt.figure(figsize=(16, 8), facecolor = "white")
plt.suptitle(f"Offset", fontsize=16)
ax1 = fig.add_subplot(111)
ax2 = ax1.twiny()  
ax1.scatter(src_x,src_y,c=colors,marker='*',cmap='jet')
ax1.scatter(rec_x,rec_y,c=colors,marker='v',cmap='jet')
ax1.scatter(spot_x,spot_y,c='k')
ax2.scatter(offset,np.array(src_y)*2000)
ax2.spines.top.set_position(("axes", 0))
ax1.set_xlabel('Distance (m)', loc='left')
ax2.set_xlabel(' Offset (m)',loc='left')
plt.ylim(-2.9,2.9)



kmeans_kwargs = {
    "init": "random",
    "n_init": 10,
    "max_iter": 300,
    "random_state": 42,
}
# A list holds the SSE values for each k
sse = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
    kmeans.fit(scaled_features)
    sse.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), sse)
plt.xticks(range(1, 11))
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()






