#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 15 17:00:56 2023

@author: spotkev
"""
# %matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# from spotfunk.res import procs
from scipy.ndimage import gaussian_filter, sobel

labelsize = 16
nt = 1001
dt = 2.08e-3
ft = -99.84e-3
nz = 151
fz = 0.0
dz = 12.49/1000.
nx = 601
fx = 0.0
dx = 12.49/1000.
no = 251
do = dx
fo = -(no-1)/2*do
ao = fo + np.arange(no)*do
at = ft + np.arange(nt)*dt
az = fz + np.arange(nz)*dz
ax = fx + np.arange(nx)*dx

pente_1 = 0.4 + 0.02*ax

pente_2 = 1.2 + 0.04*ax



plt.figure(figsize=(16,12))
plt.plot(ax,pente_1)
plt.plot(ax,pente_2)

plt.xlim(ax[0],ax[-1])
plt.ylim(az[0],az[-1])
plt.gca().invert_yaxis()

v_couche_1 = 1500
v_couche_2 = 1750
v_couche_3 = 2500




V_model = np.zeros((nx,nz))

def fill_custom_model(V_model,v_couche_1,v_couche_2,v_couche3):   
    for k in range(V_model.shape[0]):
        for i,z in enumerate(az):
            
            if z < pente_1[k]:
                V_model[k,i] = v_couche_1
                
                if  z+dz > pente_1[k]:
                    
                    V_model[k,i] = (v_couche_1*(pente_1[k]-z)+v_couche_2*(dz-pente_1[k]+z))/dz
                # print(pente_1[k],z,-pente_1[k]+dz+z)
                
            elif z < pente_2[k]:
                V_model[k,i] = v_couche_2 + 2000*(-0.5+z)
                
                if z+dz > pente_2[k]:
                    
                    V_model[k,i] = ((v_couche_2 + 2000*(-0.5+z))*(pente_2[k]-z)+v_couche_3*(dz-pente_2[k]+z))/dz
                
                
            else:
                V_model[k,i] = v_couche_3
            
        
fill_custom_model(V_model,v_couche_1,v_couche_2,v_couche_3)

        
plt.figure(figsize=(16,12))
plt.pcolor(ax,az,V_model.T)
plt.gca().invert_yaxis()
plt.colorbar()


plt.figure(figsize=(16,12))
plt.pcolor(ax,az,gaussian_filter(V_model.T,2))
plt.gca().invert_yaxis()
plt.colorbar        
        
        
        
        
        
        