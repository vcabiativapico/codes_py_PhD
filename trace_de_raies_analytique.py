#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 09:28:15 2023

@author: vcabiativapico
"""

import numpy as np
import sys
from math import pi, sqrt, exp, sin, cos, asin
import matplotlib.pyplot as plt
from tqdm import trange
import geophy_tools as gt


### ONDELETTE 
from scipy.signal import ricker


def ricker_creation(freq_principale,si=0.001):
    
   points = 1501 #prendre un nombre impair pour centrer l'ondelette
   a1=np.sqrt(2)/(2*3.14*freq_principale*si)
   le_ricker = ricker(points, a1)

   return le_ricker
    


nx    = 601 # Dimensions du modele
dx    = 12/1000
no    = 251
nt    = 1501
nz    = 151
dt    = 1.41e-3
do    = dx
dz    = 12.0/1000.
fo    = -(no-1)/2*do
ft    = -100.11e-3
fz    = 0
ao    = fo + np.arange(no)*do
at    = ft + np.arange(nt)*dt
az    = fz + np.arange(nz)*dz
fx    = 0


max_a  = 50   # Angle max
min_a  = -50  # angle min
nang   = 50     # numero d'angles
nsrc   = 50    # Numero de sources 

# ixsrct = np.transpose([ixsrc])
# ixsrc = np.append(ixsrct,ixsrct, axis=0

h1     = dz * 49+dz

h2     = dz * 101+dz
deg1   = np.linspace(min_a,max_a,nang)

v1     = 2.00
v2     = 2.06

t0     = 2*h1 / v1

t1     = np.sqrt(t0**2 + (ao/v1)**2)-ft

t_2    = 2*h2 / v2

vrms1  = np.sqrt((v1**2 * t0 + v2**2 * t_2) / (t_2 + t0))



#t01    = (2*h1 / v1) + (2*h2/v2)
t02    = (2*h1 / v1) + (2*h2/v2)

t2     = np.sqrt(t_2**2 + (ao/vrms1)**2)-ft



## PLOT LINES OF x VS t

# fig1  = plt.figure(figsize=(10,5), facecolor = "white") 
# av1   = plt.subplot(1,1,1)

# plt.plot(t0)
# plt.plot(ao,t1)
# plt.plot(ao,t2)


# #plt.ylim(0,0.5)
# plt.xlim([ao[0],ao[-1]])
# plt.ylim([at[-1],at[0]])

###### Convert into a matrix

ao_conv  = len(ao) # Read the axes to keep same size
at_conv  = len(at)
 

inp_x    = np.zeros((at_conv,ao_conv)) # initilize the matrix with zeros


## By dividing the result in t1 and t2 by dt,
# we find a position in the matrix of zeros which is replaced by ones
## A mean between two samples is calculated in order to have the peak between the two of them


for i in range(no):
    n = np.round(t1[i]/dt)
    n = n.astype(int)    # Find the index and convert to integer
    inp_x[n,i]   = ((n+1)*dt - t1[i]) / dt 
    inp_x[n+1,i] = (t1[i]- n*dt) / dt
    n_p1 = np.round(t2[i]/dt)
    n_p1 = n_p1.astype(int)
    inp_x[n_p1,i] = ((n+1)*dt - t1[i]) / dt
    inp_x[n_p1+1,i] = (t1[i]- n*dt) / dt

wlet = ricker_creation(10,si=dt)

inp_w = np.zeros_like(inp_x)

for i in range(no):
    inp_w[:,i] = np.convolve(inp_x[:,i],wlet,mode='same')






hmax = np.max(np.abs(inp_w))/2
hmax = 0.3
hmin = -hmax

fig  = plt.figure(figsize=(10,8), facecolor = "white")
av   = plt.subplot(1,1,1)
hfig = av.imshow(inp_x, extent=[ao[0],ao[-1],at[-1],at[0]], aspect='auto',\
                  vmin=hmin,vmax=hmax,cmap='seismic')


print('hmax: ',hmax)
fig  = plt.figure(figsize=(10,8), facecolor = "white")
av   = plt.subplot(1,1,1)
hfig = av.imshow(inp_w, extent=[ao[0],ao[-1],at[-1],at[0]], aspect='auto',\
                  vmin=hmin,vmax=hmax,cmap='seismic')
flout  = '../png/12_wf_simple/trace_de_raies.png'
print("Export to file:",flout)
fig.tight_layout()
fig.savefig(flout, bbox_inches='tight')

#cbar = plt.colorbar(hfig, format='%.e')
##### PLOT OBSERVED FROM BORN ####


    # Global parameters
# labelsize = 16
# nt        = 1501
# dt        = 1.41e-3
# ft        = -100.11e-3
# nz        = 151
# fz        = 0.0
# dz        = 12.0/1000.
# nx        = 601
# fx        = 0.0
# dx        = 12.0/1000.
# no        = 251
# do        = dx
# fo        = -(no-1)/2*do
# #ao        = fo + np.arange(no)*do
# #at        = ft + np.arange(nt)*dt
# az        = fz + np.arange(nz)*dz
# ax        = fx + np.arange(nx)*dx

hmin,hmax = -0.01,0.01


fl1  = '../output/05_simple/born/t1_obs_000301.dat'
# fl1  = './output//t1_obs_000301.dat'
# no   = 403

inp1 = gt.readbin(fl1,no,nt).transpose()
hmax = np.max(np.abs(inp1))/2
hmin = -hmax


## For plotting the shots after modelling

flout  = '../png/12_wf_simple/obs_0301_born_2060.png'

fig  = plt.figure(figsize=(10,8), facecolor = "white")
av   = plt.subplot(1,1,1)
hfig = av.imshow(-inp1, extent=[ao[0],ao[-1],at[-1],at[0]], \
                  vmin=hmin,vmax=hmax, aspect='auto',\
                  cmap='seismic')
# hfig = av.imshow(inp_x, extent=[ao[0],ao[-1],at[-1],at[0]], aspect='auto',\
#                       vmin=hmin,vmax=hmax,alpha=0.7,cmap='seismic')
    #plt.colorbar(hfig)
fig.tight_layout()
#cbar = plt.colorbar(hfig, format='%.0e')
# print("Export to file:",flout)
# fig.savefig(flout, bbox_inches='tight')

hmin,hmax = -0.01,0.01


fl2  = '../output/05_simple/fwi/t1_obs_000301.dat'

inp2 = gt.readbin(fl2,no,nt).transpose()
# hmax = np.max(np.abs(inp1))/2
# hmin = -hmax

inp3 =inp1+inp2
## For plotting the shots after modelling

flout  = '../png/12_wf_simple/obs_0301_fwi_2060.png'

fig  = plt.figure(figsize=(10,8), facecolor = "white")
av   = plt.subplot(1,1,1)
hfig = av.imshow(inp3, extent=[ao[0],ao[-1],at[-1],at[0]], \
                  vmin=hmin,vmax=hmax, aspect='auto',\
                  cmap='seismic')
# hfig = av.imshow(inp_x, extent=[ao[0],ao[-1],at[-1],at[0]], aspect='auto',\
#                       vmin=hmin,vmax=hmax,alpha=0.7,cmap='seismic')
    #plt.colorbar(hfig)
fig.tight_layout()
# print("Export to file:",flout)
# fig.savefig(flout, bbox_inches='tight')
#cbar = plt.colorbar(hfig, format='%.0e')

