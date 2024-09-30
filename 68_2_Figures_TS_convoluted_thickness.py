#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 21:13:51 2024

@author: vcabiativapico
"""


import csv
import numpy as np
import geophy_tools as gt
import matplotlib.pyplot as plt
from spotfunk.res import procs,visualisation
from scipy.interpolate import interpolate,InterpolatedUnivariateSpline
import sympy as sp
import tqdm
import functions_bsplines_new_kev_test_2_5D as kvbsp
from scipy.ndimage import gaussian_filter
from matplotlib.ticker import (MultipleLocator,
                               FormatStrFormatter,
                               AutoMinorLocator)
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches




# Read the results from demigration
def read_results(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
        attr = [x for x in attr if str(x) != 'nan'] 
    attr = np.array(attr)
    attr = np.nan_to_num(attr)
    return attr

ft    = -100.32e-3
title = 60

in_nam = '../output/63_evaluating_thickness/sliding_TS_plot_'+str(title)+'.csv'
in_sld = []
for i in range(6):
    in_sld.append(read_results(in_nam, i))


plt.rcParams['font.size'] = 22

fig = plt.figure(figsize=(25, 11))
gs = GridSpec(1, 4, figure=fig)

ax1 = fig.add_subplot(gs[:, 0])
ax1.plot(in_sld[0], in_sld[1], linewidth=2,label='conv')
ax1.plot(in_sld[2], in_sld[3], linewidth=2,label='FD')
ax1.plot(in_sld[4], in_sld[5], linewidth=2, label = 'theo')
# ax1.plot(in_sld[6], in_sld[7],'.')
ax1.set_title('Sliding TS\n Thickness =' + str(int(title)) + ' m')
ax1.legend()
ax1.set_xlim(-0.1,0.8)
ax1.set_ylim(1.8, ft-0.1)
ax1.set_ylabel('Time (s)')
ax1.set_xlabel('Amplitude')

rect1 = patches.Rectangle((0, 0.95), 0.12, 0.1, linewidth=1, edgecolor='black', facecolor='white', transform=ax1.transAxes)
ax1.add_patch(rect1)
ax1.text(0.03, 0.96, 'A', transform=ax1.transAxes)


title = 108

in_nam = '../output/63_evaluating_thickness/sliding_TS_plot_'+str(title)+'.csv'
in_sld108 = []
for i in range(6):
    in_sld108.append(read_results(in_nam, i))


plt.rcParams['font.size'] = 22

ax2 = fig.add_subplot(gs[:, 1])
ax2.plot(in_sld108[0], in_sld108[1],linewidth=2, label='conv')
ax2.plot(in_sld108[2], in_sld108[3], linewidth=2,label='FD')
ax2.plot(in_sld108[4], in_sld108[5],linewidth=2,label = 'theo')
# ax2.plot(in_sld[6], in_sld[7],'.')
ax2.set_title('Sliding TS\n Thickness = ' + str(int(title)) + ' m')
ax2.legend()
ax2.set_xlim(-0.1,0.8)
ax2.set_ylim(1.8, ft-0.1)

ax2.set_xlabel('Amplitude')

rect1 = patches.Rectangle((0, 0.95), 0.12, 0.1, linewidth=1, edgecolor='black', facecolor='white', transform=ax2.transAxes)
ax2.add_patch(rect1)
ax2.text(0.03, 0.96, 'B', transform=ax2.transAxes)


title = 204

in_nam = '../output/63_evaluating_thickness/sliding_TS_plot_'+str(title)+'.csv'
in_sld204 = []
for i in range(6):
    in_sld204.append(read_results(in_nam, i))


plt.rcParams['font.size'] = 22

ax3 = fig.add_subplot(gs[:, 2])
ax3.plot(in_sld204[0], in_sld204[1],linewidth=2, label='conv')
ax3.plot(in_sld204[2], in_sld204[3],linewidth=2, label='FD')
ax3.plot(in_sld204[4], in_sld204[5],linewidth=2,label = 'theo')
# ax3.plot(in_sld[6], in_sld[7],'.')
ax3.set_title('Sliding TS\n Thickness = ' + str(int(title)) + ' m')
ax3.legend()
ax3.set_xlim(-0.1,0.8)
ax3.set_ylim(1.8, ft-0.1)

ax3.set_xlabel('Amplitude')

rect1 = patches.Rectangle((0, 0.95), 0.12, 0.1, linewidth=1, edgecolor='black', facecolor='white', transform=ax3.transAxes)
ax3.add_patch(rect1)
ax3.text(0.03, 0.96, 'C', transform=ax3.transAxes)

#%%

def read_results(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
        # attr = [x for x in attr if str(x) != 'nan'] 
    attr = np.array(attr)
    attr = np.nan_to_num(attr)
    # attr.astype(float)
    return attr

title = 60

in_nam = '../output/63_evaluating_thickness/picked_TS_plot_df_'+str(title)+'.csv'
in_conv = []
for i in range(2):
    in_conv.append(read_results(in_nam, i))


in_nam = '../output/63_evaluating_thickness/picked_TS_plot_conv_'+str(title)+'.csv'
in_conv2 = []
for i in range(2):
    in_conv2.append(read_results(in_nam, i))


plt.rcParams['font.size'] = 22

fig_conv = plt.figure(figsize=(25, 11))
gs = GridSpec(1, 4, figure=fig)

ax1 = fig_conv.add_subplot(gs[:, 0])
ax1.plot(in_conv[0], in_conv[1],'-o',linewidth=2, label='conv')
ax1.plot(in_conv2[0], in_conv2[1],'-o',linewidth=2, label='DF')
ax1.plot(in_sld[4], in_sld[5],label = 'theo',linewidth=2,)
# ax1.plot(in_conv[6], in_conv[7],'.')
ax1.set_title('Picked TS \n Thickness =' + str(int(title)) + ' m')
ax1.legend()
ax1.set_xlim(-0.1,0.8)
ax1.set_ylim(1.8, ft-0.1)
ax1.set_ylabel('Time (s)')
ax1.set_xlabel('Amplitude')

rect1 = patches.Rectangle((0, 0.95), 0.12, 0.1, linewidth=1, edgecolor='black', facecolor='white', transform=ax1.transAxes)
ax1.add_patch(rect1)
ax1.text(0.03, 0.96, 'A', transform=ax1.transAxes)


title = 108

in_nam = '../output/63_evaluating_thickness/picked_TS_plot_df_'+str(title)+'.csv'
in_conv108 = []
for i in range(2):
    in_conv108.append(read_results(in_nam, i))


in_nam = '../output/63_evaluating_thickness/picked_TS_plot_conv_'+str(title)+'.csv'
in_conv2_108 = []
for i in range(2):
    in_conv2_108.append(read_results(in_nam, i))


plt.rcParams['font.size'] = 22

ax2 = fig_conv.add_subplot(gs[:, 1])
ax2.plot(in_conv108[0], in_conv108[1], '-o', label='conv',linewidth=2)
ax2.plot(in_conv2_108[0], in_conv2_108[1],'-o',  label='FD',linewidth=2)
ax2.plot(in_sld108[4], in_sld108[5],label = 'theo',linewidth=2)
# ax2.plot(in_conv[6], in_conv[7],'.')
ax2.set_title('Picked TS \n Thickness = ' + str(int(title)) + ' m')
ax2.legend()
ax2.set_xlim(-0.1,0.8)
ax2.set_ylim(1.8, ft-0.1)

ax2.set_xlabel('Amplitude')

rect1 = patches.Rectangle((0, 0.95), 0.12, 0.1, linewidth=1, edgecolor='black', facecolor='white', transform=ax2.transAxes)
ax2.add_patch(rect1)
ax2.text(0.03, 0.96, 'B', transform=ax2.transAxes)


title = 204


in_nam = '../output/63_evaluating_thickness/picked_TS_plot_df_'+str(title)+'.csv'
in_conv204 = []
for i in range(2):
    in_conv204.append(read_results(in_nam, i))


in_nam = '../output/63_evaluating_thickness/picked_TS_plot_conv_'+str(title)+'.csv'
in_conv2_204 = []
for i in range(2):
    in_conv2_204.append(read_results(in_nam, i))

plt.rcParams['font.size'] = 22

ax3 = fig_conv.add_subplot(gs[:, 2])
ax3.plot(in_conv204[0], in_conv204[1],'-o',  label='conv',linewidth=2)
ax3.plot(in_conv2_204[0], in_conv2_204[1],'-o',  label='FD',linewidth=2)
ax3.plot(in_sld204[4], in_sld204[5],label = 'theo',linewidth=2)
# ax3.plot(in_conv[6], in_conv[7],'.')
ax3.set_title('Picked TS \n Thickness = ' + str(int(title)) + ' m')
ax3.legend()
ax3.set_xlim(-0.1,0.8)
ax3.set_ylim(1.8, ft-0.1)

ax3.set_xlabel('Amplitude')

rect1 = patches.Rectangle((0, 0.95), 0.12, 0.1, linewidth=1, edgecolor='black', facecolor='white', transform=ax3.transAxes)
ax3.add_patch(rect1)
ax3.text(0.03, 0.96, 'C', transform=ax3.transAxes)

