#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:45:35 2025

@author: vcabiativapico
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


file = '../output/94_kimberlina_v4/res_vp_value_new.csv'
data_vp = pd.read_csv(file)

file1 = '../output/94_kimberlina_v4/_p2_v1_attr_max.csv'
data_attr = pd.read_csv(file1)[::-1]



year_attr = data_attr['year_number'].tolist()


existing_years = []
for i in year_attr: 
    for j in data_vp['year'].tolist():
        if i == j:
            existing_years.append(i)

filtered_data_vp = data_vp[(data_vp["year"] >= np.min(existing_years)) & (data_vp["year"] <= np.max(existing_years))]
filtered_data_attr = data_attr[(data_attr["year_number"] >= np.min(existing_years)) & (data_attr["year_number"] <= np.max(existing_years))]

res_vp_values = filtered_data_vp["mean_vp_diff"].tolist()
year_values = filtered_data_vp["year"].tolist()
vp_percentage = filtered_data_vp["vp_percentage"].tolist()
vp_anomalies_min = filtered_data_vp["vp_anomalies_min"].tolist()
vp_original_min = filtered_data_vp["vp_original_min"].tolist()



max_total_rms = filtered_data_attr["max_rms"].tolist()
max_cc = filtered_data_attr["max_cc"].tolist()



res_slwp_values =  1/(np.array(res_vp_values)*1000)

plt.rcParams['font.size'] = 26
plt.figure(figsize=(10,8))
plt.plot(year_values,max_total_rms,'o-')
plt.xlabel('year')
plt.ylabel('max RMS difference')

plt.figure(figsize=(10,8))
plt.plot(year_values,max_cc,'o-')
plt.xlabel('year')
plt.ylabel('min ts')
plt.gca().invert_yaxis()


plt.figure(figsize=(10,8))
plt.plot(year_values,vp_percentage,'o-')
plt.xlabel('year')
plt.ylabel('% Vp change')


plt.figure(figsize=(10,8))
plt.plot(vp_percentage,max_total_rms,'o-')
plt.xlabel('Diff % Vp')
plt.ylabel('maximum rms amplitude')

plt.figure(figsize=(10,8))
plt.plot(vp_percentage,max_cc,'o-')
plt.xlabel('Diff % Vp')
plt.ylabel('min ts')

#%%
# thickness = np.array([2,3,4,4,4])*0.012 # meters

# ts_theor = []
# for i in range(5):
#     t_theor_original = 2*thickness[i] / vp_original_min[i]
#     t_theor_anomaly  = 2*thickness[i] / (vp_original_min[i] - (vp_original_min[i] * vp_percentage[:5][i]/100))
#     ts_theor.append(t_theor_original - t_theor_anomaly)

# ts_theor = np.array(ts_theor)*1000

# plt.figure(figsize=(10,8))
# plt.plot(vp_percentage[:5],ts_theor,'o-')
# plt.xlabel('Diff % Vp')
# plt.ylabel('ts_theor')

# plt.figure(figsize=(10,8))
# plt.plot(max_cc[:5],ts_theor,'o-')
# plt.xlabel('ts mesuré')
# plt.ylabel('ts_theor')