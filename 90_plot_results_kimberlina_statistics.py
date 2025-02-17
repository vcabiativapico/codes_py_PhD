#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:45:35 2025

@author: vcabiativapico
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file = '../output/90_kimberlina_mod_v3_high/res_vp_value.csv'
data_vp = pd.read_csv(file)


file = '../output/90_kimberlina_mod_v3_high/test_max.csv'
data = pd.read_csv(file)


 
max_total_rms = data['max_rms'].tolist()
max_cc = data['max_cc'].tolist()
year_number = data['year_number'].tolist()

res_vp = []
existing_years = []
idx_existing_years = []
for i in year_number: 
    for j in data_vp['year'].tolist():
        if i == j:
            existing_years.append(i)


        
filtered_df = data_vp[(data_vp["year"] >= np.min(existing_years)) & (data_vp["year"] <= np.max(existing_years))]
               

res_vp_values = filtered_df["res_vp_value"].tolist()

plt.figure(figsize=(10,8))
plt.plot(res_vp_values,max_total_rms)
plt.xlabel('Vp_value at reservoir')
plt.ylabel('maximum rms amplitude')

plt.figure(figsize=(10,8))
plt.plot(res_vp_values,max_cc)
plt.xlabel('Vp_value at reservoir')
plt.ylabel('maximum ts')


plt.figure(figsize=(10,8))
plt.plot(year_number,max_total_rms)
plt.xlabel('year')
plt.ylabel('max_rms')


plt.figure(figsize=(10,8))
plt.plot(year_number,max_cc)
plt.xlabel('year')
plt.ylabel('max_cc')
