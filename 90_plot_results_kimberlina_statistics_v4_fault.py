#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 11:45:35 2025

@author: vcabiativapico
"""



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



file = '../output/93_kimberlina_angle_corr/new_attr_max.csv'
data = pd.read_csv(file)

max_total_rms = data['max_rms'].tolist()
max_cc = data['max_cc'].tolist()
year_number = data['year_number'].tolist()



plt.figure(figsize=(10,8))
plt.plot(year_number,max_total_rms,'o-')
plt.xlabel('angle (degrees)')
plt.ylabel('max_rms')

plt.figure(figsize=(10,8))
plt.plot(year_number,max_cc,'o-')
plt.xlabel('angle (degrees)')
plt.ylabel('min ts')
