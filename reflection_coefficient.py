#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 15:47:58 2023

@author: vcabiativapico
"""

import numpy as np

v = [1.5]*5
v_rc = [0]*5
vm =[0]*5
refc = [0]*4

rc = 0.10


for i in range(4):
    v[i+1] = -(v[i]*rc+v[i])/(rc-1)
    vm[i] = 4/((v[i]+v[i+1])/2)
    vm[-1] = vm[-2]
    v_rc[0]=1.5
    v_rc[i+1] = v[i] * rc + v[i] * vm[i] / (vm[i]-rc)
np.array(v)    
print(v)

# v_rc = v * (-4/np.power(v,2))
for i in range(4):
    refc[i] = (v_rc[i+1] - v_rc[i] )/ (v_rc[i] + v_rc[i+1] )
    
print(v_rc)
print(refc)

for i in range(4):
    