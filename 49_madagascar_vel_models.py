#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 21 10:55:23 2024

@author: vcabiativapico
"""

from rsf.proj import *
# Fetch Files from repository
Fetch("int_depth_vp.sgy","pluto")
Fetch("P15VPint_25f_padded.SEGY","pluto")

# Convert Files to RSF
Flow('velocityProfileStd','int_depth_vp.sgy',
     '''
     segyread read=d |
     put  d2=.025 label1=Depth o2=-34.875 
     label2=Position unit1=kft unit2=kft 
     label=Velocity unit=kft/s |
     scale rscale=0.001
     ''')

Flow('velocityProfileMetric','int_depth_vp.sgy',
     '''
     segyread read=d |
     put d1=.00760 d2=.00760 o2=-10.629
     label1=Depth label2=Position label=Velocity 
     unit1=km unit2=km unit=km/s |
     scale rscale=.0003048
     ''')

Flow('velocityProfilePadded','P15VPint_25f_padded.SEGY',
     '''
     segyread read=d |
     put  d1=.0076 d2=.00760 o2=-10.629 label1=Depth
     label2=Position unit1=km unit2=km label=Velocity |
     scale rscale=.0003048
     ''')

# Plotting Section
mins=[0,0,-10.5]
maxs=['105','32','42.5']

counter=0
for item in ['Std','Metric']:
    Result('velocityProfile' + item,
           '''
           window j1=2 j2=2 |
           grey scalebar=y color=j allpos=y bias=1 title=P-Wave Velocity Profile
           max2=%s min2=0 screenratio=.28125 screenht=2 
           labelsz=4 wanttitle=n barreverse=y
           ''' % maxs[counter])
    counter=counter+1

Result('velocityProfilePadded',
       '''
       window j1=2 j2=2 |
       grey scalebar=y color=j allpos=y bias=1 gainpanel=a title=P-Wave Velocity Profile
       screenratio=.28 125 screenht=2 labelsz=4 wanttitle=n barreverse=y 
       ''')

End()
