#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:18:22 2024

@author: vcabiativapico
"""

import csv
import numpy as np
import geophy_tools as gt
import matplotlib.pyplot as plt
from spotfunk.res import procs,visualisation
import sympy as sp

# Global parameters
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
no = 251
# no        = 2002
do = dx
fo = -(no-1)/2*do
ao = fo + np.arange(no)*do
at = ft + np.arange(nt)*dt
az = fz + np.arange(nz)*dz
ax = fx + np.arange(nx)*dx
# ## Add y dimension    
fy = -500 
ny = 21
dy = 50
ay = fy + np.arange(ny)*dy

def plot_model(inp,hmin,hmax):
    plt.rcParams['font.size'] = 16
    fig = plt.figure(figsize=(10,5), facecolor = "white")
    av  = plt.subplot(1,1,1)
    hfig = av.imshow(inp, extent=[ax[0],ax[-1],az[-1],az[0]], \
                      vmin=hmin,vmax=hmax,aspect='auto'\
                     )
    plt.colorbar(hfig)
    plt.xlabel('Distance (Km)')
    plt.ylabel('Profondeur (Km)')
    fig.tight_layout()
    return fig
    
def export_model(inp,fig,imout,flout):
    fig.savefig(imout, bbox_inches='tight')
    gt.writebin(inp,flout)  


def read_index(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        # header = next(spamreader)
        for row in spamreader:
            attr.append(int(row[srow]))
    return attr

def read_pick(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
    return attr


## Read the results from demigration
def read_results(path,srow):
    attr = []
    with open(path, newline='') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        header = next(spamreader)
        for row in spamreader:
            attr.append(float(row[srow]))
        # attr = [x for x in attr if str(x) != 'nan'] 
    attr = np.array(attr)
    attr = np.nan_to_num(attr)
    return attr

def find_nearest(array, value):
    val = np.zeros_like(array)
    for i in range(len(array)):
        val[i] = np.abs(array[i] - value)
        idx = val.argmin()
    return array[idx], idx


def extract_trace(p,idx_rt, path_shot):  
    ax = p.fx_ + np.arange(p.nx_)*p.dx_
    ao = p.fo_ + np.arange(p.no_)*p.do_
    title = str(find_nearest(ax, p.src_x_[idx_rt]/1000)[1])
    title = title.zfill(3)
    print('indice source', title)
    fl = path_shot+'/t1_obs_000'+str(title)+'.dat'
    if int(title) > int(p.nx_ - p.no_//2):
        noff = p.no_ // 2 + p.nx_ - int(title) +1
    else: 
        noff = p.no_
    inp = gt.readbin(fl, noff, p.nt_).transpose()
    tr = find_nearest(ao,p.off_x_[idx_rt]/1000)[1]
    return inp, tr

def create_nodes(diff_ind_max,idx_nr_off, idx_nr_src,nx,no):
    """
    Find the indexes of the traces that will be used as nodes for the interpolation
    """
    if idx_nr_src < 2 :
        nb_gathers = np.array([0, 1, 2, 3, 4])
    elif idx_nr_src > nx-3:
        nb_gathers = np.array([597, 598, 599, 600, 601])
    else:
        nb_gathers = np.arange(idx_nr_src-2, idx_nr_src+3)
    
    
    if idx_nr_off < 2 :
        nb_traces = np.array([0, 1, 2, 3, 4])
    elif idx_nr_off > no-3:
        nb_traces = np.array([247, 248, 249, 250, 251])
    else:
        nb_traces = np.arange(idx_nr_off-2, idx_nr_off+3)
    
    return nb_gathers, nb_traces


def read_shots_around(gather_path,nb_gathers,param):
    """
    Reads the shots around the closest shot from raytracing to the numerical born modelling grid
    """
    
    inp3 = np.zeros((len(nb_gathers),param.nt_, param.no_))
    
    for k, i in enumerate(nb_gathers):
        txt = str(i)
        title = txt.zfill(3)
        
        tr3 = gather_path+'/t1_obs_000'+str(title)+'.dat'
     
        inp3[k][:,:] = -gt.readbin(tr3, param.no_, param.nt_).transpose()
     
    return inp3

def trace_from_rt(diff_ind_max,gather_path,p):
    '''
    Finds the nearest trace to the source and offset given by the index
    Interpolates the traces so that the calculation is exact
    Exports the trace found by raytracing from the modelling data
    '''
    nr_src_x, idx_nr_src = find_nearest(p.ax_, p.src_x_[diff_ind_max]/1000)
    nr_off_x, idx_nr_off = find_nearest(p.ao_, p.off_x_[diff_ind_max]/1000)
    
    nb_gathers, nb_traces = create_nodes(diff_ind_max,idx_nr_off, idx_nr_src,p.nx_,p.no_)
    
    
    inp3 = read_shots_around(gather_path, nb_gathers, p)
   
    # fin_trace = interpolate_src_rec(nb_traces,nb_gathers,at,ao,inp3,-p.off_x_,p.src_x_,do,dx,diff_ind_max) 
    
    fin_trace = interpolate_src_rec(nb_traces,nb_gathers,p.at_,p.ao_,inp3,p.off_x_,p.src_x_,p.do_,p.dx_,diff_ind_max)    
    return fin_trace
gen_path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/'



path_adj = gen_path + '048_sm8_correction_new_solver/STD/depth_demig_out/STD/results/depth_demig_output.csv'
path_inv = gen_path + '048_sm8_correction_new_solver/QTV/depth_demig_out/QTV/results/depth_demig_output.csv'

class Param_class:
    "Class for the parameters definition"
    def __init__(self,path):
        self.src_x_ = read_results(path,1)
        self.src_y_ = read_results(path,2)
        self.src_z_ = read_results(path,3)
        self.rec_x_ = read_results(path,4)
        self.rec_y_ = read_results(path,5)
        self.rec_z_ = read_results(path,6)
        self.spot_x_ = read_results(path,7)
        self.spot_y_ = read_results(path,8)
        self.spot_z_ = read_results(path,9)
        self.off_x_ = read_results(path,16)
        self.tt_ = read_results(path,17)
        self.nt_ = 1801
        self.dt_ = 1.41e-3
        self.ft_ = -100.11e-3
        self.nz_ = 151
        self.fz_ = 0.0
        self.dz_ = 12.0/1000.
        self.nx_ = 601
        self.fx_ = 0.0
        self.dx_ = 12.0/1000.
        self.no_ = 251
        self.do_ = self.dx_
        self.fo_ = -(self.no_-1)/2*self.do_
        self.ao_ = self.fo_ + np.arange(self.no_)*self.do_
        self.at_ = self.ft_ + np.arange(self.nt_)*self.dt_
        self.az_ = self.fz_ + np.arange(self.nz_)*self.dz_
        self.ax_ = self.fx_ + np.arange(self.nx_)*self.dx_
   
        
p_adj = Param_class(path_adj)
p_inv = Param_class(path_inv)

file_index = '../input/45_marm_ano_v3/fwi_ano_114_percent.csv'
inp_index = [np.array(read_index(file_index,0)), np.array(read_index(file_index,1))]
fl2       = '../input/marm2_sm15.dat'
inp_sm    = gt.readbin(fl2,nz,nx)

inp_sm_mod = np.copy(inp_sm)

new_vel   = inp_sm_mod[inp_index]*1.14
inp_sm_mod[inp_index] = new_vel

hmin = 1.5
hmax = 4.5

fig2 = plot_model(inp_sm,hmin,hmax)
fig1 = plot_model(inp_sm_mod,hmin,hmax)

imout1 = '../png/50_ts_model/marmousi_ano_sm.png'
flout1 = '../input/50_ts_model/marmousi_ano_sm.dat'
export_model(inp_sm_mod,fig1,imout1,flout1)

'''Read the raypath'''
path_ray = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/050_TS_analytique_ano/depth_demig_out/ano1/rays/ray_0.csv'
ray_x = np.array(read_results(path_ray, 0))
ray_z = np.array(read_results(path_ray, 2))
vp =  np.array(read_results(path_ray, 6))
tt = np.array(read_results(path_ray, 8))


'''Read the velocity model '''
fl1 = '../input/45_marm_ano_v3/fwi_ano_114_percent.dat'
fl2 = '../input/45_marm_ano_v3/fwi_org.dat'
inp1 = gt.readbin(fl1,nz,nx)
inp2 = gt.readbin(fl2,nz,nx)


'''POLIGON :Read index of the anomaly'''
file_index = '../input/45_marm_ano_v3/fwi_ano_114_percent.csv'
inp_index = [np.array(read_index(file_index,0)), np.array(read_index(file_index,1))]

'''POLIGON: Compute the values of the anomaly only'''
inp_blank = np.zeros_like(inp1)


inp_blank[inp_index] = 2
inp_blank[:,0:270] = 0

new_idx = np.where(inp_blank>1)
plot_model(inp_blank,hmin,hmax)
plt.scatter(ray_x/1000,-ray_z/1000, c='k',s=0.1)

'''POLIGON: Create a polygon'''
sympy_idx = np.transpose(new_idx).tolist()

sympy_idx_tuple = []
for x in sympy_idx:
    sympy_idx_tuple.append(tuple(x))



polygon = sp.Polygon(*sympy_idx_tuple)

# Extract the coordinates
x, y = zip(*polygon.vertices)

# Close the polygon by repeating the first vertex at the end
x = list(x) + [x[0]]
y = list(y) + [y[0]]

# Plot the polygon
plt.figure(figsize=(12, 12))
plt.scatter(np.array(y)*0.012, np.array(x)*0.012)
plt.scatter(ray_x/1000,-ray_z/1000, c='k',s=1.5)
plt.axhline(1.018)
plt.title("Polygon Plot")
plt.xlabel("x")
plt.ylabel("z")
plt.grid(True)
plt.ylim(1.16,0.95)
plt.xlim(3.2,3.5)

'''Find the indexes in polygon'''
idx_in_poly = []

for i,k in enumerate(ray_z):
    if k < -1015:
        idx_in_poly.append(i)
ray_x_in_poly = ray_x[idx_in_poly]
ray_z_in_poly = ray_z[idx_in_poly]
tt_in_poly = tt[idx_in_poly]
vp_in_poly = vp[idx_in_poly]

rows = 17
cols = 2
table_org =[]
table_ano =[]
source_x_org = []
source_x_ano = []
row = []
for i in range(1,rows):
    path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/050_TS_analytique/depth_demig_out/analytique'+str(i)+'/results/depth_demig_output.csv'
    table_org.append(read_pick(path, 17)[0])
    source_x_org.append(read_pick(path,1)[0])
    
for i in range(1,rows):
    path = '/home/vcabiativapico/local/Demigration_SpotLight_Septembre2023/050_TS_analytique_ano/depth_demig_out/ano'+str(i)+'/results/depth_demig_output.csv'
    table_ano.append(read_pick(path, 17)[0])
    source_x_ano.append(read_pick(path,1)[0])

'''Create the plot of analytical time-shift'''

ts0_a = (np.array(table_org) - np.array(table_ano))*1000

first_tt_in_poly = 1.212
tt0_a =  p_inv.tt_[0] + tt_in_poly[:len(tt_in_poly)//2+1]*2 

ts0_a = np.insert(ts0_a,0,0)
tt0_a = np.insert(tt0_a,0,tt0_a[0])

ts0_a = np.insert(ts0_a,0,0)
tt0_a = np.insert(tt0_a,0,0)

ts0_a = np.append(ts0_a, ts0_a[-1])
tt0_a = np.append(tt0_a, at[-1])



gather_path_fwi_org = '../output/45_marm_ano_v3/org_1801TL'
gather_path_fwi45 = '../output/45_marm_ano_v3/ano_114_perc_1801TL'

tr_binv_fwi_org = trace_from_rt(0,gather_path_fwi_org,p_inv)
tr_binv_fwi_45 = trace_from_rt(0,gather_path_fwi45,p_inv)

tr_badj_fwi_org = trace_from_rt(0,gather_path_fwi_org,p_adj)
tr_badj_fwi_45 = trace_from_rt(0,gather_path_fwi45,p_adj)


binv_SLD_TS_fwi = procs.sliding_TS(tr_binv_fwi_org,tr_binv_fwi_45,oplen= 500,si=p_inv.dt_, taper= 30)
badj_SLD_TS_fwi = procs.sliding_TS(tr_badj_fwi_org,tr_badj_fwi_45,oplen= 500,si=p_adj.dt_, taper= 30)

plt.figure(figsize=(6, 10), facecolor="white")
plt.title('Time-shift')
plt.plot(ts0_a,tt0_a,'tab:green',label='Theoretical TS')
plt.plot(binv_SLD_TS_fwi,p_inv.at_,c='tab:orange',label='TS from traces')
plt.axhline(p_inv.tt_[0],c='tab:orange',ls='--')
plt.gca().invert_yaxis()
plt.legend()
plt.xlim(-0.7,4)
plt.ylim(at[-1],at[0])
