from spotfunk.res import procs, visualisation, input
import numpy as np
import matplotlib.pyplot as plt
import geophy_tools as gt
from matplotlib.gridspec import GridSpec


path = 'c:/users/victorcabiativapico/SpotLight/SpotLighters - SpotLight/R&D/DOSSIER_PERSO_SpotLighters_RD/SpotVictor/Data_Whitecap/Spot_12_16_Whitecap/'


path_add = path + 'spot12_processed.segy'
seismic = input.segy_reader(path_add)

data = seismic.dataset

BASE = data[0]
M1 = data[1]
M2 = data[2]
M3 = data[3]

at = np.arange(np.shape(data)[1])

t_min_qc = 780
t_max_qc = 1040
si = 0.001
oplen = 100
taper = 5
t_min_qc_udb = 1215
t_max_qc_udb = 1400
t_min = 700
t_max = 1500


int(t_max_qc_udb+(si*1000))
    
cc_ts = procs.max_cross_corr(BASE, M1, win1=t_min_qc, win2=t_max_qc)


sld_ts = procs.sliding_TS(BASE[t_min:t_max], 
                          M1[t_min:t_max], taper=taper,
                          oplen=oplen, si = 0.001)


fig = plt.figure(figsize=(4, 8))
plt.plot(sld_ts,at[t_min:t_max])
plt.xlabel('time-shift')
plt.ylabel('Time (s)')
plt.xlim(-2,2)
plt.gca().invert_yaxis()

fig = plt.figure(figsize=(4, 8))
plt.title('TS')
plt.plot(BASE[800:1400],at[800:1400], '-', color= 'tab:blue',label='baseline')
plt.plot(M1[800:1400],at[800:1400], '--', color= 'tab:orange',label='monitor')
plt.legend()
plt.xlabel('amplitude')
plt.ylabel('Time (s)')
# plt.ylim(0.9,2.0)
# plt.ylim(0.5,2.0)
# plt.xlim(-0.1,0.1)
plt.gca().invert_yaxis()
flout = '../../fortran/out2dcourse/png/ts_interpolate.png'
fig.tight_layout()
print("Export to file:", flout)
fig.savefig(flout, bbox_inches='tight')

#%%


def figure_slidingCCNRMS(data, si, text, features, t_min, t_max, oplen, taper, x_lim, x_lim_CC, x_lim_NRMS, x_lim_corrNRMS, x_lim_TS,
                         TS=True, t_min_qc=0, t_max_qc=100, t_min_qc_udb=0, t_max_qc_udb=100,
                         m1=True, m2=True, m3=True):

    
    
    spot_name, file_name, title = text
    horizons, h_colors, surveys, surv_colors = features

    fig = plt.figure(figsize=(30,20))
    grid = GridSpec(ncols = 10, nrows = 3, wspace=0.8, hspace=0.5, figure = fig)
    ax1 = fig.add_subplot(grid[: , :2])
    ax2 = fig.add_subplot(grid[:, 2:4])
    ax3 = fig.add_subplot(grid[: , 4:6])
    ax4 = fig.add_subplot(grid[: , 6:8])
    ax5 = fig.add_subplot(grid[: , 8:])
    
      
    if TS is False:
        ax1.axhline(y=t_min_qc_udb, color='purple', linestyle='-.')
        ax1.axhline(y=t_max_qc_udb-1, color='purple', linestyle='-.')
        
        ax2.set_axis_off()
        TS_ovb_m1 = procs.max_cross_corr(data[0], data[1], win1=t_min_qc, win2=t_max_qc)
        TS_ovb_m2 = procs.max_cross_corr(data[0], data[2], win1=t_min_qc, win2=t_max_qc)
        TS_ovb_m3 = procs.max_cross_corr(data[0], data[3], win1=t_min_qc, win2=t_max_qc)
        TS_udb_m1 = procs.max_cross_corr(data[0], data[1], win1=t_min_qc_udb, win2=t_max_qc_udb)
        TS_udb_m2 = procs.max_cross_corr(data[0], data[2], win1=t_min_qc_udb, win2=t_max_qc_udb)
        TS_udb_m3 = procs.max_cross_corr(data[0], data[3], win1=t_min_qc_udb, win2=t_max_qc_udb)

        ax1.text(x_lim[1]+0.01, 1100-30, f'Mean dt Overburden :\n[{t_min_qc}, {t_max_qc}]', color = 'purple', fontsize = 25)
        ax1.text(x_lim[1]+0.01, 1400-30, f'Mean dt Underburden :\n[{t_min_qc_udb}, {t_max_qc_udb}]', color = 'purple', fontsize = 25)

        if m1 is True:
            TS_ovb_m1 = TS_ovb_m1.round(2)
            TS_udb_m1 = TS_udb_m1.round(2)
            ax1.text(x_lim[1]+0.02, 900, f'dt M1 : {TS_ovb_m1}', color = surv_colors[1], fontsize = 25)
            ax1.text(x_lim[1]+0.02, 1200, f'dt M1 : {TS_udb_m1}', color = surv_colors[1], fontsize = 25)
            ax1.text(x_lim[1]+0.02, 1000, f'$\Delta$dt M1 : {TS_udb_m1-TS_ovb_m1:.2f}', color = surv_colors[1], fontsize = 30)

        if m2 is True:
            TS_ovb_m2 = TS_ovb_m2.round(2)
            TS_udb_m2 = TS_udb_m2.round(2)
            ax1.text(x_lim[1]+0.02, 900+17, f'dt M2 : {TS_ovb_m2}', color = surv_colors[2], fontsize = 25)
            ax1.text(x_lim[1]+0.02, 1200+17, f'dt M2 : {TS_udb_m2}', color = surv_colors[2], fontsize = 25)
            ax1.text(x_lim[1]+0.02, 1000+25, f'$\Delta$dt M2 : {TS_udb_m2-TS_ovb_m2:.2f}', color = surv_colors[2], fontsize = 30)

        if m3 is True:
            TS_ovb_m3 = TS_ovb_m3.round(2)
            TS_udb_m3 = TS_udb_m3.round(2)
            ax1.text(x_lim[1]+0.02, 900+34, f'dt M3 : {TS_ovb_m3}', color = surv_colors[3], fontsize = 25)
            ax1.text(x_lim[1]+0.02, 1200+34, f'dt M3 : {TS_udb_m3}', color = surv_colors[3], fontsize = 25)
            ax1.text(x_lim[1]+0.02, 1000+50, f'$\Delta$dt M3 : {TS_udb_m3-TS_ovb_m3:.2f}', color = surv_colors[3], fontsize = 30)


        print(TS_ovb_m1, TS_udb_m1)
        print(TS_ovb_m2, TS_udb_m2)
        print(TS_ovb_m3, TS_udb_m3)        
        

    if t_max_qc_udb > t_max: 
        s_cc, s_nrms, s_ts = [], [], []
        for i_mon in range(data.shape[0]):
            s_nrms.append(procs.sliding_NRMS(data[0][t_min:int(t_max_qc_udb+(si*1000))], data[i_mon][t_min:int(t_max_qc_udb+(si*1000))], oplen = oplen, si = si))
            s_cc.append(procs.sliding_corrcoeff(data[0][t_min:int(t_max_qc_udb+(si*1000))], data[i_mon][t_min:int(t_max_qc_udb+(si*1000))], oplen = oplen, si = si))
            s_ts.append(procs.sliding_TS(data[0][t_min:int(t_max_qc_udb+(si*1000))], data[i_mon][t_min:int(t_max_qc_udb+(si*1000))], taper=taper, oplen=oplen, si = si))
        
        s_corr_nrms = np.array([v_nrms/v_cc for v_cc, v_nrms in zip(s_cc, s_nrms)])
        print(len(s_cc[0]))
        visualisation.overlay(*s_cc, si = si, win1 = 0, win2 = len(s_cc[0])-1, clist=surv_colors, legend=surveys, ax=ax3)
        ax3.set_title('CC', fontsize = 15)
        ax3.set(yticks=np.arange(0,(t_max_qc_udb-t_min)+si*1000, 100), yticklabels = np.arange(t_min,t_max_qc_udb+si*1000, 100).astype(int))
        visualisation.overlay(*s_nrms, si = si, win1 = 0, win2 = len(s_cc[0])-1, clist=surv_colors, ax=ax4)
        ax4.set_title('NRMS', fontsize = 15)
        ax4.set(yticks=np.arange(0,(t_max_qc_udb-t_min)+si*1000, 100), yticklabels = np.arange(t_min,t_max_qc_udb+si*1000, 100).astype(int))
        visualisation.overlay(*s_corr_nrms, si = si, win1 = 0, win2 = len(s_cc[0])-1, clist=surv_colors, ax=ax5)
        ax5.set_title('NRMS/CC', fontsize = 15)
        ax5.set(yticks=np.arange(0,(t_max_qc_udb-t_min)+si*1000, 100), yticklabels = np.arange(t_min,t_max_qc_udb+si*1000, 100).astype(int))

        visualisation.overlay(*data, win1 = t_min, win2 = t_max_qc_udb + si*1000, clist = surv_colors, fontsize = 15, ax = ax1, si = si)
        ax1.set(xlim = [x_lim[0], x_lim[1]], xlabel = "Amplitude (a.u.)")
        ax1.set_title('Traces', fontsize = 15)
        ax1.axhline(y=t_min_qc, color='purple', linestyle='--')
        ax1.axhline(y=t_max_qc, color='purple', linestyle='--')
        
    else: 
        s_cc, s_nrms, s_ts = [], [], []
        for i_mon in range(data.shape[0]):
            s_nrms.append(procs.sliding_NRMS(data[0][t_min:int(t_max+(si*1000))], data[i_mon][t_min:int(t_max+(si*1000))], oplen = oplen, si = si))
            s_cc.append(procs.sliding_corrcoeff(data[0][t_min:int(t_max+(si*1000))], data[i_mon][t_min:int(t_max+(si*1000))], oplen = oplen, si = si))
            s_ts.append(procs.sliding_TS(data[0][t_min:int(t_max+(si*1000))], data[i_mon][t_min:int(t_max+(si*1000))], taper=taper, oplen=oplen, si = si))
        
        s_corr_nrms = np.array([v_nrms/v_cc for v_cc, v_nrms in zip(s_cc, s_nrms)])
        print(len(s_cc[0]))
        visualisation.overlay(*s_cc, si = si, win1 = 0, win2 = len(s_cc[0])-1, clist=surv_colors, legend=surveys, ax=ax3)
        ax3.set_title('CC', fontsize = 15)
        ax3.set(yticks=np.arange(0,(t_max-t_min)+si*1000, 100), yticklabels = np.arange(t_min,t_max+si*1000, 100).astype(int))
        visualisation.overlay(*s_nrms, si = si, win1 = 0, win2 = len(s_cc[0])-1, clist=surv_colors, ax=ax4)
        ax4.set_title('NRMS', fontsize = 15)
        ax4.set(yticks=np.arange(0,(t_max-t_min)+si*1000, 100), yticklabels = np.arange(t_min,t_max+si*1000, 100).astype(int))
        visualisation.overlay(*s_corr_nrms, si = si, win1 = 0, win2 = len(s_cc[0])-1, clist=surv_colors, ax=ax5)
        ax5.set_title('NRMS/CC', fontsize = 15)
        ax5.set(yticks=np.arange(0,(t_max-t_min)+si*1000, 100), yticklabels = np.arange(t_min,t_max+si*1000, 100).astype(int))

        visualisation.overlay(*data, win1 = t_min, win2 = t_max + si*1000, clist = surv_colors, fontsize = 15, ax = ax1, si = si)
        ax1.set(xlim = [x_lim[0], x_lim[1]], xlabel = "Amplitude (a.u.)")
        ax1.set_title('Traces', fontsize = 15)
        ax1.axhline(y=t_min_qc, color='purple', linestyle='--')
        ax1.axhline(y=t_max_qc, color='purple', linestyle='--')

    ax5.set(xlim = [x_lim_corrNRMS[0], x_lim_corrNRMS[1]])
    ax4.set(xlim = [x_lim_NRMS[0], x_lim_NRMS[1]])
    ax3.set(xlim = [x_lim_CC[0], x_lim_CC[1]])
    ax1.set(xlim = [x_lim[0], x_lim[1]])
        
    if TS is True:
        visualisation.overlay(*s_ts, win1 = 0, win2 = len(s_ts[0])-1, clist = surv_colors, fontsize = 15, ax = ax2, si = si)
        ax2.set_title('TS', fontsize = 15)
        ax2.set(xlim = [x_lim_TS[0], x_lim_TS[1]])
        ax2.set(yticks=np.arange(0,(t_max-t_min)+si*1000, 100), yticklabels = np.arange(t_min,t_max+si*1000, 100).astype(int))        
        # for horizon_name, color in zip(horizons.columns.to_list(), h_colors):
        #     ax1.axhline(horizons[horizon_name][spot_name], color = color, lw = 1.5)
        #     ax2.axhline(horizons[horizon_name][spot_name] - t_min, color = color, lw = 1.5)
        #     ax3.axhline(horizons[horizon_name][spot_name] - t_min, color = color, lw = 1.5)
        #     ax4.axhline(horizons[horizon_name][spot_name] - t_min, color = color, lw = 1.5)
        #     ax5.axhline(horizons[horizon_name][spot_name] - t_min, color = color, lw = 1.5)

    # else:
    #     for horizon_name, color in zip(horizons.columns.to_list(), h_colors):
    #         ax1.axhline(horizons[horizon_name][spot_name] , color = color, lw = 1.5)
    #         ax3.axhline(horizons[horizon_name][spot_name] - t_min, color = color, lw = 1.5)
    #         ax4.axhline(horizons[horizon_name][spot_name] - t_min, color = color, lw = 1.5)
    #         ax5.axhline(horizons[horizon_name][spot_name] - t_min, color = color, lw = 1.5)

    fig.savefig(file_name, bbox_inches = "tight")
    plt.close()
    
    
#%%
file_name = path+'spot12_processed.png'
text = ['12', file_name, 'whitecap']


surveys = ['Base','M1','M2','M3']
surv_colors = ['k','magenta','tab:blue','tab:orange']

features = [0,0,surveys, surv_colors]

x_lim = [-0.15,0.15]
x_lim_CC = [0.2,1.0]
x_lim_NRMS = [0, 1.5]
x_lim_corrNRMS = [0, 4]
x_lim_TS = [-6, 6]

figure_slidingCCNRMS(data,si, text, features, t_min, t_max, oplen, taper, x_lim, x_lim_CC, x_lim_NRMS, x_lim_corrNRMS, x_lim_TS,
                         TS=True, t_min_qc=t_min_qc, t_max_qc=t_max_qc, t_min_qc_udb=t_min_qc_udb, t_max_qc_udb=t_max_qc_udb,
                         m1=True, m2=True, m3=True)

