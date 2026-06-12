import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cmcrameri import cm 
import re
import glob
import os

def event_length(folder, eve_num):
    potency = np.load(folder + f'events/potency_{eve_num}.npy')
    start_time = potency[0, 0]
    end_time = potency[-1, 0]
    return (end_time- start_time)/(24*60*60)

def add_event_axes(fig, x, y, w, hm, hmain, height_inner):
    ax_moment = fig.add_axes([x, y, w, hm])
    ax_main = fig.add_axes([x, y + hm + height_inner, w, hmain])
    return ax_main, ax_moment

def cal_speed(new_list_t, new_list_x, folder, cal_simu):
    params = json.load(open(folder + 'parameters.json'))
    param_e = params['elastic']
    eps_D = param_e['eps_D']
    taup0 = param_e['tau_p0']
    Vslip = param_e['Vslip']
    param_r = params['region']
    G = param_r['G']

    Vprop_est = np.pi/4*Vslip*G/(0.5*(1+eps_D)*taup0)
    Vprop_est = Vprop_est*86400/1000

    if folder == '../results/para_1/dx50_ny50/eps_0.6/':
        return Vprop_est, 9.2

    if cal_simu:
        t1 = 2
        t2 = 3
        x1 = max(new_list_x[new_list_t == t1])
        x2 = max(new_list_x[new_list_t == t2])

        Vprop_simu = (x2-x1)/(t2-t1)
    else:
        Vprop_simu = 0
    # print(x2, x1)
    # print('estimated Vprop',Vprop_est, '\nobserved Vprop',Vprop_simu)
    return Vprop_est, Vprop_simu

def cal_base(folder, coefficient):
    params = json.load(open(folder + 'parameters.json'))
    param_e = params['elastic']
    Vslip = param_e['Vslip']

    param_r = params['region']
    my = param_r['my']
    Axy = param_r['Axy']
    nc = 4

    return my*nc*Vslip*Axy*coefficient
    

def plot_event(ax_main, ax_moment, folder, eve_num, vmax, time_window, dis_window, normalize, ene_level, label, eps_label, arrow, arrow_0, baseline):
    event_history = np.load(folder + f'events/history_X_{eve_num}.npy')
    mask = event_history > 0
    new_list_t, new_list_x = np.nonzero(mask)
    new_list_n = event_history[mask]/normalize

    if np.any(new_list_n > 1.05):
        print("There is at least one value > 1", np.max(new_list_n))

    new_list_t = new_list_t*time_window/(24*60*60)
    new_list_x = new_list_x*dis_window/1000

    potency = np.load(folder + f'events/potency_{eve_num}.npy')
    start_time = potency[0, 0]
    pottime = (potency[:, 0] - start_time)/(24*60*60)
    potrate = potency[:, 1]

    coefficient = G*1e6/10**ene_level

    cmap = cm.batlow
    sc = ax_main.scatter(new_list_t, new_list_x, cmap=cmap, vmin=0, vmax=max(1, vmax/normalize), c=new_list_n, s=0.05, rasterized=True)
    ax_main.xaxis.set_visible(False)
    ax_main.set_xlim(0, pottime[-1])
    ax_main.set_ylim(-0.5, 6)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)
    ax_main.text(-0.85, 120, label)

    ax_main.set_ylabel('Strike distance (km)', fontsize=12)

    if arrow:
        Vprop_est, Vprop_simu = cal_speed(new_list_t, new_list_x, folder, True)
        start_day = 2
        start_x = 60
        ax_main.annotate('', 
            xy=(start_day+1, start_x+Vprop_simu),
            xytext=(start_day, start_x), 
            arrowprops=dict(arrowstyle='-|>', color='tab:blue', lw=2, mutation_scale=15)
        )
        ax_main.text(start_day+1, start_x+Vprop_simu, f'{Vprop_simu} km/d', color='tab:blue', ha='left', va='center')
        ax_main.annotate('', 
            xy=(start_day+1, start_x+10+Vprop_est),
            xytext=(start_day, start_x+10), 
            arrowprops=dict(arrowstyle='-|>', color='tab:green', lw=2, linestyle='--', mutation_scale=15)
        )
        ax_main.text(start_day+1, start_x+10+Vprop_simu, f'{Vprop_est:.1f} km/d', color='tab:green', ha='left', va='center')
        ax_main.text(0.3, 105, rf'$\epsilon_D = ${eps_label}', fontsize=12)
    
    if arrow_0:
        Vprop_est, Vprop_simu = cal_speed(new_list_t, new_list_x, folder, False)
        start_day = 0.5
        start_x = 60
        ax_main.annotate('', 
            xy=(start_day+1, start_x+10+Vprop_est),
            xytext=(start_day, start_x+10), 
            arrowprops=dict(arrowstyle='-|>', color='tab:green', lw=2, linestyle='--', mutation_scale=15)
        )
        ax_main.text(start_day+1, start_x+10+Vprop_est, f'{Vprop_est:.1f} km/d', color='tab:green', ha='left', va='center')
        ax_main.text(0.3, 105, rf'$\epsilon_D = 0$', fontsize=12)
               
    if baseline:
        base_ene = cal_base(folder, coefficient)
        ax_moment.plot([1,2], [base_ene, base_ene], '--')

    if folder == '../results/para_1/dx50_ny50/eps_0.6/':
        start_day = 7
        start_x = 105
        V_bw = 96
        arrow_time = 0.3
        ax_main.annotate('', 
            xy=(start_day+arrow_time, start_x-V_bw*arrow_time),
            xytext=(start_day, start_x), 
            arrowprops=dict(arrowstyle='-|>', color='tab:orange', lw=2, mutation_scale=15)
        )
        ax_main.text(start_day+arrow_time+0.1, start_x-0.7*V_bw*arrow_time, f'{V_bw} km/d', color='tab:orange', ha='left', va='center')

        start_day = 10
        start_x = 120
        V_bw = 48
        arrow_time = 0.45
        ax_main.annotate('', 
            xy=(start_day+arrow_time, start_x-V_bw*arrow_time),
            xytext=(start_day, start_x), 
            arrowprops=dict(arrowstyle='-|>', color='tab:olive', lw=2, mutation_scale=15)
        )
        ax_main.text(start_day+arrow_time-0.3, start_x-0.7*V_bw*arrow_time, f'{V_bw} km/d', color='tab:olive', ha='right', va='center')


    ax_moment.plot(pottime, potrate*coefficient, color='black', linewidth=0.5)
    ax_moment.set_xlim(0, pottime[-1])
    y_max = np.ceil(np.max(potrate)*coefficient)
    ax_moment.set_ylim(0, y_max)
    ax_moment.set_xlabel('Time (days)')
    ax_moment.spines['top'].set_visible(False)
    ax_moment.spines['right'].set_visible(False)

    ax_moment.set_yticks([0, y_max//3, y_max//3*2])
    ax_moment.set_ylabel(r'$\dot{M_0}$ $(Nm/s)$')
    ax_moment.text(0, y_max//3*2, fr'$10^{{{ene_level}}}$', fontsize=10, ha='left', va='bottom')
    ax_moment.ticklabel_format(useOffset=False)

    ax_moment.set_xlabel('Time (days)', fontsize=12)

    return sc


if __name__ == "__main__":

    folder = '../results/test_RSQSim/test_1/'

    params = json.load(open(folder + 'parameters.json'))
    G, ny, dx = params['region']['G'], params['region']['my'], params['region']['dx']
    time_window, dis_window = params['elastic']['time_resolution'], params['elastic']['space_resolution']
    normalize = time_window*dis_window/dx*ny
    print(normalize)

    iev = 30
    ene = 11
    fig = plt.figure(figsize=(10, 4))
    ax_main = fig.add_axes([0.08, 0.24, 0.85, 0.72])
    ax_moment = fig.add_axes([0.08, 0.12, 0.8, 0.12])
    sc = plot_event(ax_main, ax_moment, folder, iev, normalize, time_window, dis_window, normalize, ene, '', 0.6, arrow=False, arrow_0=False, baseline=False)
    
    # plot the colormap
    cbar_ax = fig.add_axes([0.93, 0.25, 0.02, 0.72])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(r'Faction of slip activity')

    plt.savefig(folder + f'events/event_{iev}.png')
    plt.close('all')
