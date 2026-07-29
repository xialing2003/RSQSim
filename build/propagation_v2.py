# Note that the time step doesn't have a unit of seconds, instead, its unit is Dc/Vpl.
# Therefore, when I plot the propagation profile, I just write down 10^4 in the title directly.
# can develop an automatic way to calculate the coefficient.

# besides, to plot the propagation patterns of the original simulations, 
# the potrate should switch lines and rows, and the time/space windows also need to be specified

import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from cmcrameri import cm 
from pathlib import Path
import pandas as pd

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
    
def plot_event(ax_main, event_history, time_window, dis_window, normalize, arrow, cutoff, start_day, end_day):
    
    if cutoff:
        start_window = int(start_day*(24*60*60)/time_window)
        end_window = int(end_day*(24*60*60)/time_window)
        event_history_filter = np.copy(event_history[start_window:end_window, :])
        start_axis = 0
        end_axis = end_day-start_day
    else:
        event_history_filter = np.copy(event_history)
        start_axis = 0
        end_axis = event_history_filter.shape[0]*time_window/(24*60*60)
    del event_history

    mask = event_history_filter > 0
    new_list_t, new_list_x = np.nonzero(mask)
    new_list_n = event_history_filter[mask]/normalize
    vmax = min(1, np.max(new_list_n) * 0.8)

    if np.any(new_list_n > 1.001):
        print("There is at least one value > 1", np.max(new_list_n))

    new_list_t = new_list_t*time_window/(24*60*60)
    new_list_x = new_list_x*dis_window/1000
    del event_history_filter

    cmap = cm.batlow
    sc = ax_main.scatter(new_list_t, new_list_x, cmap=cmap, vmin=0, vmax=vmax, c=new_list_n, s=0.05, rasterized=True)
    ax_main.xaxis.set_visible(False)
    ax_main.set_xlim(start_axis, end_axis)
    ax_main.set_ylim(-0.5, 90)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    ax_main.set_ylabel('Strike distance (km)', fontsize=11)

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

    return sc

def plot_moment(ax_moment, potency, G, cutoff, start_day, end_day):

    pottime = potency[0, :]/(24*60*60)
    potrate = potency[1, :]
    pottime -= pottime[0]

    if cutoff:
        mask_pot = (pottime >= start_day) & (pottime <= end_day)
        pottime = pottime[mask_pot]
        potrate = potrate[mask_pot]
    else:
        start_day = 0
        end_day = pottime[-1]
        print('end_day', end_day)

    ene_level = int(np.log10(G*1e6*max(potrate)))
    print(ene_level)

    coefficient = G*1e6/10**ene_level
    ax_moment.plot(pottime, potrate*coefficient, color='black', linewidth=0.5)
    ax_moment.set_xlim(start_day, end_day)
    y_max = np.ceil(np.max(potrate)*coefficient)
    print('y_max', y_max)
    ax_moment.set_ylim(0, y_max)
    ax_moment.set_xlabel('Time (days)')
    ax_moment.spines['top'].set_visible(False)
    ax_moment.spines['right'].set_visible(False)

    ax_moment.set_yticks([0, (y_max*10//3)*0.1, (y_max*10//3)*0.2])
    ax_moment.set_ylabel(r'$\dot{M_0}$ $(Nm/s)$')
    ax_moment.text(start_day, y_max/3*2, fr'$10^{{{ene_level}}}$', fontsize=10, ha='left', va='bottom')
    ax_moment.ticklabel_format(useOffset=False)

    ax_moment.set_xlabel('Time (days)', fontsize=11)

def plot_alg(iev, folder, begin_days, G, time_window, dis_window, normalize):
    event_history = np.load(folder + f'/history_X_{iev}.npy')
    potency = np.load(folder + f'/potency_{iev}.npy')

    fig = plt.figure(figsize=(10, 4))
    ax_main = fig.add_axes([0.08, 0.24, 0.8, 0.72])
    ax_moment = fig.add_axes([0.08, 0.12, 0.8, 0.12])
    cutoff = False
    start_day = 616
    end_day = 620
    sc = plot_event(ax_main, event_history, time_window, dis_window, normalize, arrow=False, cutoff=cutoff, start_day=start_day, end_day=end_day)
    plot_moment(ax_moment, potency, G, cutoff=cutoff, start_day=start_day, end_day=end_day)
    
    # plot the colormap
    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.72])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label(r'Faction of slip activity')

    if cutoff:
        plt.savefig(folder + f'/event_{start_day}_{end_day}.jpg', dpi=300)
    else:
        plt.savefig(folder + f'/event_{iev}_startat_{begin_days}_days.jpg', dpi=300)
    plt.close('all')

if __name__ == "__main__":

    # folder = '../results/test_RSQSim/test_v2.2/'
    # folder = '../results/for_comp_qsdm/nx900_my5_eps0.9/'
    folder = '../results/RSQSim_stage1/test1/'
    folder_event = 'events_v3'
    events = pd.read_csv(folder + folder_event + '.csv')

    event_ids = sorted(
        int(f.stem.split("_")[-1])
        for f in Path(folder).glob(folder_event + "/history_X_*.npy")
    )

    params = json.load(open(folder + 'parameters.json'))
    G, ny, dx = params['region']['G'], params['region']['my'], params['region']['dx']
    time_window, dis_window = params['model']['time_res'], params['model']['space_res']
    normalize = time_window*dis_window/dx*ny
    print(normalize)

    for iev in event_ids:
        begin_days = int(events.iloc[iev]['start_time'] / (24 * 3600))
        plot_alg(iev, folder + folder_event, begin_days, G, time_window, dis_window, normalize)

    # iev = 33516
    # begin_days = int(events.iloc[iev]['start_time'] / (24 * 3600))
    # # begin_days = 0
    # plot_alg(iev, folder + folder_event, begin_days, G, time_window, dis_window, normalize)