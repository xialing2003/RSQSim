import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from cmcrameri import cm 

def plot_event(ax_main, plot_t, plot_x, plot_n, cutoff, start_day, end_day):
    
    if cutoff:
        mask = (plot_t >= start_day) and (plot_t <= end_day)
        plot_t = plot_t[mask]
        plot_x = plot_x[mask]
        plot_n = plot_n[mask]

    vmax = min(1, np.max(plot_n))
    if np.any(plot_n > 1.001):
        print("Warning! There is at least one value > 1", np.max(plot_n))

    cmap = cm.batlow
    sc = ax_main.scatter(plot_t, plot_x, cmap=cmap, vmin=0, vmax=vmax, c=plot_n, s=0.05, rasterized=True)
    ax_main.xaxis.set_visible(False)
    ax_main.set_xlim(min(plot_t), max(plot_t))
    ax_main.set_ylim(0, max(plot_x))
    ax_main.spines['bottom'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    ax_main.set_ylabel('Strike distance (km)', fontsize=11) 

    return sc


def update_save(day_profile, current_day, list_t, list_x, list_n):
    mask = day_profile > 0
    x = np.nonzero(mask)[0]
    n = day_profile[mask]

    list_t.extend([current_day]*len(x))
    list_x.extend(x)
    list_n.extend(n)

    return list_t, list_x, list_n

if __name__ == "__main__":

    folder = '../results/ny25_Dc2um/'

    ## read the parameters file
    params = json.load(open(folder + 'parameters.json'))
    param_r, param_e, param_m = params['region'], params['elastic'], params['model']
    Veq, Vpl, Dc = param_e['V_eq'], param_e['V_pl'], param_e['D_c']
    dx, dy, my, nx, G = param_r['dx'], param_r['dy'], param_r['my'], param_r['nx'], param_r['G']
    step_record = param_m['step_record']

    ## read the recorded simulation results
    outfile = pd.read_csv(folder + 'out_file.csv')
    jj_list = outfile['jj'].to_numpy(np.int32)
    kk_list = outfile['kk'].to_numpy(np.int32)
    new_state_list = outfile['index'].to_numpy(np.int8)
    
    times = pd.read_csv(folder + "times.csv")['time']
    times *= Dc/Vpl

    events = pd.read_csv(folder + 'events.csv')
    # save_mask = events['total_moment'] > potency_bar
    # event_list = events[save_mask].index.to_numpy(np.int32)
    event_list = events.index.to_numpy(np.int32)
    start_step_list = events['start_step'].to_numpy(np.int64)
    start_time_list = events['start_time'].to_numpy(np.int64)
    duration_list = events['duration'].to_numpy(np.float64)

    ## initial conditions of simulation profile
    time_res = 24 * 60 * 60
    time_num = 200000 # in the unit of days
    slip_x = np.zeros(nx, dtype=np.int64)
    day_profile = np.zeros(nx, dtype=np.float64)
    plot_t, plot_x, plot_n = [], [], []
    current_day = 0

    for iev in range(len(event_list)):
        start_step = start_step_list[event_list[iev]]
        start_time = start_time_list[event_list[iev]]

        if iev > 0 and start_time/time_res != current_day:
            plot_t, plot_x, plot_n = update_save(day_profile, current_day, plot_t, plot_x, plot_n)
            day_profile[:] = 0

        flag_event_on = True
        slip_x[kk_list[start_step]] += 1
        i = start_step + 1
        number_slip = 1

        while flag_event_on and i < step_record:
            dt_next = times[i] - times[i-1]
            time = times[i-1]
            jj, kk, new_state = jj_list[i], kk_list[i], new_state_list[i]

            time_unit_0 = int(time / time_res)
            time_unit_1 = int((time + dt_next) / time_res)
            if time_unit_0 == time_unit_1:
                day_profile += slip_x * dt_next
            else:
                if time_unit_1 != time_unit_0 + 1:
                    print('Warning! One timestep is longer than 1 day!')

                local_time = time
                time_left = time_unit_1 * time_res - local_time
                day_profile += slip_x * time_left

                plot_t, plot_x, plot_n = update_save(day_profile, time_unit_0, plot_t, plot_x, plot_n)
                day_profile[:] = 0

                day_profile += slip_x * (time + dt_next - time_unit_1 * time_res)

            if new_state == 2:
                slip_x[kk_list[i]] += 1
                number_slip += 1
            elif new_state == 0:
                slip_x[kk_list[i]] -= 1
                number_slip -= 1

            if number_slip == 0:
                flag_event_on = False
                current_day = time_unit_1
            else:
                i += 1

    norm = time_res * 25
    fig, axes = plt.subplots(1, 1, figsize=(10, 4))
    sc = plot_event(axes, np.asarray(plot_t), np.asarray(plot_x), np.asarray(plot_n)/norm, cutoff=False, start_day=0, end_day=0)
    plt.tight_layout()
    plt.savefig(folder + 'whole.png', dpi=300)
