import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from model.post.gene_list import generate_list

def plot_simu(ax, list_t, list_x, list_n, set_axis, axis_range, arrow=False, Tr_min=1):
    if set_axis:
        start_x, end_x = axis_range[0], axis_range[1]
        start_y, end_y = axis_range[2], axis_range[3]
    else:
        start_x, end_x = np.min(list_t), np.max(list_t)
        start_y, end_y = np.min(list_x)-2, np.max(list_x)+2
    vmax = min(1.0, 0.2*np.max(list_n))

    cmap = 'Blues'
    sc = ax.scatter(list_t, list_x, cmap=cmap, vmin=0, vmax=vmax, c=list_n, s=0.05, rasterized=True)
    ax.set_xlim(start_x, end_x)
    ax.set_ylim(start_y, end_y)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Strike distance (km)', fontsize=11)
    ax.set_xlabel('Time (years)', fontsize=11)
    return sc

if __name__ == '__main__':

    folder = 'results/ny32_Dc2um_dx50/'

    # Load parameters
    with open(folder + 'parameters.json') as f:
        params = json.load(f)

    param_r = params['region']
    param_e = params['elastic']
    param_m = params['model']

    Veq = param_e['V_eq']
    Vpl = param_e['V_pl']
    Dc = param_e['D_c']

    dx = param_r['dx']
    dy = param_r['dy']
    my = param_r['my']
    nx = param_r['nx']
    G = param_r['G']

    step_record = param_m['step_record']

    # Plot config
    time_res = 24*60*60
    space_res = 100
    space_num = int(nx * dx / space_res) + 1

    # Load data
    outfile = pd.read_csv(folder + 'out_file.csv')
    record = pd.read_csv(folder + 'record.csv')
    events = pd.read_csv(folder + 'events.csv')

    new_state_list = outfile['index'].to_numpy(np.int8)
    number_slip_list = outfile['slip_number'].to_numpy(np.int32)
    times = record['time'].to_numpy(np.float64) * Dc / Vpl
    space_index = (
        outfile['kk'].to_numpy(np.int32) * dx / space_res
    ).astype(np.int32)

    start_step_list = events['start_step'].to_numpy(np.int64)
    end_step_list = events['end_step'].to_numpy(np.int64)

    # Process simulation
    list_plot_simu = np.zeros((3, 300000000), dtype=np.float32)
    len_plot_simu = 0
    
    for iev in range(len(events)):
        start_step = start_step_list[iev]
        end_step = end_step_list[iev]

        times_event = times[start_step:end_step]
        state_event = new_state_list[start_step:end_step]
        space_event = space_index[start_step:end_step]

        cutoff = False
        window_start = 0
        window_end = 1
        unit_d2s = 24 * 60 * 60
        list_plot = generate_list(
            time_res, space_num, end_step - start_step, 
            times_event, state_event, space_event, 
            cutoff, window_start, window_end, unit_d2s
        )
        
        len_new_list = len(list_plot[0])
        list_plot_simu[:, len_plot_simu:len_plot_simu+len_new_list] = list_plot
        len_plot_simu += len_new_list

    
    list_t = list_plot_simu[0, :len_plot_simu]/365
    list_x = list_plot_simu[1, :len_plot_simu]*space_res/1000
    normalize = time_res*(space_res/dx)*my
    list_n = list_plot_simu[2, :len_plot_simu]/normalize

    fig, ax = plt.subplots(figsize=(10, 4))
    sc = plot_simu(ax, list_t, list_x, list_n, False, [0, 1, 0, 90])
    plt.tight_layout()

    plt.savefig(folder + 'whole.jpg', dpi=300)

