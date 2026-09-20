import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from model.post.gene_list import generate_list
from model.post.plot_event import plot_event, plot_moment

if __name__ == '__main__':

    folder = 'results/ny16_Dc2um/'

    # Load parameters
    with open(folder + 'parameters.json') as f:
        params = json.load(f)

    param_r = params['region']
    param_e = params['elastic']

    Veq = param_e['V_eq']
    Vpl = param_e['V_pl']
    Dc = param_e['D_c']

    dx = param_r['dx']
    dy = param_r['dy']
    my = param_r['my']
    nx = param_r['nx']
    G = param_r['G']

    # Plot config
    time_res = 60
    space_res = 100
    space_num = int(nx * dx / space_res) + 1

    # Load data
    outfile = pd.read_csv(folder + 'out_file.csv')
    record = pd.read_csv(folder + 'record.csv')
    events = pd.read_csv(folder + 'events.csv')

    new_state_list = outfile['index'].to_numpy(np.int8)
    number_slip_list = outfile['slip_number'].to_numpy(np.int32)
    nuc_vel_list = record['nuc_MR'].to_numpy(np.float64)
    times = record['time'].to_numpy(np.float64) * Dc / Vpl

    space_index = (
        outfile['kk'].to_numpy(np.int32) * dx / space_res
    ).astype(np.int32)

    unit_d2s = 24 * 60 * 60
    pot_coe = dx * dy * G * Veq * 1e6
    nuc_coe = dx * dy * G * Vpl * 1e6

    # Select events
    potency_bar = 3000
    event_list = events.index[
        events['total_moment'] > potency_bar
    ].to_numpy(np.int32)

    start_step_list = events['start_step'].to_numpy(np.int64)
    end_step_list = events['end_step'].to_numpy(np.int64)
    start_time_list = events['start_time'].to_numpy(np.int64)

    # Plot events
    iev = 60790
    flag_cut = True
    window_start = 2
    window_end = 3

    start_step = start_step_list[iev]
    end_step = end_step_list[iev]
    start_time = start_time_list[iev]

    times_event = times[start_step:end_step] - start_time
    state_event = new_state_list[start_step:end_step]
    space_event = space_index[start_step:end_step]
    potency = number_slip_list[start_step:end_step]
    potency_nuc = nuc_vel_list[start_step:end_step]

    list_t, list_x, list_n = generate_list(
        time_res, space_num, end_step - start_step,
        times_event, state_event, space_event,
        flag_cut, window_start, window_end, unit_d2s
    )

    print(
        f'event {iev}: {end_step - start_step} steps, '
        f'{len(list_t)} plot points'
    )

    list_t *= time_res / unit_d2s
    list_x *= space_res / 1000

    normalize = time_res * (space_res / dx) * my

    fig = plt.figure(figsize=(10, 4))

    ax_main = fig.add_axes([0.08, 0.24, 0.8, 0.72])
    ax_moment = fig.add_axes([0.08, 0.12, 0.8, 0.12])

    sc = plot_event(
        ax_main,
        list_t,
        list_x,
        list_n / normalize,
        flag_cut,
        [window_start, window_end, 0, 55],
        arrow=False
    )

    plot_moment(
        ax_moment,
        times_event / unit_d2s,
        potency,
        pot_coe,
        True,
        potency_nuc,
        nuc_coe,
        flag_cut,
        [window_start, window_end],
    )

    cbar_ax = fig.add_axes([0.9, 0.25, 0.02, 0.72])
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label('Fraction of slip activity')

    filename = (
        folder
        + f'events/event_{iev}_{window_start}_{window_end}'
        + f'.jpg'
    )

    plt.savefig(filename, dpi=300)
    plt.close(fig)