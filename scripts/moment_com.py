import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from model.post.plot_event import plot_moment

def plot_dist(ax, times, slip_rate, nuc_rate, coe, flag_cut, t_range):
    if flag_cut:
        mask = (times >= t_range[0]) & (times <= t_range[1])
        slip_rate = slip_rate[mask]
        nuc_rate = nuc_rate[mask]

    mask = (nuc_rate != 0) & (slip_rate != 0)
    slip_rate = slip_rate[mask]
    nuc_rate = nuc_rate[mask]

    ratio = coe * slip_rate / nuc_rate

    median = np.median(ratio)
    mean = np.mean(ratio)
    maximum = np.max(ratio)
    minimum = np.min(ratio)

    print(maximum, minimum, slip_rate[0], slip_rate[-1])

    # bins = np.arange(np.min(ratio), np.max(ratio), 0.1)
    # bins = np.arange(np.min(ratio), 100, 0.1)
    bins = np.arange(0, 50, 0.1)
    ax.hist(ratio, bins=bins)

    ax.axvline(1, linestyle='--', linewidth=1, c='tab:orange')

    text = (
        f'Median = {median:.3f}\n'
        f'Mean = {mean:.3f}\n'
        f'Max = {maximum:.3f}\n'
        f'Min = {minimum:.3f}'
    )

    ax.text(
        0.95, 0.95, text,
        transform=ax.transAxes,
        ha='right',
        va='top'
    )

if __name__ == '__main__':

    folder = 'results/ny16_Dc10um_dx100/'

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

    # Load data
    outfile = pd.read_csv(folder + 'out_file.csv')
    record = pd.read_csv(folder + 'record.csv')
    events = pd.read_csv(folder + 'events.csv')

    number_slip_list = outfile['slip_number'].to_numpy(np.int32)
    nuc_vel_list = record['nuc_MR'].to_numpy(np.float64)
    times = record['time'].to_numpy(np.float64) * Dc / Vpl

    unit_d2s = 24 * 60 * 60
    pot_coe = dx * dy * G * Veq * 1e6
    nuc_coe = dx * dy * G * Vpl * 1e6

    select_event = True
    iev = 47897
    if select_event:
        start_step = int(events['start_step'].iloc[iev])
        end_step = int(events['start_step'].iloc[iev + 1])
        times = times[start_step:end_step] - times[start_step]
        number_slip_list = number_slip_list[start_step:end_step]
        nuc_vel_list = nuc_vel_list[start_step:end_step]

    flag_cut = False
    window_start = 420
    window_end = 435
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 4))
    ax_moment = axes[0]
    plot_moment(
        ax_moment,
        times / unit_d2s,
        number_slip_list,
        pot_coe,
        True,
        nuc_vel_list,
        nuc_coe,
        flag_cut,
        [window_start, window_end],
    )

    ax_dist = axes[1]
    plot_dist(ax_dist, times[1:]/unit_d2s, number_slip_list[1:], nuc_vel_list[1:], pot_coe/nuc_coe, flag_cut, [window_start, window_end])
    plt.tight_layout()

    os.makedirs(folder + 'moment', exist_ok=True)
    if select_event:
        if flag_cut == False:
            plt.savefig(folder + f'moment/event_{iev}.png', dpi=300)
        else:
            plt.savefig(folder + f'moment/event_{iev}_{window_start}_{window_end}.png', dpi=300)
    else:
        if flag_cut == False:
            plt.savefig(folder + 'moment/all.png', dpi=300)
        else:
            plt.savefig(folder + f'moment/{window_start}_{window_end}.png', dpi=300)