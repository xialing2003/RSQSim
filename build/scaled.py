# This file is used to examine the proportionality between moment rate and squared moment acceleration

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.stats import binned_statistic
import json

def plot_dt(dt, folder, label):
    dt_log = np.log10(dt[dt>0])

    plt.figure()
    plt.hist(dt_log, bins=50, density=True)
    plt.xlabel('log10(dt)')
    plt.ylabel('Density')
    plt.text(int(min(dt_log)), 0.5, f'Mean: {np.mean(dt):.2f}\nMedian: {np.median(dt):.2f}\nMax: {np.max(dt):.2f}\nMin: 10e{np.min(dt_log):.1f}', fontsize=12)
    plt.title('Histogram of log10(dt)')
    plt.savefig(folder + 'events/' + label +'.png', dpi=300)
    plt.close()

def smooth_function_fast(t, y, window_size=100.0):
    # Define bin edges
    t_min, t_max = t[0], t[-1]
    bins = np.arange(t_min, t_max + window_size, window_size)

    # Compute average per bin
    bin_means, bin_edges, _ = binned_statistic(t, y, statistic='mean', bins=bins)

    # Use bin centers as time points
    t_points = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return t_points, bin_means

if __name__ == "__main__":
    folder = '../results/test_RSQSim/test_v2.1/'
    potrate_np = np.load(folder + f'events/potency_0.npy')
    params = json.load(open(folder + 'parameters.json'))
    G = params['region']['G']
    Axy = params['region']['dx']**2

    pottime_raw = potrate_np[0, :]
    potrate_raw = potrate_np[1, :]

    unit_d2s = 24 * 60 * 60
    start_day = 403.2
    end_day = 403.4

    mask = (pottime_raw >= start_day*unit_d2s) & (pottime_raw <= end_day*unit_d2s)
    pottime = pottime_raw[mask]
    potrate = potrate_raw[mask]

    dt = np.diff(pottime, prepend=pottime[0])
    plot_dt(dt, folder, f'dt_hist_{start_day}_{end_day}')
    dt_uniform = round(np.median(dt), 2)

    t_uniform = np.arange(pottime[0], pottime[-1],dt_uniform)
    N = len(t_uniform)
    potrate_uniform = interp1d(pottime, potrate, kind='previous', fill_value='extrapolate')(t_uniform)

    dn = np.diff(potrate_uniform)
    dn_pad = np.append(dn, dn[-1])
    potaccel = dn_pad/dt_uniform

    window_size = 100
    smooth_time, smooth_rate = smooth_function_fast(t_uniform, potrate_uniform, window_size)
    smooth_time, smooth_accel = smooth_function_fast(t_uniform, potaccel**2, window_size)
    smooth_rate *= G * 1e6
    smooth_accel *= (G*1e6)**2

    plt.figure(figsize=(10, 5))
    plt.subplot(2, 1, 1)
    plt.plot(smooth_time/unit_d2s, smooth_rate, lw=0.5, label='Smoothed Moment Rate')
    plt.subplot(2, 1, 2)
    plt.plot(smooth_time/unit_d2s, smooth_accel, lw=0.5, label='Smoothed Squared Moment Acceleration')
    plt.legend()
    plt.tight_layout()
    plt.savefig(folder + f'events/smooth_moment_rate_accel_{start_day}_{end_day}.png', dpi=300)
