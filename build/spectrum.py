# an improved version of figure 4.
# plot the estimated moment rate function in this figure as well
# The improvement here is that I replace the "np.gradient" with "np.diff" to calculate the moment acceleration
# because the center differencing method will low pass the whole time series.
import numpy as np
import pandas as pd
import os
import json
import glob
import re
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import ImageFont
from scipy.interpolate import interp1d
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.stats import binned_statistic

def smooth_function_fast(t, y, window_size=100.0):
    # Define bin edges
    t_min, t_max = t[0], t[-1]
    bins = np.arange(t_min, t_max + window_size, window_size)

    # Compute average per bin
    bin_means, bin_edges, _ = binned_statistic(t, y, statistic='mean', bins=bins)

    # Use bin centers as time points
    t_points = 0.5 * (bin_edges[1:] + bin_edges[:-1])
    return t_points, bin_means

def plot_dt(dt, folder, event_index):

    dt_log = np.log10(dt[dt>0])

    plt.figure()
    plt.hist(dt_log, bins=50, density=True)
    plt.xlabel('log10(dt)')
    plt.ylabel('Density')
    plt.text(int(min(dt_log)), 0.5, f'Mean: {np.mean(dt):.2f}\nMedian: {np.median(dt):.2f}\nMax: {np.max(dt):.2f}\nMin: 10e{np.min(dt_log):.1f}', fontsize=12)
    plt.title('Histogram of log10(dt)')
    plt.savefig(folder + f'events/dt_hist_event_{event_index}.png', dpi=300)
    plt.close()

def plot_update():
    mpl.rcParams.update({
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 12,
        # 'mathtext.fontset': 'stix',
        # 'font.family': 'serif',
    }
    )

def find_tip(folder, iev):
    outfile = pd.read_csv(folder + 'out_file.csv')
    events = pd.read_csv(folder + 'events.csv')

    start_step = int(events['start_step'].iloc[iev])
    end_step = int(events['start_step'].iloc[iev + 1])
    out_history = outfile[start_step:end_step].reset_index(drop=True)

    N = len(out_history)
    potrate_1 = np.zeros(N)
    indx = np.zeros((50, 2400))

    for i in range(N - 1):

        jj = int(out_history['jj'].iloc[i])
        kk = int(out_history['kk'].iloc[i])

        # sign = +1 if potency increases, -1 otherwise
        sign = 1 if out_history['potency_rate'][i+1] > out_history['potency_rate'][i] else -1

        # status = 0 if indx < 2 (first-slip group), else 1 (post-slip group)
        status = 0 if indx[jj, kk] < 2 else 1

        # copy previous values
        potrate_1[i+1] = potrate_1[i]

        # update the correct group
        if status == 0:
            potrate_1[i+1] += sign
            indx[jj, kk] += 1

    return potrate_1

def generate_plot(folder, event_index, window_size, ratio, label, pad = 1):
    potrate_history_np = np.load(folder + f'events/potency_{event_index}.npy')
    time = potrate_history_np[:, 0] - potrate_history_np[0, 0]
    potrate = potrate_history_np[:, 1]

    potrate_1 = find_tip(folder, event_index)

    dt = np.diff(time, prepend=time[0])
    plot_dt(dt, folder, event_index)
    dt_uniform = round(np.median(dt), 2)

    t_uniform = np.arange(time[0], time[-1], dt_uniform)
    N = len(t_uniform)
    potrate_uniform = interp1d(time, potrate, kind='previous', fill_value='extrapolate')(t_uniform).astype(np.float32)
    potrate_1_uniform = interp1d(time, potrate_1, kind='previous', fill_value='extrapolate')(t_uniform).astype(np.float32)

    # print('the length of the time series:', N)
    
    # potaccel = np.gradient(potrate_uniform, t_uniform)

    dn = np.diff(potrate_uniform)
    dn_pad = np.append(dn, dn[-1])
    potaccel = dn_pad/dt_uniform

    smooth_time, smooth_rate = smooth_function_fast(t_uniform, potrate_uniform, window_size)
    smooth_time, smooth_rate_1 = smooth_function_fast(t_uniform, potrate_1_uniform, window_size)
    smooth_time, smooth_accel = smooth_function_fast(t_uniform, potaccel**2, window_size)

    with open(folder + 'parameters.json', 'r') as f:
        params = json.load(f)
    G = params['region']['G']
    Axy = params['region']['Axy']
    Vslip = params['elastic']['Vslip']

    smooth_rate *= G*1e6*1e-11
    smooth_accel *= (G*1e6)**2*1e-18
    smooth_rate_1 *= G*1e6*1e-11*Axy*Vslip
    ref_rate = round(ratio*max(smooth_rate), 1)
    ref_accel = round(ratio*max(smooth_accel), 1)

    nfft = N * pad
    frequencies = np.fft.rfftfreq(nfft, dt_uniform)
    potrate_fft = np.fft.rfft(potrate_uniform, n=nfft)
    potrate_amplitude = np.abs(potrate_fft)

    num_intervals = 200
    bins = np.logspace(-7,1, num_intervals+1)
    ave_amplitude = []
    for i in range(num_intervals):
        mask = (frequencies>=bins[i]) & (frequencies<bins[i+1])
        avg = np.mean(potrate_amplitude[mask]) if np.any(mask) else np.nan
        ave_amplitude.append(avg)
    ave_frequencies = 10 ** ((np.log10(bins[:-1]) + np.log10(bins[1:])) /2)

    fig = plt.figure(figsize=(11, 3))
    plot_update()

    color_moment = '#1f77b4'
    color_spectrum = '#97caed'
    color_spectrum_ave = '#1f77b4'
    color_slop1 = '#1f77b4'
    color_slop2 = '#1f52b4'
    color_slopd5 = '#1f9cb4'

    ax_accel = fig.add_axes([0.02, 0.18, 0.69,0.4])
    ax_accel.plot(smooth_time/3600, smooth_accel, color=color_moment)
    ax_accel.set_xlim(0, max(smooth_time/3600))
    ax_accel.set_ylim(0, ref_accel*1.08)

    ax_accel.yaxis.set_visible(False)
    ax_accel.set_xlabel('Time (hours)')

    # reference = 0.005/2.88*ref_accel
    # text *5.76*10^22
    start_text = 10/250 * (smooth_time[-1]/3600)
    start_line = 20/250 * (smooth_time[-1]/3600)
    end_line = 65/250 * (smooth_time[-1]/3600)
    ax_accel.hlines(ref_accel, start_line, end_line, color='black', linewidth=0.5, linestyle='--')
    ax_accel.text(start_line, ref_accel*0.95, f"${ref_accel}" + r" \times 10^{18}$ $(Nm/s^2)^{2}$", va='top', ha='left')
    ax_accel.text(start_text, ref_accel, r"$\ddot{M}_0^2$", va='top', ha='left')

    ax_rate = fig.add_axes([0.02, 0.6, 0.69,0.4])
    ax_rate.plot(smooth_time/3600, smooth_rate, color=color_moment)
    ax_rate.plot(smooth_time/3600, smooth_rate_1, color='tab:green')
    ax_rate.set_xlim(0, max(smooth_time/3600))
    ax_rate.set_ylim(0, ref_rate*1.08)

    ax_rate.xaxis.set_visible(False)
    ax_rate.yaxis.set_visible(False)

    # reference = 15/3.6*ref_rate
    # text *4*10^10*6
    ax_rate.hlines(ref_rate, start_line, end_line, color='black', linewidth=0.5, linestyle='--')
    ax_rate.text(start_line, ref_rate*0.95, f"${ref_rate} " + r"\times 10^{11}$ $Nm/s$", verticalalignment='top', horizontalalignment='left')
    ax_rate.text(start_text, ref_rate, r"$\dot{M}_0$", va='top', ha='left')

    ax_spectrum = fig.add_axes([0.72, 0.18, 0.27,0.8])
    ax_spectrum.loglog(frequencies, potrate_amplitude, color = color_spectrum)
    ax_spectrum.plot(ave_frequencies, ave_amplitude, color=color_spectrum_ave, linewidth=2.5)
    ax_spectrum.loglog([1e-6, 1e-5], [1e7*10**0.5, 1e7], '--', color=color_slopd5)
    ax_spectrum.text(0.5e-5, 3e7, 'slope=-0.5', color=color_slopd5)
    ax_spectrum.loglog([0.001, 0.01],[1e7, 1e6], '--', color=color_slop1)
    ax_spectrum.text(0.02, 5e5, 'slope=-1', color=color_slop1)
    ax_spectrum.loglog([0.001, 0.01],[1e7, 1e5], '--', color=color_slop2)
    ax_spectrum.text(0.02, 5e4, 'slope=-2', color=color_slop2)
    # ax_spectrum.text(3e-6, 1, f'dx:50m\nVprop:{10/86.4:.3f}m/s\nt={50*86.4/10:.3f}s\ndt_uni:{dt_uniform:.2f}s')
    # ax_spectrum.vlines(2e-3, 1, 1e4, color='black', linewidth=0.5, linestyle='--')
    ax_spectrum.yaxis.set_visible(False)
    ax_spectrum.set_xlabel('Frequency (Hz)')
    ax_spectrum.xaxis.set_major_locator(plt.LogLocator(base=10.0, numticks=10))
    ax_spectrum.xaxis.set_minor_locator(plt.LogLocator(base=10.0, subs=np.arange(2, 10)*0.1, numticks=10))
    ax_spectrum.xaxis.set_minor_formatter(plt.NullFormatter())

    for ax in [ax_rate, ax_accel, ax_spectrum]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(False)
        # ax.spines['bottom'].set_visible(False)

    # font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    ax_rate.text(0, 0.95, label[0], transform=ax_rate.transAxes, fontsize=12, va='top', ha='left')
    ax_accel.text(0, 0.95, label[1], transform=ax_accel.transAxes, fontsize=12, va='top', ha='left')
    ax_spectrum.text(0, 0.95, label[2], transform=ax_spectrum.transAxes, fontsize=12, va='top', ha='left')

    plt.savefig(folder + f'final/4_moment_spectrum_{event_index}_{window_size}_{pad}.png', dpi=500)
    plt.close()

if __name__ == "__main__":
    folder = '../results/para_1/dx50_ny50/eps_0.6/'
    # folder = '../results/para_1/eps0.6_h500/'
    # label = ['(a)', '(b)', '(c)']
    label = ['(d)', '(e)', '(f)']
    window_size = 300
    ratio = 0.95
    pad = 5

    generate_plot(folder, 1878, window_size, ratio, label, pad)

    # event_list = np.array([7, 482, 1370, 1878, 2528])
    # event_list = np.array([3008, 3654, 4298, 5337, 5991, 6405, 6875])
    # for event_index in event_list:
    #     generate_plot(folder, event_index, window_size, coefficient, ratio, label, pad)

    # consider low modulus
    # folder = '../results/para_1/eps0.6_h500/'
    # pattern = folder + 'events/potency_*.npy'

    # files = glob.glob(pattern)
    # for file in files:
    #     match = re.search(r'potency_(\d+)\.npy', os.path.basename(file))
    #     if match:
    #         iev = int(match.group(1))
    #         # print(iev)
    #         generate_plot(folder, iev, window_size, ratio, label, pad)
    #         print(iev)