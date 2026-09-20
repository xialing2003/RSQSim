# this is a script that tries to analyze the slip distribution 
# from both the nucleation region and the slip region

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

if __name__ == "__main__":

    folder = 'results/ny16_Dc10um_dx100/'

    params = json.load(open(folder + 'parameters.json'))
    param_r, param_e, param_m = params['region'], params['elastic'], params['model']
    nx, dx = param_r['nx'], param_r['dx']
    Dc, Vpl, b, sigma = param_e['D_c'], param_e['V_pl'], param_e['b'], param_e['sigma']

    istep_record = param_m['step_record']

    out_file = pd.read_csv(folder + "out_file.csv")
    record = pd.read_csv(folder + "record.csv")
    times = record['time'].to_numpy(np.float64)

    with np.load(folder + 'slip_plot.npz') as data:
        slip_time = data['slip_time']
        slip_plot = data['slip_plot'] * Dc
        slip_nuc_plot = data['slip_nuc_plot'] * Dc
        stress_plot = data['stress_plot']
    iplot = len(slip_time)

    plt.figure(figsize=(10, 9))

    plt.subplot(3, 1, 1)
    for i in range(0, iplot, 10):
        plt.plot(np.arange(nx)*dx/1000, slip_plot[i, :]*100)
    # plt.xlabel('X (km)')
    plt.ylabel('slip distance (cm)')

    plt.subplot(3, 1, 2)
    for i in range(0, iplot, 10):
        plt.plot(np.arange(nx)*dx/1000, slip_nuc_plot[i, :]*100)
    # plt.xlabel('X (km)')
    plt.ylabel('slip distance (cm)')

    plt.subplot(3, 1, 3)
    for i in range(0, iplot, 50):
        plt.plot(np.arange(nx)*dx/1000, stress_plot[i, :])
    plt.xlabel('X (km)')
    plt.ylabel(f'shear stress ({b*sigma} MPa)')

    plt.tight_layout()
    plt.savefig(folder + 'slip_stress.png', dpi=300)
    print('slip profile saved')