# The algorithm is the same as the [0, 1] file
# And I don't use any parallel computing here
# ---------------- update date: 13 June

import numpy as np
import pandas as pd
import json
import os
import time
from numba import njit
import matplotlib.pyplot as plt
from scipy.stats import lognorm
import comp_kernel
import loadrate

def prep(folder):
    # read the parameter files
    parameters = json.load(open(folder + 'parameters.json'))
    param_r, param_e = parameters['region'], parameters['elastic']

    dx, dy, nx, my = param_r['dx'], param_r['dy'], param_r['nx'], param_r['my']
    xper, yper, sourcenum = param_r['xper'], param_r['yper'], param_r['sourcenum']
    W, hh, mu_d, mudomu = param_r['W'], param_r['hh'], param_r['mu_d'], param_r['mudomu']
    if W != 0:
        G, D = mu_d/mudomu, 1-mudomu
    else:
        G, D = mu_d, 0

    Kjk = comp_kernel.compute(W, hh, G, D, xper, yper, nx, my, dx, dy, sourcenum)

    # calculate the loading rate
    Vpl = param_e['V_pl']
    loading = 'bs+'
    taudot = loadrate.loadrate(Vpl, loading, hh, W, mu_d, G, my, nx, Kjk)
    
    # peak strength 
    tau_p0 = param_e['tau_p0']
    tau_p = np.ones((my, nx)) * tau_p0

    # initialize stress
    mu_dist = np.log(tau_p0)
    np.random.seed(1)
    sigma_dist = param_e['sigma_dist']
    tau = 0.25 * lognorm(s=sigma_dist, scale=np.exp(mu_dist)).rvs((my, nx))

    # update and save the parameters
    param_e['Kii'] = Kjk[0, 0]
    param_r['G'], param_r['D'], param_r['Axy'] = G, D, dx * dy
    parameters['elastic'], parameters['region'] = param_e, param_r
    with open(folder + 'parameters.json', 'w') as json_file:
        json.dump(parameters, json_file, indent=4)

    return Kjk, taudot, tau_p, tau

# @njit # (parallel=True)
def update_step(tau, tau_s, taudot, indx, slip, Vslip, K_sym, my, nx):
    min_val = 1e300
    jj = 0
    kk = 0
    for j in range(my):
        for k in range(nx):
            denom = taudot[j, k]
            if denom == 0.0:
                continue
            val = (tau_s[j, k] - tau[j, k]) / denom
            if val < min_val:
                min_val = val
                jj, kk = j, k
    dtnext = min_val

    slip += Vslip * indx * dtnext

    tau += dtnext * taudot
    if indx[jj, kk] == 0:
        indx[jj, kk] = 1
        taudot += Vslip * K_sym[my-jj-1:2*my-jj-1, nx-kk-1:2*nx-kk-1]     
    else:        
        indx[jj, kk] = 0
        taudot -= Vslip * K_sym[my-jj-1:2*my-jj-1, nx-kk-1:2*nx-kk-1]

    return dtnext, jj, kk, tau, taudot, indx, slip

if __name__ == "__main__":

    istep_record = 50000
    folder = '../results/test_RSQSim/test_3/'
    if not os.path.exists(folder + 'events/'):
        os.makedirs(folder + 'events/')
    
    # prepare the kernel function and stress
    start = time.time()
    Kjk, taudot, tau_p, tau = prep(folder)
    K_vert = np.concatenate((Kjk[::-1, :], Kjk[1:, :]), axis=0)
    K_sym = np.concatenate((K_vert[:, ::-1], K_vert[:, 1:]), axis=1)
    end = time.time()
    print(f"time for preparation:{end-start :.4f} seconds")

    params = json.load(open(folder + 'parameters.json'))
    param_r = params['region']
    param_e = params['elastic']

    eps_D, Vslip = param_e['eps_D'], param_e['Vslip']
    dx, dy, my, nx = param_r['dx'], param_r['dy'], param_r['my'], param_r['nx']
    Axy = dx * dy

    tau_d = (1 - eps_D) * tau_p
    tau_s = tau_p

    indx = np.zeros((my, nx), dtype=np.int64)
    

    # initiation inside the simulation
    tim = 0.0
    slip_number = 0
    istep = 0
    iev = -1
    potrate = 0.0
    total_potency = 0.0
    max_x = 0
    t_max = 0

    # initiation about plotting the slip profile
    slip = np.zeros((my, nx))
    slip_plot = np.zeros((500000, nx))
    stress_plot = np.zeros((500000, nx))
    slip_time = np.zeros(500000)
    iplotslip = 0

    # final output files
    # outfile: 0: time, 1: jj, 2: kk, 3: state(after trnsition), 4: number_1
    outfile_np = np.zeros((istep_record, 5))

    start = time.time()
    while istep < istep_record:

        if slip_number == 0:
            tau_s = tau_p.copy()

        dtnext, jj, kk, tau, taudot, indx, slip = update_step(tau, tau_s, taudot, indx, slip, Vslip, K_sym, my, nx)
        
        tim += dtnext
        total_potency += potrate * dtnext

        outfile_np[istep, 0] = tim
        outfile_np[istep, 1] = jj
        outfile_np[istep, 2] = kk
        outfile_np[istep, 3] = indx[jj, kk]
        outfile_np[istep, 4] = slip_number

        
        if indx[jj, kk] == 1:
            tau_s[jj, kk] = 0.0
            slip_number += 1
        else:
            tau_s[jj, kk] = tau_d[jj, kk]
            slip_number -= 1
        
        if dtnext < 0:
            print('Wrong!')

        if (kk == nx/2) & (jj == 10):
            slip_plot[iplotslip] = np.mean(slip, axis=0)
            stress_plot[iplotslip] = np.mean(tau, axis=0)
            iplotslip += 1
            slip_time[iplotslip] = tim
        elif istep % int(istep_record/40) == 0:
            slip_plot[iplotslip] = np.mean(slip, axis=0)
            stress_plot[iplotslip] = np.mean(tau, axis=0)
            iplotslip += 1
            slip_time[iplotslip] = tim
        
        istep += 1
        if istep % 2000000 == 0:
            print('istep:', istep)
    
    end = time.time()
    print(f"time for the loop:{end-start :.4f} seconds")

    # postprocess and save the plots

    out_file = pd.DataFrame(outfile_np, columns=['time', 'jj', 'kk', 'state', 'number_1'])
    out_file.to_csv(folder + 'out_file.csv', index=False)

    slip_plot = slip_plot[:iplotslip]
    stress_plot = stress_plot[:iplotslip]
    slip_time = slip_time[:iplotslip]
    np.savez(folder + 'slip_plot.npz', slip_time = slip_time, slip_plot = slip_plot, stress_plot=stress_plot)
    plt.figure()
    for i in range(0, iplotslip, 2):
        plt.plot(np.arange(nx)*param_r['dx']/1000, slip_plot[i, :]*100)
    plt.xlabel('X (km)')
    plt.ylabel('slip distance (cm)')
    plt.tight_layout()
    plt.savefig(folder + 'slip.png', dpi=300)

