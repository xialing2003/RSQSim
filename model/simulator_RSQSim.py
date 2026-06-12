# This code uses a 2D framework. The evolution of inside elements follows the rate-and-state friction law. 
# All the elements have an inherent parameter of a/b ratio
# The real-time character during the simulations include
# 1. index (0 healing; 1 nucleation; 2 rupture)
# 2. velocity (the rate at which the element is slipping)
# 3. state (the average time of asperity contacts)
# 4. frictional stress (under the assumption of constant normal stress, a function of slipping speed and state)

# the folder that I use
# test_RSQSim/test_1/
#       dx = 100m, the simulatd domain is 60 km * 2.5 km
#       the system is still periodic in the y direction, this only influences how we calculate the kernel function

# start with the version that doesn't consider the parallel computing

# ---------------- update date: May 13

import numpy as np
import pandas as pd
import json
import os
import time
# from numba import njit
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
    folder = '../results/test_RSQSim/test_1/'
    if not os.path.exists(folder + 'events/'):
        os.makedirs(folder + 'events/')
    
    # prepare the kernel function and stress
    start = time.time()
    Kjk, taudot, tau_p, tau = prep(folder)
    K_vert = np.concatenate((Kjk[::-1, :], Kjk[1:, :]), axis=0)
    K_sym = np.concatenate((K_vert[:, ::-1], K_vert[:, 1:]), axis=1)
    end = time.time()
    print(f"time for preparation:{end-start :.4f} seconds")

    with open(folder + 'parameters.json', 'r') as json_file:
        parameters = json.load(json_file)
    param_r = parameters['region']
    param_e = parameters['elastic']

    eps_D, Vslip = param_e['eps_D'], param_e['Vslip']
    dx, dy, my, nx = param_r['dx'], param_r['dy'], param_r['my'], param_r['nx']
    Axy = dx * dy

    tau_d = (1 - eps_D) * tau_p
    tau_s = tau_p

    indx = np.zeros((my, nx), dtype=np.int64)
    event_index = np.zeros((my, nx), dtype=np.int64)
    slip_x = np.zeros(500, dtype=np.int64)

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
    # events: 0: start step, 1: total potency, 2: duration, 3: area, 4: max x
    # outfile: 0: time, 1: potency rate, 2: jj, 3:kk
    events_np = np.zeros((500000, 5))
    outfile_np = np.zeros((istep_record, 4))

    # time to record the beginning time
    flag_end_event = False
    start_time_of_pre_event = 0.0

    # save the propagation profile
    save_history = True
    time_res, space_res = param_e['time_resolution'], param_e["space_resolution"]
    if save_history:
        prop_profile = np.zeros((80000, 500)) # hours, and 500 m
        potency_bar = 1e5

    start = time.time()
    while istep < istep_record:

        if slip_number == 0:
            iev += 1
            events_np[iev, 0] = istep
            if iev > 0:
                events_np[iev - 1, 1] = total_potency
                events_np[iev - 1, 2] = tim - start_time_of_pre_event
                events_np[iev - 1, 3] = np.sum(event_index)
                events_np[iev - 1, 4] = max_x

                if save_history and total_potency > potency_bar:
                    np.save(folder + f'events/history_X_{iev-1}.npy', prop_profile[:t_max+1])
                    np.save(folder + f'events/potency_{iev-1}.npy', outfile_np[int(events_np[iev-1, 0]):istep, :2])
            
            prop_profile.fill(0)
            tau_s = tau_p.copy()
            total_potency = 0.0
            max_x = 0
            event_index.fill(0)
            t_max = 0
            flag_end_event = True
        else:
            flag_end_event = False

        dtnext, jj, kk, tau, taudot, indx, slip = update_step(tau, tau_s, taudot, indx, slip, Vslip, K_sym, my, nx)
        
        tim += dtnext
        total_potency += potrate * dtnext

        outfile_np[istep, 0] = tim
        outfile_np[istep, 1] = potrate
        outfile_np[istep, 2] = jj
        outfile_np[istep, 3] = kk

        if indx[jj, kk] == 1:
            tau_s[jj, kk] = 0.0
            slip_number += 1
            potrate += Vslip * Axy
            slip_x[int(kk*dx//space_res)] += 1
            event_index[jj, kk] = 1
            if kk > max_x:
                max_x = kk
        else:
            tau_s[jj, kk] = tau_d[jj, kk]
            slip_number -= 1
            potrate -= Vslip * Axy
            slip_x[int(kk*dx//space_res)] -= 1

        if flag_end_event:
            start_time_of_pre_event = tim

        # save the propagation profile
        time_minute_0 = int(np.floor((tim - dtnext - start_time_of_pre_event)/time_res))
        time_minute_1 = int(np.floor((tim - start_time_of_pre_event)/time_res))
        if time_minute_0 == time_minute_1:
            prop_profile[time_minute_0, :] += slip_x*dtnext
        else:
            prop_profile[time_minute_0, :] += slip_x*(time_minute_1*time_res - (tim - dtnext - start_time_of_pre_event))
            prop_profile[time_minute_1, :] += slip_x*(tim - start_time_of_pre_event - time_minute_1*time_res)
        t_max = time_minute_1

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

    events = pd.DataFrame(events_np[:iev+1, :], columns=['start_step', 'Potency', 'Duration', 'Area', 'max_x'])
    out_file = pd.DataFrame(outfile_np, columns=['time', 'potency_rate', 'jj', 'kk'])

    events['steps'] = events['start_step'].shift(-1) - events['start_step']
    events = events[:-1]
    events.reset_index(inplace=True)

    events.to_csv(folder + 'events.csv', index=False)
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

