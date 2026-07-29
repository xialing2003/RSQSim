# almost the same as simulator_v2_parallel.py, the big difference is about how to set the stress stress to enter state 0

import numpy as np
import pandas as pd
import json
import time
from numba import njit, prange
from numba import set_num_threads, get_num_threads
from scipy.stats import lognorm
import comp_kernel
import loadrate

set_num_threads(5)
print("Numba threads:", get_num_threads())

def prep(folder):
    # read the parameter files
    parameters = json.load(open(folder + 'parameters.json'))
    param_r, param_e = parameters['region'], parameters['elastic']

    dx, dy, nx, my = param_r['dx'], param_r['dy'], param_r['nx'], param_r['my']
    xper, yper, sourcenum = param_r['xper'], param_r['yper'], param_r['sourcenum']
    W, hh, G, mu_d = param_r['W'], param_r['hh'], param_r['G'], param_r['mu_d']
    D = 1 - mu_d / G if W != 0 else 0

    Kjk = comp_kernel.compute(W, hh, G, D, xper, yper, nx, my, dx, dy, sourcenum)

    # calculate the loading rate
    Vpl = param_e['V_pl']
    loading = 'bs+'
    taudot = loadrate.loadrate(Vpl, loading, hh, W, mu_d, G, my, nx, Kjk)

    # dimensionalize Kjk and taudot
    b, sigma, Dc = param_e['b'], param_e['sigma'], param_e['D_c']
    Kjk = Kjk * Dc / (b * sigma)
    taudot = taudot * Dc / Vpl / (b * sigma)

    # update and save the parameters
    param_e['Kii'], param_r['Axy'] = Kjk[0, 0], dx * dy
    parameters['elastic'], parameters['region'] = param_e, param_r
    with open(folder + 'parameters.json', 'w') as json_file:
        json.dump(parameters, json_file, indent=4)

    return Kjk, taudot

def initiate(my, nx, Veq_n):
    indx = np.zeros((my, nx), dtype=np.int64)
    velocity = np.zeros((my, nx), dtype=np.float64)
    q = np.ones((my, nx), dtype=np.float64) / Veq_n

    return indx, velocity, q

def judge_dt(next_timestep):
    if next_timestep > 0:
        return next_timestep
    else:
        print('Wrong')
        return 1e300

@njit(parallel=True)
def update_step(indx, Dtau, Dtaup, taudot, velocity, q, Kjk, slip,
                my, nx, overshoot, Dtaupmin, aob, Veq_n):

    omaob = 1 - aob
    omKii = 1 - abs(Kjk[0, 0]) 

    # determine the next time step and the next transition element
    dtnext = 1e300
    jj = 0
    kk = 0
    
    min_vals = np.zeros(my)
    idxs = np.zeros(my, dtype=np.int64)

    for j in prange(my):

        min_vals[j] = 1e300

        for k in range(nx):
            if indx[j, k] == 0:
                Dttest = 0.0
                local_dt = (omaob * (np.log(Veq_n) + np.log(q[j,k])) - Dtau[j,k]) / taudot[j,k]
                while abs(local_dt - Dttest) > 1e-5 * abs(local_dt):
                    Dttest = local_dt
                    local_dt = (omaob * (np.log(Veq_n) + np.log(q[j,k]+Dttest)) - Dtau[j,k]) / taudot[j,k]
                # Dt[j,k] = judge_dt(Dt[j,k])
            elif indx[j, k] == 1:              
                local_dt = -(aob / taudot[j,k]) * np.log(
                    ((1.0 / Veq_n) + omKii / taudot[j,k]) / 
                    ((1.0) / velocity[j,k] + omKii / taudot[j,k])
                )
                # Dt[j,k] = judge_dt(Dt[j,k])
            else: # index == 2
                local_dt = (Dtaup[j, k] - Dtau[j, k]) / taudot[j, k]
                # Dt[j,k] = judge_dt(Dt[j,k])
            
            if local_dt < min_vals[j]:
                min_vals[j] = local_dt
                idxs[j] = j*nx + k
        
    for j in range(my):
            
        if min_vals[j] < dtnext:
            dtnext = min_vals[j]
            jj, kk = idxs[j]//nx, idxs[j]%nx

    # state switch
    if indx[jj, kk] == 0:
        indx[jj, kk] = 1
        ico = 0
    elif indx[jj, kk] == 1:
        indx[jj, kk] = 2
        ico = 1
    else:
        indx[jj, kk] = 0
        ico = -1

    taudot_jk = taudot[jj,kk]
    
    # update all the elements based on the state switch of the element (jj, kk)
    for j in prange(my):
        for k in range(nx):

            Dtau[j,k] += dtnext * taudot[j,k]

            if indx[j, k] == 0:
                if (j, k) == (jj, kk):
                    q[j,k] = 1 / Veq_n
                    slip[j,k] += Veq_n * dtnext
                else:
                    q[j,k] += dtnext
            elif indx[j, k] == 1:
                if (j, k) == (jj, kk):
                    velocity[j,k] = 1.0 / (q[j,k] + dtnext)
                else:
                    V0m1 = (
                        (1.0 / velocity[j,k] + omKii / taudot[j,k])
                        * np.exp(-taudot[j,k] * dtnext / aob)
                        - omKii / taudot[j,k]
                    )
                    velocity[j,k] = 1.0 / V0m1
            else:
                if (j, k) == (jj, kk):
                    Dtaup[j,k] = min(Dtaupmin, -overshoot*Dtau[j,k])
                else:
                    slip[j,k] += Veq_n * dtnext
                q[j,k] = 1.0 / Veq_n
        
            taudot[j,k] += ico * Veq_n * Kjk[abs(jj-j), abs(kk-k)]

    return dtnext, jj, kk, Dtau, taudot, indx, velocity, q, slip, taudot_jk

if __name__ == "__main__":

    folder = '../results/RSQSim_stage1/test1/'
    
    # prepare the kernel function and stress
    start = time.time()
    Kjk, taudot = prep(folder)
    end = time.time()
    print(f"time for preparation:{end-start :.4f} seconds")

    params = json.load(open(folder + 'parameters.json'))
    param_r, param_e, param_m = params['region'], params['elastic'], params['model']
    Veq, Vpl, my, nx = param_e['V_eq'], param_e['V_pl'], param_r['my'], param_r['nx']
    a, b, overshoot, Dtaupmin = param_e['a'], param_e['b'], param_e['overshoot'], param_e['Dtaupmin']
    aob, Veq_n = a/b, Veq / Vpl  # non-dimensionalize the velocity

    # initialize stress
    np.random.seed(1)
    Dtau_0, sigma_dist, Dtaupmin = param_e['Dtau_0'], param_e['sigma_dist'], param_e['Dtaupmin']
    # tau = -0.01 + 0.25 * lognorm(s=sigma_dist, scale=Dtau_0).rvs((my, nx))
    Dtau = np.random.normal(loc=Dtau_0, scale = sigma_dist, size=(my, nx))
    Dtaup = np.ones((my, nx), dtype=np.float64) * (Dtaupmin) 

    # initiation inside the simulation
    istep_record = param_m['step_record']
    tim = 0.0
    nucleation_number = 0
    slip_number = 0
    istep = 0

    indx, velocity, q = initiate(my, nx, Veq_n)

    # initiation about plotting the slip profile
    size_rec = param_m['size_rec']
    len_slip = int(istep_record / size_rec)
    slip = np.zeros((my, nx))
    slip_plot = np.zeros((len_slip, nx))
    stress_plot = np.zeros((len_slip, nx))
    slip_time = np.zeros(len_slip)
    iplotslip = 0

    # final output files
    # outfile: 0: time, 1: jj, 2: kk, 3: index(after trnsition), 4: number_2, 5: number_1
    times_np = np.zeros(istep_record, dtype=np.float64)
    outfile_np = np.zeros((istep_record, 5), dtype=np.int32)
    stress_np = np.zeros(istep_record, dtype=np.float64)
    dtauodt_np = np.zeros(istep_record, dtype=np.float64)

    start = time.time()
    while istep < istep_record:

        # # I feel like something should change if one big characteristic event finishes
        # if slip_number == 0:
        #     continue 

        dtnext, jj, kk, Dtau, taudot, indx, velocity, q, slip, taudot_jk = update_step(indx, Dtau, Dtaup, taudot, velocity, q, Kjk, slip,
                                                                            my, nx, overshoot, Dtaupmin, aob, Veq_n)
        
        tim += dtnext

        times_np[istep] = tim
        outfile_np[istep, 0] = jj
        outfile_np[istep, 1] = kk
        outfile_np[istep, 2] = indx[jj, kk]
        outfile_np[istep, 3] = slip_number # the number of elements that are at state 2
        outfile_np[istep, 4] = nucleation_number # the number of elements that are at state 1
        stress_np[istep] = Dtau[jj, kk]
        dtauodt_np[istep] = taudot_jk
        
        if indx[jj, kk] == 2:
            slip_number += 1
            nucleation_number -= 1
        elif indx[jj, kk] == 1:
            nucleation_number += 1
        else:
            slip_number -= 1
        
        if dtnext < 0:
            print('Wrong!')

        if istep % size_rec ==0:
            slip_plot[iplotslip] = np.mean(slip, axis=0)
            stress_plot[iplotslip] = np.mean(Dtau, axis=0)
            slip_time[iplotslip] = tim
            iplotslip += 1
        
        istep += 1
        if istep % 10000 == 0:
            print('istep:', istep)
    
    end = time.time()
    print(f"time for the loop:{end-start :.4f} seconds")

    # postprocess and save the plots
    times = pd.DataFrame(times_np, columns=['time'])
    times.to_csv(folder + 'times.csv', index=False)
    out_file = pd.DataFrame(outfile_np, columns=['jj', 'kk', 'index', 'slip_number', 'nucleation_number'])
    out_file.to_csv(folder + 'out_file.csv', index=False)
    stress_record = pd.DataFrame(stress_np, columns=['stress'])
    stress_record.to_csv(folder + 'stress_record.csv', index=False)
    dtauodt = pd.DataFrame(dtauodt_np, columns=['dtauodt'])
    dtauodt.to_csv(folder + 'dtauodt.csv', index=False)

    slip_plot = slip_plot[:iplotslip]
    stress_plot = stress_plot[:iplotslip]
    slip_time = slip_time[:iplotslip]
    np.savez(folder + 'slip_plot.npz', slip_time = slip_time, slip_plot = slip_plot, stress_plot=stress_plot)

