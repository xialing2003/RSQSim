# the matrix inside:
# 1. index (0 healing; 1 nucleation; 2 rupture)
# 2. velocity (the rate at which the element is slipping)
# 3. state (the average time of asperity contacts)
# 4. frictional stress (under the assumption of constant normal stress, a function of slipping speed and state)

# Based on v1, all the elements have a fixed a/b value
# The two remianing problems
# 1. how to fit the Kjk matrix to the dimensional code
# 2. where to start the initiation
# ---------------- update date: 20 June


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
    tau_p = np.ones((my, nx), dtype=np.float64) * tau_p0

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

def initiate():
    indx = np.zeros((my, nx), dtype=np.int64)
    tau = np.ones((my, nx), dtype=np.float64)*6
    velocity = np.zeros((my, nx), dtype=np.float64)
    q = np.zeros((my, nx), dtype=np.float64)

    return indx, tau, velocity, q

# @njit # (parallel=True)
def update_step(indx, tau, slip, Vslip, taudot, Kij, Kii, my, nx):

    # some artifial values
    aob = 0.8
    omaob = 1 - aob
    Veq = 1e3
    Kii = -0.6
    omKii = 1 - Kii
    overshoot = 0.1
    Dtaupbg = overshoot * 6
    Dtaupmin = 0.01

    # determine the next time step and the element that will transition
    dtnext = 1e300
    jj = 0
    kk = 0
    
    Dt = np.zeros((my, nx), dtype=np.float64)
    Dtaup = np.ones((my, nx), dtype=np.float64) * Dtaupbg
    for j in range(my):
        for k in range(nx):
            if indx[j, k] == 0:
                Dttest = 0.0
                Dt[j, k] = ((omaob * np.log(Veq*q[j,k])) - tau[j,k]) / taudot[j,k]
                while abs(Dt[j, k] - Dttest) > 1e-5 * abs(Dt[j, k]):
                    Dttest = Dt[j, k]
                    Dt[j, k] = ((omaob * np.log(Veq*(q[j,k]+Dttest))) - tau[j,k]) / taudot[j,k]
                Dt[j, k] = max(0, Dt[j, k])
            elif indx[j, k] == 1:              
                Dt[i] = -(aob / taudot[i]) * np.log(
                    ((1.0 / Veq) + omKii / taudot[i]) / 
                    ((1.0) / velocity[i] + omKii / taudot[i])
                )
                Dt[i] = max(0, Dt[i])
            else: # index == 2
                Dt[j, k] = (-Dtaup[j, k] - tau[j, k]) / taudot[j, k]
                Dt[j, k] = max(0, Dt[j, k])
            
            if Dt[j, k] < dtnext:
                dtnext = Dt[j, k]
                jj, kk = j, k

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
    
    # update all the elements based on the state switch of the element (jj, kk)
    for j in range(my):
        for k in range(nx):

            tau[j,k] += dtnext * taudot[j,k]

            if indx[j, k] == 0:
                if (j, k) == (jj, kk):
                    q[j,k] = 1 / Veq
                else:
                    q[j,k] += dtnext
            elif indx[j, k] == 1:
                if (j, k) == (jj, kk):
                    velocity[j,k] = 1.0 / (q[i] + dtnext)
                else:
                    V0m1 = (
                        (1.0 / velocity[i] + omKii / taudot[i])
                        * np.exp(-taudot[i] * dtnext / aob)
                        - omKii / taudot[i]
                    )
                    velocity[j,k] = 1.0 / V0m1
            else:
                if (j, k) == (jj, kk):
                    Dtaup[j,k] = max(Dtaupmin, overshoot*Dtaup[j,k])
                else:
                    slip[j,k] += Vslip * dtnext
                q[i] = 1.0 / Veq
        
            taudot[j,k] += ico * Veq * Kij[abs(jj-j), abs(kk-k)]

    return dtnext, jj, kk, tau, taudot, indx, slip

if __name__ == "__main__":

    folder = '../results/test_RSQSim/test_4/'
    if not os.path.exists(folder + 'events/'):
        os.makedirs(folder + 'events/')
    
    # prepare the kernel function and stress
    start = time.time()
    Kjk, taudot, tau_p, tau = prep(folder)
    # K_vert = np.concatenate((Kjk[::-1, :], Kjk[1:, :]), axis=0)
    # K_sym = np.concatenate((K_vert[:, ::-1], K_vert[:, 1:]), axis=1)
    end = time.time()
    print(f"time for preparation:{end-start :.4f} seconds")

    params = json.load(open(folder + 'parameters.json'))
    param_r, param_e, param_m = params['region'], params['elastic'], params['model']
    eps_D, Vslip = param_e['eps_D'], param_e['Vslip']
    dx, dy, my, nx, Axy = param_r['dx'], param_r['dy'], param_r['my'], param_r['nx'], param_r['Axy']

    # initiation inside the simulation
    istep_record = param_m['step_record']
    tim = 0.0
    slip_number = 0
    istep = 0

    indx, tau, velocity, q = initiate()

    # initiation about plotting the slip profile
    len_slip = 500000
    slip = np.zeros((my, nx))
    slip_plot = np.zeros((len_slip, nx))
    stress_plot = np.zeros((len_slip, nx))
    slip_time = np.zeros(len_slip)
    iplotslip = 0

    # final output files
    # outfile: 0: time, 1: jj, 2: kk, 3: index(after trnsition), 4: number_1
    outfile_np = np.zeros((istep_record, 5))

    start = time.time()
    while istep < istep_record:

        # I feel like the initiation should be put here
        if slip_number == 0:
            tau_s = tau_p.copy()

        dtnext, jj, kk, tau, taudot, indx, slip = update_step(indx, tau, slip, Vslip, taudot, Kjk, my, nx)
        
        tim += dtnext

        outfile_np[istep, 0] = tim
        outfile_np[istep, 1] = jj
        outfile_np[istep, 2] = kk
        outfile_np[istep, 3] = indx[jj, kk]
        outfile_np[istep, 4] = slip_number

        
        if indx[jj, kk] == 1:
            slip_number += 1
        else:
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

    out_file = pd.DataFrame(outfile_np, columns=['time', 'jj', 'kk', 'index', 'number_1'])
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

