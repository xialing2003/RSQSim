# no parallel
import numpy as np
import pandas as pd
import json
import time
from numba import njit, prange
from numba import set_num_threads, get_num_threads
from scipy.stats import lognorm
import model.simu.comp_kernel as comp_kernel
import model.simu.loadrate as loadrate

# set_num_threads(5)
# print("Numba threads:", get_num_threads())

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

@njit#(parallel=True)
def update_step(indx, Dtau, Dtaup, taudot, velocity, q, Kjk, slip, slip_nuc,
                Dt, flag_dt, my, nx, overshoot, Dtaupmin, aob, Veq_n):

    omaob = 1 - aob
    omKii = 1 - abs(Kjk[0, 0]) 

    # determine the next time step and the next transition element
    dtnext = 1e300
    jj = 0
    kk = 0

    if flag_dt == False:
        for j in range(my):
            for k in range(nx):
                if indx[j, k] == 0:
                    Dttest = 0.0
                    local_dt = (omaob * (np.log(Veq_n) + np.log(q[j,k])) - Dtau[j,k]) / taudot[j,k]
                    while abs(local_dt - Dttest) > 1e-5 * abs(local_dt):
                        Dttest = local_dt
                        local_dt = (omaob * (np.log(Veq_n) + np.log(q[j,k]+Dttest)) - Dtau[j,k]) / taudot[j,k]
                    # Dt[j,k] = judge_dt(Dt[j,k])
                    Dt[j, k] = local_dt
                elif indx[j, k] == 1:              
                    local_dt = -(aob / taudot[j,k]) * np.log(
                        ((1.0 / Veq_n) + omKii / taudot[j,k]) / 
                        ((1.0) / velocity[j,k] + omKii / taudot[j,k])
                    )
                    Dt[j, k] = local_dt
                    # Dt[j,k] = judge_dt(Dt[j,k])
                else: # index == 2
                    local_dt = (Dtaup[j, k] - Dtau[j, k]) / taudot[j, k]
                    Dt[j, k] = local_dt
                    # Dt[j,k] = judge_dt(Dt[j,k])
                
                if local_dt < dtnext:
                    dtnext = local_dt
                    jj, kk = j, k
    else:
        dtnext = np.min(Dt)
        flat_idx = np.argmin(Dt)
        jj, kk = flat_idx//nx, flat_idx%nx

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
    MR_nuc = 0
    for j in range(my):
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
                    delta_slip = aob/omKii * np.log(1.0/(V0m1 * velocity[j,k])) - taudot[j,k]*dtnext/omKii
                    velocity[j,k] = 1.0 / V0m1
                    slip_nuc[j,k] += delta_slip
                MR_nuc += velocity[j, k]
            else:
                if (j, k) == (jj, kk):
                    Dtaup[j,k] = min(Dtaupmin, -overshoot*Dtau[j,k])
                else:
                    slip[j,k] += Veq_n * dtnext
                q[j,k] = 1.0 / Veq_n
        
            taudot[j,k] += ico * Veq_n * Kjk[abs(jj-j), abs(kk-k)]

    return dtnext, jj, kk, Dtau, taudot, indx, velocity, q, slip, slip_nuc, Dt, MR_nuc

@njit
def mean_axis0(arr):
    rows, cols = arr.shape
    res = np.zeros(cols, dtype=arr.dtype)
    for j in range(cols):
        for i in range(rows):
            res[j] += arr[i, j]
    return res / rows

@njit
def run_simulation(istep_record, my, nx, aob, Veq_n, size_rec, Dtau, Dtaupmin, taudot, indx, velocity, q, Kjk):
    tim = 0.0
    nucleation_number = 0
    slip_number = 0
    istep = 0

    Dtaup = np.ones((my, nx), dtype=np.float64) * (Dtaupmin) 
    Dt = np.zeros((my, nx), dtype=np.float64)
    flag_dt = False

    # initiation about plotting the slip profile
    len_slip = int(istep_record / size_rec)
    slip = np.zeros((my, nx))
    slip_nuc = np.zeros((my, nx))
    slip_plot = np.zeros((len_slip, nx))
    slip_nuc_plot = np.zeros((len_slip, nx))
    stress_plot = np.zeros((len_slip, nx))
    slip_time = np.zeros(len_slip)
    iplotslip = 0

    # final output files
    # outfile: 0: time, 1: jj, 2: kk, 3: index(after trnsition), 4: number_2, 5: number_1
    # record: 0: time, 1: stress, 2: stressing_rate, 3: moment rate from nuc
    outfile_np = np.zeros((istep_record, 5), dtype=np.int32)
    record_np = np.zeros((istep_record, 4), dtype=np.float64)

    # record the stress and velocity during nucleation
    elements = np.array([(0, 250), (1, 300), (2, 400), (3, 500), (4, 600)])
    n_track = 5
    js, ks = elements[:, 0], elements[:, 1]
    nuc_count = 0
    flag_nuc = False
    # select_nuc: (to record the stress, velocity of selected elements durng nucleation)
    # 0: index in the 5 elements, 1: stress;, 2: velocity
    select_nuc = np.zeros((istep_record, 3), dtype=np.float64)

    while istep < istep_record:

        dtnext, jj, kk, Dtau, taudot, indx, velocity, q, slip, slip_nuc, Dt, MR_nuc \
            = update_step(indx, Dtau, Dtaup, taudot, velocity, q, Kjk, slip, slip_nuc,
                        Dt, flag_dt, my, nx, overshoot, Dtaupmin, aob, Veq_n)
        
        tim += dtnext
        Dt -= dtnext
        if indx[jj, kk] == 1:
            Dt[jj, kk] = -(aob / taudot[jj,kk]) * np.log(
                    ((1.0 / Veq_n) + omKii / taudot[jj,kk]) / 
                    ((1.0) / velocity[jj,kk] + omKii / taudot[jj,kk])
                )
            flag_dt = True
        else:
            flag_dt = False

        outfile_np[istep] = [jj, kk, indx[jj, kk], slip_number, nucleation_number]
        record_np[istep] = [tim, Dtau[jj, kk], taudot[jj, kk], MR_nuc]

        if indx[jj, kk] == 2:
            slip_number += 1
            nucleation_number -= 1
        elif indx[jj, kk] == 1:
            nucleation_number += 1
        else:
            slip_number -= 1
        
        if dtnext < 0:
            print('Wrong!')

        for t in range(n_track):
            flag_nuc = False
            jjs, kks = js[t], ks[t]
            if indx[jjs, kks] == 1:
                flag_nuc = True
            if (jjs, kks) == (jj, kk) and indx[jj, kk] == 2:
                flag_nuc = True
            if flag_nuc:
                select_nuc[nuc_count] = [t, Dtau[jjs, kks], velocity[jjs, kks]]
                nuc_count += 1

        if istep % size_rec ==0:
            slip_plot[iplotslip] = mean_axis0(slip)
            slip_nuc_plot[iplotslip] = mean_axis0(slip_nuc)
            stress_plot[iplotslip] = mean_axis0(Dtau)
            slip_time[iplotslip] = tim
            iplotslip += 1
        
        istep += 1
        if istep % 10000 == 0:
            print('istep:', istep)

    return outfile_np, record_np, slip_plot, slip_nuc_plot, stress_plot, slip_time, iplotslip, select_nuc, nuc_count

if __name__ == "__main__":

    folder = '../results/ny16_Dc2um/'
    
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
    omKii = 1 - abs(Kjk[0, 0])

    # initialize stress
    np.random.seed(1)
    Dtau_0, sigma_dist, Dtaupmin = param_e['Dtau_0'], param_e['sigma_dist'], param_e['Dtaupmin']
    # tau = -0.01 + 0.25 * lognorm(s=sigma_dist, scale=Dtau_0).rvs((my, nx))
    Dtau = np.random.normal(loc=Dtau_0, scale = sigma_dist, size=(my, nx))

    # initiation inside the simulation
    istep_record = param_m['step_record']
    size_rec = param_m['size_rec']
    indx, velocity, q = initiate(my, nx, Veq_n)
    
    start = time.time()
    outfile_np, record_np, slip_plot, slip_nuc_plot, stress_plot, slip_time, \
    iplotslip, select_nuc, nuc_count = run_simulation(istep_record, my, nx, aob, \
                            Veq_n, size_rec, Dtau, Dtaupmin, taudot, indx, velocity, q, Kjk)
    end = time.time()
    print(f"time for the loop:{end-start :.4f} seconds")

    # postprocess and save the plots
    out_file = pd.DataFrame(outfile_np, columns=['jj', 'kk', 'index', 'slip_number', 'nucleation_number'])
    out_file.to_csv(folder + 'out_file.csv', index=False)
    record = pd.DataFrame(record_np, columns=['time', 'stress', 'dtauodt', 'nuc_MR'])
    record.to_csv(folder + 'record.csv', index=False)

    slip_plot = slip_plot[:iplotslip]
    stress_plot = stress_plot[:iplotslip]
    slip_time = slip_time[:iplotslip]
    np.savez(folder + 'slip_plot.npz', slip_time = slip_time, slip_plot = slip_plot, stress_plot=stress_plot)

    nuc_indx = select_nuc[0, :nuc_count].astype(np.int8)
    nuc_stress = select_nuc[1, :nuc_count]
    nuc_vel = select_nuc[2, :nuc_count]
    np.savez(folder + 'nucleation_data.npz', element_index=nuc_indx, stress=nuc_stress, velocity=nuc_vel)