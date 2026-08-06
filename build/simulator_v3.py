# plan to separate the elements into three categories and do calculations individually
import numpy as np
import pandas as pd
import json
import time
from numba import njit, prange
from numba import set_num_threads, get_num_threads
from numba.experimental import jitclass
from scipy.stats import lognorm
import comp_kernel
import loadrate

set_num_threads(5)
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

def mirror(a):
    v = np.concatenate((a[::-1, :], a[1:, :]), axis=0)
    return np.concatenate((v[:, ::-1], v[:, 1:]), axis=1)


@njit
def update_step(state_j, state_k, state_n, Dtau, Dtaup, taudot, velocity, q, Kjk, slip,
                my, nx, overshoot, Dtaupmin, aob, Veq_n):

    omaob = 1 - aob
    omKii = 1 - abs(Kjk[0, 0]) 

    # determine the next time step and the next transition element
    dtnext = 1e300
    ii = 0
    idx_to_change = 0
    jj = 0
    kk = 0

    for i in range(state_n[0]):
        j, k = state_j[0, i], state_k[0, i]
        Dttest = 0.0
        local_dt = (omaob * (np.log(Veq_n) + np.log(q[j,k])) - Dtau[j,k]) / taudot[j,k]
        while abs(local_dt - Dttest) > 1e-5 * abs(local_dt):
            Dttest = local_dt
            local_dt = (omaob * (np.log(Veq_n) + np.log(q[j,k]+Dttest)) - Dtau[j,k]) / taudot[j,k]
        if local_dt < dtnext:
            dtnext = local_dt
            idx_to_change = 0
            ii = i

    for i in range(state_n[1]):
        j, k = state_j[1, i], state_k[1, i]
        local_dt = -(aob / taudot[j,k]) * np.log(
            ((1.0 / Veq_n) + omKii / taudot[j,k]) / 
            ((1.0) / velocity[j,k] + omKii / taudot[j,k])
        )
        if local_dt < dtnext:
            dtnext = local_dt
            idx_to_change = 1
            ii = i

    for i in range(state_n[2]):
        j, k = state_j[2, i], state_k[2, i]
        local_dt = (Dtaup[j, k] - Dtau[j, k]) / taudot[j, k]
        if local_dt < dtnext:
            dtnext = local_dt
            idx_to_change = 2
            ii = i

    jj, kk = state_j[idx_to_change, ii], state_k[idx_to_change, ii]
    
    # update all the elements based on the state switch of the element (jj, kk)
    Dtau += dtnext * taudot
    q[state_j[0, :state_n[0]], state_k[0, :state_n[0]]] += dtnext
    slip[state_j[2, :state_n[2]], state_k[2, :state_n[2]]] += Veq_n * dtnext
    for i in range(state_n[1]):
        if idx_to_change == 1 and i == ii:
            continue
        j, k = state_j[1, i], state_k[1, i]
        V0m1 = (
            (1.0 / velocity[j,k] + omKii / taudot[j,k])
            * np.exp(-taudot[j,k] * dtnext / aob)
            - omKii / taudot[j,k]
        )
        velocity[j,k] = 1.0 / V0m1

    if idx_to_change == 0:
        velocity[jj, kk] = 1.0 / q[jj, kk]
    elif idx_to_change == 1:
        Dtaup[jj, kk] = min(Dtaupmin, -overshoot*Dtau[jj,kk])
    else:
        q[jj,kk] = 1.0 / Veq_n 

    return dtnext, ii, idx_to_change, Dtau, velocity, q, slip

@njit
def update_stress_rate2(taudot, coe, jj, kk, K_sym):
    my, nx = taudot.shape
    jstart = my - jj - 1
    kstart = nx - kk - 1
    for j in range(my):
        for k in range(nx):
            taudot[j,k] += coe * K_sym[jstart+j, kstart+k]
    return taudot

@njit
def update_stress_rate1(K_sym, Kii, velocity, state_j, state_k, number_1, V_c):

    taudot1 = np.zeros((my, nx), dtype = np.float64)
    for i in range(number_1):
        jj = state_j[1, i]
        kk = state_k[1, i]
        jstart = my - jj - 1
        kstart = nx - kk - 1
        if velocity[jj, kk] <= V_c:
            continue
        for j in range(my):
            for k in range(nx):
                if j == jj and k == kk:
                    continue
                taudot1[j, k] += velocity[jj, kk] * K_sym[jstart + j, kstart + k]

        # taudot1[jj, kk] -= velocity[jj, kk] * Kii

    return taudot1

if __name__ == "__main__":

    folder = '../results/RSQSim_stage3/test_v3_2/'

    params = json.load(open(folder + 'parameters.json'))
    param_r, param_e, param_m = params['region'], params['elastic'], params['model']
    Veq, Vpl, my, nx = param_e['V_eq'], param_e['V_pl'], param_r['my'], param_r['nx']
    a, b, overshoot, Dtaupmin = param_e['a'], param_e['b'], param_e['overshoot'], param_e['Dtaupmin']
    aob, Veq_n = a/b, Veq / Vpl  # non-dimensionalize the velocity
    
    # prepare the kernel function and stress
    start = time.time()
    Kjk, taudot0 = prep(folder)
    end = time.time()
    print(f"time for preparation:{end-start :.4f} seconds")

    K_sym = mirror(Kjk)
    taudot1 = np.zeros((my, nx), dtype=np.float64)
    taudot2 = np.zeros((my, nx), dtype=np.float64)
    taudot = taudot0 + taudot1 + taudot2

    # initialize stress
    np.random.seed(1)
    Dtau_0, sigma_dist, Dtaupmin = param_e['Dtau_0'], param_e['sigma_dist'], param_e['Dtaupmin']
    # tau = -0.01 + 0.25 * lognorm(s=sigma_dist, scale=Dtau_0).rvs((my, nx))
    Dtau = np.random.normal(loc=Dtau_0, scale = sigma_dist, size=(my, nx))
    Dtaup = np.ones((my, nx), dtype=np.float64) * (Dtaupmin) 

    # initiation inside the simulation
    istep_record = param_m['step_record']
    tim = 0.0
    istep = 0

    indx, velocity, q = initiate(my, nx, Veq_n)
    state_j = np.zeros((3, my * nx), dtype=np.int64)
    state_k = np.zeros((3, my * nx), dtype=np.int64)
    state_j[0, :] = np.arange(my * nx) // nx
    state_k[0, :] = np.arange(my * nx) % nx 
    state_n = np.array([my*nx, 0, 0], dtype=np.int64)
    flag_use = False
    ico_label = [-1, 0, 1]

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
    nuc_MR_np = np.zeros(istep_record, dtype=np.float64)

    start = time.time()
    while istep < istep_record:

        dtnext, ii, idx_to_change, Dtau, velocity, q, slip = update_step(state_j, state_k, state_n, 
                                                            Dtau, Dtaup, taudot, velocity, q, Kjk, slip,
                                                            my, nx, overshoot, Dtaupmin, aob, Veq_n)

        jj = state_j[idx_to_change, ii]
        kk = state_k[idx_to_change, ii]
        taudot_jk = taudot[jj, kk]
        
        tim += dtnext

        times_np[istep] = tim
        outfile_np[istep, 0] = jj
        outfile_np[istep, 1] = kk
        outfile_np[istep, 2] = (idx_to_change + 1)%3
        outfile_np[istep, 3] = state_n[2] # the number of elements that are at state 2
        outfile_np[istep, 4] = state_n[1] # the number of elements that are at state 1
        stress_np[istep] = Dtau[jj, kk]
        dtauodt_np[istep] = taudot_jk

        state_j[idx_to_change, ii] = state_j[idx_to_change, state_n[idx_to_change]-1]
        state_k[idx_to_change, ii] = state_k[idx_to_change, state_n[idx_to_change]-1]
        state_n[idx_to_change] -= 1
        idx_to_change = (idx_to_change+1)%3
        state_j[idx_to_change, state_n[idx_to_change]] = jj
        state_k[idx_to_change, state_n[idx_to_change]] = kk
        state_n[idx_to_change] += 1

        ico = ico_label[idx_to_change]
        taudot2 = update_stress_rate2(taudot2, ico * Veq_n, jj, kk, K_sym)
        taudot1 = update_stress_rate1(K_sym, Kjk[0, 0], velocity, state_j, state_k, state_n[1], 0.1 * Veq_n)
        taudot = taudot0 + taudot1 + taudot2
        
        if dtnext < 0:
            print('Wrong!')

        nuc_MR_np[istep] = velocity[state_j[1, :state_n[1]], state_k[1, :state_n[1]]].sum()

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
    nuc_MR = pd.DataFrame(nuc_MR_np, columns=['nuc_MR'])
    nuc_MR.to_csv(folder + 'nuc_MR.csv', index=False)

    slip_plot = slip_plot[:iplotslip]
    stress_plot = stress_plot[:iplotslip]
    slip_time = slip_time[:iplotslip]
    np.savez(folder + 'slip_plot.npz', slip_time = slip_time, slip_plot = slip_plot, stress_plot=stress_plot)

