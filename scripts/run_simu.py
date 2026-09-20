import numpy as np
import pandas as pd
import json
import time

from model.simu.simulator import prep
from model.simu.simulator import initiate
from model.simu.simulator import run_simulation

if __name__ == "__main__":

    folder = 'results/ny16_Dc2um_dx100/'
    
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
    iplotslip, select_nuc, nuc_count = run_simulation(istep_record, my, nx, aob, Veq_n, overshoot, \
                            size_rec, Dtau, Dtaupmin, taudot, indx, velocity, q, Kjk)
    end = time.time()
    print(f"time for the loop:{end-start :.4f} seconds")

    # postprocess and save the plots
    out_file = pd.DataFrame(outfile_np, columns=['jj', 'kk', 'index', 'slip_number', 'nucleation_number'])
    out_file.to_csv(folder + 'out_file.csv', index=False)
    record = pd.DataFrame(record_np, columns=['time', 'stress', 'dtauodt', 'nuc_MR'])
    record.to_csv(folder + 'record.csv', index=False)

    slip_plot = slip_plot[:iplotslip]
    slip_nuc_plot = slip_nuc_plot[:iplotslip]
    stress_plot = stress_plot[:iplotslip]
    slip_time = slip_time[:iplotslip]
    np.savez(folder + 'slip_plot.npz', slip_time = slip_time, stress_plot=stress_plot,
             slip_plot = slip_plot, slip_nuc_plot = slip_nuc_plot)

    nuc_indx = select_nuc[0, :nuc_count].astype(np.int8)
    nuc_stress = select_nuc[1, :nuc_count]
    nuc_vel = select_nuc[2, :nuc_count]
    np.savez(folder + 'nucleation_data.npz', element_index=nuc_indx, stress=nuc_stress, velocity=nuc_vel)