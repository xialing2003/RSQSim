# The purpose of this file is to generate the events figure based on the newly developed simulator_v2.py

import numpy as np
import json
import pandas as pd
import os


if __name__ == "__main__":

    folder = '../results/test_RSQSim/test_6/'
    if not os.path.exists(folder + 'events/'):
        os.makedirs(folder + 'events/')

    ## load parameters and output file

    params = json.load(open(folder + 'parameters.json'))
    param_r = params['region']
    param_e = params['elastic']
    param_m = params['model']

    Veq, Dc = param_e['V_eq'], param_e['D_c']
    dx, dy, my, nx, G = param_r['dx'], param_r['dy'], param_r['my'], param_r['nx'], param_r['G']
    step_record, time_res, space_res = param_m['step_record'], param_m['time_res'], param_m['space_res']

    # outfile: time, jj(my), kk(nx), state(after trnsition), number_1
    outfile = pd.read_csv(folder + 'out_file.csv')
    times = pd.read_csv(folder + "times.csv")['time']

    ## set up the output files

    # to save the event catlog
    event_np = np.zeros((500000, 5)) # [0: start time, 1: total moment, 2: duration, 3: area, 4: max x]
    number_slip = 0
    iev = -1
    event_potency = 0.0
    event_index = np.zeros((my, nx), dtype=np.int64) # to record the entire rupture area
    
    # to save the propagation profile
    global start_time
    potency_bar = 0
    space_num = int(nx*dx/space_res) + 1
    unit_d2s = 24 * 60 * 60
    time_num = int(20*unit_d2s/time_res) + 1
    slip_x = np.zeros(space_num, dtype=np.int64)
    prop_profile = np.zeros((time_num, space_num))

    # to save the moment rate function
    global start_step
    potrate = np.array(outfile['slip_number']) * dx*dy * Veq

    ## generate output files

    global time 
    time = 0
    for i in range(step_record):
        dt_next = times[i] - time
        jj, kk = int(outfile['jj'].iloc[i]), int(outfile['kk'].iloc[i])
        new_state = outfile['index'].iloc[i]
        number_slip = outfile['slip_number'].iloc[i]
        number_nucleation = outfile['nucleation_number'].iloc[i]

        if (number_slip == 0 and number_nucleation == 0 and i > 0) or i == step_record - 1:
            iev += 1
            start_step = i
            start_time = time + dt_next
            event_np[iev, 0] = start_time
            

            if iev > 0:
                event_np[iev-1, 1] = event_potency * dx*dy * Veq * G * 1e6
                duration = time - event_np[iev-1, 0]
                event_np[iev-1, 2] = duration
                event_np[iev-1, 3] = np.sum(event_index)
                event_np[iev-1, 4] = np.where(event_index == 1)[1].max()
            
            if event_potency > potency_bar:
                np.save(folder + f'events/history_X_{iev-1}.npy', prop_profile[int(duration/time_res) + 1, :])
                np.save(folder + f'events/potency_{iev-1}.npy', [times[start_step:i+1], potrate[start_step:i+1]])
            
            prop_profile.fill(0)
            event_potency = 0.0
            event_index.fill(0)
            if not np.all(slip_x == 0):
                print('Wrong')

        if new_state == 2:
            event_index[jj, kk] = 1
        
        if number_slip != 0:
            time_minute_0 = int(np.floor((time - start_time)/time_res))
            time_minute_1 = int(np.floor((time + dt_next - start_time)/time_res))
            if time_minute_0 == time_minute_1:
                prop_profile[time_minute_0, :] += slip_x*dt_next
            else:
                prop_profile[time_minute_0, :] += slip_x*(time_minute_1*time_res - (time - start_time))
                prop_profile[time_minute_1, :] += slip_x*(time + dt_next - start_time - time_minute_1*time_res)

        
        if new_state == 2:
            slip_x[int(kk*dx/space_res)] -= 1
        elif new_state == 0:
            slip_x[int(kk*dx/space_res)] += 1

        event_potency += number_slip * dt_next * dx*dy * Veq
        time += dt_next
    
    print(f'Done! {iev} events in total')
    
    ## save the event catlog
    events = pd.DataFrame(event_np[:iev], columns=['start_time', 'total_moment', 'duration', 'area', 'max_x'])
    events.reset_index(inplace=True)
    events.to_csv(folder + 'events.csv', index=False)