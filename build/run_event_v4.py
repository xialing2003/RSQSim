# totally based on run_event_v3.py, the difference is that I generate the event catalog first and then run the propagation profile
# this file is based on run_event_v2.py
# the major difference is that events is defined as the period from one element starts slipping no elements are slipping
# The purpose of this file is to generate the events figure based on the newly developed simulator_v2.py

import numpy as np
import json
import pandas as pd
import os

if __name__ == "__main__":

    folder = '../results/test_RSQSim/test_v2.2/'
    if not os.path.exists(folder + 'events_v3/'):
        os.makedirs(folder + 'events_v3/')

    ## load parameters and output file

    params = json.load(open(folder + 'parameters.json'))
    param_r, param_e, param_m = params['region'], params['elastic'], params['model']

    Veq, Vpl, Dc = param_e['V_eq'], param_e['V_pl'], param_e['D_c']
    dx, dy, my, nx, G = param_r['dx'], param_r['dy'], param_r['my'], param_r['nx'], param_r['G']
    step_record, time_res, space_res = param_m['step_record'], param_m['time_res'], param_m['space_res']

    # outfile: time, jj(my), kk(nx), state(after trnsition), number_1
    outfile = pd.read_csv(folder + 'out_file.csv')
    times = pd.read_csv(folder + "times.csv")['time']
    times *= Dc/Vpl

    ## set up the output files

    # to save the event catlog
    flag_save = False
    event_np = np.zeros((500000, 6)) # [0: start step, 1: start time, 2: total moment, 3: duration, 4: area, 5: max x]
    number_slip = 0
    iev = -1
    event_potency = 0.0
    potency_factor = dx*dy * Veq
    event_index = np.zeros((my, nx), dtype=np.int64) # to record the entire rupture area
    event_area = 0
    event_max_x = -1

    # to save the moment rate function
    jj_list = outfile['jj'].to_numpy(np.int32)
    kk_list = outfile['kk'].to_numpy(np.int32)
    new_state_list = outfile['index'].to_numpy(np.int8)
    number_slip_list = outfile['slip_number'].to_numpy(np.int32)

    space_index = (kk_list * dx / space_res).astype(np.int32)
    ## generate output files

    time = 0
    for i in range(step_record):
        dt_next = times[i] - times[i-1] if i > 0 else times[i]
        jj, kk, new_state = jj_list[i], kk_list[i], new_state_list[i]
        number_slip = number_slip_list[i]

        if (number_slip == 1 and new_state == 0) and iev >=0:
            flag_save = True

        if i == step_record - 1:
            flag_save = True
        
        if flag_save:
            event_np[iev, 2] = event_potency * potency_factor
            duration = times[i] - event_np[iev, 1]
            event_np[iev, 3] = duration
            event_np[iev, 4] = event_area
            event_np[iev, 5] = event_max_x
            
            flag_save = False

        if (number_slip == 0 and new_state == 2):
            iev += 1
            event_np[iev, 0] = i
            event_np[iev, 1] = times[i]
            
            event_potency = 0.0
            event_index.fill(0)
            event_max_x = -1
            event_area = 0

        if new_state == 2:
            if event_index[jj, kk] == 0: 
                event_index[jj, kk] = 1
                event_area += 1
                if kk > event_max_x:
                    event_max_x = kk

        event_potency += number_slip * dt_next * potency_factor
        time += dt_next
    
    ## save the event catlog
    events = pd.DataFrame(event_np[:iev+1], columns=['start_step', 'start_time', 'total_moment', 'duration', 'area', 'max_x'])
    events.reset_index(inplace=True)
    events.to_csv(folder + 'events_v3.csv', index=False)

    print(f'Done! {iev+1} events in total')


    potency_bar = 100
    save_mask = events['total_moment'] > potency_bar
    event_list = events[save_mask].index.to_numpy(np.int32)
    start_step_list = events['start_step'].to_numpy(np.int64)
    start_time_list = events['start_time'].to_numpy(np.int64)
    duration_list = events['duration'].to_numpy(np.float64)
    
    space_num = int(nx*dx/space_res) + 1
    unit_d2s = 24 * 60 * 60
    time_res = 60
    time_num = int(2000*unit_d2s/time_res) + 1
    # time_num = 2000000
    slip_x = np.zeros(space_num, dtype=np.int64)
    prop_profile = np.zeros((time_num, space_num), dtype=np.float32)

    for iev in event_list:
        start_step = start_step_list[iev]
        start_time = start_time_list[iev]
        flag_event_on = True

        i = start_step + 1
        slip_x[space_index[start_step]] += 1
        duration = duration_list[iev]
        prop_profile.fill(0)
        while flag_event_on and i < step_record:
            dt_next = times[i] - times[i-1]
            time = times[i-1]
            jj, kk, new_state = jj_list[i], kk_list[i], new_state_list[i]
            number_slip = number_slip_list[i]

            inv_time_res = 1.0 / time_res
            time_minute_0 = int((time-start_time)*inv_time_res)
            time_minute_1 = int((time+dt_next-start_time)*inv_time_res)
            if time_minute_0 == time_minute_1:
                prop_profile[time_minute_0, :] += slip_x*dt_next
            else:
                prop_profile[time_minute_0, :] += slip_x*(time_minute_1*time_res - (time - start_time))
                prop_profile[time_minute_1, :] += slip_x*(time + dt_next - start_time - time_minute_1*time_res)

            if new_state == 2:
                slip_x[space_index[i]] += 1
            elif new_state == 0:
                slip_x[space_index[i]] -= 1

            if (number_slip == 1 and new_state == 0):
                flag_event_on = False
            else:
                i += 1

        np.save(folder + f'events_v3/history_X_{iev}.npy', prop_profile[:int(duration/time_res) + 1, :])
        np.save(folder + f'events_v3/potency_{iev}.npy', [times[start_step:i+1] - times[start_step], number_slip_list[start_step:i+1]*dx*dy * Veq])
        print(f'Event {iev}: time units of', int(duration/time_res) + 1)