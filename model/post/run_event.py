import numpy as np
import json
import pandas as pd
import os

def run_event(folder):
    if not os.path.exists(folder + 'events/'):
        os.makedirs(folder + 'events/')

    ## load parameters and output file

    with open(folder + 'parameters.json') as f:
        params = json.load(f)

    param_r = params['region']
    param_e = params['elastic']
    param_m = params['model']

    Veq = param_e['V_eq']
    Vpl = param_e['V_pl']
    Dc = param_e['D_c']

    dx = param_r['dx']
    dy = param_r['dy']
    my = param_r['my']
    nx = param_r['nx']

    step_record = param_m['step_record']

    # outfile: time, jj(my), kk(nx), state(after transition), number_1
    outfile = pd.read_csv(folder + 'out_file.csv')

    jj_list = outfile['jj'].to_numpy(np.int32)
    kk_list = outfile['kk'].to_numpy(np.int32)
    new_state_list = outfile['index'].to_numpy(np.int8)
    number_slip_list = outfile['slip_number'].to_numpy(np.int32)

    record = pd.read_csv(folder + 'record.csv')
    times = record['time'].to_numpy() * Dc / Vpl
    nuc_vel_list = record['nuc_MR'].to_numpy(np.float64)

    ## set up the output arrays

    event_np = np.zeros((1000000, 7))
    # [0: start step,
    #  1: start time,
    #  2: end step,
    #  3: total moment,
    #  4: duration,
    #  5: area,
    #  6: max x]

    iev = -1
    event_potency = 0.0
    potency_factor = dx * dy * Veq

    event_index = np.zeros((my, nx), dtype=np.int64)
    event_area = 0
    event_max_x = -1
    flag_save = False

    ## generate events

    for i in range(step_record):

        dt_next = times[i] - times[i - 1] if i > 0 else times[i]

        jj = jj_list[i]
        kk = kk_list[i]
        new_state = new_state_list[i]
        number_slip = number_slip_list[i]

        if (number_slip == 1 and new_state == 0) and iev >= 0:
            flag_save = True

        if i == step_record - 1:
            flag_save = True

        if flag_save:
            event_np[iev, 2] = i
            event_np[iev, 3] = event_potency
            event_np[iev, 4] = times[i] - event_np[iev, 1]
            event_np[iev, 5] = event_area
            event_np[iev, 6] = event_max_x

            flag_save = False

        if number_slip == 0 and new_state == 2:

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

    ## save event catalog

    events = pd.DataFrame(
        event_np[:iev + 1],
        columns=[
            'start_step',
            'start_time',
            'end_step',
            'total_moment',
            'duration',
            'area',
            'max_x'
        ]
    )

    events.reset_index(inplace=True)

    output_file = folder + 'events.csv'
    events.to_csv(output_file, index=False)

    print(f'Done! {iev + 1} events in total')

    return events
