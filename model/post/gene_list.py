import numpy as np

def update_save(window_profile, current_time, list_plot, len_list):
    # list_plot: 0: t, 1: x, 2: n
    mask = window_profile > 0
    x = np.nonzero(mask)[0]
    n = window_profile[mask]

    len_new = len(x)
    list_plot[0, len_list:len_list+len_new] = current_time
    list_plot[1, len_list:len_list+len_new] = x
    list_plot[2, len_list:len_list+len_new] = n

    len_list += len_new

    return list_plot, len_list

def generate_list(time_res, space_num, len_step, times, new_state_list, space_index,
                        cutoff, window_start, window_end, unit):
    # unit_d2s = 60 * 60 * 24
    
    # time_num = int(200*unit_d2s/time_res) + 1
    slip_x = np.zeros(space_num, dtype=np.int64)
    current_profile = np.zeros((space_num), dtype=np.float32)
    list_plot = np.zeros((3, 30000000), dtype=np.float32)
    len_list = 0
    time_unit_0 = 0

    slip_x[space_index[0]] += 1
    i = 1
    while i < len_step:
        dt_next = times[i] - times[i-1]
        time = times[i-1]
        new_state = new_state_list[i]
        # number_slip = number_slip_list[i]

        time_unit_0 = int(time/time_res)
        time_unit_1 = int((time+dt_next)/time_res)

        if time_unit_0 == time_unit_1:
            # prop_profile[time_unit_0, :] += slip_x*dt_next
            current_profile[:] += slip_x*dt_next
        else:
            
            local_time = time
            while time_unit_0 < time_unit_1:
                # prop_profile[time_unit_0, :] += slip_x*((time_unit_0+1)*time_res - local_time)
                current_profile[:] += slip_x*((time_unit_0+1)*time_res - local_time)
                if cutoff:
                    if time_unit_0*time_res >= window_start*unit and time_unit_0*time_res <= window_end*unit:
                        list_plot, len_list = update_save(current_profile, time_unit_0, list_plot, len_list)
                else:
                    list_plot, len_list = update_save(current_profile, time_unit_0, list_plot, len_list)
                time_unit_0 += 1
                local_time = time_unit_0 * time_res
                current_profile.fill(0)
             
            # prop_profile[time_unit_1, :] += slip_x*(time + dt_next - start_time - time_unit_1*time_res)
            current_profile[:] += slip_x * (time + dt_next - time_unit_1*time_res)

        if new_state == 2:
            slip_x[space_index[i]] += 1
        elif new_state == 0:
            slip_x[space_index[i]] -= 1

        # if (number_slip == 1 and new_state == 0):
        #     flag_event_on = False
        # else:
        i += 1

    list_plot, len_list = update_save(current_profile, time_unit_0, list_plot, len_list)
    return list_plot[:, :len_list]
