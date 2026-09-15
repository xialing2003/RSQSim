import numpy as np
import matplotlib.pyplot as plt
from cmcrameri import cm 

def cal_speed(new_list_t, new_list_x, folder):

    t_min = np.min(new_list_t)
    t_max = np.max(new_list_t)

    t1_target = t_min + (t_max - t_min) / 3
    t2_target = t_min + 2 * (t_max - t_min) / 3

    t1 = new_list_t[np.argmin(np.abs(new_list_t - t1_target))]
    t2 = new_list_t[np.argmin(np.abs(new_list_t - t2_target))]

    x1 = np.max(new_list_x[new_list_t == t1])
    x2 = np.max(new_list_x[new_list_t == t2])

    Vprop_simu = (x2 - x1) / (t2 - t1)

    return Vprop_simu, Vprop_simu

def plot_event(ax_main, list_t, list_x, list_n, set_axis, axis_range, arrow):

    print('maximum activity', np.max(list_n))
    if set_axis:
        start_x, end_x = axis_range[0], axis_range[1]
        start_y, end_y = axis_range[2], axis_range[3]
    else:
        start_x, end_x = np.min(list_t), np.max(list_t)
        start_y, end_y = np.min(list_x)-2, np.max(list_x)+2
    vmax = min(1.0, np.max(list_n))

    cmap = cm.batlow
    sc = ax_main.scatter(list_t, list_x, cmap=cmap, vmin=0, vmax=vmax, c=list_n, s=0.05, rasterized=True)
    ax_main.xaxis.set_visible(False)
    ax_main.set_xlim(start_x, end_x)
    ax_main.set_ylim(start_y, end_y)
    ax_main.spines['bottom'].set_visible(False)
    ax_main.spines['top'].set_visible(False)
    ax_main.spines['right'].set_visible(False)

    ax_main.set_ylabel('Strike distance (km)', fontsize=11)

    if arrow:
        Vprop_est, Vprop_simu = cal_speed(list_t, list_x)

        start_day = 2
        start_x = 60
        ax_main.annotate('', 
            xy=(start_day+1, start_x+Vprop_simu),
            xytext=(start_day, start_x), 
            arrowprops=dict(arrowstyle='-|>', color='tab:blue', lw=2, mutation_scale=15)
        )
        ax_main.text(start_day+1, start_x+Vprop_simu, f'{Vprop_simu} km/d', color='tab:blue', ha='left', va='center')
        ax_main.annotate('', 
            xy=(start_day+1, start_x+10+Vprop_est),
            xytext=(start_day, start_x+10), 
            arrowprops=dict(arrowstyle='-|>', color='tab:green', lw=2, linestyle='--', mutation_scale=15)
        )
        ax_main.text(start_day+1, start_x+10+Vprop_simu, f'{Vprop_est:.1f} km/d', color='tab:green', ha='left', va='center')   

    return sc

def plot_moment(ax_moment, pottime, potrate, pot_coe, flag_nuc, potrate_nuc, nuc_coe, set_x, axis_x):
    if set_x:
        start_x, end_x = axis_x[0], axis_x[1]
    else:
        start_x, end_x = 0, np.max(pottime)

    mask = (pottime >= start_x) & (pottime <= end_x)
    pottime = pottime[mask]
    potrate = potrate[mask]
    potrate_nuc = potrate_nuc[mask]

    ene_level = int(np.log10(pot_coe) + np.log10(max(potrate)))
    pot_coe /= 10**ene_level
    nuc_coe /= 10**ene_level

    ax_moment.plot(pottime, potrate*pot_coe, color='black', linewidth=0.5)
    if flag_nuc:
        ax_moment.plot(pottime, potrate_nuc*nuc_coe, color='blue', linewidth=0.5)
    ax_moment.set_xlim(start_x, end_x)
    y_max = np.ceil(np.max(potrate*pot_coe))
    ax_moment.set_ylim(0, y_max)
    ax_moment.spines['top'].set_visible(False)
    ax_moment.spines['right'].set_visible(False)

    ax_moment.set_yticks([0, (y_max*10//3)*0.1, (y_max*10//3)*0.2])
    ax_moment.set_ylabel(r'$\dot{M_0}$ $(Nm/s)$')
    ax_moment.text(start_x, y_max/3*2, fr'$10^{{{ene_level}}}$', fontsize=10, ha='left', va='bottom')
    ax_moment.ticklabel_format(useOffset=False)
    ax_moment.set_xlabel('Time (days)', fontsize=11)