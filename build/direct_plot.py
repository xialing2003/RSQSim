# this file reads the output file and slip history file and plots them

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json

from matplotlib.animation import FuncAnimation
from matplotlib.colors import ListedColormap, BoundaryNorm

if __name__ == "__main__":

    folder = "../results/RSQSim_stage1/test1/"
    params = json.load(open(folder + 'parameters.json'))
    param_r, param_e, param_m = params['region'], params['elastic'], params['model']
    my, nx, dx, dy = param_r['my'], param_r['nx'], param_r['dx'], param_r['dy']
    Dc, Vpl = param_e['D_c'], param_e['V_pl']

    istep_record = param_m['step_record']

    out_file = pd.read_csv(folder + "out_file.csv")
    times = pd.read_csv(folder + "times.csv")

    with np.load(folder + 'slip_plot.npz') as data:
        slip_time = data['slip_time']
        slip_plot = data['slip_plot']
        stress_plot = data['stress_plot']
    iplot = len(slip_time)

    # plot slip plots
    plt.figure()
    for i in range(0, iplot, 10):
        plt.plot(np.arange(nx)*dx/1000, slip_plot[i, :]*Dc*100)
    plt.xlabel('X (km)')
    plt.ylabel('slip distance (cm)')
    plt.tight_layout()
    plt.savefig(folder + 'slip.png', dpi=300)
    print('slip profile saved')

    # plot stress plots
    plt.figure()
    for i in range(0, iplot, 10):
        plt.plot(np.arange(nx)*dx/1000, stress_plot[i, :])
    plt.xlabel('X (km)')
    plt.ylabel('Stress')
    plt.tight_layout()
    plt.savefig(folder + 'stress.png', dpi=300)
    print('stress profile saved')

    # plot moment rate figures
    plt.subplots(2, 2, figsize=(10, 8))
    ax = plt.subplot(2, 2, 1)
    ax.plot(times['time'], out_file['slip_number'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of slip elements')
    ax = plt.subplot(2, 2, 2)
    ax.plot(range(0, len(out_file)), out_file['slip_number'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Number of slip elements')
    ax = plt.subplot(2, 2, 3)
    ax.plot(times['time'], out_file['nucleation_number'])
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of nucleation elements')
    ax = plt.subplot(2, 2, 4)
    ax.plot(range(0, len(out_file)), out_file['nucleation_number'])
    ax.set_xlabel('Step')
    ax.set_ylabel('Number of nucleation elements')
    plt.tight_layout()
    plt.savefig(folder + 'slip_nucleation.png', dpi=300)
    print('slip and nucleation plot saved')

    # plot the animation of events
    plot_animation = False
    if plot_animation:
        # Define your colormap and normalization
        cmap = ListedColormap(["white", "#ecb78d", "red"])
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5], cmap.N)

        # Your timestep range
        timestep_range = np.arange(11000, 12000, 1)  # Adjust based on your timestep range
        ani_time = slip_time[timestep_range]
        ani_stress = stress_plot[timestep_range]
        ani_slip = slip_plot[timestep_range]
        ani_state = np.zeros((len(timestep_range), my, nx), dtype=np.int8)
        state_plot = np.zeros((my, nx), dtype=np.int8)
        i_timestep = 0
        for i in range(iplot):
            j = int(out_file['jj'].iloc[i])
            k = int(out_file['kk'].iloc[i])
            if out_file['index'].iloc[i] == 2:
                state_plot[j, k] = 2
            elif out_file['index'].iloc[i] == 1:
                state_plot[j, k] = 1
            else:
                state_plot[j, k] = 0

            if i == timestep_range[i_timestep]:
                ani_state[i_timestep] = state_plot
                i_timestep += 1
                if i_timestep == len(timestep_range):
                    break


        # Create a figure and axes for the plots
        fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True, gridspec_kw={'height_ratios': [1, 1, 2]})

        # Plot elements
        stress_line, = axs[0].step([], [], where='post', c="tab:blue")
        slip_line, = axs[1].step([], [], where='post', c="tab:green")
        state_imshow = axs[2].imshow(np.zeros((my, nx)), aspect='auto', cmap=cmap, norm=norm, origin="lower", extent=[0, nx*dx/1000, 0, my*dy/1000])
        title = fig.suptitle(f"Time = {slip_time[1]*Dc/Vpl/3600:.2f} h", fontsize=12)

        # Adjust axes labels
        axs[0].set_xlim(0, 6)
        axs[0].set_ylim(-2, 6)
        axs[0].set_ylabel('Normalized stress ' + r'$\tau / \tau_p$')
        axs[1].set_ylabel('Slip distance (cm)')
        axs[1].set_ylim(0, 800)
        axs[2].set_xlabel('Location along strike (km)')
        axs[2].set_ylabel('Location along dip (km)')

        plt.tight_layout()

        # Update function to modify plot elements at each timestep
        def update(timestep):
            # Update stress plot
            stress_line.set_data(np.arange(nx)*dx/1000, ani_stress[timestep])
            
            # Update slip plot
            slip_line.set_data(np.arange(nx)*dx/1000, ani_slip[timestep] * 100 * Dc)
            
            # Update state plot
            state_imshow.set_data(ani_state[timestep])

            # Update title
            title.set_text(f"Time = {(ani_time[timestep]*Dc/Vpl)/3600:.2f} h")
            
            return stress_line, slip_line, state_imshow

        # Create the animation
        ani = FuncAnimation(fig, update, frames=range(0, len(timestep_range)), interval=250, blit=True)

        # To save the animation as a video (optional)
        ani.save(folder + 'animation.mp4', writer='ffmpeg', fps=20)