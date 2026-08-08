import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_m(folder, iev):
    potency = np.load(folder + f'number_{iev}.npy')

    unit = 24 * 60 * 60
    time = potency[0, :] / unit
    number_slip = potency[1, :]
    number_nuc = potency[2, :]
    pot_nuc = potency[3, :]

    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(time, number_slip, label='slip')
    plt.plot(time, number_nuc, label='nucleation')
    plt.legend()
    plt.ylabel('Number')

    plt.subplot(3, 1, 2)
    plt.plot(time, number_slip*1000*1e-9, label='slip')
    plt.plot(time, pot_nuc*1e-9, label='nucleation')
    plt.plot([time[0], time[-1]], [0, 0], '--', c='tab:red')
    plt.ylabel('velocity sum (m/s)')

    plt.subplot(3, 1, 3)
    plt.plot(time, number_slip*1000/pot_nuc)
    plt.plot([time[0], time[-1]], [1, 1], '--', c='tab:red')
    plt.text(time[-1]*0.75, 9, f'max ratio: {max(number_slip*1000/pot_nuc):.1f}', c='k')
    plt.ylim([0, 10])
    plt.ylabel('Ratio between slip and nuc')
    plt.xlabel('Time (days)')
    plt.tight_layout()

    plt.savefig(folder + f'number_{iev}.jpg', dpi = 300)

folder = '../results/RSQSim_stage3/test_v3_2_0.5/events/'

event_ids = sorted(
        int(f.stem.split("_")[-1])
        for f in Path(folder).glob("number_*.npy")
    )
print(event_ids)
for iev in event_ids:
    plot_m(folder, iev)