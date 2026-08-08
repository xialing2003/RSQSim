import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

folder = '../results/RSQSim_stage3/test_3_Dc5um/'
data = np.load(folder + 'nucleation_data.npz')

nuc_indx = data['element_index']
nuc_stress = data['stress']
nuc_velocity = data['velocity']


fig, axes = plt.subplots(1, 2, figsize=(14, 6))


# ----------------------
# Left: scatter plot
# ----------------------
ax = axes[0]

for i in range(5):
    mask = nuc_indx == i

    ax.scatter(
        nuc_velocity[mask],
        nuc_stress[mask],
        s=3,
        label=f'Element {i}',
        alpha=0.3
    )

ax.set_xscale('log')
ax.set_xlabel('Velocity')
ax.set_ylabel('Stress')
ax.set_title('Stress-Velocity Scatter')
ax.legend()


# ----------------------
# Right: density plot
# ----------------------
ax = axes[1]

mask = nuc_velocity > 0   # avoid log problem

h = ax.hist2d(
    np.log10(nuc_velocity[mask]),
    nuc_stress[mask],
    bins=300,
    density=True,
    cmap='Blues',
    norm=LogNorm()
)

fig.colorbar(h[3], ax=ax, label='Density (log scale)')

ax.set_xlabel(r'$\log_{10}$(Velocity)')
ax.set_ylabel('Stress')
ax.set_title('Stress-Velocity Density')


plt.tight_layout()
plt.savefig(folder + 'nucleation.png', dpi=300)