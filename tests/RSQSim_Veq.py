import time
import numpy as np
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================

istepmax = 1_000_00
# a/b
aob = 0.8
# 1 - a/b
omaob = 1 - aob
# a'/b
apob = 0.4
Dtaubg = 6
overshoot = 0.1
Dtaupinit = overshoot * Dtaubg
Dtaupmin = 0.01
LboDx = 1
Veq = 1e3
qbg = 3e15 / Veq
taudotbg = 0.0
nel = 356

# =========================
# Load elastic kernel
# =========================

Kijfile = np.loadtxt("../results/Kij_32_1")
Kij = Kijfile[:, 1] * LboDx
Kii = Kij[0]
omKii = 1.0 - Kii

# =========================
# Initialization
# =========================

indices = np.arange(nel)
indx = np.zeros(nel, dtype=int)
Dtau = np.ones(nel) * Dtaubg
Dtaup = np.ones(nel) * Dtaupinit
taudot = np.ones(nel) * taudotbg
q = np.ones(nel) * qbg
slip = np.zeros(nel)
Dt = np.ones(nel) * 9999
rate = np.ones(nel) * aob
V0 = np.ones(nel)

timistep = np.zeros(istepmax)
ifront = -1
hist2 = []
hist1 = []

m = 50
indx[:m] = 2

# initial stressing rate
for i in range(nel):
    for j in range(m):
        taudot[i] += Veq * Kij[abs(i-j)]

tim = 0.0
istep = 0
stopped = False

# snapshots plotting
Dtau_snapshots = []
slip_snapshots = []
snapshot_fronts = []
snapshot_times = []


# =========================
# Time stepping loop
# =========================

start_time = time.time()

while istep < istepmax and not stopped:
    istep += 1

    if np.max(indx) < 2:
        stopped = True

    # -------- compute Dt --------
    # mask0 = indx == 0
    # mask1 = indx == 1
    # mask2 = indx == 2

    # # transition from 2 to 0
    # Dt[mask2] = (-Dtaup[mask2] - Dtau[mask2]) / taudot[mask2]
    
    # # transition from 1 to 2
    # neighbour_left = np.roll(indx, 1)
    # neighbour_right = np.roll(indx, -1)
    # mask_neighbour2 = (neighbour_left == 2) | (neighbour_right == 2)

    # rate[:] = aob
    # rate[mask_neighbour2] = apob
    # rate[0] = rate[-1] = aob

    # Dt[mask1] = -(rate[mask1] / taudot[mask1]) * np.log(
    #             ((1.0 / Veq) + omKii / taudot[mask1]) / 
    #             ((1.0) / V0[mask1] + omKii / taudot[mask1])
    #         )
    # # transition from 0 to 1 
    # for i in np.where(indx==0)[0]:
    #     Dttest = 0.0
    #     Dt[i] = ((omaob * np.log(Veq * q[i])) - Dtau[i]) / taudot[i]
    #     while abs(Dt[i] - Dttest) > 1e-5 * abs(Dt[i]):
    #         Dttest = Dt[i]
    #         Dt[i] = ((omaob * np.log(Veq * (q[i] + Dttest))) - Dtau[i]) / taudot[i]

    # -------- compute Dt --------
    
    for i in range(nel):

        if indx[i] == 0: # Dt01
            Dttest = 0.0
            Dt[i] = ((omaob * np.log(Veq * q[i])) - Dtau[i]) / taudot[i]
            while abs(Dt[i] - Dttest) > 1e-5 * abs(Dt[i]):
                Dttest = Dt[i]
                Dt[i] = ((omaob * np.log(Veq * (q[i] + Dttest))) - Dtau[i]) / taudot[i]
        
        elif indx[i] == 1:

            if i == 0 or i == nel - 1:
                rate[i] = aob
                if i == nel - 1:
                    stopped = True
            else:
                if indx[i-1] == 2 or indx[i+1] == 2:
                    rate[i] = apob
                else:
                    rate[i] = aob
            
            Dt[i] = -(rate[i] / taudot[i]) * np.log(
                ((1.0 / Veq) + omKii / taudot[i]) / 
                ((1.0) / V0[i] + omKii / taudot[i])
            )

        else: 
            Dt[i] = (-Dtaup[i] - Dtau[i]) / taudot[i]

    Dtmin = np.min(Dt)
    ichange = np.argmin(Dt)

    tim += Dtmin
    timistep[istep - 1] = tim

    # -------- state switching --------

    if indx[ichange] == 0:
        indx[ichange] = 1
        ico = 0
    
    elif indx[ichange] == 1:
        indx[ichange] = 2
        ico = 1
        if ichange == nel - 1:
            stopped = True
        
    else:
        indx[ichange] = 0
        ico = -1

    # -------- update all the elements --------

    for i in range(nel):

        Dtau[i] += Dtmin * taudot[i]

        if indx[i] == 0:
            if i == ichange:
                q[i] = 1 / Veq
                slip[i] += Veq * Dtmin
            else:
                q[i] += Dtmin
            
        elif indx[i] == 1:
            if i == ichange:
                V0[i] = 1.0 / (q[i] + Dtmin)
            else:
                V0m1 = (
                    (1.0 / V0[i] + omKii / taudot[i])
                    * np.exp(-taudot[i] * Dtmin / rate[i])
                    - omKii / taudot[i]
                )
                V0[i] = 1.0 / V0m1
            
        else:
            if i == ichange:
                Dtaup[i] = max(Dtaupmin, overshoot * Dtau[i])
            else:
                slip[i] += Veq * Dtmin
            q[i] = 1.0 / Veq

        # taudot[i] += ico * Veq * Kij[abs(i - ichange)]

        if indx[i] == 2:
            ifront = i

    taudot += ico * Veq * Kij[(indices - ichange)]

    # -------- record --------

    for i in range(nel):
        if indx[i] == 2:
            hist2.append((i, tim))
        if indx[i] == 1:
            hist1.append((i, tim))

    if ifront >= 0:
        if ifront % 50 == 0:
            Dtau_snapshots.append(Dtau.copy())
            slip_snapshots.append(slip.copy())
            snapshot_fronts.append(ifront)
            snapshot_times.append(tim)

    if Dtmin < 0:
        print("Negative Dt:", istep, ichange, Dtmin)

end_time = time.time()
print(f"Total computation time: {end_time - start_time:.2f} seconds")

# =========================
# Plot
# =========================

hist2 = np.array(hist2)
hist1 = np.array(hist1)

plt.figure(figsize=(8, 10))

plt.subplot(2, 1, 2)
plt.plot(hist2[:, 0], hist2[:, 1], 'r.', markersize=1)
plt.plot(hist1[:, 0], hist1[:, 1], 'k.', markersize=1)
plt.xlabel("Position")
plt.ylabel("Time")

plt.savefig('../results/hist.jpg')

plt.figure(figsize=(8,6))
for k in range(len(Dtau_snapshots)):
    plt.plot(Dtau_snapshots[k], alpha = 0.5)

plt.xlabel("Position")
plt.ylabel('Delta tau')
plt.title("Stress evolution snapshots")
plt.savefig("../results/stress.jpg")

plt.figure(figsize=(8,6))
for k in range(len(slip_snapshots)):
    plt.plot(slip_snapshots[k], alpha = 0.5)

plt.xlabel("Position")
plt.ylabel('Delta tau')
plt.title("Slip evolution snapshots")
plt.savefig("../results/slip.jpg")
