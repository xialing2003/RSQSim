# This is a direct translation from RSQSim.m without any impovements in running speeds

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os

istepmax = 2_000_000
nel = 300
Nvs = 50

# =========================
# Parameters
# =========================

aob = 0.8 * np.ones(nel)
aob[:Nvs] = 2.0

omaob = 1 - aob
aobm1 = aob - 1
apob = 0.1 * aob

LboDx = 1
Veq = 1e3
Vbg = 0.5
DxoCs = 5e-5
DlnV = 1.0

# Load stiffness kernel 
Kijfile = np.loadtxt("Kij_50_1")
Kij_raw = Kijfile[:, 1] 
Kij = Kij_raw * LboDx
Kii = Kij[0]
omKii = 1 - Kii

DtauRD = Veq * DxoCs * LboDx
RefCo = 0.1
Dtauref1 = RefCo * DtauRD

Dtaupmin = 0.01
overshoot = 0.1

# =========================
# State Variable initiation
# =========================

indx = np.zeros(nel, dtype=int)
indx[:Nvs] = -1

Dtau = aobm1 * np.log(Vbg)

tautarget = np.zeros((Nvs, 2))
tautarget[:, 0] = Dtau[:Nvs] + DlnV * aobm1[:Nvs]
tautarget[:, 1] = Dtau[:Nvs] - DlnV * aobm1[:Nvs]

Dtaup = np.zeros(nel)

qbg = 1e1 / Veq
q = qbg * np.ones(nel)
q[:Nvs] = 1 / Veq

slip = np.zeros(nel)

taudotbg = (-0.5 * Kii) / 50
taudot = taudotbg * np.ones(nel)
V0 = Vbg * np.ones(nel) ###
for i in range(nel):
    for j in range(Nvs):
        taudot[i] += V0[j] * Kij[abs(i-j)]

Dt = np.ones(nel) * 9999
timistep = np.zeros(istepmax)
a2use = aob.copy()

# =========================
# Save and plot
# =========================

hist2 = np.zeros((int(istepmax*nel/5), 2))
hist1 = np.zeros((int(istepmax*nel/5), 2))
tot2 = 0
tot1 = 0
tim = 0.0
mrate = np.zeros((istepmax, 3))
ifront = -1
iplot = 0
Dtau_plot = np.zeros((500, nel))
slip_plot = np.zeros((500, nel))
slip_time = np.zeros(500)

# =========================
# Time stepping loop
# =========================

Thold = []
istep = 0

start_time = time.time()

while istep < istepmax:
    istep += 1

    notthis2_set = {int(row[1]) for row in Thold if int(row[2]) == 2}
    yesthis0_set = {int(row[1]) for row in Thold if int(row[2]) == 0}

    ## compute Dt for each element
    for i in range(Nvs, nel):
        
        if indx[i] == 0:
            Dt_i = ((omaob[i]*np.log(Veq*q[i])) - Dtau[i]) / taudot[i]
            if Dt_i > 0:
                Dttest = 0
                while (Dt_i - Dttest) > 1e-5 * Dt_i:
                    Dttest = Dt_i
                    Dt_i = ((omaob[i]*np.log(Veq*(q[i]+Dttest))) - Dtau[i]) / taudot[i]
            else:
                Dt_i = 0

            Dt[i]= Dt_i
        
        elif indx[i] == 1:
            use_aprime = False
            if i == nel - 1:
                if (indx[i-1] == 2 and (i-1) not in notthis2_set) or \
                   (indx[i-1] == 0 and (i-1) in yesthis0_set):
                    use_aprime = True
            else:
                if (indx[i-1] == 2 and (i-1) not in notthis2_set) or \
                   (indx[i-1] == 0 and (i-1) in yesthis0_set) or \
                   (indx[i+1] == 2 and (i+1) not in notthis2_set) or \
                   (indx[i+1] == 0 and (i+1) in yesthis0_set):
                    use_aprime = True
 
            a2use[i] = apob[i] if use_aprime else aob[i]
 
            Dt[i] = -(a2use[i] / taudot[i]) * np.log(
                ((1.0 / Veq) + omKii / taudot[i]) /
                ((1.0 / V0[i]) + omKii / taudot[i])
            )

        elif indx[i] == 2:
            Dt[i] = (-Dtaup[i] - Dtau[i]) / taudot[i]
            if Dt[i] < 0:
                Dt[i] = 0

    Dt[:Nvs] = 9e4

    ## determine next event: state switch or delay stress transfer

    iDtswitchmin = int(np.argmin(Dt))
    Dtswitchmin = Dt[iDtswitchmin]

    if len(Thold) == 0:
        nextT = 'switch'
        Dtmin = Dtswitchmin
    elif Dtswitchmin <= Thold[0][0] - tim:
        nextT = 'switch'
        Dtmin = Dtswitchmin
    else:
        nextT = 'reverse'
        Dtmin = Thold[0][0] - tim

    tim += Dtmin
    timistep[istep-1] = tim

    ## apply the state switch (if nextT == 'switch')

    if nextT == 'switch':

        ichange = iDtswitchmin

        if indx[ichange] == 0:
            indx[ichange] = 1
        elif indx[ichange] == 1:
            indx[ichange] = 2
            Thold.append([tim + DxoCs, ichange, 2])
            Thold.sort(key=lambda r: r[0])
        elif indx[ichange] == 2:
            indx[ichange] = 0
            Thold.append([tim + DxoCs, ichange, 0])
            Thold.sort(key=lambda r: r[0])

    ## update state variable
    for i in range(nel):
        Dtau[i] += Dtmin * taudot[i]

        if indx[i] == 0:
            if nextT == 'switch' and i == ichange:
                q[i] = 1/Veq
                slip[i] += Veq * Dtmin
            else:
                q[i] += Dtmin
        
        elif indx[i] == 1:
            if nextT == 'switch' and i == ichange:
                V0[i] = 1/(q[i] + Dtmin)
            else:
                V0m1 = (1./V0[i] + omKii/taudot[i]) * \
                        np.exp(-taudot[i]*Dtmin/a2use[i]) - omKii/taudot[i]
                V0[i] = 1/V0m1
        
        elif indx[i] == 2:
            if nextT == 'switch' and i == ichange:
                Dtaup[i] = max(Dtaupmin, overshoot*Dtau[i]) # I think there may be an issue here. Dtau[i] is literally -Daup[i]
            else:
                slip[i] += Veq * Dtmin
            q[i] = 1/Veq
        
        elif indx[i] == -1:
            slip[i] += V0[i] * Dtmin
            if i == ichange: # which should not apply to this version
                V0prior = V0[i]
                V0[i] = np.exp(np.log(V0prior) + np.sign(taudot[i])*DlnV)
                tautarget[i, :] = [Dtau[i] + DlnV*aobm1[i], Dtau[i] - DlnV*aobm1[i]]

    ## Update stressing rates for the next time step
    if nextT == 'switch':

        if indx[ichange] == 2:
            taudot[ichange] += Veq*Kij[0]

        elif indx[ichange] == 0:
            taudot[ichange] -= Veq*Kij[0]

        elif indx[ichange] == -1: ## current code doesn't satisfy this
            taudot[ichange] += (V0[ichange] - V0prior)*Kij[0]
    else: 
        # nextT == 'reverse' -- apply the delayed stress transfer

        row = Thold.pop(0)
        ipriorchange = row[1]
        priorindx = row[2]

        for i in range(nel):

            if i == ipriorchange:
                # reflection on the element that originally switched
                if priorindx == 2:
                    Dtau[i] -= Dtauref1

                elif priorindx == 0:
                    Dtau[i] += Dtauref1
            
            else:
                # far-field stress transfer to all other elements
                # I think a basic assumption here is that all the other elements 
                # feel the state transfer at the same time
                if priorindx == 2:
                    taudot[i] += Veq*Kij[abs(i-ipriorchange)]

                elif priorindx == 0:
                    taudot[i] -= Veq*Kij[abs(i-ipriorchange)]

                elif priorindx == -1: ## current code doesn't satisfy this
                    taudot[i] += (V0[ipriorchange] - V0prior)*Kij[abs(i-ipriorchange)]

            if indx[i] == 2:
                ifront = i

    ## record history

    if ifront % 50 == 0 and iplot < 500:
        iplot += 1
        Dtau_plot[iplot] = Dtau
        slip_plot[iplot] = slip
        slip_time[iplot] = tim

    aa = indx.copy()
    aa[indx == 1] = 0
    aa[indx == -1] = 0
    loc2 = np.where(aa)[0]

    bb = indx.copy()
    bb[indx == 2] = 0
    bb[indx == -1] = 0
    loc1 = np.where(bb)[0]

    for loc in loc2:
        hist2.append((loc, tim))
    for loc in loc1:
        hist1.append((loc, tim))

    mrate[istep-1, :] = [tim, len(loc2), ichange]

    ## Debug / pause on negative Dtmin
    if Dtmin < 0:
        print(f"WARNING: Dtmin={Dtmin:.4e} at step {istep}, element {ichange}, "
              f"state {indx[ichange]}")
        break
 
print(f"Simulation finished in {time.time() - start_time:.1f} s  ({istep} steps)")

# =========================
# Post processing
# =========================

folder = '../results/direct_test1/'
if not os.path.exists(folder):
        os.makedirs(folder)

Dtau_plot = Dtau_plot[:iplot]
slip_plot = slip_plot[:iplot]
slip_time = slip_time[:iplot]

plt.figure()
plt.subplot(2, 1, 1)
for i in range(iplot):
    plt.plot(np.range(nel), Dtau_plot[i])
    plt.ylabel(r'$\Delta\tau$ (d''less)')
    plt.xlabel('Position (d''less)')
plt.subplot(2, 1, 2)
for i in range(iplot):
    plt.plot(np.range(nel), slip_plot[i])
    plt.ylabel('Slip (d''less)')
    plt.xlabel('Position (d''less)')

plt.savefig(folder + 'stress_slip.png')

hist2 = np.array(hist2) if hist2 else np.empty((0, 2))
hist1 = np.array(hist1) if hist1 else np.empty((0, 2))

plt.figure()
if hist2.size:
    plt.plot(hist2[:, 0], hist2[:, 1], 'rs', markersize=1, markerfacecolor='r')
if hist1.size:
    plt.plot(hist1[:, 0], hist1[:, 1], 'ks', markersize=1, markerfacecolor='k')
plt.xlabel("Position (d'less)")
plt.ylabel("Time (d'less)")
plt.savefig(folder + 'hist.png')

mrate_csv = pd.DataFrame(mrate, columns=['time', 'len2', ichange])
mrate_csv.to_csv(folder + 'mrate.csv', index=False)
np.savez(folder + 'hist.npz', hist1=hist1, hist2=hist2)
np.savez(folder + 'slip_plot.npz', slip_time = slip_time, slip_plot = slip_plot, Dtau_plot=Dtau_plot)