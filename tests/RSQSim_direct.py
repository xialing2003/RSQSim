# This is a direct translation from RSQSim.m without any impovements in running speeds

import numpy as np
import matplotlib.pyplot as plt
import time

istepmax = 2_000_000
Kijfile = np.loadtxt("Kij_50_1")
nel = 300

# =========================
# Parameters
# =========================

aob = 0.8 * np.ones(nel)
Nvs = 50
aob[:Nvs] = 2.0

omaob = 1 - aob
aobm1 = aob - 1
apob = 0.1 * aob

LboDx = 1
Veq = 1e3
Vbg = 0.5
DxoCs = 5e-5
DlnV = 1

Kij = Kijfile[:, 1] * LboDx
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
V0 = Vbg * np.ones(nel)
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
Thold = []
istep = 0

# =========================
# Time stepping loop
# =========================

start_time = time.time()

while istep < istepmax:
    istep += 1

    for i in range(Nvs, nel):
        
        if indx[i] == 0:
            Dt_i = ((omaob[i]*np.log(Veq*q[i])) - Dtau[i]) / taudot[i]
            Dttest = 0

            while (Dt_i - Dttest) > 1e-5 * Dt_i:
                Dttest = Dt_i
                Dt_i = ((omaob[i]*np.log(Veq*(q[i]+Dttest))) - Dtau[i]) / taudot[i]

            if Dt_i < 0:
                Dt_i = 0
            Dt[i]= Dt_i
        
        elif indx[i] == 1:
            Dt[i] = -(a2use[i]/taudot[i])*np.log(
                (1./Veq + omKii/taudot[i]) /
                (1./V0[i] + omKii/taudot[i])
            )
        
        elif indx[i] == 2:
            Dt_i = (-Dtaup[i] - Dtau[i]) / taudot[i]
            if Dt_i < 0:
                Dt_i = 0
            Dt[i] = Dt_i

    Dt[:Nvs] = 9e4

    iDtswitchmin = np.argmin(Dt)
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

    if nextT == 'switch':

        ichange = iDtswitchmin

        if indx[ichange] == 0:
            indx[ichange] = 1
        elif indx[ichange] == 1:
            indx[ichange] = 2
            Thold.append([tim + DxoCs, ichange, 2])
        elif indx[ichange] == 2:
            indx[ichange] = 0
            Thold.append([tim + DxoCs, ichange, 0])

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
                        np.exp(-taudot[i]*Dtmin/a2use[i]) - \
                        omKii/taudot[i]
                V0[i] = 1/V0m1
        
        elif indx[i] == 2:
            if nextT == 'switch' and i == ichange:
                Dtaup[i] = max(Dtaupmin, overshoot*Dtau[i])
            else:
                slip[i] += Veq * Dtmin
            q[i] = 1/Veq
        
        elif indx[i] == -1:
            slip[i] += V0[i] * Dtmin

    if nextT == 'switch':

        if indx[ichange] == 2:
            taudot[ichange] += Veq*Kij[0]

        elif indx[ichange] == 0:
            taudot[ichange] -= Veq*Kij[0]
    else:

        ipriorchange = Thold[0][1]
        priorindx = Thold[0][2]
        Thold.pop(0)

        for i in range(nel):

            if i == ipriorchange:

                if priorindx == 2:
                    Dtau[i] -= Dtauref1

                elif priorindx == 0:
                    Dtau[i] += Dtauref1
            
            else:
                
                if priorindx == 2:
                    taudot[i] += Veq*Kij[abs(i-ipriorchange)]

                elif priorindx == 0:
                    taudot[i] -= Veq*Kij[abs(i-ipriorchange)]

            if indx[i] == 2:
                ifront = i

end_time = time.time()
print('runtime:', end_time - start_time)
