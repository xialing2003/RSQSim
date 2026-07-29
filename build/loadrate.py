import numpy as np
from scipy.signal import fftconvolve

def loadrate(Vpl, loading, hh, W, mu_d, G, my, nx, Kjk):
    np.random.seed(1)

    # Stiffness calculations
    k1 = mu_d / (2 * hh) if hh != 0 else np.inf  # Handle hh = 0 case
    k2 = G / (2 * W) if W != 0 else np.inf       # Handle W = 0 case
    k_eff = k1 * k2 / (k1 + k2) if (k1 != np.inf and k2 != np.inf) else (k1 if k2 == np.inf else k2)

    # Initialize taudotbg based on loading type
    if loading == 'rw':
        taudot0 = Vpl * k_eff
        taudotbg = taudot0 * np.ones((my, nx))
    else:
        if loading == 'bs+':
            taudotbg = 0.5 * Vpl * k_eff * np.ones((my, nx))
        elif loading == 'bs':
            taudotbg = np.zeros((my, nx))

        # Add loading from the "left side"        
        V_plate = np.zeros((my, nx))
        V_plate[:, :nx//2] = 0.5 * Vpl
        K_vert = np.concatenate((Kjk[::-1, :], Kjk[1:, :]), axis=0)
        K_sym = np.concatenate((K_vert[:, ::-1], K_vert[:, 1:]), axis=1)
        conv_full = fftconvolve(V_plate, K_sym, mode = 'same')

        taudotbg[:, :nx//2] += conv_full[:, nx//2:]

    return taudotbg

if __name__ == "__main__":
    Vpl = 1.0
    loading = 'rw'
    hh = 0.1
    W = 1.0
    mu_d = 0.5
    G = 10.0
    my = 5
    nx = 10
    Kjk = np.random.rand(my, nx // 2)  # Example stiffness matrix


    # Call the function
    taudotbg = loadrate(Vpl, loading, hh, W, mu_d, G, my, nx, Kjk)
    print(taudotbg)
