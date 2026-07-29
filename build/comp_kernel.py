import numpy as np
from scipy.fft import fft2, ifft2, fftfreq, next_fast_len

def Kspec(k, W, h, mu, D):
    k = np.abs(k)
    tanh_kW = np.tanh(W * k)
    tanh_kh = np.tanh(h * k)

    mask = k > 0
    K = np.empty_like(k, dtype=np.float64)
    K[mask] = 0.5 * (1 - D) * mu * k[mask] * (
        1 + (1 - D) * tanh_kW[mask] * tanh_kh[mask]
        ) / ((1 - D) * tanh_kW[mask] + tanh_kh[mask])

    K[0, 0] = 0.5 * mu / (W + h / (1 - D))

    return K

def kernel_3D(Lx, Ly, W, h, mu, D, Nx, Ny, sourcenum):
    # define wavenumbers
    kx = 2 * np.pi / Lx * Nx * np.fft.fftfreq(Nx)
    ky = 2 * np.pi / Ly * Ny * np.fft.fftfreq(Ny)
    kx, ky = np.meshgrid(kx, ky, indexing='ij')
    k = np.sqrt(kx**2 + ky**2)

    # Compute spectral kernel
    K_k = Kspec(k, W, h, mu, D)

    # Source distribution(constructed directly in frequency domain)
    source = np.ones((sourcenum, sourcenum))
    fftsource = fft2(source, s=(Nx, Ny))

    # Convolution in Fourier space
    Kk = fftsource * K_k
    K_xy = ifft2(Kk).real

    return K_xy

def compute(W, hh, G, D, xper, yper, nx, my, dx, dy, sourcenum):
    # define domain dimensions
    Lx_domain = nx * dx
    Ly_domain = my * dy
    Lx = Lx_domain * (2 if not xper else 1)
    Ly = Ly_domain * (2 if not yper else 1)

    Nx = int(Lx / dx * sourcenum)
    Ny = int(Ly / dy * sourcenum)
    littledx = dx / sourcenum
    littledy = dy / sourcenum

    Kjk = kernel_3D(Lx, Ly, W, hh, G, D, Nx, Ny, sourcenum)

    # average Kjk to obtain Kjkav
    Kjk = -Kjk.T
    kses = np.arange(sourcenum // 2 - 1, Nx, sourcenum) if xper else np.arange(sourcenum // 2 - 1, Nx //2, sourcenum)
    jses = np.arange(sourcenum // 2 - 1, Ny, sourcenum) if yper else np.arange(sourcenum // 2 - 1, Ny //2, sourcenum)

    J, K = np.ix_(jses, kses)
    Kjkav = 0.25 * (Kjk[J, K] + Kjk[J, K+1] + Kjk[J+1, K] + Kjk[J+1, K+1])

    return Kjkav