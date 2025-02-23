import numpy as np
from numba import jit, prange, int64, float64, complex128

@jit(complex128[:](float64, float64, float64, float64, float64[:]), nopython=True, cache=True)
def get_frf(wu: float, zu: float, wl: float, zl: float, freq):
    """
    Frequency response function for the filter

    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency upto Nyq.
    """
    # Displacement - low pass
    frfu = 1 / ((wu ** 2 - freq ** 2) + (2j * zu * wu * freq))
    # Acceleration - high pass
    frfl = -1 * freq ** 2 / ((wl ** 2 - freq ** 2) + (2j * zl * wl * freq))
    return frfu * frfl

@jit(float64[:](float64, float64, float64, float64, float64[:]), nopython=True, cache=True)
def get_psd(wu: float, zu: float, wl: float, zl: float, freq):
    """
    Non-normalized Power Spectral Density (PSD) for the filter

    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency up to Nyq.
    """
    # Displacement - low pass
    psdu = 1 / ((wu ** 2 - freq ** 2) ** 2 + (2 * zu * wu * freq) ** 2)
    # Acceleration - high pass
    psdl = freq ** 4 / ((wl ** 2 - freq ** 2) ** 2 + (2 * zl * wl * freq) ** 2)
    return psdu * psdl

@jit((float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True, parallel=True, cache=True)
def get_stats(wu, zu, wl, zl, freq, variance, variance_dot, variance_2dot, variance_bar, variance_2bar):
    """
    The evolutionary statistics of the stochastic model using Power Spectral Density (PSD)
    Ignoring the modulating function and the unit-variance White noise

    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency up to Nyq.

    statistics:
        variance :     variance                   using power 0
        variance_dot:  variance 1st derivative    using power 2
        variance_2dot: variance 2nd derivative    using power 4
        variance_bar:  variance 1st integral      using power -2
        variance_2bar: variance 2nd integral      using power -4
    """
    freq_p2 = freq ** 2
    freq_p4 = freq ** 4
    freq_n2 = freq[1:] ** -2
    freq_n4 = freq[1:] ** -4
    for i in prange(len(wu)):
        psdb = get_psd(wu[i], zu[i], wl[i], zl[i], freq)
        variance[i] = np.sum(psdb)
        variance_dot[i] = np.sum(freq_p2 * psdb)
        variance_2dot[i] = np.sum(freq_p4 * psdb)
        variance_bar[i] = np.sum(freq_n2 * psdb[1:])
        variance_2bar[i] = np.sum(freq_n4 * psdb[1:])

@jit((float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True, cache=True)
def get_fas(mdl, wu, zu, wl, zl, freq, fas):
    """
    The Fourier amplitude spectrum (FAS) of the stochastic model using PSD
    """
    for i in range(len(wu)):
        psd_i = get_psd(wu[i], zu[i], wl[i], zl[i], freq)
        fas += mdl[i] ** 2 * psd_i / np.sum(psd_i)
    fas[:] = np.sqrt(fas)

@jit(complex128[:, :](int64, int64, float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:, :]), nopython=True, parallel=True, cache=True)
def simulate_fourier_series(n, npts, t, freq_sim, mdl, wu, zu, wl, zl, variance, white_noise):
    """
    The Fourier series of n number of simulations
    """
    fourier = np.zeros((n, len(freq_sim)), dtype=np.complex128)
    for sim in prange(n):
        for i in range(npts):
            fourier[sim, :] += (get_frf(wu[i], zu[i], wl[i], zl[i], freq_sim)
                                * np.exp(-1j * freq_sim * t[i]) * white_noise[sim][i] * mdl[i] / np.sqrt(variance[i] * 2 / npts))
    return fourier