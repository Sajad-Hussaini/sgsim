import numpy as np
from numba import njit, prange, int64, float64, complex128

@njit(complex128[::1](float64, float64, float64, float64, float64[::1], float64[::1]), fastmath=True, cache=True)
def get_frf(wu, zu, wl, zl, freq, freq_p2):
    """
    Frequency response function for the filter
    using the displacement and acceleration transfer function of the 2nd order system

    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency upto Nyq.
    freq_p2: freq ** 2
    """
    return -freq_p2 / (((wl ** 2 - freq_p2) + (2j * zl * wl * freq)) *
                       ((wu ** 2 - freq_p2) + (2j * zu * wu * freq)))

@njit(float64[::1](float64, float64, float64, float64, float64[::1], float64[::1]), fastmath=True, cache=True)
def get_psd(wu, zu, wl, zl, freq_p2, freq_p4):
    """
    Non-normalized Power Spectral Density (PSD) for the filter
    using the displacement and acceleration transfer function of the 2nd order system

    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency up to Nyq.
    freq_p2: freq ** 2
    freq_p4: freq ** 4
    """
    return freq_p4 / ((wl ** 4 + freq_p4 + 2 * wl ** 2 * freq_p2 * (2 * zl ** 2 - 1)) *
                      (wu ** 4 + freq_p4 + 2 * wu ** 2 * freq_p2 * (2 * zu ** 2 - 1)))

@njit((float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1]), parallel=True, fastmath=True, cache=True)
def get_stats(wu, zu, wl, zl, freq_p2, freq_p4, freq_n2, freq_n4, variance, variance_dot, variance_2dot, variance_bar, variance_2bar):
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
    for i in prange(len(wu)):
        psdb = get_psd(wu[i], zu[i], wl[i], zl[i], freq_p2, freq_p4)
        variance[i] = 0.0
        variance_dot[i] = 0.0
        variance_2dot[i] = 0.0
        variance_bar[i] = 0.0
        variance_2bar[i] = 0.0
        for j in range(len(psdb)):  # one passage with scalar operation through the loop is faster than using np.sum and np.dot (5 passages)
            variance[i] += psdb[j]
            variance_dot[i] += freq_p2[j] * psdb[j]
            variance_2dot[i] += freq_p4[j] * psdb[j]
            variance_bar[i] += freq_n2[j] * psdb[j]
            variance_2bar[i] += freq_n4[j] * psdb[j]
        # variance[i] = np.sum(psdb)  # slower than the loop
        # variance_dot[i] = np.dot(freq_p2, psdb)
        # variance_2dot[i] = np.dot(freq_p4, psdb)
        # variance_bar[i] = np.dot(freq_n2, psdb)
        # variance_2bar[i] = np.dot(freq_n4, psdb)

@njit((float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1]), parallel=True, fastmath=True, cache=True)
def get_fas(mdl, wu, zu, wl, zl, freq_p2, freq_p4, fas):
    """
    The Fourier amplitude spectrum (FAS) of the stochastic model using PSD
    """
    fas.fill(0.0)
    for i in prange(len(wu)):
        psd_i = get_psd(wu[i], zu[i], wl[i], zl[i], freq_p2, freq_p4)
        scale = mdl[i] ** 2 / np.sum(psd_i)
        for j in range(len(psd_i)):
            fas[j] += scale * psd_i[j]
    fas[:] = np.sqrt(fas)

@njit(complex128[:, ::1](int64, int64, float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[::1], float64[:, ::1]), parallel=True, fastmath=True, cache=True)
def simulate_fourier_series(n, npts, t, freq_sim, freq_sim_p2, mdl, wu, zu, wl, zl, variance, white_noise):
    """
    The Fourier series of n number of simulations
    """
    fourier = np.zeros((n, len(freq_sim)), dtype=np.complex128)
    for sim in prange(n):
        for i in range(npts):
            fourier[sim, :] += (get_frf(wu[i], zu[i], wl[i], zl[i], freq_sim, freq_sim_p2) *
                                np.exp(-1j * freq_sim * t[i]) * white_noise[sim][i] * mdl[i] / np.sqrt(variance[i] * 2 / npts))
    return fourier

@njit((float64, float64[::1], float64[::1], float64[::1]), fastmath=True, cache=True)
def cumulative_rate(dt, numerator, denominator, out):
    scale = dt / (2 * np.pi)
    cumsum = 0.0
    for i in range(len(numerator)):
        cumsum += np.sqrt(numerator[i] / denominator[i]) * scale
        out[i] = cumsum

@njit((float64, float64[::1], float64[::1], float64[::1], float64[::1]), fastmath=True, cache=True)
def pmnm_rate(dt, first, middle, last, out):
    scale = dt / (4 * np.pi)
    cumsum = 0.0
    for i in range(len(first)):
        cumsum += (np.sqrt(first[i] / middle[i]) - np.sqrt(middle[i] / last[i])) * scale
        out[i] = cumsum