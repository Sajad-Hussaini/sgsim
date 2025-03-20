import numpy as np
from numba import njit, prange, int64, float64, complex128

@njit(complex128[:](float64, float64, float64, float64, float64[:], float64[:]), cache=True)
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

@njit(float64[:](float64, float64, float64, float64, float64[:], float64[:]), cache=True)
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

@njit((float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), parallel=True, cache=True)
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
        variance[i] = np.sum(psdb)
        variance_dot[i] = np.sum(freq_p2 * psdb)
        variance_2dot[i] = np.sum(freq_p4 * psdb)
        variance_bar[i] = np.sum(freq_n2 * psdb[1:])
        variance_2bar[i] = np.sum(freq_n4 * psdb[1:])

@njit((float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), cache=True)
def get_fas(mdl, wu, zu, wl, zl, freq_p2, freq_p4, fas):
    """
    The Fourier amplitude spectrum (FAS) of the stochastic model using PSD
    """
    fas.fill(0.0)
    for i in range(len(wu)):
        psd_i = get_psd(wu[i], zu[i], wl[i], zl[i], freq_p2, freq_p4)
        fas += mdl[i] ** 2 * psd_i / psd_i.sum()
    fas[:] = np.sqrt(fas)

@njit(complex128[:, :](int64, int64, float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:, :]), parallel=True, cache=True)
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