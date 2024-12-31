import numpy as np
from numba import jit, prange, float64, complex128

@jit(complex128[:](float64, float64, float64, float64, float64[:]), nopython=True, cache=True)
def get_frf(wu: float, zu: float, wl: float, zl: float, freq):
    """
    Frequency response function for the filter

    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency upto Nyq.
    """
    # Displacement as low pass
    frfu = 1 / ((wu ** 2 - freq ** 2) + (2j * zu * wu * freq))
    # Acceleration as high pass
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

@jit(float64[:](float64, float64, float64, float64, float64[:]), nopython=True, cache=True)
def get_variances(wu: float, zu: float, wl: float, zl: float, freq):
    """
    Calculate statistics using Power Spectral Density (PSD)

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
    psdb = get_psd(wu, zu, wl, zl, freq)
    variances = np.empty(5)
    variances[0] = np.sum(psdb)
    variances[1] = np.sum(freq ** 2 * psdb)
    variances[2] = np.sum(freq ** 4 * psdb)
    variances[3] = np.sum(freq[1:] ** -2 * psdb[1:])
    variances[4] = np.sum(freq[1:] ** -4 * psdb[1:])
    return variances

@jit(float64[:, :](float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True, parallel=True, cache=True)
def get_stats(wu, zu, wl, zl, freq):
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
    npts = len(wu)
    variances = np.empty((5, npts))
    for i in prange(npts):
        variances[..., i] = get_variances(wu[i], zu[i], wl[i], zl[i], freq)[:5]
    return variances

@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True, cache=True)
def get_fas(mdl, wu, zu, wl, zl, freq):
    """
    The Fourier amplitude spectrum (FAS) of the stochastic model using PSD
    """
    psd = np.zeros(len(freq))
    for i in range(len(wu)):
        psd_i = get_psd(wu[i], zu[i], wl[i], zl[i], freq)
        psd += mdl[i] ** 2 * psd_i / np.sum(psd_i)
    return np.sqrt(psd)
