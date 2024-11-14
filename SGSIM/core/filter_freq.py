import numpy as np
from numba import jit, prange, float64, complex128

@jit(float64[:](float64, float64, float64, float64, float64[:]), nopython=True)
def get_psd(wu: float, zu: float, wl: float, zl: float, freq: np.array) -> np.array:
    """
    Calculate the non-normalized Power Spectral Density (PSD) of the stochastic model

    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency up to Nyq.

    Returns:
        Non-normalized PSD.
    """
    # Displacement - low pass
    psdu = 1 / ((wu ** 2 - freq ** 2) ** 2 + (2 * zu * wu * freq) ** 2)
    # Acceleration - high pass
    psdl = freq ** 4 / ((wl ** 2 - freq ** 2) ** 2 + (2 * zl * wl * freq) ** 2)
    return psdu * psdl

@jit(float64[:](float64, float64, float64, float64, float64[:]), nopython=True)
def get_variances(wu: float, zu: float, wl: float, zl: float, freq: np.array) -> np.array:
    """
    Calculate statistical measures based on the Power Spectral Density (PSD)

    psdb: Power Spectral Density
    freq: Angular frequency array
    statistics:
        variance :     variance                   using 0
        variance_dot:  variance 1st derivative    using 2
        variance_2dot: variance 2nd derivative    using 4
        variance_bar:  variance 1st integral      using -2
        variance_2bar: variance 2nd integral      using -4
    Returns:
        Computed statistics based on the given types
    """
    psdb = get_psd(wu, zu, wl, zl, freq)
    variances = np.empty(5)
    variances[0] = np.sum(psdb)
    variances[1] = np.sum((freq ** 2) * psdb)
    variances[2] = np.sum((freq ** 4) * psdb)
    variances[3] = np.sum((freq[1:] ** -2) * psdb[1:])
    variances[4] = np.sum((freq[1:] ** -4) * psdb[1:])
    return variances

@jit(complex128[:](float64, float64, float64, float64, float64[:]), nopython=True)
def get_frf(wu: float, zu: float, wl: float, zl: float, freq: np.array) -> np.array:
    """
    FRF of the stocahstic model
    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency upto Nyq.
    return:
        FRF
    """
    # Displacement - low pass
    frfu = 1 / ((wu ** 2 - freq ** 2) + (2j * zu * wu * freq))
    # Acceleration - high pass
    frfl = -1 * freq ** 2 / ((wl ** 2 - freq ** 2) + (2j * zl * wl * freq))
    return frfu * frfl

@jit(float64[:, :](float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True, parallel=True)
def get_stats(wu: np.array, zu: np.array, wl: np.array, zl: np.array, freq: np.array):
    """
    The statistics of the stochastic model using frequency domain
    ignoring the modulating function and the variance of White noise (i.e., 1)
    variance :     variance                   using 0
    variance_dot:  variance 1st derivative    using 2
    variance_2dot: variance 2nd derivative    using 4
    variance_bar:  variance 1st integral      using -2
    variance_2bar: variance 2nd integral      using -4
    """
    npts = len(wu)
    variances = np.empty((5, npts))
    for i in prange(npts):
        stats_i = get_variances(wu[i], zu[i], wl[i], zl[i], freq)
        variances[..., i] = stats_i[:5]
    return variances

@jit(float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:]), nopython=True)
def get_fas(mdl: np.array, wu: np.array, zu: np.array, wl: np.array, zl: np.array, freq: np.array):
    """
    The FAS of the stochastic model using frequency domain
    """
    npts = len(wu)
    psd = np.zeros(len(freq))
    for i in range(npts):
        psd_i = get_psd(wu[i], zu[i], wl[i], zl[i], freq)
        psd += (psd_i / np.sum(psd_i)) * mdl[i] ** 2
    return np.sqrt(psd)
