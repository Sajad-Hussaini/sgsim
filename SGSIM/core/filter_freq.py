import numpy as np
from numba import jit, prange

@jit(nopython=True)
def get_psd(wu: float, zu: float, wl: float, zl: float, freq: np.array) -> np.array:
    """
    Calculate the non-normalized Power Spectral Density (PSD) of the stochastic model.

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
    psdb = psdu * psdl
    return psdb

@jit(nopython=True)
def get_variances(wu: float, zu: float, wl: float, zl: float, freq: np.array,
              statistics: tuple[int, ...]) -> np.array:
    """
    Calculate statistical measures based on the Power Spectral Density (PSD).

    psdb: Power Spectral Density.
    freq: Angular frequency array.
    statistics: Tuple indicating which statistics to compute.

    Returns:
        Computed statistics based on the given types.
    """
    psdb = get_psd(wu, zu, wl, zl, freq)
    stats = np.zeros(len(statistics))
    for i, r in enumerate(statistics):
        if r < 0:
            stats[i] = np.sum((freq[1:] ** r) * psdb[1:])
        else:
            stats[i] = np.sum((freq ** r) * psdb)
    return stats

@jit(nopython=True)
def get_frf(wu: float, zu: float, wl: float, zl: float, freq: np.array) -> np.array:
    """
    FRF of the stocahstic model.
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

@jit(nopython=True, parallel=True)
def get_stats(wu: np.array, zu: np.array, wl: np.array, zl: np.array, freq: np.array, statistics=(0, 2, 4, -2, -4)):
    """
    The statistics of the stochastic model using frequency domain.
    ignoring the modulating function and the variance of White noise (i.e., 1)
    variance :     variance                   using 0
    variance_dot:  variance 1st derivative    using 2
    variance_2dot: variance 2nd derivative    using 4
    variance_bar:  variance 1st integral      using -2
    variance_2bar: variance 2nd integral      using -4
    """
    npts = len(wu)
    variance, variance_dot, variance_2dot, variance_bar, variance_2bar = np.zeros((5, npts))
    for i in prange(npts):
        stats_i = get_variances(wu[i], zu[i], wl[i], zl[i], freq, statistics=statistics)
        variance[i], variance_dot[i], variance_2dot[i], variance_bar[i], variance_2bar[i] = stats_i[:5]
    return variance, variance_dot, variance_2dot, variance_bar, variance_2bar

@jit(nopython=True, parallel=True)
def get_fas(mdl: np.array, wu: np.array, zu: np.array, wl: np.array, zl: np.array, freq: np.array):
    """
    The FAS of the stochastic model using frequency domain.
    """
    npts = len(wu)
    psd = np.zeros((npts, len(freq)))
    for i in prange(npts):
        psd_i = get_psd(wu[i], zu[i], wl[i], zl[i], freq)
        psd[i] = (psd_i / np.sum(psd_i)) * mdl[i] ** 2
    psd = np.sum(psd, axis=0)
    return np.sqrt(psd)
