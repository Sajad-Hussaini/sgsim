import numpy as np

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

def get_stats(wu: float, zu: float, wl: float, zl: float, freq: np.array,
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
