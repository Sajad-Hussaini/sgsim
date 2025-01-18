import numpy as np
from scipy.fft import rfft, rfftfreq
from numba import jit, float64

def get_mzc(rec):
    """
    The mean cumulative number of zero up and down crossings
    """
    cross_mask = rec[..., :-1] * rec[..., 1:] < 0
    cross_vec = np.empty_like(rec)
    cross_vec[..., :-1] = cross_mask * 0.5
    cross_vec[..., -1] = cross_vec[..., -2]
    return np.cumsum(cross_vec, axis=-1)

def get_mpc(rec):
    """
    The mean cumulative number of pi / 2 changes for an orientation array
    """
    change_vec = np.where(np.abs(np.diff(rec, append=0))>np.pi/2, 0.5, 0)
    return np.cumsum(change_vec, axis=-1)

def get_pmnm(rec):
    """
    The mean cumulative number of positive-minima and negative-maxima
    """
    pmnm_mask =((rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) & (rec[..., 1:-1] < 0) |
               (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]) & (rec[..., 1:-1] > 0))
    pmnm_vec = np.empty_like(rec)
    pmnm_vec[..., 1:-1] = pmnm_mask * 0.5
    pmnm_vec[..., 0] = pmnm_vec[..., 1]
    pmnm_vec[..., -1] = pmnm_vec[..., -2]
    return np.cumsum(pmnm_vec, axis=-1)

def get_mle(rec):
    """
    The mean cumulative number of local extrema (peaks and valleys)
    """
    mle_mask = ((rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) |
                (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]))
    mle_vec = np.empty_like(rec)
    mle_vec[..., 1:-1] = mle_mask * 0.5
    mle_vec[..., 0] = mle_vec[..., 1]
    mle_vec[..., -1] = mle_vec[..., -2]
    return np.cumsum(mle_vec, axis=-1)

@jit(float64[:, :, :, :](float64, float64[:, :], float64[:], float64, float64), nopython=True, cache=True)
def sdof_lin_model(dt, rec, period, zeta, mass):
    """
    linear analysis of a SDOF model using newmark method

    For ground motion excitation and not an arbitraty force
        we use: p = -m * rec

    The total acceleration are computed as ac_tot = ac + rec
    disp, vel, ac are relative to the ground

    Parameters
    ----------
    dt : float
        time step.
    rec
        input ground motions 2d-array.
    period : np.array
        period array.
    zeta : float, optional
        damping ration of SDOF. The default is 0.05.
    mass : float, optional
        mass of SDOF. The default is 1.0.

    Returns
    -------
    sdf_responses : 4d-array (response_type, n_rec, npts, n_period)
        response_type corresponds to disp, vel, ac, ac_tot
    """
    rec_3d = rec[:, :, None]
    # p = -mass * rec if excitation == 'GM' else rec
    p = -mass * rec_3d

    # SDOF properties
    wn = 2 * np.pi / period
    k = mass * wn ** 2
    c = 2 * mass * wn * zeta

    n_records, npts, _ = p.shape  # Number of records and excitation points
    n_sdf = len(period)           # number of sdf periods

    sdf_responses = np.empty((4, n_records, npts, n_sdf))  # disp, vel, ac, ac_tot

    # coefficients of numerical solution
    gamma = np.full(n_sdf, 0.5)
    beta = np.full(n_sdf, 1.0 / 6.0)  # The linear acceleration method
    beta[dt / period > 0.551] = 0.25  # The constant average acceleration
    a1 = mass / (beta * dt ** 2) + c * gamma / (beta * dt)
    a2 = mass / (beta * dt) + c * (gamma / beta - 1)
    a3 = mass * (1 / (2 * beta) - 1) + c * dt * (gamma / (2 * beta) - 1)
    k_hat = k + a1

    # system at rest
    sdf_responses[0, :, 0] = 0.0  # disp
    sdf_responses[1, :, 0] = 0.0  # vel
    sdf_responses[2, :, 0] = (p[:, 0] - c * sdf_responses[1, :, 0]
                              - k * sdf_responses[0, :, 0]) / mass  # ac
    sdf_responses[3, :, 0] = sdf_responses[2, :, 0] + rec_3d[:, 0]  # ac_tot
    for i in range(npts - 1):
        dp = (p[:, i + 1] + a1 * sdf_responses[0, :, i] +
              a2 * sdf_responses[1, :, i] + a3 * sdf_responses[2, :, i])

        sdf_responses[0, :, i + 1] = dp / k_hat

        sdf_responses[1, :, i + 1] =(
            (gamma / (beta * dt)) *
            (sdf_responses[0, :, i + 1] - sdf_responses[0, :, i]) +
            (1 - gamma / beta) *
            sdf_responses[1, :, i] + dt * sdf_responses[2, :, i] *
            (1 - gamma / (2 * beta)))

        sdf_responses[2, :, i + 1] = (
            (sdf_responses[0, :, i + 1] -
             sdf_responses[0, :, i]) / (beta * dt ** 2) -
            sdf_responses[1, :, i] / (beta * dt) -
            sdf_responses[2, :, i] * (1 / (2 * beta) - 1))

        sdf_responses[3, :, i + 1] = sdf_responses[2, :, i + 1] + rec_3d[:, i + 1]
    return sdf_responses

def get_spectra(dt: float, rec, period, zeta: float=0.05):
    """
    displacement, velocity, and total acceleration response spectra (SD, SV, SA)
    """
    n_rec = rec.shape[0]
    n_period = len(period)
    sd = np.empty((n_rec, n_period))
    sv = np.empty((n_rec, n_period))
    sa = np.empty((n_rec, n_period))

    chunk_size = 5
    for start in range(0, n_rec, chunk_size):
        end = min(start + chunk_size, n_rec)
        disp_sdf, vel_sdf, _, act_sdf = sdof_lin_model(dt=dt, rec=rec[start:end], period=period, zeta=zeta, mass=1.0)

        sd[start:end] = np.max(np.abs(disp_sdf), axis=1)
        sv[start:end] = np.max(np.abs(vel_sdf), axis=1)
        sa[start:end] = np.max(np.abs(act_sdf), axis=1)
    return sd, sv, sa

def get_energy_mask(dt: float, rec, target_range: tuple[float, float]=(0.001, 0.999)):
    " A boolean mask for slicing the input motion using total energy percentages"
    cumulative_energy = get_ce(dt, rec)
    return ((cumulative_energy >= target_range[0] * cumulative_energy[-1]) &
            (cumulative_energy <= target_range[1] * cumulative_energy[-1]))

def get_amplitude_mask(rec, threshold: float):
    " A boolean mask for slicing the input motion using a threshold of amplitude"
    indices = np.where(np.abs(rec) > threshold)[0]
    if len(indices) == 0:
        return slice(0, 0)
    return slice(indices[0], indices[-1] + 1)

def get_freq_mask(freq: np.array, target_range: tuple[float, float]=(0.1, 25.0)):
    " A boolean mask for slicing the frequency array using input freqs in Hz"
    return (freq >= target_range[0] * 2 * np.pi) & (freq <= target_range[1] * 2 * np.pi)

def get_ce(dt: float, rec):
    """
    Compute the cumulative energy
    """
    return np.cumsum(rec ** 2, axis=-1) * dt

def get_integral(dt: float, rec):
    """
    Compute the velocity of an acceleration input
    """
    return np.cumsum(rec, axis=-1) * dt

def get_integral_detrend(dt: float, rec):
    """
    Compute the integral with linear detrending
    """
    uvec = get_integral(dt, rec)
    return uvec - np.linspace(0.0, uvec[-1], len(uvec))

def get_pgp(rec):
    " Peak ground motion parameter"
    return np.max(np.abs(rec), axis=-1)

def get_cav(dt: float, rec):
    " Cumulative absolute velocity"
    return np.sum(np.abs(rec), axis=-1) * dt

def get_fas(npts, rec):
    " Fourier amplitude spectrum "
    return np.abs(rfft(rec)) / np.sqrt(npts / 2)

def get_freq(npts, dt):
    " Angular frequency upto Nyq "
    return rfftfreq(npts, dt) * 2 * np.pi

def get_time(npts, dt):
    " time array "
    return np.linspace(0, (npts - 1) * dt, npts)
