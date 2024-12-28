import numpy as np
from scipy.fft import rfft, rfftfreq
from numba import jit, float64

def get_mzc(rec: np.ndarray) -> np.ndarray:
    """
    The mean cumulative number of zero up/ down crossings.
    """
    cross_vec = np.where(np.diff(np.sign(rec), append=0), 0.5, 0)
    return np.cumsum(cross_vec, axis=-1)

def get_pmnm(rec: np.ndarray) -> np.ndarray:
    """
    The mean cumulative number of positive-minima and negative-maxima.
    # TODO : It is mean velue check if zu or zl can directly fit to none-average
    """
    pmnm_vec = np.where(
        (rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) &
        (rec[..., 1:-1] < 0) |
        (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]) &
        (rec[..., 1:-1] > 0), 0.5, 0)
    pmnm_vec = np.concatenate((pmnm_vec[..., :1],
                               pmnm_vec, pmnm_vec[..., -1:]), axis=-1)
    return np.cumsum(pmnm_vec, axis=-1)

def get_mle(rec: np.ndarray) -> np.ndarray:
    """
    The mean cumulative number of local extrema (all peaks and valleys).
    """
    mle_vec = np.where(
        (rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) |
        (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]),
        0.5, 0)
    mle_vec = np.concatenate((mle_vec[..., :1],
                              mle_vec, mle_vec[..., -1:]), axis=-1)
    return np.cumsum(mle_vec, axis=-1)

@jit(float64[:, :, :, :](float64, float64[:, :], float64[:], float64, float64), nopython=True, cache=True)
def sdof_lin_model(dt: float, rec: np.ndarray, period: np.array, zeta: float, mass: float) -> np.ndarray:
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
    rec : np.ndarray
        input ground motions 2d-array.
    period : np.array
        period array.
    zeta : float, optional
        damping ration of SDOF. The default is 0.05.
    mass : float, optional
        mass of SDOF. The default is 1.0.

    Returns
    -------
    sdf_responses : 4d-array (response, n_rec, npts, n_period)
        first dimension correspond to disp, vel, ac, ac_tot responses of SDOF.

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

def get_spectra(dt: float, rec: np.ndarray, period: np.array, zeta: float=0.05):
    """
    displacement, velocity, and total acceleration spectra (SD, SV, SA)
    """
    n_rec = rec.shape[0]
    n_period = len(period)
    sd = np.empty((n_rec, n_period))
    sv = np.empty((n_rec, n_period))
    sa = np.empty((n_rec, n_period))

    chunk_size = 5  # 5 record per loop to avoid memory issue
    for start in range(0, n_rec, chunk_size):
        end = min(start + chunk_size, n_rec)
        rec_chunk = rec[start:end]
        disp_sdf, vel_sdf, _, act_sdf = sdof_lin_model(dt=dt, rec=rec_chunk, period=period, zeta=zeta, mass=1.0)

        sd_chunk = np.max(np.abs(disp_sdf), axis=1)
        sv_chunk = np.max(np.abs(vel_sdf), axis=1)
        sa_chunk = np.max(np.abs(act_sdf), axis=1)

        sd[start:end] = sd_chunk
        sv[start:end] = sv_chunk
        sa[start:end] = sa_chunk
    return sd, sv, sa

def get_energy_slice(dt: float, rec: np.array, target_range: tuple[float, float]=(0.001, 0.999)):
    """
    A slice of the input motion based on a range of total energy percentages
    i.e. (0.001, 0.999)
    """
    cumulative_energy = get_ce(dt, rec)
    return ((cumulative_energy >= target_range[0] * cumulative_energy[-1]) &
            (cumulative_energy <= target_range[1] * cumulative_energy[-1]))

def get_amplitude_slice(rec: np.array, threshold: float):
    " A slice of the input motion based on passing a threshold of amplitude"
    indices = np.where(np.abs(rec) > threshold)[0]
    if len(indices) == 0:
        return slice(0, 0)
    return slice(indices[0], indices[-1] + 1)

def get_freq_slice(freq: np.array, target_range: tuple[float, float]=(0.1, 25.0)):
    " A slice of angular frequency array between input frequnecies in Hz"
    return (freq >= target_range[0] * 2 * np.pi) & (freq <= target_range[1] * 2 * np.pi)

def get_ce(dt: float, rec: np.ndarray) -> np.ndarray:
    """
    Compute cumulative energy of an input
    """
    return np.cumsum(rec ** 2, axis=-1) * dt

def get_vel(dt: float, rec: np.ndarray) -> np.ndarray:
    """
    Compute the velocity of an acceleration input
    """
    return np.cumsum(rec, axis=-1) * dt

def get_disp(dt: float, rec: np.ndarray) -> np.ndarray:
    """
    Compute the displacement of an acceleration input
    """
    return np.cumsum(np.cumsum(rec, axis=-1), axis=-1) * dt ** 2

def get_disp_detrend(dt: float, rec: np.ndarray) -> np.ndarray:
    """
    Compute the displacement of an acceleration input with detrending
    """
    uvec = get_disp(dt, rec)
    return uvec - np.linspace(0.0, uvec[-1], len(uvec))

def get_pga(rec: np.ndarray):
    " Peak ground acceleration"
    return np.max(np.abs(rec), axis=-1)

def get_pgv(dt: float, rec: np.ndarray):
    " Peak ground velocity"
    vel = get_vel(dt, rec)
    return np.max(np.abs(vel), axis=-1)

def get_pgd(dt: float, rec: np.ndarray):
    " Peak ground displacement"
    disp = get_disp(dt, rec)
    return np.max(np.abs(disp), axis=-1)

def get_cav(dt: float, rec: np.ndarray):
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