import numpy as np
from scipy.signal import butter, sosfilt
from numba import jit, float64

def find_error(rec: np.array, model: np.array) -> float:
    """
    evaluate the error between record and model feature (e.g., mzc)
    a relative error based on overall area ratio
    # TODO later to take this out in another module
    """
    return np.sum(np.abs(rec - model)) / np.sum(rec)

def find_mzc(rec: np.ndarray) -> np.ndarray:
    """
    The mean cumulative number of zero up/ down crossings.
    """
    cross_vec = np.where(np.diff(np.sign(rec), append=0), 0.5, 0)
    return np.cumsum(cross_vec, axis=-1)

def find_pmnm(rec: np.ndarray) -> np.ndarray:
    """
    The mean cumulative number of positive-minima and negative-maxima.
    """
    pmnm_vec = np.where(
        (rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) &
        (rec[..., 1:-1] < 0) |
        (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]) &
        (rec[..., 1:-1] > 0), 0.5, 0)
    pmnm_vec = np.concatenate((pmnm_vec[..., :1],
                               pmnm_vec, pmnm_vec[..., -1:]), axis=-1)
    return np.cumsum(pmnm_vec, axis=-1)

def find_mle(rec: np.ndarray) -> np.ndarray:
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

def find_slice(dt: float, t: np.array, rec: np.array, target_range: tuple[float, float] = (0.001, 0.999)):
    """
    A slice of the input motion based on a range of total energy percentages i.e. (0.001, 0.999)
    """
    cumulative_energy = get_ce(dt, rec)
    return ((cumulative_energy >= target_range[0] * cumulative_energy[-1]) &
            (cumulative_energy <= target_range[1] * cumulative_energy[-1]))

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

def bandpass_filter(dt, rec, lowcut=0.1, highcut=25.0, order=4):
    """
    Apply a high-pass Butterworth filter to remove low-frequency drift.
    """
    nyquist = 0.5 / dt  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    n = len(rec)
    next_pow2 = int(2 ** np.ceil(np.log2(2 * n)))
    pad_width = next_pow2 - n
    signal_padded = np.pad(rec, (pad_width // 2, pad_width - pad_width // 2), mode='constant')
    filtered_rec = sosfilt(sos, signal_padded)
    filtered_rec = filtered_rec[pad_width // 2: -pad_width // 2]
    return filtered_rec

def baseline_correction(rec, degree=1):
    " Baseline correction using polynomial fit "
    n = len(rec)
    x = np.arange(n)
    baseline_coefficients = np.polyfit(x, rec, degree)
    baseline = np.polyval(baseline_coefficients, x)
    corrected_signal = rec - baseline
    return corrected_signal

def moving_average(rec, window_size=9):
    """
    Perform a moving average smoothing on the input data with the specified window size.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd.")
    window = np.ones(window_size) / window_size
    if rec.ndim == 1:
        smoothed_rec = np.convolve(rec, window, mode='same')
    elif rec.ndim == 2:
        smoothed_rec = np.apply_along_axis(lambda x: np.convolve(x, window, mode='same'), axis=0, arr=rec)
    else:
        raise ValueError("Input must be a 1D or 2D array.")
    return smoothed_rec

@jit(float64[:, :, :, :](float64, float64[:, :], float64[:], float64, float64), nopython=True)
def linear_analysis_sdf(dt: float, rec: np.ndarray, period: np.array, zeta: float, mass: float) -> np.ndarray:
    """
    linear analysis of a SDOF using newmark method

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

    sdf_responses = np.empty((4, n_records, npts, n_sdf))

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
    displacement, velocity, and total acceleration spectra
    """
    n_rec = rec.shape[0]
    n_period = len(period)
    sd = np.empty((n_rec, n_period))
    sv = np.empty((n_rec, n_period))
    sa = np.empty((n_rec, n_period))

    chunk_size = 5  # 5 record per loop to avoid memory issue
    for i in range(0, n_rec, chunk_size):
        rec_chunk = rec[i:i + chunk_size]
        disp_sdf, vel_sdf, _, act_sdf = linear_analysis_sdf(dt=dt, rec=rec_chunk, period=period, zeta=zeta, mass=1.0)

        sd_chunk = np.max(np.abs(disp_sdf), axis=1)
        sv_chunk = np.max(np.abs(vel_sdf), axis=1)
        sa_chunk = np.max(np.abs(act_sdf), axis=1)

        sd[i:i + chunk_size] = sd_chunk
        sv[i:i + chunk_size] = sv_chunk
        sa[i:i + chunk_size] = sa_chunk
    return sd, sv, sa
