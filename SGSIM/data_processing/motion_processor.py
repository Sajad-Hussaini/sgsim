import numpy as np
from scipy.signal import butter, sosfilt
from numba import jit

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
    pmnm_vec = np.where((rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) &
                        (rec[..., 1:-1] < 0) |
                        (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]) &
                        (rec[..., 1:-1] > 0), 0.5, 0)
    pmnm_vec = np.concatenate((pmnm_vec[..., :1], pmnm_vec, pmnm_vec[..., -1:]), axis=-1)
    return np.cumsum(pmnm_vec, axis=-1)

def find_mle(rec: np.ndarray) -> np.ndarray:
    """
    The mean cumulative number of local extrema (all peaks and valleys).
    """
    mle_vec = np.where((rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) |
                        (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]), 0.5, 0)
    mle_vec = np.concatenate((mle_vec[..., :1], mle_vec, mle_vec[..., -1:]), axis=-1)
    return np.cumsum(mle_vec, axis=-1)

def find_slice(dt: float, t: np.array, rec: np.array, target_range: tuple[float, float] = (0.001, 0.999)):
    """
    A slice of the input motion based on a range of total energy percentages i.e. (0.001, 0.999)
    """
    cumulative_energy = get_ce(dt, rec)
    return (cumulative_energy >= target_range[0] * cumulative_energy[-1]) & (cumulative_energy <= target_range[1] * cumulative_energy[-1])

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
    next_pow2 = 2**np.ceil(np.log2(n*2)).astype(int)  # Find next power of 2
    pad_width = next_pow2 - n  # Calculate padding size
    signal_padded = np.pad(rec, (pad_width // 2, pad_width - pad_width // 2), mode='constant')
    filtered_rec = sosfilt(sos, signal_padded)
    filtered_rec = filtered_rec[pad_width // 2: -pad_width // 2]
    return filtered_rec

def baseline_correction(rec, degree = 1):
    n = len(rec)
    x = np.arange(n)

    # Fit a polynomial of the specified degree to the signal
    baseline_coefficients = np.polyfit(x, rec, degree)
    baseline = np.polyval(baseline_coefficients, x)

    # Subtract the baseline from the original signal to correct it
    corrected_signal = rec - baseline
    return corrected_signal

def moving_average(rec, window_size=9):
    """
    Perform a moving average smoothing on the input data with the specified window size.
    """
    # Ensure the window size is an odd number for symmetry
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd.")
    window = np.ones(window_size) / window_size
    # Check the number of dimensions and apply the moving average
    if rec.ndim == 1:
        # For 1D array, apply moving average directly
        smoothed_rec = np.convolve(rec, window, mode='same')
    elif rec.ndim == 2:
        # For 2D array, apply the moving average to each column (axis=0)
        smoothed_rec = np.apply_along_axis(lambda x: np.convolve(x, window, mode='same'), axis=0, arr=rec)
    else:
        raise ValueError("Input must be a 1D or 2D array.")
    return smoothed_rec

@jit(nopython=True)
def linear_analysis_sdf(dt: float, rec: np.ndarray, period_range: tuple[float, float, float] = (0.05, 4.05, 0.01),
                        zeta: float = 0.05,
                        mass: float = 1.0,
                        excitation: str = 'GM') -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    linear analysis of a single degree of freedom system using newmark method
    For excitation as ground acceleration (GM) and not an arbitraty force
    use: p = -m * ac
    uVec, vVec, aVec are relative to the ground
    the total velocity and acceleration are computed as atVec=aVec+ac(grd)
    """
    if rec.ndim == 1:
        rec_2d = rec[None, :]
    else:
        rec_2d = rec
    p = -mass * rec_2d if excitation == 'GM' else rec_2d
    # properties of the SDF systems for each period
    T = np.arange(*period_range)
    wn = 2 * np.pi / T
    k = mass * wn ** 2
    c = 2 * mass * wn * zeta

    n_records, n_exc = p.shape  # Number of records and excitation points
    n_sdf = len(T)  # number of sdf corresponding to each period
    disp, vel, ac, ac_total = np.zeros((4, n_records, n_exc, n_sdf))

    # coefficients of numerical solution
    gamma = np.full(n_sdf, 0.5)
    beta = np.full(n_sdf, 1.0 / 6.0)  # The linear acceleration method
    beta[dt / T > 0.551] = 0.25  # The constant average acceleration
    a1 = mass / (beta * dt ** 2) + c * gamma / (beta * dt)
    a2 = mass / (beta * dt) + c * (gamma / beta - 1)
    a3 = mass * (1 / (2 * beta) - 1) + c * dt * (gamma / (2 * beta) - 1)
    k_hat = k + a1

    # system at rest disp[:, 0] = 0.0 and vel[:, 0] = 0.0
    ac[:, 0] = (p[:, 0, np.newaxis] - c * vel[:, 0] - k * disp[:, 0]) / mass
    ac_total[:, 0] = ac[:, 0] + rec_2d[:, 0, np.newaxis]
    for i in range(n_exc - 1):
        dp = p[:, i + 1, np.newaxis] + a1 * disp[:, i] + a2 * vel[:, i] + a3 * ac[:, i]
        disp[:, i + 1] = dp / k_hat
        vel[:, i + 1] = ((gamma / (beta * dt)) * (disp[:, i + 1] - disp[:, i]) +
                         (1 - gamma / beta) * vel[:, i] + dt * ac[:, i] *
                         (1 - gamma / (2 * beta)))
        ac[:, i + 1] = ((disp[:, i + 1] - disp[:, i]) / (beta * dt ** 2) -
                        vel[:, i] / (beta * dt) - ac[:, i] * (1 / (2 * beta) - 1))
        ac_total[:, i + 1] = ac[:, i + 1] + rec_2d[:, i + 1, np.newaxis]
    return disp, vel, ac, ac_total

def get_spectra(dt: float, rec: np.ndarray, period_range: tuple[float, float, float] = (0.05, 4.05, 0.01),
           zeta: float = 0.05):
    """
    Displacement, Velocity, and Acceleratin Spectra
    """
    disp_sdf, vel_sdf, _, act_sdf = linear_analysis_sdf(dt=dt, rec=rec, period_range=period_range, zeta=zeta)
    sd = np.max(np.abs(disp_sdf), axis=1)  # max over each sdf period
    sv = np.max(np.abs(vel_sdf), axis=1)
    sa = np.max(np.abs(act_sdf), axis=1)
    return sd, sv, sa
