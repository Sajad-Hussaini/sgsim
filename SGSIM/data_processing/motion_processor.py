import numpy as np

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
    if np.ndim(rec) == 1:
        rec = rec[np.newaxis, :]  # Convert to 2D array with one row for consistency
    p = -mass * rec if excitation == 'GM' else rec
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
    ac_total[:, 0] = ac[:, 0] + rec[:, 0, np.newaxis]
    for i in range(n_exc - 1):
        dp = p[:, i + 1, np.newaxis] + a1 * disp[:, i] + a2 * vel[:, i] + a3 * ac[:, i]
        disp[:, i + 1] = dp / k_hat
        vel[:, i + 1] = ((gamma / (beta * dt)) * (disp[:, i + 1] - disp[:, i]) +
                         (1 - gamma / beta) * vel[:, i] + dt * ac[:, i] *
                         (1 - gamma / (2 * beta)))
        ac[:, i + 1] = ((disp[:, i + 1] - disp[:, i]) / (beta * dt ** 2) -
                        vel[:, i] / (beta * dt) - ac[:, i] * (1 / (2 * beta) - 1))
        ac_total[:, i + 1] = ac[:, i + 1] + rec[:, i + 1, np.newaxis]
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