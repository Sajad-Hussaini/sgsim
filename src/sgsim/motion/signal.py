"""Signal processing and spectral analysis tools for ground motion data."""
import numpy as np
from numba import njit, prange
from scipy.fft import rfft, rfftfreq
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, resample as sp_resample, sosfilt
from scipy.signal.windows import tukey


__all__ = [
    "butterworth_filter",
    "baseline_correction",
    "taper",
    "resample",
    "smooth",
    
    "response_spectra",
    "sdof_response",
    "fas",
    "fps",
    "frequency",
    "time",

    "principal_angle",
    "rotate",

    "slice_energy",
    "slice_amplitude",
    "slice_freq",
    
    "integrate",
    "integrate_detrend",
    "ce",
    "cav",
    "le",
    "zc",
    "pmnm",
    "peak_abs_value",
]


# Signal Processing =============================================================

def butterworth_filter(dt, rec, lowcut=0.1, highcut=25.0, order=4):
    """
    Apply bandpass Butterworth filter using second-order sections.

    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input signal.
    lowcut : float, optional
        Low cutoff frequency (default is 0.1).
    highcut : float, optional
        High cutoff frequency (default is 25.0).
    order : int, optional
        Filter order (default is 4).

    Returns
    -------
    ndarray
        Filtered signal.
    """
    nyquist = 0.5 / dt  # Nyquist frequency
    low = lowcut / nyquist
    highcut = min(highcut, nyquist * 0.99)
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered_rec = sosfilt(sos, rec, axis=-1)
    return filtered_rec

def baseline_correction(rec, degree=1):
    """
    Remove baseline drift using polynomial fitting.

    Parameters
    ----------
    rec : ndarray
        Input signal.
    degree : int, optional
        Polynomial degree (default is 1).

    Returns
    -------
    ndarray
        Corrected signal.
    """
    rec = np.atleast_2d(rec)
    x = np.arange(rec.shape[-1])
    corrected = np.empty_like(rec)
    for i, signal in enumerate(rec):
        p = np.polynomial.Polynomial.fit(x, signal, deg=degree)
        corrected[i] = signal - p(x)
    return corrected.squeeze()

def smooth(rec: np.ndarray, window_size: int = 9) -> np.ndarray:
    """
    Apply moving average smoothing.

    Parameters
    ----------
    rec : ndarray
        Input signal.
    window_size : int, optional
        Window size (default is 9).

    Returns
    -------
    ndarray
        Smoothed signal.
    """
    return uniform_filter1d(rec, size=window_size, axis=-1, mode='constant', cval=0.0)

def taper(rec: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Apply Tukey window tapering to signal ends.

    Parameters
    ----------
    rec : ndarray
        Input signal.
    alpha : float, optional
        Taper fraction (default is 0.05).

    Returns
    -------
    ndarray
        Tapered signal.
    """
    window = tukey(rec.shape[-1], alpha=alpha)
    return rec * window

def resample(dt, dt_new, rec):
    """
    Resample signal to new time step.

    Parameters
    ----------
    dt : float
        Original time step.
    dt_new : float
        Target time step.
    rec : ndarray
        Input signal.

    Returns
    -------
    npts_new : int
        Number of points after resampling.
    dt_new : float
        New time step.
    resampled : ndarray
        Resampled signal.
    """
    npts = rec.shape[-1]
    duration = (npts - 1) * dt
    npts_new = int(np.floor(duration / dt_new)) + 1
    ac_new = sp_resample(rec, npts_new, axis=-1)
    return npts_new, dt_new, ac_new

# Signal Analysis ============================================================

@njit('float64[:, :, :, :](float64, float64[:, :], float64[:], float64, float64)', fastmath=True, parallel=True, cache=True)
def _sdof_response_kernel(dt, rec, period, zeta, mass):
    """
    Compute SDOF response time histories using Newmark method.

    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input acceleration (2D).
    period : ndarray
        SDOF natural periods.
    zeta : float
        Damping ratio.
    mass : float
        SDOF mass.

    Returns
    -------
    ndarray
        Response array (4, n_rec, npts, n_period): [Disp, Vel, Acc, Acc_total].
    """
    n_rec = rec.shape[0]
    npts = rec.shape[1]
    n_sdf = len(period)
    
    # 4D output: (response_type, n_rec, npts, n_period)
    # response_type indices: 0=disp, 1=vel, 2=acc, 3=acc_tot
    out_responses = np.empty((4, n_rec, npts, n_sdf))
    
    # Newmark Constants (Linear Acceleration Method)
    gamma = 0.5
    beta = 1.0 / 6.0
    MIN_STEPS_PER_CYCLE = 20.0 

    for j in prange(n_sdf):
        T = period[j]
        
        # Safety for T=0
        if T <= 1e-6:
            # Rigid body: Disp=0, Acc = Ground Acc
            out_responses[0, :, :, j] = 0.0 # Disp
            out_responses[1, :, :, j] = 0.0 # Vel
            # Acc relative = -Ground
            out_responses[2, :, :, j] = -rec 
            # Acc total = Acc relative + Ground = 0 relative to inertial frame? 
            # Actually Acc Total = Ground Acc for rigid structure.
            # Let's just output trivial zeros for disp/vel and handle acc:
            for r in range(n_rec):
                out_responses[3, r, :, j] = rec[r, :] # Total Acc = Ground Acc
            continue

        wn = 2 * np.pi / T
        k = mass * wn**2
        c = 2 * mass * wn * zeta
        
        # Sub-stepping Logic
        if dt > (T / MIN_STEPS_PER_CYCLE):
            n_sub = int(np.ceil(dt / (T / MIN_STEPS_PER_CYCLE)))
        else:
            n_sub = 1
        dt_sub = dt / n_sub
        
        # Newmark Coefficients
        a1 = mass / (beta * dt_sub**2) + c * gamma / (beta * dt_sub)
        a2 = mass / (beta * dt_sub) + c * (gamma / beta - 1)
        a3 = mass * (1 / (2 * beta) - 1) + c * dt_sub * (gamma / (2 * beta) - 1)
        k_hat = k + a1
        
        for r in range(n_rec):
            # Views for cleaner indexing
            disp = out_responses[0, r, :, j]
            vel = out_responses[1, r, :, j]
            acc = out_responses[2, r, :, j]
            acc_tot = out_responses[3, r, :, j]

            # --- CRITICAL FIX START ---
            # Explicitly zero out the initial state in the output array
            disp[0] = 0.0
            vel[0] = 0.0
            
            # Initial acceleration: ma + cv + kd = p  => ma = p => a = -rec[0]
            acc[0] = -rec[r, 0]
            acc_tot[0] = acc[0] + rec[r, 0] # Should be 0
            
            # Init Temp variables from these explicit zeros
            d_curr = 0.0
            v_curr = 0.0
            a_curr = acc[0]
            # --- CRITICAL FIX END ---

            for i in range(npts - 1):
                ug_start = rec[r, i]
                ug_end = rec[r, i+1]
                
                for sub in range(n_sub):
                    alpha = (sub + 1) / n_sub 
                    ug_now = ug_start + (ug_end - ug_start) * alpha
                    p_eff = -mass * ug_now
                    
                    dp = p_eff + a1 * d_curr + a2 * v_curr + a3 * a_curr
                    d_next = dp / k_hat
                    
                    v_next = ((gamma / (beta * dt_sub)) * (d_next - d_curr) +
                              (1 - gamma / beta) * v_curr +
                              dt_sub * a_curr * (1 - gamma / (2 * beta)))
                    
                    a_next = ((d_next - d_curr) / (beta * dt_sub**2) -
                              v_curr / (beta * dt_sub) -
                              a_curr * (1 / (2 * beta) - 1))
                    
                    d_curr = d_next
                    v_curr = v_next
                    a_curr = a_next

                # Save State
                disp[i+1] = d_curr
                vel[i+1] = v_curr
                acc[i+1] = a_curr
                acc_tot[i+1] = a_curr + ug_end

    return out_responses

def sdof_response(dt: float, rec: np.ndarray, period: np.ndarray, zeta: float = 0.05, mass: float = 1.0):
    """
    Compute SDOF response time histories.

    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input acceleration (1D or 2D).
    period : ndarray
        SDOF natural periods.
    zeta : float, optional
        Damping ratio (default is 0.05).
    mass : float, optional
        SDOF mass (default is 1.0).

    Returns
    -------
    disp : ndarray
        Displacement time histories.
    vel : ndarray
        Velocity time histories.
    acc : ndarray
        Relative acceleration time histories.
    acc_total : ndarray
        Total acceleration time histories.
    """
    if rec.ndim == 1:
        n = 1
        rec = rec[None, :]
    else:
        n = rec.shape[0]

    resp = _sdof_response_kernel(dt, rec, period, zeta, mass)
    
    # Unpack: (4, n_rec, npts, n_sdf) -> disp, vel, acc, acc_tot
    d, v, a, at = resp[0], resp[1], resp[2], resp[3]
    
    if n == 1:
        return d[0], v[0], a[0], at[0]
    return d, v, a, at

@njit('float64[:, :, :](float64, float64[:, :], float64[:], float64, float64)', fastmath=True, parallel=True, cache=True)
def _spectra_kernel(dt, rec, period, zeta, mass):
    """
    Compute response spectra using Newmark method.

    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input acceleration (2D).
    period : ndarray
        Spectral periods.
    zeta : float
        Damping ratio.
    mass : float
        SDOF mass.

    Returns
    -------
    ndarray
        Spectra array (3, n_rec, n_period): [SD, SV, SA].
    """
    n_rec = rec.shape[0]
    npts = rec.shape[1]
    n_sdf = period.shape[-1]
    
    # Output: (SD, SV, SA)
    spectra_vals = np.zeros((3, n_rec, n_sdf))
    
    # Constants
    gamma = 0.5
    beta = 1.0 / 6.0 
    MIN_STEPS_PER_CYCLE = 20.0 

    for j in prange(n_sdf):
        T = period[j]
        
        # SAFETY: Handle T=0 or negative periods
        if T <= 1e-6:
            # For T=0 (Rigid), Response = Ground Motion
            # SD=0, SV=0, SA = Max Ground Acc (PGA)
            for r in range(n_rec):
                pga = 0.0
                for i in range(npts):
                    val = abs(rec[r, i])
                    if val > pga: pga = val
                spectra_vals[2, r, j] = pga
            continue # Skip to next period

        wn = 2 * np.pi / T
        k = mass * wn**2
        c = 2 * mass * wn * zeta
        
        # Sub-stepping Logic
        if dt > (T / MIN_STEPS_PER_CYCLE):
            n_sub = int(np.ceil(dt / (T / MIN_STEPS_PER_CYCLE)))
        else:
            n_sub = 1
        dt_sub = dt / n_sub
        
        # Newmark Coefficients (Linear Acceleration)
        # use dt_sub for all dynamic stiffness calculations
        a1 = mass / (beta * dt_sub**2) + c * gamma / (beta * dt_sub)
        a2 = mass / (beta * dt_sub) + c * (gamma / beta - 1)
        a3 = mass * (1 / (2 * beta) - 1) + c * dt_sub * (gamma / (2 * beta) - 1)
        k_hat = k + a1
        
        for r in range(n_rec):
            # We must ensure previous state is exactly 0.0
            disp_prev = 0.0
            vel_prev = 0.0
            
            # Initial acceleration (assuming starting from rest)
            # ma + cv + kd = p  -> ma = p -> a = p/m = -ug
            acc_prev = -rec[r, 0]
            
            # Initialize Max Trackers
            sd_max = 0.0
            sv_max = 0.0
            # SA is Total Acceleration: a_rel + a_ground
            # At t=0: -rec[0] + rec[0] = 0.0
            sa_max = 0.0 
            
            for i in range(npts - 1):
                ug_start = rec[r, i]
                ug_end = rec[r, i+1]
                
                # Temp variables for sub-stepping
                d_curr = disp_prev
                v_curr = vel_prev
                a_curr = acc_prev
                
                # Sub-step Loop
                for sub in range(n_sub):
                    # Interpolate Ground Motion
                    alpha = (sub + 1) / n_sub 
                    ug_now = ug_start + (ug_end - ug_start) * alpha
                    
                    p_eff = -mass * ug_now
                    
                    # Newmark Step
                    dp = p_eff + a1 * d_curr + a2 * v_curr + a3 * a_curr
                    d_next = dp / k_hat
                    
                    v_next = ((gamma / (beta * dt_sub)) * (d_next - d_curr) +
                              (1 - gamma / beta) * v_curr +
                              dt_sub * a_curr * (1 - gamma / (2 * beta)))
                    
                    a_next = ((d_next - d_curr) / (beta * dt_sub**2) -
                              v_curr / (beta * dt_sub) -
                              a_curr * (1 / (2 * beta) - 1))
                    
                    # Update state
                    d_curr = d_next
                    v_curr = v_next
                    a_curr = a_next
                    
                    # Track Maxima (inside sub-steps for precision)
                    if abs(d_curr) > sd_max: sd_max = abs(d_curr)
                    if abs(v_curr) > sv_max: sv_max = abs(v_curr)
                    
                    # Total Acceleration = Relative Acc + Ground Acc
                    val_sa = abs(a_curr + ug_now)
                    if val_sa > sa_max: sa_max = val_sa

                # End of Sub-loop
                disp_prev = d_curr
                vel_prev = v_curr
                acc_prev = a_curr
            
            # Save final spectra values
            spectra_vals[0, r, j] = sd_max
            spectra_vals[1, r, j] = sv_max
            spectra_vals[2, r, j] = sa_max

    return spectra_vals

def response_spectra(dt: float, rec: np.ndarray, period: np.ndarray, zeta: float = 0.05):
    """
    Calculate response spectra.
    
    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input acceleration (1D or 2D).
    period : ndarray
        Spectral periods.
    zeta : float, optional
        Damping ratio (default is 0.05).

    Returns
    -------
    sd : ndarray
        Spectral displacement.
    sv : ndarray
        Spectral velocity.
    sa : ndarray
        Spectral acceleration.
    """
    if rec.ndim == 1:
        n = 1
        rec = rec[None, :]
    else:
        n = rec.shape[0]

    specs = _spectra_kernel(dt, rec, period, zeta, 1.0)
    
    sd, sv, sa = specs[0], specs[1], specs[2]

    if n == 1:
        sd = sd[0]
        sv = sv[0]
        sa = sa[0]
    return sd, sv, sa

def slice_energy(ce: np.ndarray, target_range: tuple[float, float] = (0.001, 0.999)):
    """
    Create slice for cumulative energy range.

    Parameters
    ----------
    ce : ndarray
        Cumulative energy array.
    target_range : tuple of float, optional
        Energy fraction range (default is (0.001, 0.999)).

    Returns
    -------
    slice
        Slice object.
    """
    total_energy = ce[-1]
    start_idx = np.searchsorted(ce, target_range[0] * total_energy)
    end_idx = np.searchsorted(ce, target_range[1] * total_energy)
    return slice(start_idx, end_idx + 1)

def slice_amplitude(rec: np.ndarray, threshold: float):
    """
    Create slice based on amplitude threshold.

    Parameters
    ----------
    rec : ndarray
        Input signal.
    threshold : float
        Amplitude threshold.

    Returns
    -------
    slice
        Slice object.

    Raises
    ------
    ValueError
        If no values exceed threshold.
    """
    indices = np.nonzero(np.abs(rec) > threshold)[0]
    if len(indices) == 0:
        raise ValueError("No values exceed the threshold. Consider using a lower threshold value.")
    return slice(indices[0], indices[-1] + 1)

def slice_freq(freq: np.ndarray, target_range: tuple[float, float] = (0.1, 25.0)):
    """
    Create slice for frequency range.

    Parameters
    ----------
    freq : ndarray
        Frequency array.
    target_range : tuple of float, optional
        Frequency range (default is (0.1, 25.0)).

    Returns
    -------
    slice
        Slice object.
    """
    start_idx = np.searchsorted(freq, target_range[0])
    end_idx = np.searchsorted(freq, target_range[1])
    return slice(start_idx, end_idx + 1)

def fas(dt: float, rec: np.ndarray):
    """
    Calculate Fourier amplitude spectrum.

    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Fourier amplitude spectrum.

    Notes
    -----
    Scaled by dt for seismological convention. Units are input units × time
    (e.g., g·s for acceleration in g, or m/s for acceleration in m/s²).
    """
    return np.abs(rfft(rec)) * dt

def fps(rec: np.ndarray):
    """
    Calculate Fourier phase spectrum (unwrapped).

    Parameters
    ----------
    rec : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Unwrapped phase in radians.

    Notes
    -----
    Unlike fas(), phase spectrum is independent of time step.
    """
    complex_coeffs = rfft(rec)
    return np.unwrap(np.angle(complex_coeffs))

def frequency(npts, dt):
    """
    Generate frequency array.

    Parameters
    ----------
    npts : int
        Number of points.
    dt : float
        Time step.

    Returns
    -------
    ndarray
        Frequency array.
    """
    return rfftfreq(npts, dt)

def time(npts, dt):
    """
    Generate time array.

    Parameters
    ----------
    npts : int
        Number of points.
    dt : float
        Time step.

    Returns
    -------
    ndarray
        Time array.
    """
    return np.linspace(0, (npts - 1) * dt, npts, dtype=np.float64)

def zc(rec):
    """
    Calculate zero-crossing rate.

    Parameters
    ----------
    rec : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Cumulative zero crossings.
    """
    cross_mask = rec[..., :-1] * rec[..., 1:] < 0
    cross_vec = np.empty_like(rec, dtype=np.float64)
    cross_vec[..., :-1] = cross_mask * 0.5
    cross_vec[..., -1] = cross_vec[..., -2]
    return np.cumsum(cross_vec, axis=-1)

def pmnm(rec):
    """
    Calculate positive-minima and negative-maxima count.
    
    Parameters
    ----------
    rec : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Cumulative PMNM count.
    """
    pmnm_mask =((rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) & (rec[..., 1:-1] < 0) |
               (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]) & (rec[..., 1:-1] > 0))
    pmnm_vec = np.empty_like(rec, dtype=np.float64)
    pmnm_vec[..., 1:-1] = pmnm_mask * 0.5
    pmnm_vec[..., 0] = pmnm_vec[..., 1]
    pmnm_vec[..., -1] = pmnm_vec[..., -2]
    return np.cumsum(pmnm_vec, axis=-1)

def le(rec):
    """
    Calculate local extrema count.
    
    Parameters
    ----------
    rec : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Cumulative local extrema.
    """
    mle_mask = ((rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) |
                (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]))
    mle_vec = np.empty_like(rec, dtype=np.float64)
    mle_vec[..., 1:-1] = mle_mask * 0.5
    mle_vec[..., 0] = mle_vec[..., 1]
    mle_vec[..., -1] = mle_vec[..., -2]
    return np.cumsum(mle_vec, axis=-1)

def ce(dt: float, rec: np.ndarray):
    """
    Compute cumulative energy.

    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Cumulative energy array.
    """
    return np.cumsum(rec ** 2, axis=-1) * dt

def integrate(dt: float, rec: np.ndarray):
    """
    Integrate signal using cumulative sum.

    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Integrated signal.
    """
    return np.cumsum(rec, axis=-1) * dt

def integrate_detrend(dt: float, rec: np.ndarray):
    """
    Integrate signal with linear detrending.

    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input signal.

    Returns
    -------
    ndarray
        Integrated and detrended signal.
    """
    uvec = integrate(dt, rec)
    return uvec - np.linspace(0.0, uvec[-1], len(uvec))

def peak_abs_value(rec: np.ndarray):
    """
    Calculate peak absolute value.

    Parameters
    ----------
    rec : ndarray
        Input signal.

    Returns
    -------
    float or ndarray
        Peak value.
    """
    return np.max(np.abs(rec), axis=-1)

def cav(dt: float, rec: np.ndarray):
    """
    Calculate cumulative absolute velocity.

    Parameters
    ----------
    dt : float
        Time step.
    rec : ndarray
        Input signal.

    Returns
    -------
    float or ndarray
        CAV value.
    """
    return np.sum(np.abs(rec), axis=-1) * dt

def rotate(rec1, rec2, angle):
    """
    Rotate two-component signal.

    Parameters
    ----------
    rec1 : ndarray
        First component.
    rec2 : ndarray
        Second component.
    angle : float
        Rotation angle.

    Returns
    -------
    rotated_1 : ndarray
        Rotated first component.
    rotated_2 : ndarray
        Rotated second component.
    """
    xr = rec1 * np.cos(angle) - rec2 * np.sin(angle)
    yr = rec1 * np.sin(angle) + rec2 * np.cos(angle)
    return xr, yr


def principal_angle(rec1: np.ndarray, rec2: np.ndarray) -> float:
    """
    Find rotation angle for principal axes (max/min energy).

    Computes the angle that rotates two orthogonal components to their
    principal directions, where one component has maximum energy and
    the other has minimum energy.

    Parameters
    ----------
    rec1 : ndarray
        First orthogonal component.
    rec2 : ndarray
        Second orthogonal component.

    Returns
    -------
    float
        Rotation angle in radians.

    Examples
    --------
    >>> theta = principal_angle(rec_ns, rec_ew)
    >>> rec_major, rec_minor = rotate(rec_ns, rec_ew, theta)
    """
    cross = 2 * np.sum(rec1 * rec2)
    diff = np.sum(rec1**2) - np.sum(rec2**2)
    return 0.5 * np.arctan2(cross, diff)
