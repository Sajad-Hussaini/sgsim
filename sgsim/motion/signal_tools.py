import numpy as np
from scipy.signal import butter, sosfilt, resample as sp_resample
from scipy.fft import rfft, rfftfreq
from numba import njit, prange

# Signal processing functions

def bandpass_filter(dt, rec, lowcut=0.1, highcut=25.0, order=4):
    """
    Apply a band-pass Butterworth filter to remove low-frequency drift.
    """
    nyquist = 0.5 / dt  # Nyquist frequency
    low = lowcut / nyquist
    highcut = min(highcut, nyquist * 0.99)
    high = highcut / nyquist
    sos = butter(order, [low, high], btype='band', output='sos')
    n = len(rec)
    next_pow2 = int(2 ** np.ceil(np.log2(2 * n)))
    pad_width = next_pow2 - n
    signal_padded = np.pad(rec, (pad_width // 2, pad_width - pad_width // 2), mode='constant')
    filtered_rec = sosfilt(sos, signal_padded)
    filtered_rec = filtered_rec[pad_width // 2: -(pad_width - pad_width // 2)]
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
    Perform a moving average smoothing on the input records.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size should be odd.")
    window = np.ones(window_size) / window_size
    if rec.ndim == 1:
        smoothed_rec = np.convolve(rec, window, mode='same')
    elif rec.ndim == 2:
        smoothed_rec = np.apply_along_axis(lambda x: np.convolve(x, window, mode='same'), axis=1, arr=rec)
    else:
        raise ValueError("Input must be a 1D or 2D array.")
    return smoothed_rec

def resample(dt, dt_new, rec):
    """
    resample a time series from an original time step dt to a new one dt_new.
    """
    npts = len(rec)
    duration = (npts - 1) * dt
    npts_new = int(np.floor(duration / dt_new)) + 1
    ac_new = sp_resample(rec, npts_new)
    return npts_new, dt_new, ac_new

# Properties of the signal

def get_mzc(rec):
    """
    The mean cumulative number of zero up and down crossings
    """
    cross_mask = rec[..., :-1] * rec[..., 1:] < 0
    cross_vec = np.empty_like(rec, dtype=np.float64)
    cross_vec[..., :-1] = cross_mask * 0.5
    cross_vec[..., -1] = cross_vec[..., -2]
    return np.cumsum(cross_vec, axis=-1)

def get_pmnm(rec):
    """
    The mean cumulative number of positive-minima and negative-maxima
    """
    pmnm_mask =((rec[..., :-2] < rec[..., 1:-1]) & (rec[..., 1:-1] > rec[..., 2:]) & (rec[..., 1:-1] < 0) |
               (rec[..., :-2] > rec[..., 1:-1]) & (rec[..., 1:-1] < rec[..., 2:]) & (rec[..., 1:-1] > 0))
    pmnm_vec = np.empty_like(rec, dtype=np.float64)
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
    mle_vec = np.empty_like(rec, dtype=np.float64)
    mle_vec[..., 1:-1] = mle_mask * 0.5
    mle_vec[..., 0] = mle_vec[..., 1]
    mle_vec[..., -1] = mle_vec[..., -2]
    return np.cumsum(mle_vec, axis=-1)

@njit('float64[:, :, :, :](float64, float64[:, :], float64[:], float64, float64)', fastmath=True, parallel=True, cache=True)
def run_sdof_linear(dt, rec, period, zeta, mass):
    """
    Computes full TIME HISTORIES (Disp, Vel, Acc) for SDOF systems.
    Useful for visualizing the vibration of a specific structure.
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

@njit('float64[:, :, :](float64, float64[:, :], float64[:], float64, float64)', fastmath=True, parallel=True, cache=True)
def calc_spectra_kernel(dt, rec, period, zeta, mass):
    """
    Computes Response Spectra (SD, SV, SA).
    
    Returns:
    --------
    spectra : 3D array (3, n_rec, n_period) -> [SD, SV, SA]
    """
    n_rec = rec.shape[0]
    npts = rec.shape[1]
    n_sdf = len(period)
    
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

def get_spectra(dt: float, rec: np.ndarray, period: np.ndarray, zeta: float = 0.05):
    """
    Calculates response spectra (SD, SV, SA).
    """
    if rec.ndim == 1:
        rec = rec[None, :]
        
    # The kernel returns (3, n_rec, n_period)
    spectra_array = calc_spectra_kernel(dt, rec, period, zeta, 1.0)
    
    sd = spectra_array[0]
    sv = spectra_array[1]
    sa = spectra_array[2]

    return sd, sv, sa

def slice_energy(ce: np.ndarray, target_range: tuple[float, float] = (0.001, 0.999)):
    " A slicer of the input motion using a target cumulative energy range (as a fraction of total energy) "
    total_energy = ce[-1]
    start_idx = np.searchsorted(ce, target_range[0] * total_energy)
    end_idx = np.searchsorted(ce, target_range[1] * total_energy)
    return slice(start_idx, end_idx + 1)

def slice_amplitude(rec: np.ndarray, threshold: float):
    " A slicer of the input motion using an amplitude threshold. "
    indices = np.nonzero(np.abs(rec) > threshold)[0]
    if len(indices) == 0:
        raise ValueError("No values exceed the threshold. Consider using a lower threshold value.")
    return slice(indices[0], indices[-1] + 1)

def slice_freq(freq: np.ndarray, target_range: tuple[float, float] = (0.1, 25.0)):
    " A slicer of the frequencies using a frequency range in Hz"
    start_idx = np.searchsorted(freq, target_range[0] * 2 * np.pi)
    end_idx = np.searchsorted(freq, target_range[1] * 2 * np.pi)
    return slice(start_idx, end_idx + 1)

def get_ce(dt: float, rec: np.ndarray):
    """
    Compute the cumulative energy
    """
    return np.cumsum(rec ** 2, axis=-1) * dt

def get_integral(dt: float, rec: np.ndarray):
    """
    Compute the velocity of an acceleration input
    """
    return np.cumsum(rec, axis=-1) * dt

def get_integral_detrend(dt: float, rec: np.ndarray):
    """
    Compute the integral with linear detrending
    """
    uvec = get_integral(dt, rec)
    return uvec - np.linspace(0.0, uvec[-1], len(uvec))

def get_peak_param(rec: np.ndarray):
    " Peak ground motion parameter"
    return np.max(np.abs(rec), axis=-1)

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
    return np.linspace(0, (npts - 1) * dt, npts, dtype=np.float64)

def get_magnitude(rec1, rec2):
    " magnitude of a vector that is indepednent of coordinate system"
    return np.sqrt(np.abs(rec1) ** 2 + np.abs(rec2) ** 2)

def get_angle(rec1, rec2):
    " angle of a vector that is depednent on coordinate system"
    return np.unwrap(np.arctan2(rec2, rec1))

def get_turning_rate(dt, rec1, rec2):
    " turning rate or angular velocity of a vector that is indepednent of coordinate system"
    anlges = get_angle(rec1, rec2)
    if len(anlges.shape) == 1:
        return np.diff(anlges, prepend=anlges[0]) / dt
    else:
        return np.diff(anlges, prepend=anlges[..., 0][:, None]) / dt

def rotate_records(rec1, rec2, angle):
    " rotated components in the new coordinate system"
    xr = rec1 * np.cos(angle) - rec2 * np.sin(angle)
    yr = rec1 * np.sin(angle) + rec2 * np.cos(angle)
    return xr, yr

def get_correlation(rec1, rec2):
    " correlation between two signals"
    return np.sum(rec1 * rec2) / np.sqrt(np.sum(rec1 ** 2) * np.sum(rec2 ** 2))