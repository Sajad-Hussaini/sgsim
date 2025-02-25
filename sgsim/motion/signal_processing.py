import numpy as np
from scipy.signal import butter, sosfilt, resample

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

def upsample(dt, dt_new, rec):
    """
    Upsample a time series from an original time step dt to a finer dt_new.
    """
    if dt_new >= dt:
        raise ValueError("The new time step should be smaller than the original time step.")
    npts = len(rec)
    duration = (npts - 1) * dt
    npts_new = int(np.floor(duration / dt_new)) + 1
    ac_new = resample(rec, npts_new)
    return npts_new, dt_new, ac_new
