import numpy as np
from numba import njit, prange

@njit('complex128[:](float64, float64, float64, float64, float64[:], float64[:])', fastmath=True, cache=True)
def get_frf(wu, zu, wl, zl, freq, freq_p2):
    """
    Frequency response function for the filter
    using the displacement and acceleration transfer function of the 2nd order system

    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency upto Nyq.
    freq_p2: freq ** 2
    """
    out = np.empty_like(freq, dtype=np.complex128)
    for i in range(len(freq)):
        denom = ((wl ** 2 - freq_p2[i]) + (2j * zl * wl * freq[i])) * \
                ((wu ** 2 - freq_p2[i]) + (2j * zu * wu * freq[i]))
        out[i] = -freq_p2[i] / denom
    return out

@njit('float64[:](float64, float64, float64, float64, float64[:], float64[:])', fastmath=True, cache=True)
def get_psd(wu, zu, wl, zl, freq_p2, freq_p4):
    """
    Non-normalized Power Spectral Density (PSD) for the filter
    using the displacement and acceleration transfer function of the 2nd order system

    wu, zu: upper angular frequency and damping ratio
    wl, zl: lower angular frequency and damping ratio
    freq: angular frequency up to Nyq.
    freq_p2: freq ** 2
    freq_p4: freq ** 4
    """
    out = np.empty_like(freq_p2)
    wu2 = wu * wu
    wu4 = wu2 * wu2
    wl2 = wl * wl
    wl4 = wl2 * wl2
    scalar_l = 2 * wl2 * (2 * zl * zl - 1)
    scalar_u = 2 * wu2 * (2 * zu * zu - 1)
    for i in range(len(freq_p2)):
        val_p2 = freq_p2[i]
        val_p4 = freq_p4[i]
        denom = (wl4 + val_p4 + scalar_l * val_p2) * \
                (wu4 + val_p4 + scalar_u * val_p2)
        out[i] = val_p4 / denom
    return out

@njit('Tuple((float64[:], float64[:], float64[:], float64[:], float64[:]))(float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)', parallel=True, fastmath=True, cache=True)
def get_stats(wu, zu, wl, zl, freq_p2, freq_p4, freq_n2, freq_n4, dw):
    """
    The evolutionary statistics of the stochastic model using Power Spectral Density (PSD)
    Optimized with Kernel Fusion (inlined PSD calculation) to avoid array allocations.
    """
    n = len(wu)
    variance = np.empty(n)
    variance_dot = np.empty(n)
    variance_2dot = np.empty(n)
    variance_bar = np.empty(n)
    variance_2bar = np.empty(n)
    scale = 2 * dw
    
    for i in prange(n):
        wui = wu[i]
        zui = zu[i]
        wli = wl[i]
        zli = zl[i]
        wu2 = wui * wui
        wu4 = wu2 * wu2
        wl2 = wli * wli
        wl4 = wl2 * wl2
        scalar_l = 2 * wl2 * (2 * zli * zli - 1)
        scalar_u = 2 * wu2 * (2 * zui * zui - 1)
        # Accumulators
        var, var_dot, var_2dot, var_bar, var_2bar = 0.0, 0.0, 0.0, 0.0, 0.0
        # 2. Single pass loop: Calculate PSD value and stats simultaneously
        for j in range(len(freq_p2)):
            val_p2 = freq_p2[j]
            val_p4 = freq_p4[j]
            # Inline PSD Calculation
            denom = (wl4 + val_p4 + scalar_l * val_p2) * \
                    (wu4 + val_p4 + scalar_u * val_p2)   
            psd_val = val_p4 / denom
            # Accumulate
            var += psd_val
            var_dot += val_p2 * psd_val
            var_2dot += val_p4 * psd_val
            var_bar += freq_n2[j] * psd_val
            var_2bar += freq_n4[j] * psd_val
        # Final scaling
        variance[i] = var * scale
        variance_dot[i] = var_dot * scale
        variance_2dot[i] = var_2dot * scale
        variance_bar[i] = var_bar * scale
        variance_2bar[i] = var_2bar * scale
    return variance, variance_dot, variance_2dot, variance_bar, variance_2bar

@njit('float64[:](float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64)', fastmath=True, cache=True)
def get_fas(mdl, wu, zu, wl, zl, freq_p2, freq_p4, variance, dt):
    """
    The Fourier amplitude spectrum (FAS) of the stochastic model using PSD
    Optimized with Kernel Fusion to eliminate temporary array allocations inside the loop.
    """
    fas = np.zeros_like(freq_p2, dtype=np.float64)
    final_scale = dt * 2 * np.pi
    for i in range(len(wu)):
        wui = wu[i]
        zui = zu[i]
        wli = wl[i]
        zli = zl[i]
        wu2 = wui * wui
        wu4 = wu2 * wu2
        wl2 = wli * wli
        wl4 = wl2 * wl2
        scalar_l = 2 * wl2 * (2 * zli * zli - 1)
        scalar_u = 2 * wu2 * (2 * zui * zui - 1)
        # Calculate the amplitude scaling factor for this time step
        scale = (mdl[i] * mdl[i]) / variance[i]
        # 2. Inner Loop: Compute PSD scalar and add directly to FAS
        for j in range(len(freq_p2)):
            val_p2 = freq_p2[j]
            val_p4 = freq_p4[j]
            denom = (wl4 + val_p4 + scalar_l * val_p2) * \
                    (wu4 + val_p4 + scalar_u * val_p2)
            psd_val = val_p4 / denom
            # Accumulate directly
            fas[j] += scale * psd_val
    
    # Final conversion to magnitude
    for j in range(len(fas)):
        fas[j] = np.sqrt(fas[j] * final_scale)
    return fas

@njit('complex128[:, :](int64, int64, float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:], float64[:, :], float64)', parallel=True, fastmath=True, cache=True)
def simulate_fourier_series(n, npts, t, freq_sim, freq_sim_p2, mdl, wu, zu, wl, zl, variance, white_noise, dt):
    """
    The Fourier series of n number of simulations
    """
    n_freq = len(freq_sim)
    fourier = np.zeros((n, n_freq), dtype=np.complex128)
    discrete_correction = np.sqrt(2 * np.pi / dt)
    transfer_vec = np.empty(n_freq, dtype=np.complex128)
    for i in range(npts):
        # --- PHASE 1: COMPUTE PHYSICS FOR TIME STEP i ---
        ti = t[i]
        scalei = (mdl[i] / np.sqrt(variance[i])) * discrete_correction 
        wui, zui = wu[i], zu[i]
        wli, zli = wl[i], zl[i]
        wu2, wl2 = wui*wui, wli*wli
        for k in range(n_freq):
            w = freq_sim[k]
            w2 = freq_sim_p2[k]
            denom = ((wl2 - w2) + (2j * zli * wli * w)) * \
                    ((wu2 - w2) + (2j * zui * wui * w))
            frf_val = -w2 / denom            
            # Inline Exp Math: exp(-j * w * t)
            # cos/sin is often slightly faster/cleaner for purely imaginary exp
            arg = w * ti
            exp_val = np.cos(arg) - 1j * np.sin(arg)
            
            transfer_vec[k] = frf_val * exp_val * scalei

        # --- PHASE 2: DISTRIBUTE TO SIMULATIONS ---
        # Now apply this vector to all simulations in parallel
        for sim in prange(n):
            noise = white_noise[sim, i]
            # Add to the specific simulation row
            for k in range(n_freq):
                fourier[sim, k] += transfer_vec[k] * noise

    return fourier

@njit('float64[:](float64, float64[:], float64[:])', fastmath=True, cache=True)
def cumulative_rate(dt, numerator, denominator):
    scale = dt / (2 * np.pi)
    cumsum = 0.0
    out = np.empty_like(numerator, dtype=np.float64)
    for i in range(len(numerator)):
        cumsum += np.sqrt(numerator[i] / denominator[i]) * scale
        out[i] = cumsum
    return out

@njit('float64[:](float64, float64[:], float64[:], float64[:])', fastmath=True, cache=True)
def pmnm_rate(dt, first, middle, last):
    scale = dt / (4 * np.pi)
    cumsum = 0.0
    out = np.empty_like(first, dtype=np.float64)
    for i in range(len(first)):
        cumsum += (np.sqrt(first[i] / middle[i]) - np.sqrt(middle[i] / last[i])) * scale
        out[i] = cumsum
    return out