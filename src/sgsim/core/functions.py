"""
Parametric functions for stochastic ground motion simulation.

This module provides pure functions for time-varying parametric models.
Use `functools.partial` to bind parameters before passing to ModelConfig.

Examples
--------
>>> from functools import partial
>>> from sgsim.core.functions import beta_single, linear, constant
>>> from sgsim.core.model_config import ModelConfig
>>>
>>> config = ModelConfig(
...     npts=4000,
...     dt=0.01,
...     modulating=partial(beta_single, peak=0.3, concentration=5.0, energy=100.0, duration=40.0),
...     upper_frequency=partial(linear, start=10.0, end=5.0),
...     upper_damping=partial(constant, c=0.3),
...     lower_frequency=partial(constant, c=0.1),
...     lower_damping=partial(constant, c=0.5),
... )
"""
import numpy as np
from scipy.special import betaln

__all__ = [
    "beta_basic",
    "beta_single",
    "beta_dual",
    "gamma",
    "housner",
    "linear",
    "bilinear",
    "exponential",
    "constant",]


# =============================================================================
# Modulating Functions (Envelope shapes for ground motion)
# =============================================================================

def beta_basic(t: np.ndarray, peak: float, concentration: float,
               energy: float, duration: float) -> np.ndarray:
    """
    Basic Beta modulating function for earthquake ground motion simulation.
    
    Provides a smooth envelope function based on the Beta distribution,
    suitable for modeling single-phase earthquake strong motion.

    Parameters
    ----------
    t : ndarray
        Time array.
    peak : float
        Peak location as fraction of duration (0 < peak < 1).
    concentration : float
        Concentration parameter controlling sharpness (> 0).
    energy : float
        Total energy under the envelope (> 0).
    duration : float
        Total duration of the function (> 0).
    
    Returns
    -------
    ndarray
        Modulating function values at each time point.

    See Also
    --------
    beta_single : Beta function with weak motion baseline.
    beta_dual : Beta function with two strong phases.

    References
    ----------  
    Hussaini SS, Karimzadeh S, Rezaeian S, Lourenço PB. Broadband stochastic
    simulation of earthquake ground motions with multiple strong phases.
    Earthquake Spectra. 2025;41(3):2399-2435.
    """
    mdl = np.zeros(len(t))
    mdl[1:-1] = np.exp((concentration * peak) * np.log(t[1:-1]) +
                       (concentration * (1 - peak)) * np.log(duration - t[1:-1]) -
                       betaln(1 + concentration * peak, 1 + concentration * (1 - peak)) -
                       (1 + concentration) * np.log(duration))
    return np.sqrt(energy * mdl)

def beta_single(t: np.ndarray, peak: float, concentration: float,
                energy: float, duration: float) -> np.ndarray:
    """
    Beta modulating function with weak motion baseline.
    
    Combines a parabolic weak motion component (5% energy) with a Beta
    distribution strong motion component (95% energy) for realistic
    earthquake ground motion envelopes.

    Parameters
    ----------
    t : ndarray
        Time array.
    peak : float
        Peak location as fraction of duration (0 < peak < 1).
    concentration : float
        Concentration parameter controlling sharpness (> 0).
    energy : float
        Total energy under the envelope (> 0).
    duration : float
        Total duration of the function (> 0).
    
    Returns
    -------
    ndarray
        Modulating function values at each time point.

    See Also
    --------
    beta_basic : Pure Beta function without weak motion.
    beta_dual : Beta function with two strong phases.

    References
    ----------  
    Hussaini SS, Karimzadeh S, Rezaeian S, Lourenço PB. Broadband stochastic
    simulation of earthquake ground motions with multiple strong phases.
    Earthquake Spectra. 2025;41(3):2399-2435.
    """
    mdl = np.zeros(len(t))
    mdl[1:-1] += 0.05 * (6 * (t[1:-1] * (duration - t[1:-1])) / (duration ** 3))
    mdl[1:-1] += 0.95 * np.exp((concentration * peak) * np.log(t[1:-1]) +
                               (concentration * (1 - peak)) * np.log(duration - t[1:-1]) -
                               betaln(1 + concentration * peak, 1 + concentration * (1 - peak)) -
                               (1 + concentration) * np.log(duration))
    return np.sqrt(energy * mdl)

def beta_dual(t: np.ndarray, peak: float, concentration: float,
              peak_2: float, concentration_2: float, energy_ratio: float,
              energy: float, duration: float) -> np.ndarray:
    """
    Beta modulating function with two distinct strong phases.
    
    Models earthquakes with multiple strong motion packets, combining weak
    motion baseline (5% energy) with two independent Beta distributions
    representing separate strong motion phases.

    Parameters
    ----------
    t : ndarray
        Time array.
    peak : float
        Peak location of first strong phase as fraction of duration (0 < peak < 1).
    concentration : float
        Concentration parameter of first phase (> 0).
    peak_2 : float
        Peak location of second strong phase as fraction of duration (0 < peak_2 < 1).
    concentration_2 : float
        Concentration parameter of second phase (> 0).
    energy_ratio : float
        Energy fraction allocated to first strong phase (0 < energy_ratio < 0.95).
    energy : float
        Total energy under the envelope (> 0).
    duration : float
        Total duration of the function (> 0).
    
    Returns
    -------
    ndarray
        Modulating function values at each time point.

    See Also
    --------
    beta_basic : Single Beta function without weak motion.
    beta_single : Single strong phase with weak motion.

    References
    ----------  
    Hussaini SS, Karimzadeh S, Rezaeian S, Lourenço PB. Broadband stochastic
    simulation of earthquake ground motions with multiple strong phases.
    Earthquake Spectra. 2025;41(3):2399-2435.
    """
    mdl = np.zeros(len(t))
    mdl[1:-1] += 0.05 * (6 * (t[1:-1] * (duration - t[1:-1])) / (duration ** 3))
    mdl[1:-1] += energy_ratio * np.exp((concentration * peak) * np.log(t[1:-1]) +
                                       (concentration * (1 - peak)) * np.log(duration - t[1:-1]) -
                                       betaln(1 + concentration * peak, 1 + concentration * (1 - peak)) -
                                       (1 + concentration) * np.log(duration))
    mdl[1:-1] += (0.95 - energy_ratio) * np.exp((concentration_2 * peak_2) * np.log(t[1:-1]) +
                                                 (concentration_2 * (1 - peak_2)) * np.log(duration - t[1:-1]) -
                                                 betaln(1 + concentration_2 * peak_2, 1 + concentration_2 * (1 - peak_2)) -
                                                 (1 + concentration_2) * np.log(duration))
    return np.sqrt(energy * mdl)

def gamma(t: np.ndarray, scale: float, shape: float, decay: float) -> np.ndarray:
    """
    Gamma distribution modulating function.
    
    Classical envelope function for earthquake ground motion based on
    Gamma distribution with exponential decay.

    Parameters
    ----------
    t : ndarray
        Time array.
    scale : float
        Amplitude scaling factor (> 0).
    shape : float
        Shape parameter controlling rise time (> 0).
    decay : float
        Decay rate parameter (> 0).
    
    Returns
    -------
    ndarray
        Modulating function values at each time point.

    See Also
    --------
    beta_single : Alternative Beta-based envelope.
    housner : Piecewise envelope function.
    """
    return scale * t ** shape * np.exp(-decay * t)

def housner(t: np.ndarray, amplitude: float, decay: float, shape: float, 
            tp: float, td: float) -> np.ndarray:
    """
    Housner piecewise modulating function.
    
    Three-phase envelope function: quadratic rise, constant plateau,
    and exponential decay. Classic model for earthquake strong motion.

    Parameters
    ----------
    t : ndarray
        Time array.
    amplitude : float
        Constant amplitude during plateau phase (> 0).
    decay : float
        Decay rate during tail phase (> 0).
    shape : float
        Decay shape exponent (> 0).
    tp : float
        Time to reach peak amplitude (> 0).
    td : float
        Time to start decay phase (td > tp).
    
    Returns
    -------
    ndarray
        Modulating function values at each time point.

    See Also
    --------
    gamma : Alternative smooth envelope.
    beta_single : Beta-based envelope function.
    """
    return np.piecewise(t, [(t >= 0) & (t < tp), (t >= tp) & (t <= td), t > td],
                        [lambda t_val: amplitude * (t_val / tp) ** 2, amplitude,
                        lambda t_val: amplitude * np.exp(-decay * ((t_val - td) ** shape))])

# =============================================================================
# Interpolation Functions (Time-varying filter parameters)
# =============================================================================

def linear(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """
    Linear interpolation function.
    
    Provides linear transition between start and end values over
    the time domain.

    Parameters
    ----------
    t : ndarray
        Time array.
    start : float
        Starting value at t=0.
    end : float
        Ending value at t=max(t).
    
    Returns
    -------
    ndarray
        Linearly interpolated values at each time point.

    See Also
    --------
    bilinear : Piecewise linear with midpoint.
    exponential : Exponential interpolation.
    """
    return start + (end - start) * (t / t.max())


def bilinear(t: np.ndarray, start: float, mid: float, end: float, t_mid: float) -> np.ndarray:
    """
    Piecewise linear interpolation function.
    
    Provides two-segment linear transition through a specified midpoint,
    useful for modeling parameters with intermediate changes.

    Parameters
    ----------
    t : ndarray
        Time array.
    start : float
        Starting value at t=0.
    mid : float
        Value at midpoint time.
    end : float
        Ending value at t=max(t).
    t_mid : float
        Time at midpoint (0 < t_mid < max(t)).
    
    Returns
    -------
    ndarray
        Piecewise linearly interpolated values at each time point.

    See Also
    --------
    linear : Simple linear interpolation.
    exponential : Smooth exponential transition.
    """
    t_max = t.max()
    return np.piecewise(t, [t <= t_mid, t > t_mid],
                        [lambda t_val: start - (start - mid) * t_val / t_mid,
                        lambda t_val: mid - (mid - end) * (t_val - t_mid) / (t_max - t_mid)])

def exponential(t: np.ndarray, start: float, end: float) -> np.ndarray:
    """
    Exponential interpolation function.
    
    Provides smooth exponential transition between start and end values,
    useful for parameters varying over orders of magnitude.

    Parameters
    ----------
    t : ndarray
        Time array.
    start : float
        Starting value at t=0 (> 0).
    end : float
        Ending value at t=max(t) (> 0).
    
    Returns
    -------
    ndarray
        Exponentially interpolated values at each time point.

    See Also
    --------
    linear : Linear interpolation.
    bilinear : Piecewise linear with midpoint.
    """
    return start * np.exp(np.log(end / start) * (t / t.max()))

def constant(t: np.ndarray, c: float) -> np.ndarray:
    """
    Constant value function.
    
    Provides time-invariant parameter values throughout the time domain.

    Parameters
    ----------
    t : ndarray
        Time array.
    c : float
        Constant value for all time points.
    
    Returns
    -------
    ndarray
        Array of constant values with same length as t.

    See Also
    --------
    linear : Time-varying linear function.
    """
    return np.full(len(t), c)
