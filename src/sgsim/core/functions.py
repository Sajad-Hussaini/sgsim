"""
Parametric functions for stochastic ground motion simulation.

Each class exposes its required parameter names via the `param_names` property for easy introspection and automation.

Examples
--------
>>> import numpy as np
>>> from sgsim.functions import BetaSingle, Linear, Constant
>>> t = np.linspace(0, 40, 4000)
>>> # Functional (stateless) usage
>>> y_env = BetaSingle.compute(t, peak=0.3, concentration=5.0, energy=100.0, duration=40.0)
>>> y_freq = Linear.compute(t, start=10.0, end=5.0)
>>> y_damp = Constant.compute(t, value=0.3)
>>>
>>> # Object-oriented (stateful) usage
>>> env = BetaSingle()
>>> y_env2 = env(t, peak=0.3, concentration=5.0, energy=100.0, duration=40.0)
>>> print(env.params)  # {'peak': 0.3, 'concentration': 5.0, 'energy': 100.0, 'duration': 40.0}
"""
import numpy as np
from scipy.special import betaln
from abc import ABC, abstractmethod

__all__ = [
    "BetaBasic",
    "BetaSingle",
    "BetaDual",
    "Gamma",
    "Housner",
    "Linear",
    "Bilinear",
    "Exponential",
    "Constant"
    ]

class ParametricFunction(ABC):
    """
    Abstract base class for parametric functions.
    
    Provides interface for time-varying parametric functions used in
    stochastic ground motion simulation.
    
    Attributes
    ----------
    params : dict
        Dictionary of function parameters.
    value : ndarray
        Last computed function values.
    """
    _pnames = []
    def __init__(self):
        self.params = {k: None for k in self._pnames}

    @property
    def param_names(self):
        return list(self._pnames)

    @property
    def n_params(self):
        return len(self._pnames)

    @abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @staticmethod
    @abstractmethod
    def compute(*args, **kwargs):
        pass

    def __repr__(self):
        p = getattr(self, "params", {})
        param_str = ', '.join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in p.items())
        return f"{self.__class__.__name__}({param_str})"


class BetaBasic(ParametricFunction):
    """
    Basic Beta modulating function for earthquake ground motion simulation.
    
    Provides a smooth envelope function based on the Beta distribution,
    suitable for modeling single-phase earthquake strong motion.

    Parameters
    ----------
    peak : float
        Peak location as fraction of duration (0 < peak < 1).
    concentration : float
        Concentration parameter controlling sharpness (> 0).
    energy : float
        Total energy under the envelope (> 0).
    duration : float
        Total duration of the function (> 0).
    
    See Also
    --------
    BetaSingle : Beta function with weak motion baseline.
    BetaDual : Beta function with two strong phases.

    References
    ----------  
    Hussaini SS, Karimzadeh S, Rezaeian S, Lourenço PB. Broadband stochastic
    simulation of earthquake ground motions with multiple strong phases.
    Earthquake Spectra. 2025;41(3):2399-2435.
    """
    _pnames = ['peak', 'concentration', 'energy', 'duration']
    def __call__(self, t, peak, concentration, energy, duration):
        self.params = dict(peak=peak, concentration=concentration, energy=energy, duration=duration)
        return self.compute(t, peak, concentration, energy, duration)
    
    @staticmethod
    def compute(t, peak, concentration, energy, duration):
        mdl = np.zeros(len(t))
        mdl[1:-1] = np.exp((concentration * peak) * np.log(t[1:-1]) +
                             (concentration * (1 - peak)) * np.log(duration - t[1:-1]) -
                             betaln(1 + concentration * peak, 1 + concentration * (1 - peak)) -
                             (1 + concentration) * np.log(duration))
        return np.sqrt(energy * mdl)

class BetaSingle(ParametricFunction):
    """
    Beta modulating function with weak motion baseline.
    
    Combines a parabolic weak motion component (5% energy) with a Beta
    distribution strong motion component (95% energy) for realistic
    earthquake ground motion envelopes.

    Parameters
    ----------
    peak : float
        Peak location as fraction of duration (0 < peak < 1).
    concentration : float
        Concentration parameter controlling sharpness (> 0).
    energy : float
        Total energy under the envelope (> 0).
    duration : float
        Total duration of the function (> 0).
    
    See Also
    --------
    BetaBasic : Pure Beta function without weak motion.
    BetaDual : Beta function with two strong phases.

    References
    ----------  
    Hussaini SS, Karimzadeh S, Rezaeian S, Lourenço PB. Broadband stochastic
    simulation of earthquake ground motions with multiple strong phases.
    Earthquake Spectra. 2025;41(3):2399-2435.
    """
    _pnames = ['peak', 'concentration', 'energy', 'duration']
    def __call__(self, t, peak, concentration, energy, duration):
        self.params = dict(peak=peak, concentration=concentration, energy=energy, duration=duration)
        return self.compute(t, peak, concentration, energy, duration)
    
    @staticmethod
    def compute(t, peak, concentration, energy, duration):
        mdl = np.zeros(len(t))
        mdl[1:-1] += 0.05 * (6 * (t[1:-1] * (duration - t[1:-1])) / (duration ** 3))
        mdl[1:-1] += 0.95 * np.exp((concentration * peak) * np.log(t[1:-1]) +
                             (concentration * (1 - peak)) * np.log(duration - t[1:-1]) -
                             betaln(1 + concentration * peak, 1 + concentration * (1 - peak)) -
                             (1 + concentration) * np.log(duration))
        return np.sqrt(energy * mdl)

class BetaDual(ParametricFunction):
    """
    Beta modulating function with two distinct strong phases.
    
    Models earthquakes with multiple strong motion packets, combining weak
    motion baseline (5% energy) with two independent Beta distributions
    representing separate strong motion phases.

    Parameters
    ----------
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
    
    See Also
    --------
    BetaBasic : Single Beta function without weak motion.
    BetaSingle : Single strong phase with weak motion.

    References
    ----------  
    Hussaini SS, Karimzadeh S, Rezaeian S, Lourenço PB. Broadband stochastic
    simulation of earthquake ground motions with multiple strong phases.
    Earthquake Spectra. 2025;41(3):2399-2435.
    """
    _pnames = ['peak', 'concentration', 'peak_2', 'concentration_2', 'energy_ratio', 'energy', 'duration']
    def __call__(self, t, peak, concentration, peak_2, concentration_2, energy_ratio, energy, duration):
        self.params = dict(peak=peak, concentration=concentration,
                           peak_2=peak_2, concentration_2=concentration_2,
                           energy_ratio=energy_ratio, energy=energy, duration=duration)
        return self.compute(t, peak, concentration, peak_2, concentration_2, energy_ratio, energy, duration)
    
    @staticmethod
    def compute(t, peak, concentration, peak_2, concentration_2, energy_ratio, energy, duration):
        # Original formula:
        # mdl1 = 0.05 * (6 * (t * (duration - t)) / (duration ** 3))
        # mdl2 = energy_ratio * ((t ** (concentration * peak) * (duration - t) ** (concentration * (1 - peak))) / (beta(1 + concentration * peak, 1 + concentration * (1 - peak)) * duration ** (1 + concentration)))
        # mdl3 = (1 - 0.05 - energy_ratio) * ((t ** (concentration_2 * peak_2) * (duration - t) ** (concentration_2 * (1 - peak_2))) / (beta(1 + concentration_2 * peak_2, 1 + concentration_2 * (1 - peak_2)) * duration ** (1 + concentration_2)))
        # multi_mdl = mdl1 + mdl2 + mdl3
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

class Gamma(ParametricFunction):
    """
    Gamma distribution modulating function.
    
    Classical envelope function for earthquake ground motion based on
    Gamma distribution with exponential decay.

    Parameters
    ----------
    scale : float
        Amplitude scaling factor (> 0).
    shape : float
        Shape parameter controlling rise time (> 0).
    decay : float
        Decay rate parameter (> 0).
    
    See Also
    --------
    BetaSingle : Alternative Beta-based envelope.
    Housner : Piecewise envelope function.
    """
    _pnames = ['scale', 'shape', 'decay']
    def __call__(self, t, scale, shape, decay):
        self.params = dict(scale=scale, shape=shape, decay=decay)
        return self.compute(t, scale, shape, decay)
    
    @staticmethod
    def compute(t, scale, shape, decay):
        return scale * t ** shape * np.exp(-decay * t)

class Housner(ParametricFunction):
    """
    Housner piecewise modulating function.
    
    Three-phase envelope function: quadratic rise, constant plateau,
    and exponential decay. Classic model for earthquake strong motion.

    Parameters
    ----------
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
    
    See Also
    --------
    Gamma : Alternative smooth envelope.
    BetaSingle : Beta-based envelope function.
    """
    _pnames = ['amplitude', 'decay', 'shape', 'tp', 'td']
    def __call__(self, t, amplitude, decay, shape, tp, td):
        self.params = dict(amplitude=amplitude, decay=decay, shape=shape, tp=tp, td=td)
        return self.compute(t, amplitude, decay, shape, tp, td)
    
    @staticmethod
    def compute(t, amplitude, decay, shape, tp, td):
        return np.piecewise(t, [(t >= 0) & (t < tp), (t >= tp) & (t <= td), t > td],
                            [lambda t_val: amplitude * (t_val / tp) ** 2, amplitude,
                             lambda t_val: amplitude * np.exp(-decay * ((t_val - td) ** shape))])

class Linear(ParametricFunction):
    """
    Linear interpolation function.
    
    Provides linear transition between start and end values over
    the time domain.

    Parameters
    ----------
    start : float
        Starting value at t=0.
    end : float
        Ending value at t=max(t).
    
    See Also
    --------
    Bilinear : Piecewise linear with midpoint.
    Exponential : Exponential interpolation.
    """
    _pnames = ['start', 'end']
    def __call__(self, t, start, end):
        self.params = dict(start=start, end=end)
        return self.compute(t, start, end)
    
    @staticmethod
    def compute(t, start, end):
        return start + (end - start) * (t / t.max())

class Bilinear(ParametricFunction):
    """
    Piecewise linear interpolation function.
    
    Provides two-segment linear transition through a specified midpoint,
    useful for modeling parameters with intermediate changes.

    Parameters
    ----------
    start : float
        Starting value at t=0.
    mid : float
        Value at midpoint time.
    end : float
        Ending value at t=max(t).
    t_mid : float
        Time at midpoint (0 < t_mid < max(t)).
    
    See Also
    --------
    Linear : Simple linear interpolation.
    Exponential : Smooth exponential transition.
    """
    _pnames = ['start', 'mid', 'end', 't_mid']
    def __call__(self, t, start, mid, end, t_mid):
        self.params = dict(start=start, mid=mid, end=end, t_mid=t_mid)
        return self.compute(t, start, mid, end, t_mid)
    
    @staticmethod
    def compute(t, start, mid, end, t_mid):
        return np.piecewise(t, [t <= t_mid, t > t_mid],
                            [lambda t_val: start - (start - mid) * t_val / t_mid,
                             lambda t_val: mid - (mid - end) * (t_val - t_mid) / (t.max() - t_mid)])

class Exponential(ParametricFunction):
    """
    Exponential interpolation function.
    
    Provides smooth exponential transition between start and end values,
    useful for parameters varying over orders of magnitude.

    Parameters
    ----------
    start : float
        Starting value at t=0 (> 0).
    end : float
        Ending value at t=max(t) (> 0).
    
    See Also
    --------
    Linear : Linear interpolation.
    Bilinear : Piecewise linear with midpoint.
    """
    _pnames = ['start', 'end']
    def __call__(self, t, start, end):
        self.params = dict(start=start, end=end)
        return self.compute(t, start, end)
    
    @staticmethod
    def compute(t, start, end):
        return start * np.exp(np.log(end / start) * (t / t.max()))

class Constant(ParametricFunction):
    """
    Constant value function.
    
    Provides time-invariant parameter values throughout the time domain.

    Parameters
    ----------
    value : float
        Constant value for all time points.
    
    See Also
    --------
    Linear : Time-varying linear function.
    """
    _pnames = ['value']
    def __call__(self, t, value):
        self.params = dict(value=value)
        return self.compute(t, value)
    
    @staticmethod
    def compute(t, value):
        return np.full(len(t), value)
