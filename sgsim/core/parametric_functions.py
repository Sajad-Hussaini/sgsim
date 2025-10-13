import numpy as np
from scipy.special import beta
from abc import ABC, abstractmethod

class ParametricFunction(ABC):
    """
    Abstract base class for parametric functions.
    """
    def _trigger_callback(self):
        """Trigger callback if it exists."""
        if hasattr(self, 'callback'):
            self.callback()

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

class BetaSingle(ParametricFunction):
    """Beta single modulating function.

    Parameters
    ----------
    peak : float
        Peak location as a fraction of duration (0 < peak < 1).
    concentration : float
        Concentration parameter (> 0).
    energy : float
        Total energy of the modulating function (> 0).
    duration : float
        Duration of the modulating function (> 0).

    References
    ----------  
    - Hussaini SS, Karimzadeh S, Rezaeian S, Lourenço PB. Broadband stochastic simulation of earthquake ground motions with multiple strong phases with an application to the 2023 Kahramanmaraş, Turkey (Türkiye), earthquake. Earthquake Spectra. 2025;41(3):2399-2435. doi:10.1177/87552930251331981

    """
    def __call__(self, t, peak, concentration, energy, duration):
        self.values = self.compute(t, peak, concentration, energy, duration)
        self.params = dict(peak=peak, concentration=concentration, energy=energy, duration=duration)
        self._trigger_callback()
        return self.values
    
    @staticmethod
    def compute(t, peak, concentration, energy, duration):
        mdl1 = 0.05 * (6 * (t[1:-1] * (duration - t[1:-1])) / (duration ** 3))
        mdl2 = 0.95 * np.exp((concentration * peak) * np.log(t[1:-1]) +
                             (concentration * (1 - peak)) * np.log(duration - t[1:-1]) -
                             np.log(beta(1 + concentration * peak, 1 + concentration * (1 - peak))) -
                             (1 + concentration) * np.log(duration))
        multi_mdl = np.zeros_like(t)
        multi_mdl[1:-1] = mdl1 + mdl2
        return np.sqrt(energy * multi_mdl)

class BetaDual(ParametricFunction):
    """Beta dual modulating function.

    Parameters
    ----------
    peak : float
        Peak location of the first strong phase as a fraction of duration (0 < peak < 1).
    concentration : float
        Concentration parameter of the first phase (> 0).
    peak_2 : float
        Peak location of the second strong phase as a fraction of duration (0 < peak_2 < 1).
    concentration_2 : float
        Concentration parameter of the second phase (> 0).
    energy_ratio : float
        Energy ratio of the first strong phase (0 < energy_ratio < 1).
    energy : float
        Total energy of the modulating function (> 0).
    duration : float
        Duration of the modulating function (> 0).

    References
    ----------  
    - Hussaini SS, Karimzadeh S, Rezaeian S, Lourenço PB. Broadband stochastic simulation of earthquake ground motions with multiple strong phases with an application to the 2023 Kahramanmaraş, Turkey (Türkiye), earthquake. Earthquake Spectra. 2025;41(3):2399-2435. doi:10.1177/87552930251331981

    """
    def __call__(self, t, peak, concentration, peak_2, concentration_2, energy_ratio, energy, duration):
        self.values = self.compute(t, peak, concentration, peak_2, concentration_2, energy_ratio, energy, duration)
        self.params = dict(peak=peak, concentration=concentration,
                           peak_2=peak_2, concentration_2=concentration_2,
                           energy_ratio=energy_ratio, energy=energy, duration=duration)
        self._trigger_callback()
        return self.values
    
    @staticmethod
    def compute(t, peak, concentration, peak_2, concentration_2, energy_ratio, energy, duration):
        # Original formula:
        # mdl1 = 0.05 * (6 * (t * (duration - t)) / (duration ** 3))
        # mdl2 = energy_ratio * ((t ** (concentration * peak) * (duration - t) ** (concentration * (1 - peak))) / (beta(1 + concentration * peak, 1 + concentration * (1 - peak)) * duration ** (1 + concentration)))
        # mdl3 = (1 - 0.05 - energy_ratio) * ((t ** (concentration_2 * peak_2) * (duration - t) ** (concentration_2 * (1 - peak_2))) / (beta(1 + concentration_2 * peak_2, 1 + concentration_2 * (1 - peak_2)) * duration ** (1 + concentration_2)))
        # multi_mdl = mdl1 + mdl2 + mdl3
        mdl1 = 0.05 * (6 * (t[1:-1] * (duration - t[1:-1])) / (duration ** 3))
        mdl2 = energy_ratio * np.exp((concentration * peak) * np.log(t[1:-1]) +
                                     (concentration * (1 - peak)) * np.log(duration - t[1:-1]) -
                                     np.log(beta(1 + concentration * peak, 1 + concentration * (1 - peak))) -
                                     (1 + concentration) * np.log(duration))
        mdl3 = (0.95 - energy_ratio) * np.exp((concentration_2 * peak_2) * np.log(t[1:-1]) +
                                              (concentration_2 * (1 - peak_2)) * np.log(duration - t[1:-1]) -
                                              np.log(beta(1 + concentration_2 * peak_2, 1 + concentration_2 * (1 - peak_2))) -
                                              (1 + concentration_2) * np.log(duration))
        multi_mdl = np.zeros_like(t)
        multi_mdl[1:-1] = mdl1 + mdl2 + mdl3
        return np.sqrt(energy * multi_mdl)

class BetaBasic(ParametricFunction):
    """Basic Beta modulating function.

    Parameters
    ----------
    peak : float
        Peak location as a fraction of duration (0 < peak < 1).
    concentration : float
        Concentration parameter (> 0).
    energy : float
        Total energy of the modulating function (> 0).
    duration : float
        Duration of the modulating function (> 0).

    References
    ----------  
    - Hussaini SS, Karimzadeh S, Rezaeian S, Lourenço PB. Broadband stochastic simulation of earthquake ground motions with multiple strong phases with an application to the 2023 Kahramanmaraş, Turkey (Türkiye), earthquake. Earthquake Spectra. 2025;41(3):2399-2435. doi:10.1177/87552930251331981

    """
    def __call__(self, t, peak, concentration, energy, duration):
        self.values = self.compute(t, peak, concentration, energy, duration)
        self.params = dict(peak=peak, concentration=concentration, energy=energy, duration=duration)
        self._trigger_callback()
        return self.values
    
    @staticmethod
    def compute(t, peak, concentration, energy, duration):
        mdl = ((t ** (concentration * peak) * (duration - t) ** (concentration * (1 - peak))) /
               (beta(1 + concentration * peak, 1 + concentration * (1 - peak)) * duration ** (1 + concentration)))
        return np.sqrt(energy * mdl)

class Gamma(ParametricFunction):
    """Gamma modulating function.

    Parameters
    ----------
    scale : float
        Amplitude scaling factor (> 0).
    shape : float
        Shape parameter (> 0).
    decay : float
        Decay parameter (> 0).
    """
    def __call__(self, t, scale, shape, decay):
        self.values = self.compute(t, scale, shape, decay)
        self.params = dict(scale=scale, shape=shape, decay=decay)
        self._trigger_callback()
        return self.values
    
    @staticmethod
    def compute(t, scale, shape, decay):
        return scale * t ** shape * np.exp(-decay * t)

class Housner(ParametricFunction):
    """Housner modulating function.

    Parameters
    ----------
    amplitude : float
        Constant amplitude (> 0).
    decay : float
        Decay scale (> 0).
    shape : float
        Shape parameter (> 0).
    tp : float
        Time to peak amplitude (> 0).
    td : float
        Time to start of decay phase (td > tp).
    """
    def __call__(self, t, amplitude, decay, shape, tp, td):
        self.values = self.compute(t, amplitude, decay, shape, tp, td)
        self.params = dict(amplitude=amplitude, decay=decay, shape=shape, tp=tp, td=td)
        self._trigger_callback()
        return self.values
    
    @staticmethod
    def compute(t, amplitude, decay, shape, tp, td):
        return np.piecewise(t, [(t >= 0) & (t < tp), (t >= tp) & (t <= td), t > td],
                            [lambda t_val: amplitude * (t_val / tp) ** 2, amplitude,
                             lambda t_val: amplitude * np.exp(-decay * ((t_val - td) ** shape))])

class Linear(ParametricFunction):
    """Linear function.

    Parameters
    ----------
    start : float
        Starting value.
    end : float
        Ending value.
    """
    def __call__(self, t, start, end):
        self.values = self.compute(t, start, end)
        self.params = dict(start=start, end=end)
        self._trigger_callback()
        return self.values
    
    @staticmethod
    def compute(t, start, end):
        return start + (end - start) * (t / t.max())

class Bilinear(ParametricFunction):
    """Bilinear function.

    Parameters
    ----------
    start : float
        Starting value.
    mid : float
        Midpoint value.
    end : float
        Ending value.
    t_mid : float
        Time at which the midpoint occurs (0 < t_mid < max(t)).
    """
    def __call__(self, t, start, mid, end, t_mid):
        self.values = self.compute(t, start, mid, end, t_mid)
        self.params = dict(start=start, mid=mid, end=end, t_mid=t_mid)
        self._trigger_callback()
        return self.values
    
    @staticmethod
    def compute(t, start, mid, end, t_mid):
        return np.piecewise(t, [t <= t_mid, t > t_mid],
                            [lambda t_val: start - (start - mid) * t_val / t_mid,
                             lambda t_val: mid - (mid - end) * (t_val - t_mid) / (t.max() - t_mid)])

class Exponential(ParametricFunction):
    """Exponential function.

    Parameters
    ----------
    start : float
        Starting value.
    end : float
        Ending value.
    """
    def __call__(self, t, start, end):
        self.values = self.compute(t, start, end)
        self.params = dict(start=start, end=end)
        self._trigger_callback()
        return self.values
    
    @staticmethod
    def compute(t, start, end):
        return start * np.exp(np.log(end / start) * (t / t.max()))

class Constant(ParametricFunction):
    """Constant function.

    Parameters
    ----------
    value : float
        Constant value.
    """
    def __call__(self, t, value):
        self.values = self.compute(t, value)
        self.params = dict(value=value)
        self._trigger_callback()
        return self.values
    
    @staticmethod
    def compute(t, value):
        return np.full_like(t, value)
