import numpy as np
from . import parametric_functions
from ..motion import signal_analysis

class ModelConfig:
    """
    This class allows configuring time, frequency arrays using npts, dt (from an input motion or user input)
    and configuring model parameters or parametric functions.
    """
    def __init__(self, npts: int, dt: float,
                 modulating_function: str,
                 upper_dominant_frequency_function: str, upper_damping_ratio_function: str,
                 lower_dominant_frequency_function: str, lower_damping_ratio_function: str):
        self.npts = npts
        self.dt = dt
        self.mdl_func = getattr(parametric_functions, modulating_function.lower(), None)
        self.wu_func = getattr(parametric_functions, upper_dominant_frequency_function.lower(), None)
        self.zu_func = getattr(parametric_functions, upper_damping_ratio_function.lower(), None)
        self.wl_func = getattr(parametric_functions, lower_dominant_frequency_function.lower(), None)
        self.zl_func = getattr(parametric_functions, lower_damping_ratio_function.lower(), None)

    def reset_attributes(self):
        """ Reset specific attributes when model parameters change. """
        self._fas = self._ce = None
        self._mle_ac = self._mle_vel = self._mle_disp = None
        self._mzc_ac = self._mzc_vel = self._mzc_disp = None
        self._pmnm_ac = self._pmnm_vel = self._pmnm_disp = None
        self.variance = self.variance_dot = self.variance_2dot = self.variance_bar = self.variance_2bar = None

    @property
    def t(self):
        if not hasattr(self, '_t'):
            self._t = signal_analysis.get_time(self.npts, self.dt)
        return self._t
    
    @t.setter
    def t(self, value):
        self._t = value

    @property
    def freq(self):
        if not hasattr(self, '_freq'):
            self._freq = signal_analysis.get_freq(self.npts, self.dt)
        return self._freq  # Nyquist frequency for fitting

    @property
    def freq_sim(self):
        if not hasattr(self, '_freq_sim'):
            npts_sim = int(2 ** np.ceil(np.log2(2 * self.npts)))
            self._freq_sim = signal_analysis.get_freq(npts_sim, self.dt)
        return self._freq_sim  # Nyquist frequency for simulations and avoiding aliasing

    @property
    def freq_mask(self):
        if not hasattr(self, '_freq_mask'):
            self.freq_mask = (0.1, 25.0)
        return self._freq_mask

    @freq_mask.setter
    def freq_mask(self, target_range: tuple[float, float]):
        self._freq_mask = signal_analysis.get_freq_mask(self.freq, target_range)

    @property
    def mdl(self):
        """ Modulating function """
        if self._mdl is None:
            raise ValueError("Time-Modulating parameters have not been set.")
        return self._mdl

    @mdl.setter
    def mdl(self, params):
        self._mdl = self.mdl_func(self.t, params)
        self.mdl_params = params
        self.reset_attributes()

    @property
    def wu(self):
        """ Upper dominant frequency """
        if self._wu is None:
            raise ValueError("Upper frequency parameters have not been set.")
        return self._wu

    @wu.setter
    def wu(self, params):
        self._wu = self.wu_func(self.t, params)
        self.wu_params = params
        self._wu *= 2 * np.pi  # Convert to angular frequency
        self.reset_attributes()

    @property
    def wl(self):
        """ Lower dominant frequency """
        if self._wl is None:
            raise ValueError("Lower frequency parameters have not been set.")
        return self._wl

    @wl.setter
    def wl(self, params):
        self._wl = self.wl_func(self.t, params)
        self.wl_params = params
        self._wl *= 2 * np.pi  # Convert to angular frequency
        self.reset_attributes()

    @property
    def zu(self):
        """ Upper damping ratio """
        if self._zu is None:
            raise ValueError("Upper damping parameters have not been set.")
        return self._zu

    @zu.setter
    def zu(self, params):
        self._zu = self.zu_func(self.t, params)
        self.zu_params = params
        self.reset_attributes()

    @property
    def zl(self):
        """ Lower damping ratio """
        if self._zl is None:
            raise ValueError("Lower damping parameters have not been set.")
        return self._zl

    @zl.setter
    def zl(self, params):
        self._zl = self.zl_func(self.t, params)
        self.zl_params = params
        self.reset_attributes()

    def parameters(self):
        """ Print formatted parameters """
        def format_params(values):
            return ', '.join([f"{round(val, 3)}" for val in values])
        print()
        print(f"Modulating function (mdl): {format_params(self.mdl_params)}")
        print(f"Upper frequency function (wu): {format_params(self.wu_params)}")
        print(f"Lower frequency function (wl): {format_params(self.wl_params)}")
        print(f"Upper damping function (zu): {format_params(self.zu_params)}")
        print(f"Lower damping function (zl): {format_params(self.zl_params)}")
