import numpy as np
from . import parametric_functions
from ..motion import signal_analysis

class ModelConfig:
    """
    This class allows configuring time, frequency arrays using npts, dt (from an input motion or user input)
    and configuring model parameters or parametric functions.
    """
    def __init__(self, npts: int, dt: float,
                 mdl_type: str,
                 wu_type: str, zu_type: str,
                 wl_type: str, zl_type: str):

        # Configure time and frequency
        self.get_time_freq(npts, dt)

        # Set model functions
        self.mdl_func = getattr(parametric_functions, mdl_type.lower(), None)
        self.wu_func = getattr(parametric_functions, wu_type.lower(), None)
        self.zu_func = getattr(parametric_functions, zu_type.lower(), None)
        self.wl_func = getattr(parametric_functions, wl_type.lower(), None)
        self.zl_func = getattr(parametric_functions, zl_type.lower(), None)

    def get_time_freq(self, npts, dt):
        """ Time and frequency arrays """
        self.dt = dt
        self.npts = npts
        self.t = signal_analysis.get_time(npts, dt)
        self.freq = signal_analysis.get_freq(npts, dt)  # Nyquist frequency for fitting
        self.freq_mask = signal_analysis.get_freq_mask(self.freq, (0.1, 25.0))  # TODO: Hard-coded frequency range
        npts_sim = int(2 ** np.ceil(np.log2(2 * npts)))
        self.freq_sim = signal_analysis.get_freq(npts_sim, dt)  # Nyquist frequency for simulations and avoiding aliasing

    def reset_attributes(self):
        """ Intended to be overridden in child classes. """
        pass  # No implementation here; it's a placeholder

    @property
    def mdl(self):
        """ Modulating function """
        if self._mdl is None:
            raise ValueError("Time-Modulating parameters have not been set.")
        return self._mdl

    @mdl.setter
    def mdl(self, params):
        self._mdl, self.mdl_param = self.mdl_func(self.t, params)
        self.reset_attributes()

    @property
    def wu(self):
        """ Upper dominant frequency """
        if self._wu is None:
            raise ValueError("Upper frequency parameters have not been set.")
        return self._wu

    @wu.setter
    def wu(self, params):
        self._wu, self.wu_param = self.wu_func(self.t, params)
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
        self._wl, self.wl_param = self.wl_func(self.t, params)
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
        self._zu, self.zu_param = self.zu_func(self.t, params)
        self.reset_attributes()

    @property
    def zl(self):
        """ Lower damping ratio """
        if self._zl is None:
            raise ValueError("Lower damping parameters have not been set.")
        return self._zl

    @zl.setter
    def zl(self, params):
        self._zl, self.zl_param = self.zl_func(self.t, params)
        self.reset_attributes()

    def parameters(self):
        """ Print formatted parameters """
        def format_dict(d):
            return ', '.join([f"{key}: {round(value, 6) if key == 'Et' else round(value, 2)}" for key, value in d.items()])
        print()
        print(f"Modulating function (mdl) ({self.mdl_func.__name__}): {format_dict(self.mdl_param)}")
        print(f"Upper frequency function (wu) ({self.wu_func.__name__}): {format_dict(self.wu_param)}")
        print(f"Lower frequency function (wl) ({self.wl_func.__name__}): {format_dict(self.wl_param)}")
        print(f"Upper damping function (zu) ({self.zu_func.__name__}): {format_dict(self.zu_param)}")
        print(f"Lower damping function (zl) ({self.zl_func.__name__}): {format_dict(self.zl_param)}")
