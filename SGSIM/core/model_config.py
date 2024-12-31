import numpy as np
from . import parametric_functions
from ..motion import signal_analysis

class ModelConfig:
    """
    This class allows to
        configure time, frequency arrays using npts, dt (from an input motion or an user input)
        configure model parameters or parametric functions
    """
    def __init__(self, npts: int, dt: float,
                 mdl_type: str = 'beta_single',
                 wu_type: str = 'linear', zu_type: str = 'linear',
                 wl_type: str = 'linear', zl_type: str = 'linear'):

        self.get_time_freq(npts, dt)
        self.mdl_func = getattr(parametric_functions, mdl_type.lower())
        self.wu_func = getattr(parametric_functions, wu_type.lower())
        self.zu_func = getattr(parametric_functions, zu_type.lower())
        self.wl_func = getattr(parametric_functions, wl_type.lower())
        self.zl_func = getattr(parametric_functions, zl_type.lower())

    def get_time_freq(self, npts, dt):
        """ Time and frequency arrays """
        self.dt = dt
        self.npts = npts
        self.t = signal_analysis.get_time(npts, dt)
        self.freq = signal_analysis.get_freq(npts, dt)  # Nyq freq for fitting
        self.freq_mask = signal_analysis.get_freq_mask(self.freq, (0.1, 25.0))  # TODO Hard-coded freq range
        npts_sim = int(2 ** np.ceil(np.log2(2 * npts)))
        self.freq_sim = signal_analysis.get_freq(npts_sim, dt)  # Nyq freq for simulations and avoiding aliasing
        return self

    def get_mdl(self, *params):
        """ Modulating function """
        self.mdl, self.mdl_param = self.mdl_func(self.t, *params)
        return self

    def get_wu(self, *params):
        """ Upper dominant frequency """
        self.wu, self.wu_param = self.wu_func(self.t, *params)
        self.wu *= 2 * np.pi  # in angular freq
        return self

    def get_wl(self, *params):
        """ Lower dominant frequency """
        self.wl, self.wl_param = self.wl_func(self.t, *params)
        self.wl *= 2 * np.pi  # in angular freq
        return self

    def get_zu(self, *params):
        """ Upper damping ratio """
        self.zu, self.zu_param = self.zu_func(self.t, *params)
        return self

    def get_zl(self, *params):
        """ Upper damping ratio """
        self.zl, self.zl_param = self.zl_func(self.t, *params)
        return self

    def print_parameters(self):
        def format_dict(d):
            return ', '.join([f"{key}: {round(value, 6) if key == 'Et' else round(value, 2)}" for key, value in d.items()])
        print(f"Modulating ({self.mdl_func.__name__}): {format_dict(self.mdl_param)}")
        print(f"wu ({self.wu_func.__name__}): {format_dict(self.wu_param)}")
        print(f"wl ({self.wl_func.__name__}): {format_dict(self.wl_param)}")
        print(f"zu ({self.zu_func.__name__}): {format_dict(self.zu_param)}")
        print(f"zl ({self.zl_func.__name__}): {format_dict(self.zl_param)}")
        return self
