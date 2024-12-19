import numpy as np
from . import parametric_functions
from ..motion import signal_props as sps

class ModelConfig:
    """
    This class allows to
        config time, frequency arrays using npts, dt (from a target motion or user)
        config functions and parameters (if already have model parameters)
        flexible assignment of parametric functions using string names (i.e. linear)
    """
    def __init__(self, npts: int, dt: float, mdl_type: str = 'beta_multi',
                 wu_type: str = 'linear', zu_type: str = 'linear',
                 wl_type: str = 'linear', zl_type: str = 'linear'):

        self.get_time_freq(npts, dt)
        self.mdl_func = getattr(parametric_functions, mdl_type.lower())
        self.wu_func = getattr(parametric_functions, wu_type.lower())
        self.zu_func = getattr(parametric_functions, zu_type.lower())
        self.wl_func = getattr(parametric_functions, wl_type.lower())
        self.zl_func = getattr(parametric_functions, zl_type.lower())

    def get_time_freq(self, npts, dt):
        self.dt = dt
        self.npts = npts
        self.t = sps.get_time(npts, dt)
        self.freq = sps.get_freq(npts, dt)  # Nyq freq in fitting
        self.slicer_freq = sps.get_freq_slice(self.freq, (0.1, 25.0))

        npts_sim = int(2 ** np.ceil(np.log2(2 * npts)))
        self.freq_sim = sps.get_freq(npts_sim, dt)  # Nyq freq in simulations
        return self

    def get_mdl(self, *params):
        self.mdl_param = params
        self.mdl = self.mdl_func(self.t, *params)
        return self

    def get_wu(self, *params):
        self.wu_param = params
        self.wu = self.wu_func(self.t, *params)  * 2 * np.pi  # in angular freq
        return self

    def get_wl(self, *params):
        self.wl_param = params
        self.wl = self.wl_func(self.t, *params)  * 2 * np.pi  # in angular freq
        return self

    def get_zu(self, *params):
        self.zu_param = params
        self.zu = self.zu_func(self.t, *params)
        return self

    def get_zl(self, *params):
        self.zl_param = params
        self.zl = self.zl_func(self.t, *params)
        return self

    def print_parameters(self):
        def format_array(arr):
            return np.array2string(np.array(arr), precision=2, separator=', ', suppress_small=True)
        print(f"Modulating ({self.mdl_func.__name__}): {format_array(self.mdl_param)}")
        print(f"wu ({self.wu_func.__name__}): {format_array(self.wu_param)}")
        print(f"wl ({self.wl_func.__name__}): {format_array(self.wl_param)}")
        print(f"zu ({self.zu_func.__name__}): {format_array(self.zu_param)}")
        print(f"zl ({self.zl_func.__name__}): {format_array(self.zl_param)}")