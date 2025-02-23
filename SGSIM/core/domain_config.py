import numpy as np
from ..motion import signal_analysis

class DomainConfig:
    """ Time and frequency domain configuration """
    def __init__(self, npts, dt):
        """
        npts: int
            Number of points in the time series.
        dt: float
            Time step between points.
        """
        self._npts = npts
        self._dt = dt
        self._t = None
        self._freq = None
        self._freq_sim = None
        self._tp = None
        self._freq_mask = None

    @property
    def npts(self):
        return self._npts

    @npts.setter
    def npts(self, num_points):
        self._npts = num_points
        self._t = None
        self._freq = None
        self._freq_sim = None
        self._freq_mask = None

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, time_step):
        self._dt = time_step
        self._t = None
        self._freq = None
        self._freq_sim = None
        self._freq_mask = None

    @property
    def t(self):
        if self._t is None:
            self._t = signal_analysis.get_time(self._npts, self._dt)
        return self._t

    @property
    def freq(self):
        if self._freq is None:
            self._freq = signal_analysis.get_freq(self._npts, self._dt)
        return self._freq

    @property
    def freq_sim(self):
        if self._freq_sim is None:
            npts_sim = int(2 ** np.ceil(np.log2(2 * self._npts)))
            self._freq_sim = signal_analysis.get_freq(npts_sim, self._dt)
        return self._freq_sim  # Nyquist frequency for avoiding aliasing in simulations

    @property
    def freq_mask(self):
        if self._freq_mask is None:
            self.freq_mask = (0.1, 25.0)  # Default frequency range in Hz
        return self._freq_mask

    @freq_mask.setter
    def freq_mask(self, target_range: tuple[float, float]):
        self._freq_mask = signal_analysis.get_freq_mask(self.freq, target_range)

    @property
    def tp(self):
        if self._tp is None:
            self.tp = (0.04, 10.04, 0.01)  # Default period range in seconds
        return self._tp

    @tp.setter
    def tp(self, period_range: tuple[float, float, float]):
        self._tp = np.arange(*period_range)