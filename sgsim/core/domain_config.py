from functools import cached_property
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
        self.npts = npts
        self.dt = dt

    @cached_property
    def t(self):
        return signal_analysis.get_time(self.npts, self.dt)

    @cached_property
    def freq(self):
        return signal_analysis.get_freq(self.npts, self.dt)
    
    @cached_property
    def freq_sim(self):
        npts_sim = int(2 ** np.ceil(np.log2(2 * self.npts)))
        return signal_analysis.get_freq(npts_sim, self.dt)  # Nyquist freq to avoid aliasing in simulations

    @property
    def freq_slice(self):
        if not hasattr(self, '_freq_slice'):
            self._freq_slice = signal_analysis.slice_freq(self.freq, (0.1, 25.0))
        return self._freq_slice

    @freq_slice.setter
    def freq_slice(self, freq_range: tuple[float, float]):
        self._freq_slice = signal_analysis.slice_freq(self.freq, freq_range)

    @property
    def tp(self):
        return self._tp if hasattr(self, '_tp') else np.arange(0.04, 10.04, 0.01)

    @tp.setter
    def tp(self, period_range: tuple[float, float, float]):
        self._tp = np.arange(*period_range)

    @cached_property
    def freq_sim_p2(self):
        return self.freq_sim ** 2

    @cached_property
    def freq_p2(self):
        return self.freq ** 2

    @cached_property
    def freq_p4(self):
        return self.freq ** 4

    @cached_property
    def freq_n2(self):
        _freq_n2 = np.zeros_like(self.freq)
        _freq_n2[1:] = self.freq[1:] ** -2
        return _freq_n2

    @cached_property
    def freq_n4(self):
        _freq_n4 = np.zeros_like(self.freq)
        _freq_n4[1:] = self.freq[1:] ** -4
        return _freq_n4