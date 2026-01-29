from functools import cached_property
import numpy as np
from ..motion import signal_tools

class DomainConfig:
    """
    Time and frequency domain configuration.

    Parameters
    ----------
    npts : int
        Number of points in the time series.
    dt : float
        Time step between points.
    """
    def __init__(self, npts: int, dt: float):
        self.npts = npts
        self.dt = dt

    @cached_property
    def t(self):
        """
        ndarray: Time array for the configured number of points and time step.
        """
        return signal_tools.time(self.npts, self.dt)

    @cached_property
    def freq(self):
        """
        ndarray: Angular frequency array for the configured number of points and time step.
        """
        return signal_tools.frequency(self.npts, self.dt) * 2 * np.pi
    
    @cached_property
    def df(self):
        """
        float: Angular frequency step.
        """
        return self.freq[2] - self.freq[1]
    
    @cached_property
    def freq_sim(self):
        """
        ndarray: Angular frequency array for simulation (zero-padded to avoid aliasing).
        Uses Nyquist frequency to avoid aliasing in simulations.
        """
        npts_sim = int(2 ** np.ceil(np.log2(2 * self.npts)))
        return signal_tools.frequency(npts_sim, self.dt) * 2 * np.pi

    @cached_property
    def freq_sim_p2(self):
        """
        ndarray: Square of the simulation angular frequency array.
        """
        return self.freq_sim ** 2

    @cached_property
    def freq_p2(self):
        """
        ndarray: Square of the angular frequency array.
        """
        return self.freq ** 2

    @cached_property
    def freq_p4(self):
        """
        ndarray: Fourth power of the angular frequency array.
        """
        return self.freq ** 4

    @cached_property
    def freq_n2(self):
        """
        ndarray: Negative second power of the angular frequency array (0 for freq=0).
        """
        _freq_n2 = np.zeros_like(self.freq)
        _freq_n2[1:] = self.freq[1:] ** -2
        return _freq_n2

    @cached_property
    def freq_n4(self):
        """
        ndarray: Negative fourth power of the angular frequency array (0 for freq=0).
        """
        _freq_n4 = np.zeros_like(self.freq)
        _freq_n4[1:] = self.freq[1:] ** -4
        return _freq_n4