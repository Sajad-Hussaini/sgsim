from functools import cached_property
import numpy as np
from ..motion import signal_tools

class DomainConfig:
    """
    Time and frequency domain configuration.

    Provides properties and methods for configuring time and frequency domains for signal analysis and simulation.
    """

    _CORE_ATTRS = frozenset(['_npts', '_dt'])

    def __init__(self, npts: int, dt: float):
        """
        Initialize a DomainConfig instance.

        Parameters
        ----------
        npts : int
            Number of points in the time series.
        dt : float
            Time step between points.
        """
        self._npts = npts
        self._dt = dt
    
    @property
    def npts(self):
        return self._npts

    @npts.setter
    def npts(self, value: int):
        if value != self._npts:
            self._npts = value
            self.clear_cache()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value: float):
        if value != self._dt:
            self._dt = value
            self.clear_cache()

    def clear_cache(self):
        """
        Clear cached properties, preserving core attributes.
        """
        core_values = {attr: getattr(self, attr) for attr in self._CORE_ATTRS}
        self.__dict__.clear()
        self.__dict__.update(core_values)

    @cached_property
    def t(self):
        """
        ndarray: Time array for the configured number of points and time step.
        """
        return signal_tools.get_time(self.npts, self.dt)

    @cached_property
    def freq(self):
        """
        ndarray: Frequency array for the configured number of points and time step.
        """
        return signal_tools.get_freq(self.npts, self.dt)
    
    @cached_property
    def freq_sim(self):
        """
        ndarray: Frequency array for simulation (zero-padded to avoid aliasing).
        """
        npts_sim = int(2 ** np.ceil(np.log2(2 * self.npts)))
        return signal_tools.get_freq(npts_sim, self.dt)  # Nyquist freq to avoid aliasing in simulations

    @property
    def freq_slice(self):
        """
        slice: Slice object corresponding to the specified frequency range.
        """
        if not hasattr(self, '_freq_slice'):
            self._freq_slice = signal_tools.slice_freq(self.freq, (0.1, 25.0))
        return self._freq_slice

    @freq_slice.setter
    def freq_slice(self, freq_range: tuple[float, float]):
        """
        Set the frequency slice range.

        Parameters
        ----------
        freq_range : tuple of float
            (start_freq, end_freq) for frequency range.
        """
        self._freq_slice = signal_tools.slice_freq(self.freq, freq_range)

    @property
    def tp(self):
        """
        ndarray: Period array for response spectra (default: 0.04 to 10.0 s).
        """
        return self._tp if hasattr(self, '_tp') else np.arange(0.04, 10.01, 0.01)

    @tp.setter
    def tp(self, period_range: tuple[float, float, float]):
        """
        Set the period array for response spectra.

        Parameters
        ----------
        period_range : tuple of float
            (start, stop, step) for period array.
        """
        self._tp = np.arange(*period_range)

    @cached_property
    def freq_sim_p2(self):
        """
        ndarray: Square of the simulation frequency array.
        """
        return self.freq_sim ** 2

    @cached_property
    def freq_p2(self):
        """
        ndarray: Square of the frequency array.
        """
        return self.freq ** 2

    @cached_property
    def freq_p4(self):
        """
        ndarray: Fourth power of the frequency array.
        """
        return self.freq ** 4

    @cached_property
    def freq_n2(self):
        """
        ndarray: Negative second power of the frequency array (0 for freq=0).
        """
        _freq_n2 = np.zeros_like(self.freq)
        _freq_n2[1:] = self.freq[1:] ** -2
        return _freq_n2

    @cached_property
    def freq_n4(self):
        """
        ndarray: Negative fourth power of the frequency array (0 for freq=0).
        """
        _freq_n4 = np.zeros_like(self.freq)
        _freq_n4[1:] = self.freq[1:] ** -4
        return _freq_n4