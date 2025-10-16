import numpy as np
from .domain_config import DomainConfig
from .parametric_functions import ParametricFunction
from . import model_engine
from ..motion import signal_tools

class ModelConfig(DomainConfig):
    """
    Base class for stochastic model configuration and dependent attribute management.

    Parameters
    ----------
    npts : int
        Number of time points in the simulation.
    dt : float
        Time step of the simulation.
    modulating : ParametricFunction
        Time-varying modulating function.
    upper_frequency : ParametricFunction
        Upper frequency parameter function.
    upper_damping : ParametricFunction
        Upper damping parameter function.
    lower_frequency : ParametricFunction
        Lower frequency parameter function.
    lower_damping : ParametricFunction
        Lower damping parameter function.
    """
    _CORE_ATTRS = DomainConfig._CORE_ATTRS | frozenset(['modulating', 'upper_frequency', 'upper_damping', 'lower_frequency', 'lower_damping'])

    def __init__(self, modulating: ParametricFunction,
                 upper_frequency: ParametricFunction, upper_damping: ParametricFunction,
                 lower_frequency: ParametricFunction, lower_damping: ParametricFunction):
        self.modulating = modulating
        self.upper_frequency = upper_frequency
        self.upper_damping = upper_damping
        self.lower_frequency = lower_frequency
        self.lower_damping = lower_damping

        for func in [self.modulating, self.upper_frequency, self.upper_damping, 
                     self.lower_frequency, self.lower_damping]:
            func.callback = self._clear_cache

    @property
    def _stats(self):
        """Compute and store the variances for internal use."""
        if not hasattr(self, '_variance'):
            self._variance, self._variance_dot, self._variance_2dot, self._variance_bar, self._variance_2bar = model_engine.get_stats(
                self.upper_frequency.values * 2 * np.pi, self.upper_damping.values,
                self.lower_frequency.values * 2 * np.pi, self.lower_damping.values,
                self.freq_p2, self.freq_p4, self.freq_n2, self.freq_n4)

    @property
    def fas(self):
        """
        Fourier amplitude spectrum (FAS) of the stochastic model.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        if not hasattr(self, '_fas'):
            self._fas = model_engine.get_fas(self.modulating.values,
                                             self.upper_frequency.values * 2 * np.pi, self.upper_damping.values,
                                             self.lower_frequency.values * 2 * np.pi, self.lower_damping.values,
                                             self.freq_p2, self.freq_p4, self._variance)
        return self._fas
    
    @property
    def ce(self):
        """
        Cumulative energy of the stochastic model.

        Returns
        -------
        ndarray
            Cumulative energy time history.
        """
        return signal_tools.get_ce(self.dt, self.modulating.values)

    @property
    def mle_ac(self):
        """
        Mean cumulative number of local extrema (peaks and valleys) of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration local extrema.
        """
        self._stats
        return model_engine.cumulative_rate(self.dt, self._variance_2dot, self._variance_dot)

    @property
    def mle_vel(self):
        """
        Mean cumulative number of local extrema (peaks and valleys) of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity local extrema.
        """
        self._stats
        return model_engine.cumulative_rate(self.dt, self._variance_dot, self._variance)

    @property
    def mle_disp(self):
        """
        Mean cumulative number of local extrema (peaks and valleys) of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement local extrema.
        """
        self._stats
        return model_engine.cumulative_rate(self.dt, self._variance, self._variance_bar)

    @property
    def mzc_ac(self):
        """
        Mean cumulative number of zero crossings (up and down) of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration zero crossings.
        """
        self._stats
        return model_engine.cumulative_rate(self.dt, self._variance_dot, self._variance)

    @property
    def mzc_vel(self):
        """
        Mean cumulative number of zero crossings (up and down) of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity zero crossings.
        """            
        self._stats
        return model_engine.cumulative_rate(self.dt, self._variance, self._variance_bar)

    @property
    def mzc_disp(self):
        """
        Mean cumulative number of zero crossings (up and down) of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement zero crossings.
        """
        self._stats
        return model_engine.cumulative_rate(self.dt, self._variance_bar, self._variance_2bar)

    @property
    def pmnm_ac(self):
        """
        Mean cumulative number of positive-minima and negative maxima of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration positive-minima and negative maxima.
        """
        self._stats
        return model_engine.pmnm_rate(self.dt, self._variance_2dot, self._variance_dot, self._variance)

    @property
    def pmnm_vel(self):
        """
        Mean cumulative number of positive-minima and negative maxima of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity positive-minima and negative maxima.
        """
        self._stats
        return model_engine.pmnm_rate(self.dt, self._variance_dot, self._variance, self._variance_bar)

    @property
    def pmnm_disp(self):
        """
        Mean cumulative number of positive-minima and negative maxima of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement positive-minima and negative maxima.
        """
        self._stats
        return model_engine.pmnm_rate(self.dt, self._variance, self._variance_bar, self._variance_2bar)
