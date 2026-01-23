import numpy as np
from .domain_config import DomainConfig
from .functions import ParametricFunction
from . import engine
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
            self._variance, self._variance_dot, self._variance_2dot, self._variance_bar, self._variance_2bar = engine.get_stats(
                self.upper_frequency.value * 2 * np.pi, self.upper_damping.value,
                self.lower_frequency.value * 2 * np.pi, self.lower_damping.value,
                self.freq_p2, self.freq_p4, self.freq_n2, self.freq_n4, self.dw)

    def _compute_fas(self):
        """
        Fourier amplitude spectrum (FAS) of the stochastic model (acceleration, velocity, displacement).

        Returns
        -------
        self
            The ModelConfig instance with computed FAS attributes.
        """
        self._fas, self._fas_vel, self._fas_disp = engine.get_fas(
            self.modulating.value, self.upper_frequency.value * 2 * np.pi, self.upper_damping.value,
            self.lower_frequency.value * 2 * np.pi, self.lower_damping.value,
            self.freq_p2, self.freq_p4, self._variance, self.dt)
        return self
    
    @property
    def fas(self):
        """
        Fourier amplitude spectrum (FAS) of the stochastic model (acceleration).

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        if not hasattr(self, '_fas'):
            self._compute_fas()
        return self._fas
    
    @property
    def fas_vel(self):
        """
        Fourier amplitude spectrum (FAS) of the velocity of the stochastic model.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        if not hasattr(self, '_fas_vel'):
            self._compute_fas()
        return self._fas_vel
    
    @property
    def fas_disp(self):
        """
        Fourier amplitude spectrum (FAS) of the displacement of the stochastic model.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        if not hasattr(self, '_fas_disp'):
            self._compute_fas()
        return self._fas_disp
    
    @property
    def ce(self):
        """
        Cumulative energy of the stochastic model.

        Returns
        -------
        ndarray
            Cumulative energy time history.
        """
        return signal_tools.ce(self.dt, self.modulating.value)

    @property
    def le_ac(self):
        """
        Mean cumulative number of local extrema (peaks and valleys) of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration local extrema.
        """
        self._stats
        return engine.cumulative_rate(self.dt, self._variance_2dot, self._variance_dot)

    @property
    def le_vel(self):
        """
        Mean cumulative number of local extrema (peaks and valleys) of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity local extrema.
        """
        self._stats
        return engine.cumulative_rate(self.dt, self._variance_dot, self._variance)

    @property
    def le_disp(self):
        """
        Mean cumulative number of local extrema (peaks and valleys) of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement local extrema.
        """
        self._stats
        return engine.cumulative_rate(self.dt, self._variance, self._variance_bar)

    @property
    def zc_ac(self):
        """
        Mean cumulative number of zero crossings (up and down) of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration zero crossings.
        """
        self._stats
        return engine.cumulative_rate(self.dt, self._variance_dot, self._variance)

    @property
    def zc_vel(self):
        """
        Mean cumulative number of zero crossings (up and down) of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity zero crossings.
        """            
        self._stats
        return engine.cumulative_rate(self.dt, self._variance, self._variance_bar)

    @property
    def zc_disp(self):
        """
        Mean cumulative number of zero crossings (up and down) of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement zero crossings.
        """
        self._stats
        return engine.cumulative_rate(self.dt, self._variance_bar, self._variance_2bar)

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
        return engine.pmnm_rate(self.dt, self._variance_2dot, self._variance_dot, self._variance)

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
        return engine.pmnm_rate(self.dt, self._variance_dot, self._variance, self._variance_bar)

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
        return engine.pmnm_rate(self.dt, self._variance, self._variance_bar, self._variance_2bar)
