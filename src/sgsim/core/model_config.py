from functools import cached_property
from typing import Callable
import numpy as np
from .domain_config import DomainConfig
from . import engine
from ..motion import signal_tools


class ModelConfig(DomainConfig):
    """
    Stochastic model configuration with cached derived properties.

    Parameters
    ----------
    npts : int
        Number of time points.
    dt : float
        Time step.
    modulating : Callable
        Time-varying modulating function (use functools.partial to bind params).
    upper_frequency : Callable
        Upper frequency function.
    upper_damping : Callable
        Upper damping function.
    lower_frequency : Callable
        Lower frequency function.
    lower_damping : Callable
        Lower damping function.

    Examples
    --------
    >>> from functools import partial
    >>> from sgsim.core.functions import beta_single, linear, constant
    >>> config = ModelConfig(
    ...     npts=4000, dt=0.01,
    ...     modulating=partial(beta_single, peak=0.3, concentration=5.0, energy=100.0, duration=40.0),
    ...     upper_frequency=partial(linear, start=10.0, end=5.0),
    ...     upper_damping=partial(constant, c=0.3),
    ...     lower_frequency=partial(constant, c=0.1),
    ...     lower_damping=partial(constant, c=0.5))
    """
    def __init__(self, npts: int, dt: float, modulating: Callable,
                 upper_frequency: Callable, upper_damping: Callable,
                 lower_frequency: Callable, lower_damping: Callable):
        super().__init__(npts, dt)
        self.modulating = modulating
        self.upper_frequency = upper_frequency
        self.upper_damping = upper_damping
        self.lower_frequency = lower_frequency
        self.lower_damping = lower_damping

    # =========================================================================
    # Computed Function Values (deferred, cached)
    # =========================================================================

    @cached_property
    def mdl(self) -> np.ndarray:
        """Computed modulating function values."""
        return self.modulating(self.t)

    @cached_property
    def wu(self) -> np.ndarray:
        """Computed upper frequency values."""
        return self.upper_frequency(self.t)

    @cached_property
    def zu(self) -> np.ndarray:
        """Computed upper damping values."""
        return self.upper_damping(self.t)

    @cached_property
    def wl(self) -> np.ndarray:
        """Computed lower frequency values."""
        return self.lower_frequency(self.t)

    @cached_property
    def zl(self) -> np.ndarray:
        """Computed lower damping values."""
        return self.lower_damping(self.t)


    # =========================================================================
    # Core Statistics (cached tuple unpacking)
    # =========================================================================

    @cached_property
    def _stats(self):
        """Variance statistics for acceleration, velocity, and displacement."""
        return engine.get_stats(self.wu * 2 * np.pi, self.zu, self.wl * 2 * np.pi, self.zl,
                                self.freq_p2, self.freq_p4, self.freq_n2, self.freq_n4, self.df)

    @property
    def _variance(self):
        return self._stats[0]

    @property
    def _variance_dot(self):
        return self._stats[1]

    @property
    def _variance_2dot(self):
        return self._stats[2]

    @property
    def _variance_bar(self):
        return self._stats[3]

    @property
    def _variance_2bar(self):
        return self._stats[4]

    # =========================================================================
    # Fourier Amplitude Spectra (cached tuple unpacking)
    # =========================================================================

    @cached_property
    def _fas_all(self):
        """FAS for acceleration, velocity, and displacement."""
        return engine.get_fas(self.mdl, self.wu * 2 * np.pi, self.zu, self.wl * 2 * np.pi, self.zl,
                              self.freq_p2, self.freq_p4, self._variance, self.dt)

    @property
    def fas(self):
        """
        Fourier amplitude spectrum (FAS) of acceleration.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        return self._fas_all[0]

    @property
    def fas_vel(self):
        """
        Fourier amplitude spectrum (FAS) of velocity.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        return self._fas_all[1]

    @property
    def fas_disp(self):
        """
        Fourier amplitude spectrum (FAS) of displacement.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        return self._fas_all[2]

    # =========================================================================
    # Cumulative Energy
    # =========================================================================

    @cached_property
    def ce(self):
        """
        Cumulative energy of the stochastic model.

        Returns
        -------
        ndarray
            Cumulative energy time history.
        """
        return signal_tools.ce(self.dt, self.mdl)

    # =========================================================================
    # Local Extrema Counts
    # =========================================================================

    @cached_property
    def le_ac(self):
        """
        Mean cumulative local extrema count of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration local extrema.
        """
        return engine.cumulative_rate(self.dt, self._variance_2dot, self._variance_dot)

    @cached_property
    def le_vel(self):
        """
        Mean cumulative local extrema count of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity local extrema.
        """
        return engine.cumulative_rate(self.dt, self._variance_dot, self._variance)

    @cached_property
    def le_disp(self):
        """
        Mean cumulative local extrema count of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement local extrema.
        """
        return engine.cumulative_rate(self.dt, self._variance, self._variance_bar)

    # =========================================================================
    # Zero Crossing Counts
    # =========================================================================

    @cached_property
    def zc_ac(self):
        """
        Mean cumulative zero crossing count of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration zero crossings.
        """
        return engine.cumulative_rate(self.dt, self._variance_dot, self._variance)

    @cached_property
    def zc_vel(self):
        """
        Mean cumulative zero crossing count of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity zero crossings.
        """
        return engine.cumulative_rate(self.dt, self._variance, self._variance_bar)

    @cached_property
    def zc_disp(self):
        """
        Mean cumulative zero crossing count of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement zero crossings.
        """
        return engine.cumulative_rate(self.dt, self._variance_bar, self._variance_2bar)

    # =========================================================================
    # Positive-Minima / Negative-Maxima Counts
    # =========================================================================

    @cached_property
    def pmnm_ac(self):
        """
        Mean cumulative PMNM count of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration positive-minima and negative-maxima.
        """
        return engine.pmnm_rate(self.dt, self._variance_2dot, self._variance_dot, self._variance)

    @cached_property
    def pmnm_vel(self):
        """
        Mean cumulative PMNM count of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity positive-minima and negative-maxima.
        """
        return engine.pmnm_rate(self.dt, self._variance_dot, self._variance, self._variance_bar)

    @cached_property
    def pmnm_disp(self):
        """
        Mean cumulative PMNM count of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement positive-minima and negative-maxima.
        """
        return engine.pmnm_rate(self.dt, self._variance, self._variance_bar, self._variance_2bar)
