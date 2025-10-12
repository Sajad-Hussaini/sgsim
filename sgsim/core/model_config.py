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
    _FAS, _VARIANCE, _MLE, _MZC, _PMNM = (1 << i for i in range(5))  # bit flags for dependent attributes

    def __init__(self, npts: int, dt: float, modulating: ParametricFunction,
                 upper_frequency: ParametricFunction, upper_damping: ParametricFunction,
                 lower_frequency: ParametricFunction, lower_damping: ParametricFunction):
        super().__init__(npts, dt)
        self.modulating = modulating
        self.upper_frequency = upper_frequency
        self.upper_damping = upper_damping
        self.lower_frequency = lower_frequency
        self.lower_damping = lower_damping

        (self._variance, self._variance_dot, self._variance_2dot, self._variance_bar, self._variance_2bar,
         self._mle_ac, self._mle_vel, self._mle_disp, self._mzc_ac, self._mzc_vel, self._mzc_disp,
         self._pmnm_ac, self._pmnm_vel, self._pmnm_disp) = np.empty((14, self.npts))
        self._fas = np.empty_like(self.freq)

        for func in [self.modulating, self.upper_frequency, self.upper_damping, 
                     self.lower_frequency, self.lower_damping]:
            func.callback = self._set_dirty_flag
    
    def _set_dirty_flag(self):
        """Set a dirty flag on dependent attributes upon core attribute changes."""
        self._dirty_flags = (1 << 5) - 1

    @property
    def _stats(self):
        """Compute and store the variances for internal use."""
        if self._dirty_flags & self._VARIANCE:  # check bit flag
            model_engine.get_stats(self.upper_frequency.values * 2 * np.pi, self.upper_damping.values, self.lower_frequency.values * 2 * np.pi, self.lower_damping.values,
                                   self.freq_p2, self.freq_p4, self.freq_n2, self.freq_n4,
                                   self._variance, self._variance_dot, self._variance_2dot, self._variance_bar, self._variance_2bar)
            self._dirty_flags &= ~self._VARIANCE  # clear bit flag (set to 0)

    @property
    def fas(self):
        """
        Fourier amplitude spectrum (FAS) of the stochastic model.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        if self._dirty_flags & self._FAS:
            model_engine.get_fas(self.modulating.values, self.upper_frequency.values * 2 * np.pi, self.upper_damping.values,
                                 self.lower_frequency.values * 2 * np.pi, self.lower_damping.values, self.freq_p2, self.freq_p4, self._variance, self._fas)
            self._dirty_flags &= ~self._FAS
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
    def nce(self):
        """
        Normalized cumulative energy of the stochastic model.

        Returns
        -------
        ndarray
            Normalized cumulative energy time history.
        """
        ce = self.ce
        return ce / ce[-1]
    
    def _compute_mzc(self):
        """Compute and store all mean zero-crossing rates."""
        if self._dirty_flags & self._MZC:
            self._stats
            model_engine.cumulative_rate(self.dt, self._variance_dot, self._variance, self._mzc_ac)
            model_engine.cumulative_rate(self.dt, self._variance, self._variance_bar, self._mzc_vel)
            model_engine.cumulative_rate(self.dt, self._variance_bar, self._variance_2bar, self._mzc_disp)
            self._dirty_flags &= ~self._MZC
    
    def _compute_mle(self):
        """Compute and store all mean local extrema rates."""
        if self._dirty_flags & self._MLE:
            self._stats
            model_engine.cumulative_rate(self.dt, self._variance_2dot, self._variance_dot, self._mle_ac)
            model_engine.cumulative_rate(self.dt, self._variance_dot, self._variance, self._mle_vel)
            model_engine.cumulative_rate(self.dt, self._variance, self._variance_bar, self._mle_disp)
            self._dirty_flags &= ~self._MLE
    
    def _compute_pmnm(self):
        """Compute and store all mean positive minima and negative maxima rates."""
        if self._dirty_flags & self._PMNM:
            self._stats
            model_engine.pmnm_rate(self.dt, self._variance_2dot, self._variance_dot, self._variance, self._pmnm_ac)
            model_engine.pmnm_rate(self.dt, self._variance_dot, self._variance, self._variance_bar, self._pmnm_vel)
            model_engine.pmnm_rate(self.dt, self._variance, self._variance_bar, self._variance_2bar, self._pmnm_disp)
            self._dirty_flags &= ~self._PMNM

    @property
    def mle_ac(self):
        """
        Mean cumulative number of local extrema (peaks and valleys) of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration local extrema.
        """
        self._compute_mle()
        return self._mle_ac

    @property
    def mle_vel(self):
        """
        Mean cumulative number of local extrema (peaks and valleys) of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity local extrema.
        """
        self._compute_mle()
        return self._mle_vel

    @property
    def mle_disp(self):
        """
        Mean cumulative number of local extrema (peaks and valleys) of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement local extrema.
        """
        self._compute_mle()
        return self._mle_disp

    @property
    def mzc_ac(self):
        """
        Mean cumulative number of zero crossings (up and down) of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration zero crossings.
        """
        self._compute_mzc()
        return self._mzc_ac

    @property
    def mzc_vel(self):
        """
        Mean cumulative number of zero crossings (up and down) of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity zero crossings.
        """            
        self._compute_mzc()
        return self._mzc_vel

    @property
    def mzc_disp(self):
        """
        Mean cumulative number of zero crossings (up and down) of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement zero crossings.
        """
        self._compute_mzc()
        return self._mzc_disp

    @property
    def pmnm_ac(self):
        """
        Mean cumulative number of positive-minima and negative maxima of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration positive-minima and negative maxima.
        """
        self._compute_pmnm()
        return self._pmnm_ac

    @property
    def pmnm_vel(self):
        """
        Mean cumulative number of positive-minima and negative maxima of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity positive-minima and negative maxima.
        """
        self._compute_pmnm()
        return self._pmnm_vel

    @property
    def pmnm_disp(self):
        """
        Mean cumulative number of positive-minima and negative maxima of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement positive-minima and negative maxima.
        """
        self._compute_pmnm()
        return self._pmnm_disp
