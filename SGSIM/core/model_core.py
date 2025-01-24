import numpy as np
from . import filter_engine
from .model_config import ModelConfig
from ..motion import signal_analysis

class ModelCore(ModelConfig):
    """
    This class allows to
        calculate properties required in fitting to or simulating ground motions
        i.e., variances, FAS, CE, zero crossing, and local extrema of the stochastic model
    """
    def __init__(self, npts: int, dt: float,
                 mdl_func: str,
                 wu_func: str, zu_func: str,
                 wl_func: str, zl_func: str):
        super().__init__(npts, dt, mdl_func, wu_func, zu_func, wl_func, zl_func)
        self.reset_attributes()

    @property
    def stats(self):
        """ Computes and stores the variances for internal use. """
        if self.variance is None:
            self.variance, self.variance_dot, self.variance_2dot, self.variance_bar, self.variance_2bar = filter_engine.get_stats(self.wu, self.zu, self.wl, self.zl, self.freq)

    def reset_attributes(self):
        """ Reset specific attributes when model parameters change. """
        self._fasx = self._fas = self._ce = None
        self._mle_ac = self._mle_vel = self._mle_disp = None
        self._mzc_ac = self._mzc_vel = self._mzc_disp = None
        self._pmnm_ac = self._pmnm_vel = self._pmnm_disp = None
        self.variancex = self.variance = self.variance_dot = self.variance_2dot = self.variance_bar = self.variance_2bar = None

    @property
    def fas(self):
        """ The Fourier amplitude spectrum (FAS) of the stochastic model using model's PSD """
        if self._fas is None:
            self._fas = filter_engine.get_fas(self.mdl, self.wu, self.zu, self.wl, self.zl, self.freq)
        return self._fas

    @property
    def ce(self):
        """ The Cumulative energy of the stochastic model. """
        if self._ce is None:
            self._ce = signal_analysis.get_ce(self.dt, self.mdl)
        return self._ce

    @property
    def mle_ac(self):
        """ The mean cumulative number of local extream (peaks and valleys) of the acceleration model """
        self.stats
        if self._mle_ac is None:
            mle_rate_ac = (np.sqrt(self.variance_2dot / self.variance_dot) / (2 * np.pi))
            self._mle_ac = np.cumsum(mle_rate_ac) * self.dt
        return self._mle_ac

    @property
    def mle_vel(self):
        """ The mean cumulative number of local extream (peaks and valleys) of the velocity model """
        self.stats
        if self._mle_vel is None:
            mle_rate_vel = (np.sqrt(self.variance_dot / self.variance) / (2 * np.pi))
            self._mle_vel = np.cumsum(mle_rate_vel) * self.dt
        return self._mle_vel

    @property
    def mle_disp(self):
        """ The mean cumulative number of local extream (peaks and valleys) of the displacement model """
        self.stats
        if self._mle_disp is None:
            mle_rate_disp = (np.sqrt(self.variance / self.variance_bar) / (2 * np.pi))
            self._mle_disp = np.cumsum(mle_rate_disp) * self.dt
        return self._mle_disp

    @property
    def mzc_ac(self):
        """ The mean cumulative number of zero crossing (up and down) of the acceleration model """
        self.stats
        if self._mzc_ac is None:
            mzc_rate_ac = (np.sqrt(self.variance_dot / self.variance) / (2 * np.pi))
            self._mzc_ac = np.cumsum(mzc_rate_ac) * self.dt
        return self._mzc_ac

    @property
    def mzc_vel(self):
        """ The mean cumulative number of zero crossing (up and down) of the velocity model """
        self.stats
        if self._mzc_vel is None:
            mzc_rate_vel = (np.sqrt(self.variance / self.variance_bar) / (2 * np.pi))
            self._mzc_vel = np.cumsum(mzc_rate_vel) * self.dt
        return self._mzc_vel

    @property
    def mzc_disp(self):
        """ The mean cumulative number of zero crossing (up and down) of the displacement model """
        self.stats
        if self._mzc_disp is None:
            mzc_rate_disp = (np.sqrt(self.variance_bar / self.variance_2bar) / (2 * np.pi))
            self._mzc_disp = np.cumsum(mzc_rate_disp) * self.dt
        return self._mzc_disp

    @property
    def pmnm_ac(self):
        """ The mean cumulative number of positive-minima and negative maxima of the acceleration model """
        self.stats
        if self._pmnm_ac is None:
            pmnm_rate_ac = (np.sqrt(self.variance_2dot / self.variance_dot) -
                            np.sqrt(self.variance_dot / self.variance)) / (4 * np.pi)
            self._pmnm_ac = np.cumsum(pmnm_rate_ac) * self.dt
        return self._pmnm_ac

    @property
    def pmnm_vel(self):
        """ The mean cumulative number of positive-minima and negative maxima of the velocity model """
        self.stats
        if self._pmnm_vel is None:
            pmnm_rate_vel = (np.sqrt(self.variance_dot / self.variance) -
                            np.sqrt(self.variance / self.variance_bar)) / (4 * np.pi)
            self._pmnm_vel = np.cumsum(pmnm_rate_vel) * self.dt
        return self._pmnm_vel

    @property
    def pmnm_disp(self):
        """ The mean cumulative number of positive-minima and negative maxima of the displacement model """
        self.stats
        if self._pmnm_disp is None:
            pmnm_rate_disp = (np.sqrt(self.variance / self.variance_bar) -
                              np.sqrt(self.variance_bar / self.variance_2bar)) / (4 * np.pi)
            self._pmnm_disp = np.cumsum(pmnm_rate_disp) * self.dt
        return self._pmnm_disp
