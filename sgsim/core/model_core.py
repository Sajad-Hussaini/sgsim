import numpy as np
from . import model_engine
from .model_config import ModelConfig
from ..motion import signal_analysis

class ModelCore(ModelConfig):
    """
    This class allows to calculate core features of the stochastic model
    i.e., variances, FAS, CE, zero crossing, and local extrema of the stochastic model
    """
    @property
    def stats(self):
        """ Computes and stores the variances for internal use. """
        if not np.any(self.variance):
            model_engine.get_stats(self.wu, self.zu, self.wl, self.zl, self.freq, self.variance, self.variance_dot, self.variance_2dot, self.variance_bar, self.variance_2bar)

    @property
    def fas(self):
        """ The Fourier amplitude spectrum (FAS) of the stochastic model using model's PSD """
        if not np.any(self._fas):
            model_engine.get_fas(self.mdl, self.wu, self.zu, self.wl, self.zl, self.freq, self._fas)
        return self._fas

    @property
    def ce(self):
        """ The Cumulative energy of the stochastic model. """
        if not np.any(self._ce):
            self._ce[:] = signal_analysis.get_ce(self.dt, self.mdl)
        return self._ce

    @property
    def mle_ac(self):
        """ The mean cumulative number of local extream (peaks and valleys) of the acceleration model """
        self.stats
        if not np.any(self._mle_ac):
            np.cumsum((np.sqrt(self.variance_2dot / self.variance_dot) / (2 * np.pi)) * self.dt, out=self._mle_ac)
        return self._mle_ac

    @property
    def mle_vel(self):
        """ The mean cumulative number of local extream (peaks and valleys) of the velocity model """
        self.stats
        if not np.any(self._mle_vel):
            np.cumsum((np.sqrt(self.variance_dot / self.variance) / (2 * np.pi)) * self.dt, out=self._mle_vel)
        return self._mle_vel

    @property
    def mle_disp(self):
        """ The mean cumulative number of local extream (peaks and valleys) of the displacement model """
        self.stats
        if not np.any(self._mle_disp):
            np.cumsum((np.sqrt(self.variance / self.variance_bar) / (2 * np.pi)) * self.dt, out=self._mle_disp)
        return self._mle_disp

    @property
    def mzc_ac(self):
        """ The mean cumulative number of zero crossing (up and down) of the acceleration model """
        self.stats
        if not np.any(self._mzc_ac):
            np.cumsum((np.sqrt(self.variance_dot / self.variance) / (2 * np.pi)) * self.dt, out=self._mzc_ac)
        return self._mzc_ac

    @property
    def mzc_vel(self):
        """ The mean cumulative number of zero crossing (up and down) of the velocity model """
        self.stats
        if not np.any(self._mzc_vel):
            np.cumsum((np.sqrt(self.variance / self.variance_bar) / (2 * np.pi)) * self.dt, out=self._mzc_vel)
        return self._mzc_vel

    @property
    def mzc_disp(self):
        """ The mean cumulative number of zero crossing (up and down) of the displacement model """
        self.stats
        if not np.any(self._mzc_disp):
            np.cumsum((np.sqrt(self.variance_bar / self.variance_2bar) / (2 * np.pi)) * self.dt, out=self._mzc_disp)
        return self._mzc_disp

    @property
    def pmnm_ac(self):
        """ The mean cumulative number of positive-minima and negative maxima of the acceleration model """
        self.stats
        if not np.any(self._pmnm_ac):
            np.cumsum(((np.sqrt(self.variance_2dot / self.variance_dot) -
                        np.sqrt(self.variance_dot / self.variance)) / (4 * np.pi)) * self.dt, out=self._pmnm_ac)
        return self._pmnm_ac

    @property
    def pmnm_vel(self):
        """ The mean cumulative number of positive-minima and negative maxima of the velocity model """
        self.stats
        if not np.any(self._pmnm_vel):
            np.cumsum(((np.sqrt(self.variance_dot / self.variance) -
                       np.sqrt(self.variance / self.variance_bar)) / (4 * np.pi)) * self.dt, out=self._pmnm_vel)
        return self._pmnm_vel

    @property
    def pmnm_disp(self):
        """ The mean cumulative number of positive-minima and negative maxima of the displacement model """
        self.stats
        if not np.any(self._pmnm_disp):
            np.cumsum(((np.sqrt(self.variance / self.variance_bar) -
                       np.sqrt(self.variance_bar / self.variance_2bar)) / (4 * np.pi)) * self.dt, out=self._pmnm_disp)
        return self._pmnm_disp
