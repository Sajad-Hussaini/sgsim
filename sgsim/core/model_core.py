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
        if self._dirty_flags & self.VARIANCES:  # check bit flag
            model_engine.get_stats(self.wu, self.zu, self.wl, self.zl, self.freq_p2, self.freq_p4, self.freq_n2, self.freq_n4, self.variance, self.variance_dot, self.variance_2dot, self.variance_bar, self.variance_2bar)
            self._dirty_flags &= ~self.VARIANCES  # clear bit flag (set to 0)

    @property
    def fas(self):
        """ The Fourier amplitude spectrum (FAS) of the stochastic model using model's PSD """
        if self._dirty_flags & self.FAS:
            model_engine.get_fas(self.mdl, self.wu, self.zu, self.wl, self.zl, self.freq_p2, self.freq_p4, self._fas)
            self._dirty_flags &= ~self.FAS
        return self._fas

    @property
    def ce(self):
        """ The Cumulative energy of the stochastic model. """
        if self._dirty_flags & self.CE:
            self._ce[:] = signal_analysis.get_ce(self.dt, self.mdl)
            self._dirty_flags &= ~self.CE
        return self._ce
    
    @property
    def mle_ac(self):
        """ The mean cumulative number of local extream (peaks and valleys) of the acceleration model """
        if self._dirty_flags & self.MLE_AC:
            self.stats
            model_engine.cumulative_rate(self.dt, self.variance_2dot, self.variance_dot, self._mle_ac)
            self._dirty_flags &= ~self.MLE_AC
        return self._mle_ac

    @property
    def mle_vel(self):
        """ The mean cumulative number of local extream (peaks and valleys) of the velocity model """
        if self._dirty_flags & self.MLE_VEL:
            self.stats
            model_engine.cumulative_rate(self.dt, self.variance_dot, self.variance, self._mle_vel)
            self._dirty_flags &= ~self.MLE_VEL
        return self._mle_vel

    @property
    def mle_disp(self):
        """ The mean cumulative number of local extream (peaks and valleys) of the displacement model """
        if self._dirty_flags & self.MLE_DISP:
            self.stats
            model_engine.cumulative_rate(self.dt, self.variance, self.variance_bar, self._mle_disp)
            self._dirty_flags &= ~self.MLE_DISP
        return self._mle_disp

    @property
    def mzc_ac(self):
        """ The mean cumulative number of zero crossing (up and down) of the acceleration model """
        if self._dirty_flags & self.MZC_AC:
            self.stats
            model_engine.cumulative_rate(self.dt, self.variance_dot, self.variance, self._mzc_ac)
            self._dirty_flags &= ~self.MZC_AC
        return self._mzc_ac

    @property
    def mzc_vel(self):
        """ The mean cumulative number of zero crossing (up and down) of the velocity model """
        if self._dirty_flags & self.MZC_VEL:
            self.stats
            model_engine.cumulative_rate(self.dt, self.variance, self.variance_bar, self._mzc_vel)
            self._dirty_flags &= ~self.MZC_VEL
        return self._mzc_vel

    @property
    def mzc_disp(self):
        """ The mean cumulative number of zero crossing (up and down) of the displacement model """
        if self._dirty_flags & self.MZC_DISP:
            self.stats
            model_engine.cumulative_rate(self.dt, self.variance_bar, self.variance_2bar, self._mzc_disp)
            self._dirty_flags &= ~self.MZC_DISP
        return self._mzc_disp

    @property
    def pmnm_ac(self):
        """ The mean cumulative number of positive-minima and negative maxima of the acceleration model """
        if self._dirty_flags & self.PMNM_AC:
            self.stats
            model_engine.pmnm_rate(self.dt, self.variance_2dot, self.variance_dot, self.variance, self._pmnm_ac)
            self._dirty_flags &= ~self.PMNM_AC
        return self._pmnm_ac

    @property
    def pmnm_vel(self):
        """ The mean cumulative number of positive-minima and negative maxima of the velocity model """
        if self._dirty_flags & self.PMNM_VEL:
            self.stats
            model_engine.pmnm_rate(self.dt, self.variance_dot, self.variance, self.variance_bar, self._pmnm_vel)
            self._dirty_flags &= ~self.PMNM_VEL
        return self._pmnm_vel

    @property
    def pmnm_disp(self):
        """ The mean cumulative number of positive-minima and negative maxima of the displacement model """
        if self._dirty_flags & self.PMNM_DISP:
            self.stats
            model_engine.pmnm_rate(self.dt, self.variance, self.variance_bar, self.variance_2bar, self._pmnm_disp)
            self._dirty_flags &= ~self.PMNM_DISP
        return self._pmnm_disp
