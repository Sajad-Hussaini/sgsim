import numpy as np
from SGSIM.core import filter_freq as ff
from SGSIM.core.model_config import ModelConfig
import SGSIM.data_processing.motion_processor as mps

class ModelCore(ModelConfig):
    """
    This class allows to
        calculate variance stats, FAS, CE, level crossing  and local extrema of the stochastic model
        access to ParametricFunction class methods and attributes (i.e., parameters)
    """
    def __init__(self, npts: int, dt: float, mdl_type: str = 'beta_multi',
                 wu_type: str = 'linear', zu_type: str = 'linear',
                 wl_type: str = 'linear', zl_type: str = 'linear'):
        super().__init__(npts, dt, mdl_type, wu_type, zu_type, wl_type, zl_type)

    def get_stats(self):
        """
        The statistics of the stochastic model using frequency domain.
        ignoring the modulating function and the variance of White noise (i.e., 1)
        self.variance :     variance                   using 0
        self.variance_dot:  variance 1st derivative    using 2
        self.variance_2dot: variance 2nd derivative    using 4
        self.variance_bar:  variance 1st integral      using -2
        self.variance_2bar: variance 2nd integral      using -4
        """
        self.variance, self.variance_dot, self.variance_2dot, self.variance_bar, self.variance_2bar = np.zeros((5, self.npts))
        for i in range(self.npts):
            stats_i = ff.get_stats(self.wu[i], self.zu[i], self.wl[i], self.zl[i], self.freq,
                                   statistics=(0, 2, 4, -2, -4))
            self.variance[i], self.variance_dot[i], self.variance_2dot[i], self.variance_bar[i], self.variance_2bar[i] = stats_i[:5]
        return self

    def get_fas(self):
        """
        The FAS of the stochastic model using frequency domain.
        """
        psd = np.zeros(len(self.freq))
        for i, mdl_i in enumerate(self.mdl):
            psd_i = ff.get_psd(self.wu[i], self.zu[i], self.wl[i], self.zl[i], self.freq)
            psd += (psd_i / np.sum(psd_i)) * mdl_i ** 2
        self.fas = np.sqrt(psd)
        return self.fas

    def get_ce(self):
        """
        The Cumulative energy of the stochastic model.
        """
        self.ce = mps.get_ce(self.dt, self.mdl)
        return self.ce

    def get_mle_ac(self) -> np.array:
        """
        The mean cumulative number of local extream (peaks and valleys)
        of the acceleration model
        without considering the modulation function
        """
        mle_rate_ac = (np.sqrt(self.variance_2dot / self.variance_dot) / (2 * np.pi))
        self.mle_ac = np.cumsum(mle_rate_ac) * self.dt
        return self.mle_ac

    def get_mle_vel(self) -> np.array:
        """
        The mean cumulative number of local extream (peaks and valleys)
        of the velocity model
        without considering the modulation function
        """
        mle_rate_vel = (np.sqrt(self.variance_dot / self.variance) / (2 * np.pi))
        self.mle_vel = np.cumsum(mle_rate_vel) * self.dt
        return self.mle_vel

    def get_mle_disp(self) -> np.array:
        """
        The mean cumulative number of local extream (peaks and valleys)
        of the displacement model
        without considering the modulation function
        """
        mle_rate_disp = (np.sqrt(self.variance / self.variance_bar) / (2 * np.pi))
        self.mle_disp = np.cumsum(mle_rate_disp) * self.dt
        return self.mle_disp

    def get_mzc_ac(self) -> np.array:
        """
        The mean cumulative number of zero crossing of the acceleration model
        without considering the modulation function
        """
        mzc_rate_ac = (np.sqrt(self.variance_dot / self.variance) / (2 * np.pi))
        self.mzc_ac = np.cumsum(mzc_rate_ac) * self.dt
        return self.mzc_ac

    def get_mzc_vel(self) -> np.array:
        """
        The mean cumulative number of zero crossing of the velocity model
        without considering the modulation function
        """
        mzc_rate_vel = (np.sqrt(self.variance / self.variance_bar) / (2 * np.pi))
        self.mzc_vel = np.cumsum(mzc_rate_vel) * self.dt
        return self.mzc_vel

    def get_mzc_disp(self) -> np.array:
        """
        The mean cumulative number of zero crossing of the displacement model
        without considering the modulation function
        """
        mzc_rate_disp = (np.sqrt(self.variance_bar / self.variance_2bar) / (2 * np.pi))
        self.mzc_disp = np.cumsum(mzc_rate_disp) * self.dt
        return self.mzc_disp

    def get_pmnm_ac(self) -> np.array:
        """
        The mean cumulative number of positive-minima / negative maxima
        of the acceleration model without considering the modulation function.
        """
        pmnm_rate_ac = (np.sqrt(self.variance_2dot / self.variance_dot) -
                       np.sqrt(self.variance_dot / self.variance)) / (4 * np.pi)
        self.pmnm_ac = np.cumsum(pmnm_rate_ac) * self.dt
        return self.pmnm_ac

    def get_pmnm_vel(self) -> np.array:
        """
        The mean cumulative number of positive-minima / negative maxima
        of the velocity model without considering the modulation function.
        """
        pmnm_rate_vel = (np.sqrt(self.variance_dot / self.variance) -
                        np.sqrt(self.variance / self.variance_bar)) / (4 * np.pi)
        self.pmnm_vel = np.cumsum(pmnm_rate_vel) * self.dt
        return self.pmnm_vel

    def get_pmnm_disp(self) -> np.array:
        """
        The mean cumulative number of positive-minima / negative maxima
        of the displacement model without considering the modulation function.
        """
        pmnm_rate_disp = (np.sqrt(self.variance / self.variance_bar) -
                         np.sqrt(self.variance_bar / self.variance_2bar)) / (4 * np.pi)
        self.pmnm_disp = np.cumsum(pmnm_rate_disp) * self.dt
        return self.pmnm_disp

    def get_properties(self):
        """
        Runs all statistical methods of the stochastic model and
        stores their results in instance attributes.
        """
        self.get_stats()
        self.get_fas()
        self.get_ce()
        self.get_mle_ac()
        self.get_mle_vel()
        self.get_mle_disp()
        self.get_mzc_ac()
        self.get_mzc_vel()
        self.get_mzc_disp()
        self.get_pmnm_ac()
        self.get_pmnm_vel()
        self.get_pmnm_disp()