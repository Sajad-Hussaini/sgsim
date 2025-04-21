import numpy as np
from .domain_config import DomainConfig

class ModelConfig(DomainConfig):
    """
    This class allows to configure time, frequency, and model parametric functions.
    """
    FAS, CE, VARIANCES, MLE_AC, MLE_VEL, MLE_DISP, MZC_AC, MZC_VEL, MZC_DISP, PMNM_AC, PMNM_VEL, PMNM_DISP = (
    1 << i for i in range(12))  # bit flags for dependent attributes

    def __init__(self, npts: int, dt: float, modulating,
                 upper_frequency, upper_damping,
                 lower_frequency, lower_damping):
        super().__init__(npts, dt)
        self.mdl_func = modulating
        self.wu_func = upper_frequency
        self.zu_func = upper_damping
        self.wl_func = lower_frequency
        self.zl_func = lower_damping

        (self._mdl, self._wu, self._zu, self._wl, self._zl,
         self.variance, self.variance_dot, self.variance_2dot, self.variance_bar, self.variance_2bar,
         self._ce, self._mle_ac, self._mle_vel, self._mle_disp, self._mzc_ac, self._mzc_vel, self._mzc_disp,
         self._pmnm_ac, self._pmnm_vel, self._pmnm_disp) = np.empty((20, self.npts))
        self._fas = np.empty_like(self.freq)
    
    def _set_dirty_flag(self):
        " Set a dirty flag on dependent attributes upon core attribute changes. "
        self._dirty_flags = (1 << 12) - 1  # Set all flags to dirty using bit flag 0b111111111111 (fun overkill!)

    @property
    def mdl(self):
        """ Modulating function """
        return self._mdl

    @mdl.setter
    def mdl(self, params):
        self._mdl[:] = self.mdl_func(self.t, *params)
        self.mdl_params = tuple(p for p in params if np.isscalar(p))
        self._set_dirty_flag()

    @property
    def wu(self):
        """ Upper dominant frequency """
        return self._wu

    @wu.setter
    def wu(self, params):
        self._wu[:] = self.wu_func(self.t, *params)
        self.wu_params = tuple(p for p in params if np.isscalar(p))
        self._wu *= 2 * np.pi  # Convert to angular frequency
        self._set_dirty_flag()

    @property
    def wl(self):
        """ Lower dominant frequency """
        return self._wl

    @wl.setter
    def wl(self, params):
        self._wl[:] = self.wl_func(self.t, *params)
        self.wl_params = tuple(p for p in params if np.isscalar(p))
        self._wl *= 2 * np.pi  # Convert to angular frequency
        self._set_dirty_flag()

    @property
    def zu(self):
        """ Upper damping ratio """
        return self._zu

    @zu.setter
    def zu(self, params):
        self._zu[:] = self.zu_func(self.t, *params)
        self.zu_params = tuple(p for p in params if np.isscalar(p))
        self._set_dirty_flag()

    @property
    def zl(self):
        """ Lower damping ratio """
        return self._zl

    @zl.setter
    def zl(self, params):
        self._zl[:] = self.zl_func(self.t, *params)
        self.zl_params = tuple(p for p in params if np.isscalar(p))
        self._set_dirty_flag()