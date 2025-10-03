import numpy as np
from .domain_config import DomainConfig

class ModelConfig(DomainConfig):
    """
    Configure time, frequency, and model parametric functions for simulation models.

    Inherits from `DomainConfig` and provides properties for setting and retrieving model parameter functions and their values.
    """
    FAS, CE, NCE, VARIANCES, MLE_AC, MLE_VEL, MLE_DISP, MZC_AC, MZC_VEL, MZC_DISP, PMNM_AC, PMNM_VEL, PMNM_DISP = (1 << i for i in range(13))  # bit flags for dependent attributes

    def __init__(self, npts: int, dt: float, modulating: callable,
                 upper_frequency: callable, upper_damping: callable,
                 lower_frequency: callable, lower_damping: callable):
        super().__init__(npts, dt)
        self.mdl_func = modulating
        self.wu_func = upper_frequency
        self.zu_func = upper_damping
        self.wl_func = lower_frequency
        self.zl_func = lower_damping

        (self._mdl, self._wu, self._zu, self._wl, self._zl,
         self.variance, self.variance_dot, self.variance_2dot, self.variance_bar, self.variance_2bar,
         self._ce, self._nce, self._mle_ac, self._mle_vel, self._mle_disp, self._mzc_ac, self._mzc_vel, self._mzc_disp,
         self._pmnm_ac, self._pmnm_vel, self._pmnm_disp) = np.empty((21, self.npts))
        self._fas = np.empty_like(self.freq)
    
    def _set_dirty_flag(self):
        """
        Set a dirty flag on dependent attributes upon core attribute changes.
        """
        self._dirty_flags = (1 << 13) - 1

    @property
    def mdl(self):
        """
        ndarray: Modulating function using time as the clocking variable.
        """
        return self._mdl

    @mdl.setter
    def mdl(self, params):
        """
        Set the modulating function parameters.

        Parameters
        ----------
        params : tuple
            Parameters for the modulating function.
        """
        self._mdl[:] = self.mdl_func(self.t, *params)
        self.mdl_params = tuple(p for p in params if np.isscalar(p))
        self._set_dirty_flag()

    @property
    def wu(self):
        """
        ndarray: Upper dominant frequency using normalized cumulative energy as the clocking variable.
        """
        return self._wu

    @wu.setter
    def wu(self, params):
        """
        Set the upper dominant frequency parameters.

        Parameters
        ----------
        params : tuple
            Parameters for the upper frequency function.
        """
        self._wu[:] = self.wu_func(self.nce, *params)
        self._wu[:] = np.clip(self._wu, 0.1, 30.0)
        self.wu_params = tuple(p for p in params if np.isscalar(p))
        self._wu *= 2 * np.pi  # Convert to angular frequency
        self._set_dirty_flag()

    @property
    def wl(self):
        """
        ndarray: Lower dominant frequency using normalized cumulative energy as the clocking variable.
        """
        return self._wl

    @wl.setter
    def wl(self, params):
        """
        Set the lower dominant frequency parameters.

        Parameters
        ----------
        params : tuple
            Parameters for the lower frequency function.
        """
        self._wl[:] = self.wl_func(self.nce, *params)
        self._wl[:] = np.clip(self._wl, 0.1, 10.0)
        self.wl_params = tuple(p for p in params if np.isscalar(p))
        self._wl *= 2 * np.pi  # Convert to angular frequency
        self._set_dirty_flag()

    @property
    def zu(self):
        """
        ndarray: Upper damping ratio using normalized cumulative energy as the clocking variable.
        """
        return self._zu

    @zu.setter
    def zu(self, params):
        """
        Set the upper damping ratio parameters.

        Parameters
        ----------
        params : tuple
            Parameters for the upper damping function.
        """
        self._zu[:] = self.zu_func(self.nce, *params)
        self._zu[:] = np.clip(self._zu, 0.1, 5.0)
        self.zu_params = tuple(p for p in params if np.isscalar(p))
        self._set_dirty_flag()

    @property
    def zl(self):
        """
        ndarray: Lower damping ratio using normalized cumulative energy as the clocking variable.
        """
        return self._zl

    @zl.setter
    def zl(self, params):
        """
        Set the lower damping ratio parameters.

        Parameters
        ----------
        params : tuple
            Parameters for the lower damping function.
        """
        self._zl[:] = self.zl_func(self.nce, *params)
        self._zl[:] = np.clip(self._zl, 0.1, 5.0)
        self.zl_params = tuple(p for p in params if np.isscalar(p))
        self._set_dirty_flag()