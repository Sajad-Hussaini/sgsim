import numpy as np
from scipy.fft import irfft
from numba import jit, prange, float64, complex128, int64
from . import filter_engine
from .model_core import ModelCore

class StochasticModel(ModelCore):
    """
    This class allows to initiate a stochatic simulation model
        to calibrate model parameters
        to simulate ground motions using calibrated parameters
    """
    def __init__(self, npts: int, dt: float,
                 mdl_type: str = 'beta_single',
                 wu_type: str = 'linear', zu_type: str = 'linear',
                 wl_type: str = 'linear', zl_type: str = 'linear'):
        super().__init__(npts, dt, mdl_type, wu_type, zu_type, wl_type, zl_type)
        self.rng = np.random.default_rng()
        self._seed = None

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value
        self.rng = np.random.default_rng(value)

    @staticmethod
    @jit(complex128[:, :](int64, int64, float64[:], float64[:], float64[:],
                       float64[:], float64[:], float64[:], float64[:], float64[:],
                       float64[:, :]), nopython=True, parallel=True, cache=True)
    def _simulate_fourier(n, npts, t, freq_sim, mdl, wu, zu, wl, zl, variance, white_noise):
        """
        The Fourier series of n number of simulations
        """
        fourier = np.zeros((n, len(freq_sim)), dtype=np.complex128)
        for sim in prange(n):
            for i in range(npts):
                fourier[sim, :] += (
                    filter_engine.get_frf(wu[i], zu[i], wl[i], zl[i], freq_sim)
                    * np.exp(-1j * freq_sim * t[i])
                    * white_noise[sim][i] * mdl[i] / np.sqrt(variance[i] * 2 / npts))
        return fourier

    def simulate(self, n: int):
        """
        Simulate ground motions using the calibrated stochastic model
            acceleration, velocity, displacement time series
        """
        self.stats
        n = int(n)
        chunk_size = 5
        self.ac = np.empty((n, self.npts))
        self.vel = np.empty((n, self.npts))
        self.disp = np.empty((n, self.npts))
        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            n_chunk = end - start
            white_noise = self.rng.standard_normal((n_chunk, self.npts))
            fourier = self._simulate_fourier(n_chunk, self.npts, self.t, self.freq_sim,
                                                 self.mdl, self.wu, self.zu, self.wl, self.zl,
                                                 self.variance, white_noise)
            self.ac[start:end] = irfft(fourier)[..., :self.npts]  # to avoid aliasing
            # FT(w)/jw + pi*delta(w)*FT(0)  integration in freq domain
            self.vel[start:end] = irfft(fourier[..., 1:] / (1j * self.freq_sim[1:]))[..., :self.npts]
            self.disp[start:end] = irfft(-fourier[..., 1:] / (self.freq_sim[1:] ** 2))[..., :self.npts]
        return self
