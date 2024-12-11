import numpy as np
from scipy.fft import irfft
from numba import jit, prange, float64, complex128, int64
from . import freq_filter_engine
from .model_core import ModelCore

class StochasticModel(ModelCore):
    """
    This class allows to
        Simulate ground motion time series (acceleration, velocity, displacement)
    """
    def __init__(self, npts: int, dt: float, mdl_type: str = 'beta_multi',
                 wu_type: str = 'linear', zu_type: str = 'linear',
                 wl_type: str = 'linear', zl_type: str = 'linear'):
        super().__init__(npts, dt, mdl_type, wu_type, zu_type, wl_type, zl_type)
        self.seed = None

    def set_seed(self, number):
        """
        A fixed seed for White noise to avoid stochastic behavior
        None value produces variability
        """
        self.seed = number
        return self

    @staticmethod
    @jit(complex128[:, :](int64, int64, float64[:], float64[:], float64[:],
                       float64[:], float64[:], float64[:], float64[:], float64[:],
                       float64[:, :]), nopython=True, parallel=True)
    def _simulate_fourier(nsim, npts, t, freq_sim, mdl, wu, zu, wl, zl, variance, white_noise):
        """
        The simulated Fourier of nsim number of simulation.
        """
        sim_fourier = np.zeros((nsim, len(freq_sim)), dtype=np.complex128)
        for sim in prange(nsim):
            for i in range(npts):
                sim_fourier[sim, :] += (freq_filter_engine.get_frf(wu[i], zu[i], wl[i], zl[i], freq_sim)
                                        * np.exp(-1j * freq_sim * t[i])
                                        * white_noise[sim][i] * mdl[i] / np.sqrt(variance[i] * 2 / npts))
        return sim_fourier

    def simulate(self, nsim) -> tuple[np.array, np.array, np.array]:
        """
        Simulate ground-motions using fitted model parameters
        Based on the frequency representation for `nsim` number of simulations.
        Update: ac, vel, disp of simulations.
        """
        self.ac = self.vel = self.disp = None
        nsim = int(nsim)
        chunk_size = 5
        self.ac = np.empty((nsim, self.npts))
        self.vel = np.empty((nsim, self.npts))
        self.disp = np.empty((nsim, self.npts))
        # Process simulations in chunks
        for start in range(0, nsim, chunk_size):
            end = min(start + chunk_size, nsim)
            nsim_chunk = end - start
            white_noise = np.random.default_rng(seed=self.seed).standard_normal((nsim_chunk, self.npts))
            sim_fourier = self._simulate_fourier(nsim_chunk, self.npts, self.t, self.freq_sim,
                                                 self.mdl, self.wu, self.zu, self.wl, self.zl,
                                                 self.variance, white_noise)
            self.ac[start:end] = irfft(sim_fourier)[..., :self.npts]  # to avoid aliasing
            # FT(w)/jw + pi*delta(w)*FT(0)  integration in freq domain
            self.vel[start:end] = irfft(sim_fourier[..., 1:] / (1j * self.freq_sim[1:]))[..., :self.npts]
            self.disp[start:end] = irfft(-sim_fourier[..., 1:] / (self.freq_sim[1:] ** 2))[..., :self.npts]
        return self