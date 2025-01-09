import numpy as np
from scipy.fft import irfft
from numba import jit, prange, float64, complex128, int64
import h5py
from . import filter_engine
from .model_core import ModelCore

class StochasticModel(ModelCore):
    """
    This class allows to initiate a stochatic simulation model
        to calibrate model parameters
        to simulate ground motions using calibrated parameters
    """
    def __init__(self, npts: int, dt: float,
                 mdl_func: str = 'beta_single',
                 wu_func: str = 'linear', zu_func: str = 'linear',
                 wl_func: str = 'linear', zl_func: str = 'linear'):
        super().__init__(npts, dt, mdl_func, wu_func, zu_func, wl_func, zl_func)
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

    def save_hdf5(self, filename: str):
        """
        Save all relevant data to an HDF5 file with improved structure.

        filename: The name of the HDF5 file to save the data to.
        """
        with h5py.File(filename, 'w') as hdf:
            # Save parameters
            parameter_group = hdf.create_group('parameters')
            parameter_group.create_dataset('mdl', data=self.mdl_params)
            parameter_group.create_dataset('wu', data=self.wu_params)
            parameter_group.create_dataset('zu', data=self.zu_params)
            parameter_group.create_dataset('wl', data=self.wl_params)
            parameter_group.create_dataset('zl', data=self.zl_params)
            parameter_group.create_dataset('npts', data=self.npts)
            parameter_group.create_dataset('dt', data=self.dt)
            # Save function types as attributes
            parameter_group['mdl'].attrs['func'] = self.mdl_func.__name__
            parameter_group['wu'].attrs['func'] = self.wu_func.__name__
            parameter_group['zu'].attrs['func'] = self.zu_func.__name__
            parameter_group['wl'].attrs['func'] = self.wl_func.__name__
            parameter_group['zl'].attrs['func'] = self.zl_func.__name__
            # Save parameter names as attributes
            parameter_group['mdl'].attrs['vars'] = self.mdl_params_name
            parameter_group['wu'].attrs['vars'] = self.wu_params_name
            parameter_group['zu'].attrs['vars'] = self.zu_params_name
            parameter_group['wl'].attrs['vars'] = self.wl_params_name
            parameter_group['zl'].attrs['vars'] = self.zl_params_name
            parameter_group.attrs['description'] = 'Parameters of the Stochastic model: Modulating, Upper and lower frequencies and damping ratios'
        return self

    @classmethod
    def from_hdf5(cls, filename: str):
        """
        Load data from an HDF5 file and create a Motion instance.

        filename: The name of the HDF5 file to load the data from.
        """
        with h5py.File(filename, 'r') as hdf:
            # Load parameters
            mdl_params = hdf['parameters/mdl'][:]
            wu_params = hdf['parameters/wu'][:]
            zu_params = hdf['parameters/zu'][:]
            wl_params = hdf['parameters/wl'][:]
            zl_params = hdf['parameters/zl'][:]
            npts = hdf['parameters/npts'][:]
            dt = hdf['parameters/dt'][:]
            # Load function types
            mdl_func = hdf['parameters/mdl'].attrs['func']
            wu_func = hdf['parameters/wu'].attrs['func']
            zu_func = hdf['parameters/zu'].attrs['func']
            wl_func = hdf['parameters/wl'].attrs['func']
            zl_func = hdf['parameters/zl'].attrs['func']
        # Create a new Stochastic Model instance with the loaded function types
        model = cls(npts=npts, dt=dt, mdl_func=mdl_func, wu_func=wu_func, zu_func=zu_func, wl_func=wl_func, zl_func=zl_func)
        model.mdl = mdl_params
        model.wu = wu_params
        model.zu = zu_params
        model.wl = wl_params
        model.zl = zl_params
        return model
