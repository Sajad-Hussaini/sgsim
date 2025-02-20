import numpy as np
from scipy.fft import irfft
import h5py
from . import model_engine
from .model_core import ModelCore

class StochasticModel(ModelCore):
    """
    This class allows to initiate a stochastic simulation model
        to calibrate model parameters
        to simulate ground motions using calibrated parameters
    """
    def __init__(self, npts: int, dt: float,
                 modulating_function: str = 'beta_single',
                 upper_dominant_frequency_function: str = 'linear', upper_damping_ratio_function: str = 'linear',
                 lower_dominant_frequency_function: str = 'linear', lower_damping_ratio_function: str = 'linear'):
        super().__init__(npts, dt, modulating_function,
                         upper_dominant_frequency_function, upper_damping_ratio_function,
                         lower_dominant_frequency_function, lower_damping_ratio_function)
        self._seed = None

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, value):
        self._seed = value

    def simulate(self, n: int):
        """
        Simulate ground motions using the calibrated stochastic model
            acceleration, velocity, displacement time series
        """
        self.stats
        n = int(n)
        white_noise = np.random.default_rng(self._seed).standard_normal((n, self.npts))
        fourier = model_engine.simulate_fourier_series(n, self.npts, self.t, self.freq_sim,
                                                        self.mdl, self.wu, self.zu, self.wl, self.zl,
                                                        self.variance, white_noise)
        self.ac = irfft(fourier, workers=-1)[..., :self.npts]  # to avoid aliasing
        # FT(w)/jw + pi*delta(w)*FT(0)  integration in freq domain
        self.vel = irfft(fourier[..., 1:] / (1j * self.freq_sim[1:]), workers=-1)[..., :self.npts]
        self.disp = irfft(-fourier[..., 1:] / (self.freq_sim[1:] ** 2), workers=-1)[..., :self.npts]
        return self

    def save_parameters(self, filename: str):
        """
        Save all model parameters to an HDF5 file with improved structure.
        A stochastic model can be initiated from the saved file using the class method from_hdf5.
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
            parameter_group.create_dataset('t', data=self.t)
            # Save function types as attributes
            parameter_group['mdl'].attrs['func'] = self.mdl_func.__name__
            parameter_group['wu'].attrs['func'] = self.wu_func.__name__
            parameter_group['zu'].attrs['func'] = self.zu_func.__name__
            parameter_group['wl'].attrs['func'] = self.wl_func.__name__
            parameter_group['zl'].attrs['func'] = self.zl_func.__name__
            parameter_group.attrs['description'] = 'Parameters of the Stochastic model: Modulating, Upper and lower frequencies and damping ratios'
        return self

    @classmethod
    def from_file(cls, filename: str) -> 'StochasticModel':
        """
        Load model parameters from an HDF5 file.
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
        model = cls(npts=int(npts), dt=float(dt), modulating_function=mdl_func, 
                upper_dominant_frequency_function=wu_func, upper_damping_ratio_function=zu_func, 
                lower_dominant_frequency_function=wl_func, lower_damping_ratio_function=zl_func)
        model.mdl = mdl_params
        model.wu = wu_params
        model.zu = zu_params
        model.wl = wl_params
        model.zl = zl_params
        return model
