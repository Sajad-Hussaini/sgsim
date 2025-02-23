import numpy as np
from scipy.fft import irfft
from . import model_engine
from . import parametric_functions
from .model_core import ModelCore

class StochasticModel(ModelCore):
    """
    This class allows to construct a stochastic simulation model
    for calibrattion of model parameters and simulation of ground motions
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # Pass all arguments to the superclass
        self._seed = None  # Initialize seed without modifying superclass behavior

    @property
    def seed(self):
        return self._seed

    @seed.setter
    def seed(self, fixed_seed: int):
        self._seed = fixed_seed

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
    
    def show_parameters(self):
        """ Print all model parameters to the console. """
        print()
        print(f"modulating_func (mdl): {self.mdl_func.__name__} {", ".join(map(str, self.mdl_params))}")
        print(f"upper_frequency_func (wu): {self.wu_func.__name__} {", ".join(map(str, self.wu_params))}")
        print(f"lower_frequency_func (wl): {self.wl_func.__name__} {", ".join(map(str, self.wl_params))}")
        print(f"upper_damping_func (zu): {self.zu_func.__name__} {", ".join(map(str, self.zu_params))}")
        print(f"lower_damping_func (zl): {self.zl_func.__name__} {", ".join(map(str, self.zl_params))}")

    def save_parameters(self, filename: str):
        """
        Save all model parameters to a plain text file.
        A stochastic model can be initiated from the saved file using the class method from_file.
        filename: The name of the text file to save the data to.
        """
        with open(filename, 'w') as file:
            file.write("SGSIM: Stochastic Simulation Model Parameters\n")
            file.write(f"npts={self.npts}\n")
            file.write(f"dt={self.dt}\n")

            file.write(f"modulating_func={self.mdl_func.__name__}\n")
            file.write(f"modulating_params={','.join(map(str, self.mdl_params))}\n")

            file.write(f"upper_frequency_func={self.wu_func.__name__}\n")
            file.write(f"upper_frequency_params={','.join(map(str, self.wu_params))}\n")

            file.write(f"upper_damping_func={self.zu_func.__name__}\n")
            file.write(f"upper_damping_params={','.join(map(str, self.zu_params))}\n")

            file.write(f"lower_frequency_func={self.wl_func.__name__}\n")
            file.write(f"lower_frequency_params={','.join(map(str, self.wl_params))}\n")

            file.write(f"lower_damping_func={self.zl_func.__name__}\n")
            file.write(f"lower_damping_params={','.join(map(str, self.zl_params))}\n")
        return self

    @classmethod
    def from_file(cls, filename: str) -> 'StochasticModel':
        """
        Construct a stochastic model using loaded model parameters from a plain text file.
        filename: The name of the text file to load the data from.
        """
        params = {}
        with open(filename, 'r') as file:
            # Skip the header line
            next(file)
            for line in file:
                key, value = line.strip().split('=')
                params[key] = value

        # Create a new Stochastic Model instance with the loaded function types
        model = cls(npts=int(params['npts']), dt=float(params['dt']),
                    modulating=getattr(parametric_functions, params['modulating_func']),
                    upper_frequency=getattr(parametric_functions, params['upper_frequency_func']),
                    upper_damping=getattr(parametric_functions, params['upper_damping_func']),
                    lower_frequency=getattr(parametric_functions, params['lower_frequency_func']),
                    lower_damping=getattr(parametric_functions, params['lower_damping_func']))
        model.mdl = tuple(map(float, params['modulating_params'].split(',')))
        model.wu = tuple(map(float, params['upper_frequency_params'].split(',')))
        model.zu = tuple(map(float, params['upper_damping_params'].split(',')))
        model.wl = tuple(map(float, params['lower_frequency_params'].split(',')))
        model.zl = tuple(map(float, params['lower_damping_params'].split(',')))
        return model
