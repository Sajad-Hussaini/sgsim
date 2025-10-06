import numpy as np
from scipy.fft import irfft
from . import model_engine
from . import parametric_functions
from .model_core import ModelCore
from ..motion.ground_motion import GroundMotion

class StochasticModel(ModelCore):
    """
    Construct a stochastic simulation model for calibration of model parameters and simulation of ground motions.

    Inherits from `ModelCore` and provides methods for simulating ground motions and summarizing or loading model parameters.
    """
    def simulate(self, n: int, seed: int = None):
        """
        Simulate ground motions using the calibrated stochastic model.

        Parameters
        ----------
        n : int
            Number of simulations to generate.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        ac : ndarray
            Simulated acceleration time series, shape (n, npts).
        vel : ndarray
            Simulated velocity time series, shape (n, npts).
        disp : ndarray
            Simulated displacement time series, shape (n, npts).
        """
        self.stats
        n = int(n)
        white_noise = np.random.default_rng(seed).standard_normal((n, self.npts))
        fourier = model_engine.simulate_fourier_series(n, self.npts, self.t, self.freq_sim, self.freq_sim_p2,
                                                        self.mdl, self.wu, self.zu, self.wl, self.zl,
                                                        self.variance, white_noise)
        ac = irfft(fourier, workers=-1)[..., :self.npts]  # anti-aliasing
        # FT(w)/jw + pi*delta(w)*FT(0)  integration in freq domain
        vel = irfft(fourier[..., 1:] / (1j * self.freq_sim[1:]), workers=-1)[..., :self.npts]
        disp = irfft(-fourier[..., 1:] / (self.freq_sim[1:] ** 2), workers=-1)[..., :self.npts]
        return GroundMotion(self.npts, self.dt, ac, vel, disp)

    def summary(self, filename: str = None):
        """
        Print all model parameters to the console and optionally save to a plain text file.

        A stochastic model can be initiated from the saved file using the class method `from_file`.

        Parameters
        ----------
        filename : str, optional
            The name of the text file to save the data to. If None, only prints to console.

        Returns
        -------
        self : StochasticModel
            The StochasticModel instance (for chaining).
        """
        param_lines = {
            "Time Step (dt)": f"{self.dt}",
            "Number of Points (npts)": f"{self.npts}",
            "Duration (tn)": f"{self.t[-1]:.2f}",}

        func_lines = {}
        for name, func, params in [
            ("Modulating (mdl)", self.mdl_func, self.mdl_params),
            ("Upper Frequency (wu)", self.wu_func, self.wu_params),
            ("Lower Frequency (wl)", self.wl_func, self.wl_params),
            ("Upper Damping (zu)", self.zu_func, self.zu_params),
            ("Lower Damping (zl)", self.zl_func, self.zl_params)]:
            param_str = ', '.join(f'{p:.3f}' for p in params)
            func_lines[name] = f"{func.__name__} ({param_str})"

        max_label_len = max(len(label) for label in list(param_lines.keys()) + list(func_lines.keys()))

        title = "Stochastic Model Summary " + "=" * 30
        print(title)

        for label, value in param_lines.items():
            print(f"{label:<{max_label_len}} : {value}")
        print("-" * len(title))

        for label, value in func_lines.items():
            print(f"{label:<{max_label_len}} : {value}")
        print("-" * len(title))

        if filename:
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
    def from_file(cls, filename: str):
        """
        Construct a stochastic model using loaded model parameters from a plain text file.

        Parameters
        ----------
        filename : str
            The name of the text file to load the data from.

        Returns
        -------
        StochasticModel
            An instance of StochasticModel initialized from the file.
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
