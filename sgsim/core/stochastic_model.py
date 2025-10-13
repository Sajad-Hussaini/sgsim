import json
import numpy as np
from scipy.fft import irfft
from . import model_engine
from . import parametric_functions
from .model_config import ModelConfig
from ..motion.ground_motion import GroundMotion
from ..optimization.fit_eval import goodness_of_fit

class StochasticModel(ModelConfig):
    """
    Stochastic ground motion simulation model.

    Parameters
    ----------
    npts : int
        Number of time points in the simulation.
    dt : float
        Time step of the simulation.
    modulating : ParametricFunction
        Time-varying modulating function.
    upper_frequency : ParametricFunction
        Upper frequency parameter function.
    upper_damping : ParametricFunction
        Upper damping parameter function.
    lower_frequency : ParametricFunction
        Lower frequency parameter function.
    lower_damping : ParametricFunction
        Lower damping parameter function.
    """

    def fit(self, component: str, motion: GroundMotion, fit_range: tuple = (0.01, 0.99),
                  initial_guess=None, bounds=None, method='L-BFGS-B'):
        """
        Fit stochastic model parameters to match target motion.

        Parameters
        ----------
        component : str
            Component to fit ('modulating', 'frequency', or 'damping').
        motion : GroundMotion
            The target ground motion.
        fit_range : tuple, optional
            Fitting range as (min, max).
        initial_guess : array-like, optional
            Initial parameter values.
        bounds : list of tuples, optional
            Parameter bounds.
        method : str, optional
            Optimization method.

        Returns
        -------
        StochasticModel
            Self for method chaining.
        """
        from ..optimization import model_fit
        model_fit.fit(component=component, model=self, motion=motion, fit_range=fit_range,
                      initial_guess=initial_guess, bounds=bounds, method=method)
        return self

    def simulate(self, n, tag=None, seed=None):
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
        GroundMotion
            Simulated ground motions with acceleration, velocity, and displacement.
        """
        self._stats
        n = int(n)
        white_noise = np.random.default_rng(seed).standard_normal((n, self.npts))
        fourier = model_engine.simulate_fourier_series(n, self.npts, self.t, self.freq_sim, self.freq_sim_p2,
                                                        self.modulating.values, self.upper_frequency.values * 2 * np.pi, self.upper_damping.values,
                                                        self.lower_frequency.values * 2 * np.pi, self.lower_damping.values, self._variance, white_noise)
        ac = irfft(fourier, workers=-1)[..., :self.npts]  # anti-aliasing
        # FT(w)/jw + pi*delta(w)*FT(0)  integration in freq domain
        vel = irfft(fourier[..., 1:] / (1j * self.freq_sim[1:]), workers=-1)[..., :self.npts]
        disp = irfft(-fourier[..., 1:] / (self.freq_sim_p2[1:]), workers=-1)[..., :self.npts]
        return GroundMotion(self.npts, self.dt, ac, vel, disp, tag=tag)

    def simulate_conditional(self, n: int, target: GroundMotion, metrics: dict, max_iter: int = 100):
        """
        Conditionally simulate ground motions until all GoF metrics conditions are met or exceed their thresholds.

        Parameters
        ----------
        n : int
            Number of simulations to generate.
        target : GroundMotion
            The target ground motion to compare against.
        metrics : dict
            Conditioning metrics with GoF thresholds, e.g., {'sa': 0.9, 'sv': 0.85}.
        max_iter : int, optional
            Maximum number of simulation attempts.

        Returns
        -------
        GroundMotion
            Simulated ground motions meeting all GoF thresholds.

        Raises
        ------
        RuntimeError
            If not enough simulations meet the thresholds within max_iter * n attempts.
        """
        successful = []
        attempts = 0
        while len(successful) < n and attempts < max_iter * n:
            simulated = self.simulate(1, tag=attempts)
            gof_scores = {}
            for metric in metrics:
                sim_attr = getattr(simulated, metric)
                target_attr = getattr(target, metric)
                gof_scores[metric] = goodness_of_fit(sim_attr, target_attr)
            if all(gof_scores[m] >= metrics[m] for m in metrics):
                successful.append(simulated)
            attempts += 1

        if len(successful) < n:
            raise RuntimeError(f"Only {len(successful)} simulations met all GoF thresholds after {attempts} attempts.")

        ac = np.concatenate([gm.ac for gm in successful], axis=0)
        vel = np.concatenate([gm.vel for gm in successful], axis=0)
        disp = np.concatenate([gm.disp for gm in successful], axis=0)
        return GroundMotion(self.npts, self.dt, ac, vel, disp, tag=len(successful))

    def summary(self, filename=None):
        """
        Print model parameters and optionally save to JSON file.

        Parameters
        ----------
        filename : str, optional
            Path to JSON file for saving model data.

        Returns
        -------
        StochasticModel
            Self for method chaining.
        """
        title = "Stochastic Model Summary " + "=" * 30
        print(title)
        print(f"{'Time Step (dt)':<25} : {self.dt}")
        print(f"{'Number of Points (npts)':<25} : {self.npts}")
        print("-" * len(title))
        print(f"{'Modulating':<25} : {self.modulating}")
        print(f"{'Upper Frequency':<25} : {self.upper_frequency}")
        print(f"{'Lower Frequency':<25} : {self.lower_frequency}")
        print(f"{'Upper Damping':<25} : {self.upper_damping}")
        print(f"{'Lower Damping':<25} : {self.lower_damping}")
        print("-" * len(title))

        if filename:
            model_data = {
                'npts': self.npts,
                'dt': self.dt,
                'modulating': {
                    'func': self.modulating.__class__.__name__,
                    'params': self.modulating.params
                },
                'upper_frequency': {
                    'func': self.upper_frequency.__class__.__name__,
                    'params': self.upper_frequency.params
                },
                'upper_damping': {
                    'func': self.upper_damping.__class__.__name__,
                    'params': self.upper_damping.params
                },
                'lower_frequency': {
                    'func': self.lower_frequency.__class__.__name__,
                    'params': self.lower_frequency.params
                },
                'lower_damping': {
                    'func': self.lower_damping.__class__.__name__,
                    'params': self.lower_damping.params
                }
            }
            with open(filename, 'w') as file:
                json.dump(model_data, file, indent=2)
            print(f"Model saved to: {filename}")
        return self

    @classmethod
    def load_from(cls, filename):
        """
        Construct a stochastic model from a JSON file.

        Parameters
        ----------
        filename : str
            Path to JSON file containing model data.

        Returns
        -------
        StochasticModel
            Loaded stochastic model instance.
        """
        with open(filename, 'r') as file:
            data = json.load(file)
        
        model = cls(
            npts=data['npts'],
            dt=data['dt'],
            modulating=getattr(parametric_functions, data['modulating']['func']),
            upper_frequency=getattr(parametric_functions, data['upper_frequency']['func']),
            upper_damping=getattr(parametric_functions, data['upper_damping']['func']),
            lower_frequency=getattr(parametric_functions, data['lower_frequency']['func']),
            lower_damping=getattr(parametric_functions, data['lower_damping']['func']))
        
        model.modulating(model.t, **data['modulating']['params'])
        model.upper_frequency(model.t, **data['upper_frequency']['params'])
        model.upper_damping(model.t, **data['upper_damping']['params'])
        model.lower_frequency(model.t, **data['lower_frequency']['params'])
        model.lower_damping(model.t, **data['lower_damping']['params'])
        
        return model
