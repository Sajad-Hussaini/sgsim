import numpy as np
from scipy.fft import irfft
from . import engine
from .model_config import ModelConfig
from ..motion.ground_motion import GroundMotion
from ..optimization.fit_eval import goodness_of_fit

class StochasticModel:
    """
    Stochastic ground motion simulation model.

    Parameters
    ----------
    config : ModelConfig
        Model configuration.
    """
    def __init__(self, config: ModelConfig):
        self.config = config

    @classmethod
    def load_from(cls, filename):
        pass

    def simulate(self, n, tag=None, seed=None):
        """
        Simulate ground motions using the calibrated stochastic model.

        Parameters
        ----------
        n : int
            Number of simulations to generate.
        tag : any, optional
            Identifier for the simulation batch.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        GroundMotion
            Simulated ground motions with acceleration, velocity, and displacement.
        """
        n = int(n)
        white_noise = np.random.default_rng(seed).standard_normal((n, self.config.npts))
        fourier = engine.simulate_fourier_series(
            n, self.config.npts, self.config.t,
            self.config.freq_sim, self.config.freq_sim_p2,
            self.config.mdl,
            self.config.wu * 2 * np.pi, self.config.zu,
            self.config.wl * 2 * np.pi, self.config.zl,
            self.config._variance, white_noise, self.config.dt)
        # Default backward 1/N scaling is correct here
        ac = irfft(fourier, workers=-1)[..., :self.config.npts]  # Manual anti-aliasing
        # FT(w)/jw + pi*delta(w)*FT(0)  integration in freq domain
        fourier[..., 1:] /= (1j * self.config.freq_sim[1:])
        vel = irfft(fourier, workers=-1)[..., :self.config.npts]

        fourier[..., 1:] /= (1j * self.config.freq_sim[1:])
        disp = irfft(fourier, workers=-1)[..., :self.config.npts]
        
        return GroundMotion(self.config.npts, self.config.dt, ac, vel, disp, tag=tag)

    def simulate_conditional(self, n: int, target: GroundMotion, metrics: dict, max_iter: int = 100):
        """
        Conditionally simulate ground motions until all GoF metrics conditions are met.
        
        Uses the model's current `npts` and `dt` values.

        Parameters
        ----------
        n : int
            Number of simulations to generate.
        target : GroundMotion
            The target ground motion to compare against.
        metrics : dict
            Conditioning metrics with GoF thresholds, e.g., {'sa': 0.9, 'sv': 0.85}.
        max_iter : int, optional
            Maximum number of simulation attempts per required simulation.

        Returns
        -------
        GroundMotion
            Simulated ground motions meeting all GoF thresholds.

        Raises
        ------
        RuntimeError
            If not enough simulations meet the thresholds within max_iter * n attempts.
        """
        successful: list[GroundMotion] = []
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
        return GroundMotion(self.config.npts, self.config.dt, ac, vel, disp, tag=len(successful))

    def summary(self):
        """
        Print model parameters.

        Returns
        -------
        StochasticModel
            Self for method chaining.
        """
        def _format_fn(fn):
            """Format a partial function for display."""
            params = ', '.join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}"
                                for k, v in fn.keywords.items())
            return f"{fn.func.__name__}({params})"

        title = "Stochastic Model Summary " + "=" * 30
        print(title)
        print(f"{'Time Step (dt)':<25} : {self.config.dt}")
        print(f"{'Number of Points (npts)':<25} : {self.config.npts}")
        print("-" * len(title))
        print(f"{'Modulating':<25} : {_format_fn(self.config.modulating)}")
        print(f"{'Upper Frequency':<25} : {_format_fn(self.config.upper_frequency)}")
        print(f"{'Lower Frequency':<25} : {_format_fn(self.config.lower_frequency)}")
        print(f"{'Upper Damping':<25} : {_format_fn(self.config.upper_damping)}")
        print(f"{'Lower Damping':<25} : {_format_fn(self.config.lower_damping)}")
        print("-" * len(title))
        return self