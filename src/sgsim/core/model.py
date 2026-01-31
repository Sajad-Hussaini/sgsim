"""Stochastic ground motion model."""
import json
from functools import cached_property

import numpy as np
from scipy.fft import irfft

from . import engine
from .domain import Domain
from ..motion import signal
from ..motion.ground_motion import GroundMotion
from ..optimization.fit_eval import goodness_of_fit
from . import functions


class StochasticModel(Domain):
    """
    Stochastic ground motion model.


    Parameters
    ----------
    npts : int
        Number of time points.
    dt : float
        Time step.
    modulating : np.ndarray
        Time-varying modulating function.
    upper_frequency : np.ndarray
        Upper frequency function in Hz.
    upper_damping : np.ndarray
        Upper damping function.
    lower_frequency : np.ndarray
        Lower frequency function in Hz.
    lower_damping : np.ndarray
        Lower damping function.
    params : dict, optional
        Original parameters dictionary used to create the model.

    Examples
    --------
    >>> params = {
    ...     'modulating': {'type': 'BetaBasic', 'params': [0.1, 10.0, 1.0, 25.0]},
    ...     'upper_frequency': {'type': 'Linear', 'params': [8.0, 1.5]},
    ...     'upper_damping': {'type': 'Constant', 'params': [0.5]},
    ...     'lower_frequency': {'type': 'Linear', 'params': [1.0, 0.5]},
    ...     'lower_damping': {'type': 'Constant', 'params': [0.3]},
    ... }
    >>> model = StochasticModel.load_from(params, npts, dt)
    >>> model.fas.shape
    (1000,)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(model.t, model.ac)
    >>> plt.show()
    >>> plt.plot(model.freq / (2 * np.pi), model.fas)
    >>> plt.show()
    """
    def __init__(self, npts: int, dt: float, modulating: np.ndarray,
                 upper_frequency: np.ndarray, upper_damping: np.ndarray,
                 lower_frequency: np.ndarray, lower_damping: np.ndarray,
                 params: dict = None):
        super().__init__(npts, dt)
        self.q = modulating
        self.wu = upper_frequency
        self.zu = upper_damping
        self.wl = lower_frequency
        self.zl = lower_damping
        self.params = params

    @classmethod
    def load_from(cls, params: dict, npts: int, dt: float):
        """
        Create StochasticModel from a parameters dictionary.

        Parameters
        ----------
        params : dict
            Dictionary containing model parameters (usually from ModelFitter.fit()).
        npts : int
            Number of time points.
        dt : float
            Time step.

        Returns
        -------
        StochasticModel
            Initialized stochastic model.
        """
        t = signal.time(npts, dt)
        def compute_array(param_group: dict):
            name = param_group['type']
            fn_class = functions.REGISTRY.get(name)
            if fn_class is None:
                raise ValueError(f"Unknown function type: '{name}'. Available: {list(functions.REGISTRY.keys())}")
            return fn_class.compute(t, **param_group['params'])

        modulating = compute_array(params['modulating'])
        upper_frequency = compute_array(params['upper_frequency'])
        upper_damping = compute_array(params['upper_damping'])
        lower_frequency = compute_array(params['lower_frequency'])
        lower_damping = compute_array(params['lower_damping'])

        return cls(npts, dt, modulating, upper_frequency, upper_damping, lower_frequency, lower_damping, params)

    def save(self, filename):
        """
        Save model parameters to a JSON file.

        Parameters
        ----------
        filename : str
            Path to the output JSON file (Recommeded to use .json extension).

        Examples
        --------
        >>> model.save("my_model.json")
        """
        with open(filename, "w") as f:
            json.dump(self.params, f, indent=2)

    # =========================================================================
    # Core Statistics (cached tuple unpacking)
    # =========================================================================

    @cached_property
    def _stats(self):
        """Variance statistics."""
        return engine.stats(self.wu * 2 * np.pi, self.zu, self.wl * 2 * np.pi, self.zl,
                            self.freq_p2, self.freq_p4, self.freq_n2, self.freq_n4, self.df)

    @property
    def _variance(self):
        return self._stats[0]

    @property
    def _variance_dot(self):
        return self._stats[1]

    @property
    def _variance_2dot(self):
        return self._stats[2]

    @property
    def _variance_bar(self):
        return self._stats[3]

    @property
    def _variance_2bar(self):
        return self._stats[4]

    # =========================================================================
    # Fourier Amplitude Spectra (cached tuple unpacking)
    # =========================================================================

    @cached_property
    def _fas_all(self):
        """FAS for acceleration, velocity, and displacement."""
        return engine.fas(self.q, self.wu * 2 * np.pi, self.zu, self.wl * 2 * np.pi, self.zl,
                          self.freq_p2, self.freq_p4, self._variance, self.dt)

    @property
    def fas(self):
        """
        Fourier amplitude spectrum (FAS) of acceleration.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        return self._fas_all[0]

    @property
    def fas_vel(self):
        """
        Fourier amplitude spectrum (FAS) of velocity.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        return self._fas_all[1]

    @property
    def fas_disp(self):
        """
        Fourier amplitude spectrum (FAS) of displacement.

        Returns
        -------
        ndarray
            FAS computed using the model's PSD.
        """
        return self._fas_all[2]

    # =========================================================================
    # Cumulative Energy
    # =========================================================================

    @cached_property
    def ce(self):
        """
        Cumulative energy of the stochastic model.

        Returns
        -------
        ndarray
            Cumulative energy time history.
        """
        return signal.ce(self.dt, self.q)

    # =========================================================================
    # Local Extrema Counts
    # =========================================================================

    @cached_property
    def le_ac(self):
        """
        Mean cumulative local extrema count of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration local extrema.
        """
        return engine.cumulative_rate(self.dt, self._variance_2dot, self._variance_dot)

    @cached_property
    def le_vel(self):
        """
        Mean cumulative local extrema count of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity local extrema.
        """
        return engine.cumulative_rate(self.dt, self._variance_dot, self._variance)

    @cached_property
    def le_disp(self):
        """
        Mean cumulative local extrema count of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement local extrema.
        """
        return engine.cumulative_rate(self.dt, self._variance, self._variance_bar)

    # =========================================================================
    # Zero Crossing Counts
    # =========================================================================

    @cached_property
    def zc_ac(self):
        """
        Mean cumulative zero crossing count of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration zero crossings.
        """
        return engine.cumulative_rate(self.dt, self._variance_dot, self._variance)

    @cached_property
    def zc_vel(self):
        """
        Mean cumulative zero crossing count of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity zero crossings.
        """
        return engine.cumulative_rate(self.dt, self._variance, self._variance_bar)

    @cached_property
    def zc_disp(self):
        """
        Mean cumulative zero crossing count of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement zero crossings.
        """
        return engine.cumulative_rate(self.dt, self._variance_bar, self._variance_2bar)

    # =========================================================================
    # Positive-Minima / Negative-Maxima Counts
    # =========================================================================

    @cached_property
    def pmnm_ac(self):
        """
        Mean cumulative PMNM count of acceleration.

        Returns
        -------
        ndarray
            Cumulative count of acceleration positive-minima and negative-maxima.
        """
        return engine.pmnm_rate(self.dt, self._variance_2dot, self._variance_dot, self._variance)

    @cached_property
    def pmnm_vel(self):
        """
        Mean cumulative PMNM count of velocity.

        Returns
        -------
        ndarray
            Cumulative count of velocity positive-minima and negative-maxima.
        """
        return engine.pmnm_rate(self.dt, self._variance_dot, self._variance, self._variance_bar)

    @cached_property
    def pmnm_disp(self):
        """
        Mean cumulative PMNM count of displacement.

        Returns
        -------
        ndarray
            Cumulative count of displacement positive-minima and negative-maxima.
        """
        return engine.pmnm_rate(self.dt, self._variance, self._variance_bar, self._variance_2bar)

    # =========================================================================
    # Simulation Methods
    # =========================================================================

    def simulate(self, n: int, tag=None, seed: int = None) -> GroundMotion:
        """
        Simulate ground motions using the stochastic model.

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
        rng = np.random.default_rng(seed)
        white_noise = rng.standard_normal((n, self.npts))
        fourier = engine.fourier_series(n, self.npts, self.t,
                                        self.freq_sim, self.freq_sim_p2,
                                        self.q,
                                        self.wu * 2 * np.pi, self.zu,
                                        self.wl * 2 * np.pi, self.zl,
                                        self._variance, white_noise, self.dt)
        # Using default backward 1/N scaling and manual anti-aliasing
        ac = irfft(fourier, workers=-1)[..., :self.npts]
        # FT(w)/jw + pi*delta(w)*FT(0)  integration in freq domain
        fourier[..., 1:] /= (1j * self.freq_sim[1:])
        vel = irfft(fourier, workers=-1)[..., :self.npts]
        fourier[..., 1:] /= (1j * self.freq_sim[1:])
        disp = irfft(fourier, workers=-1)[..., :self.npts]
        return GroundMotion(self.npts, self.dt, ac, vel, disp, tag=tag)

    def simulate_conditional(self, n: int, target: GroundMotion, metrics: dict, max_iter: int = 100) -> GroundMotion:
        """
        Conditionally simulate ground motions until GoF metrics are met.

        Parameters
        ----------
        n : int
            Number of simulations to generate.
        target : GroundMotion
            Target ground motion to compare against.
        metrics : dict
            Conditioning metrics with GoF thresholds, e.g., {'sa': 0.9}.
        max_iter : int, optional
            Maximum attempts per required simulation.

        Returns
        -------
        GroundMotion
            Simulated ground motions meeting all GoF thresholds.

        Raises
        ------
        RuntimeError
            If not enough simulations meet thresholds within max_iter * n attempts.
        """
        successful: list[GroundMotion] = []
        attempts = 0
        while len(successful) < n and attempts < max_iter * n:
            simulated = self.simulate(1, tag=attempts)
            gof_scores = {metric: goodness_of_fit(getattr(simulated, metric), getattr(target, metric)) for metric in metrics}
            if all(gof_scores[m] >= metrics[m] for m in metrics):
                successful.append(simulated)
            attempts += 1

        if len(successful) < n:
            raise RuntimeError(f"Only {len(successful)} simulations met thresholds after {attempts} attempts.")

        ac = np.concatenate([gm.ac for gm in successful], axis=0)
        vel = np.concatenate([gm.vel for gm in successful], axis=0)
        disp = np.concatenate([gm.disp for gm in successful], axis=0)
        return GroundMotion(self.npts, self.dt, ac, vel, disp, tag=len(successful))

    def summary(self):
        """
        Print model parameters.

        Returns
        -------
        Model
            Self for method chaining.
        """
        def _format_fn(fn):
            params = ', '.join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in fn['params'].items())
            return f"{fn['type']}({params})"

        title = "Model Summary " + "=" * 40
        print(title)
        print(f"{'Time Step (dt)':<25} : {self.dt}")
        print(f"{'Number of Points (npts)':<25} : {self.npts}")
        print("-" * len(title))
        print(f"{'Modulating':<25} : {_format_fn(self.params['modulating'])}")
        print(f"{'Upper Frequency':<25} : {_format_fn(self.params['upper_frequency'])}")
        print(f"{'Lower Frequency':<25} : {_format_fn(self.params['lower_frequency'])}")
        print(f"{'Upper Damping':<25} : {_format_fn(self.params['upper_damping'])}")
        print(f"{'Lower Damping':<25} : {_format_fn(self.params['lower_damping'])}")
        print("-" * len(title))
        return self
