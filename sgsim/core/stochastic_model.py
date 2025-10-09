import json
import numpy as np
from scipy.fft import irfft
from . import model_engine
from . import parametric_functions
from .model_config import ModelConfig
from ..motion.ground_motion import GroundMotion

class StochasticModel(ModelConfig):
    """
    Stochastic ground motion simulation model.

    This class provides methods to simulate ground motions using calibrated
    stochastic parameters, save/load model configurations, and generate
    model summaries.

    Attributes
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

    See Also
    --------
    ModelConfig : Base configuration class
    GroundMotion : Ground motion container class
    """

    def simulate(self, n, seed=None):
        """
        Simulate ground motions using the calibrated stochastic model.

        Parameters
        ----------
        n : int
            Number of simulations to generate.
        seed : int, optional
            Random seed for reproducibility. If None, the random number
            generator is not seeded.

        Returns
        -------
        GroundMotion
            An instance of GroundMotion containing the simulated ground motions
            with acceleration, velocity, and displacement time series.

        Notes
        -----
        The simulation process involves:
        1. Generating white noise samples
        2. Computing Fourier series with stochastic model parameters
        3. Applying inverse FFT to obtain time-domain signals
        4. Integrating in frequency domain for velocity and displacement

        Examples
        --------
        >>> model = StochasticModel(npts=1000, dt=0.01, ...)
        >>> gm = model.simulate(n=100, seed=42)
        >>> gm.acceleration.shape
        (100, 1000)
        """
        self._stats
        n = int(n)
        white_noise = np.random.default_rng(seed).standard_normal((n, self.npts))
        fourier = model_engine.simulate_fourier_series(n, self.npts, self.t, self.freq_sim, self.freq_sim_p2,
                                                        self.modulating.values, self.upper_frequency.values, self.upper_damping.values,
                                                        self.lower_frequency.values, self.lower_damping.values, self._variance, white_noise)
        ac = irfft(fourier, workers=-1)[..., :self.npts]  # anti-aliasing
        # FT(w)/jw + pi*delta(w)*FT(0)  integration in freq domain
        vel = irfft(fourier[..., 1:] / (1j * self.freq_sim[1:]), workers=-1)[..., :self.npts]
        disp = irfft(-fourier[..., 1:] / (self.freq_sim_p2[1:]), workers=-1)[..., :self.npts]
        return GroundMotion(self.npts, self.dt, ac, vel, disp)

    def summary(self, filename=None):
        """
        Print model parameters and optionally save to JSON file.

        Displays a formatted summary of the stochastic model configuration
        including time step, number of points, and all parametric functions.
        Optionally saves the complete model configuration to a JSON file
        for later reconstruction.

        Parameters
        ----------
        filename : str, optional
            Path to JSON file for saving model data (e.g., 'model.json'). 
            If None, only prints to console without saving.

        Returns
        -------
        self : StochasticModel
            The StochasticModel instance for method chaining.

        See Also
        --------
        load_from : Load a model from JSON file

        Notes
        -----
        The saved JSON file contains:
        - Time discretization parameters (npts, dt)
        - Function types for all parametric components
        - Parameter values for each function

        A stochastic model can be reconstructed from the saved file using
        the `load_from` class method.

        Examples
        --------
        >>> model = StochasticModel(npts=1000, dt=0.01, ...)
        >>> model.summary()  # Print only
        >>> model.summary('my_model.json')  # Print and save
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

        Loads a previously saved stochastic model configuration from a JSON
        file and reconstructs the complete model with all parametric functions
        and their calibrated parameters.

        Parameters
        ----------
        filename : str
            Path to JSON file containing model data (e.g., 'model.json').
            The file should have been created using the `summary` method
            with a filename argument.

        Returns
        -------
        StochasticModel
            An instance of StochasticModel initialized from the file with
            all parametric functions and parameters restored.

        Raises
        ------
        FileNotFoundError
            If the specified file does not exist.
        json.JSONDecodeError
            If the file is not valid JSON.
        AttributeError
            If the parametric function types specified in the file are not
            found in the parametric_functions module.

        See Also
        --------
        summary : Save a model to JSON file

        Examples
        --------
        >>> # Save a model
        >>> model = StochasticModel(npts=1000, dt=0.01, ...)
        >>> model.summary('my_model.json')
        
        >>> # Load it back
        >>> loaded_model = StochasticModel.load_from('my_model.json')
        >>> loaded_model.npts
        1000

        Notes
        -----
        The method performs the following steps:
        1. Loads JSON data from file
        2. Creates a model instance with function types
        3. Restores all parameter values by calling each function
        """
        with open(filename, 'r') as file:
            data = json.load(file)
        
        # Create model with function types
        model = cls(
            npts=data['npts'],
            dt=data['dt'],
            modulating=getattr(parametric_functions, data['modulating']['func']),
            upper_frequency=getattr(parametric_functions, data['upper_frequency']['func']),
            upper_damping=getattr(parametric_functions, data['upper_damping']['func']),
            lower_frequency=getattr(parametric_functions, data['lower_frequency']['func']),
            lower_damping=getattr(parametric_functions, data['lower_damping']['func'])
        )
        
        # Set parameters using the stored param dicts
        model.modulating(model.t, **data['modulating']['params'])
        model.upper_frequency(model.t, **data['upper_frequency']['params'])
        model.upper_damping(model.t, **data['upper_damping']['params'])
        model.lower_frequency(model.t, **data['lower_frequency']['params'])
        model.lower_damping(model.t, **data['lower_damping']['params'])
        
        return model
