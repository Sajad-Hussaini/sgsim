import numpy as np
from scipy.optimize import minimize
from ..core.stochastic_model import StochasticModel
from ..motion.ground_motion import GroundMotion

def fit(model: StochasticModel, motion: GroundMotion, component: str, fit_range: tuple = (0.01, 0.99),
        initial_guess=None, bounds=None, method='L-BFGS-B', jac="3-point"):
    """
    Fit stochastic model parameters to match target motion.

    Parameters
    ----------
    component : str
        Component to fit ('modulating', 'frequency', or 'damping').
    model : StochasticModel
        The stochastic model to fit.
    motion : GroundMotion
        The target ground motion.
    fit_range : tuple, optional
        Tuple specifying the fractional energy range (start, end) over which to fit.
        If None, the full range is used.
    initial_guess : array-like, optional
        Initial parameter values. If None, uses defaults.
    bounds : list of tuples, optional
        Parameter bounds as [(min1, max1), (min2, max2), ...]. If None, uses defaults.
    method : str, optional
        Optimization method. Default is 'L-BFGS-B'.

    Returns
    -------
    model : StochasticModel
        The calibrated model (modified in-place).
    result : OptimizeResult
        Optimization result with success status, final parameters, etc.
    """
    if initial_guess is None or bounds is None:
        default_guess, default_bounds = get_default_parameters(component, model)
        initial_guess = initial_guess or default_guess
        bounds = bounds or default_bounds

    objective_func = get_objective_function(component, model, motion, fit_range)

    result = minimize(objective_func, initial_guess, bounds=bounds, method=method, jac=jac)

    if result.success:
        objective_func(result.x)

def get_objective_function(component: str, model: StochasticModel, motion: GroundMotion, fit_range: tuple):
    """Create objective function for the specified component."""
    if component == 'modulating':
        def objective(params):
            model_ce = update_modulating(params, model, motion)
            target_ce = motion.ce
            return np.sum(np.square((model_ce - target_ce) / target_ce.max()))

    elif component == 'frequency':
        motion.energy_slicer = fit_range
        def objective(params):
            model_output = update_frequency(params, model, motion)
            target = np.concatenate((motion.mzc_ac[motion.energy_slicer],
                                     motion.mzc_vel[motion.energy_slicer],
                                     motion.mzc_disp[motion.energy_slicer],
                                     motion.pmnm_vel[motion.energy_slicer],
                                     motion.pmnm_disp[motion.energy_slicer]))

            return np.sum(np.square((model_output - target) / target.max()))
    
    else:
        raise ValueError(f"Unknown component: {component}")
    
    return objective

def update_modulating(params, model: StochasticModel, motion: GroundMotion):
    """Update modulating function and return model cumulative energy."""
    modulating_type = type(model.modulating).__name__
    et, tn = motion.ce.max(), motion.t.max()
    
    if modulating_type == 'BetaDual':
        p1, c1, dp2, c2, a1 = params
        model_params = (p1, c1, p1 + dp2, c2, a1, et, tn)
    elif modulating_type == 'BetaSingle':
        p1, c1 = params
        model_params = (p1, c1, et, tn)
    else:
        raise ValueError(f"Unknown modulating type: {modulating_type}")
    
    model.modulating(motion.t, *model_params)

    return model.ce

def update_frequency(params, model: StochasticModel, motion: GroundMotion):
    """Update damping functions and return statistics."""
    fitables = [model.upper_frequency, model.lower_frequency, model.upper_damping, model.lower_damping]
    param_counts = [len(f.params) for f in fitables]
    param_slices = np.cumsum([0] + param_counts)
    fitable_params = [params[param_slices[i]:param_slices[i+1]] for i in range(len(fitables))]

    if (type(model.upper_frequency).__name__ in ("Linear", "Exponential") and
        type(model.lower_frequency).__name__ in ("Linear", "Exponential")):
        fitable_params[0] = [fitable_params[0][0] + fitable_params[1][0], fitable_params[0][1] + fitable_params[1][1]]
        fitable_params[1] = fitable_params[1]

    for freq_model, model_params in zip(fitables, fitable_params):
        freq_model(motion.t, *model_params)

    return np.concatenate((model.mzc_ac[motion.energy_slicer],
                           model.mzc_vel[motion.energy_slicer],
                           model.mzc_disp[motion.energy_slicer],
                           model.pmnm_vel[motion.energy_slicer],
                           model.pmnm_disp[motion.energy_slicer]))

def get_default_parameters(component: str, model: StochasticModel):
    """Get default initial guess and bounds for parameters."""
    if component == 'modulating':
        model_type = type(model.modulating).__name__
    elif component == 'frequency':
        model_type = type(model.upper_frequency).__name__
    else:
        raise ValueError(f"Unknown component: {component}")

    defaults = {
        ('modulating', 'BetaDual'): (
            [0.1, 20.0, 0.2, 10.0, 0.6],
            [(0.01, 0.7), (1.0, 1000.0), (0.0, 0.8), (1.0, 1000.0), (0.0, 0.95)]
        ),
        ('modulating', 'BetaSingle'): (
            [0.1, 20.0],
            [(0.01, 0.8), (1.0, 1000.0)]
        ),
        ('frequency', 'Linear'): (
            [3.0, 2.0, 0.2, 0.5, 0.1, 0.1, 0.1, 0.1],
            [(0.1, 30.0), (0.1, 30.0), (0.1, 10.0), (0.1, 10.0), (0.05, 8.0), (0.05, 8.0), (0.05, 8.0), (0.05, 8.0)]
        ),
        ('frequency', 'Exponential'): (
            [3.0, 2.0, 0.2, 0.5, 0.1, 0.1, 0.1, 0.1],
            [(0.1, 30.0), (0.1, 30.0), (0.1, 10.0), (0.1, 10.0), (0.05, 8.0), (0.05, 8.0), (0.05, 8.0), (0.05, 8.0)]
        ),
        ('frequency', 'Constant'): (
            [5.0, 1.0, 0.3, 0.2],
            [(0.1, 30.0), (0.1, 10.0), (0.05, 8.0), (0.05, 8.0)]
        ),
    }
    
    key = (component, model_type)
    if key not in defaults:
        raise ValueError(f'No default parameters for {key}, Please provide initial_guess and bounds.')
    
    return defaults[key]
