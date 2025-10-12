import numpy as np
from scipy.optimize import minimize
from ..core.stochastic_model import StochasticModel
from ..motion.ground_motion import GroundMotion

def fit(component: str, model: StochasticModel, motion: GroundMotion, fit_range: tuple = (0.01, 0.99),
        initial_guess=None, bounds=None, method='L-BFGS-B'):
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

    result = minimize(objective_func, initial_guess, bounds=bounds, method=method)

    if result.success:
        objective_func(result.x)

def get_objective_function(component: str, model: StochasticModel, motion: GroundMotion, fit_range: tuple):
    """Create objective function for the specified component."""
    if component == 'modulating':
        def objective(params):
            model_ce = update_modulating(params, model, motion)
            target_ce = motion.ce
            return np.mean((model_ce - target_ce)**2)

    elif component == 'frequency':
        motion.energy_slicer = fit_range
        def objective(params):
            model_output = update_frequency(params, model, motion)
            target = np.concatenate((motion.mzc_ac[motion.energy_slicer], motion.mzc_vel[motion.energy_slicer], motion.mzc_disp[motion.energy_slicer],
                                     motion.pmnm_vel[motion.energy_slicer], motion.pmnm_disp[motion.energy_slicer]))
            return np.mean((model_output - target)**2)
    
    else:
        raise ValueError(f"Unknown component: {component}")
    
    return objective

def update_modulating(params, model: StochasticModel, motion: GroundMotion):
    """Update modulating function and return model cumulative energy."""
    modulating_type = type(model.modulating).__name__
    et, tn = motion.ce[-1], motion.t[-1]
    
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
    quarter_param = len(params) // 4
    wu_param = params[:quarter_param]
    wl_param = params[quarter_param:quarter_param*2]
    zu_param = params[quarter_param*2:quarter_param*3]
    zl_param = params[quarter_param*3:]
    
    freq_type = type(model.upper_frequency).__name__
    if freq_type == "Linear":
        # wl_param = (wl_param[0], wl_param[0] + wl_param[1])
        # wu_param = (wu_param[0] + wu_param[1] + wl_param[1], wu_param[1] + wl_param[1])
        # zl_param = (zl_param[0], zl_param[0] + zl_param[1])
        # zu_param = (zu_param[0], zu_param[0] + zu_param[1])
        wu_param = (wu_param[0] + wl_param[0], wu_param[1] + wl_param[1])
    
    model.upper_frequency(motion.t, *wu_param)
    model.lower_frequency(motion.t, *wl_param)
    model.upper_damping(motion.t, *zu_param)
    model.lower_damping(motion.t, *zl_param)
    
    return np.concatenate((model.mzc_ac[motion.energy_slicer], model.mzc_vel[motion.energy_slicer], model.mzc_disp[motion.energy_slicer],
                           model.pmnm_vel[motion.energy_slicer], model.pmnm_disp[motion.energy_slicer]))

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
            [(0.1, 30.0), (0.1, 30.0), (0.1, 10.0), (0.1, 10.0), (0.05, 5.0), (0.05, 5.0), (0.05, 5.0), (0.05, 5.0)]
        ),
        ('frequency', 'Exponential'): (
            [3.0, 2.0, 0.2, 0.5, 0.1, 0.1, 0.1, 0.1],
            [(0.1, 30.0), (0.1, 30.0), (0.1, 10.0), (0.1, 10.0), (0.05, 5.0), (0.05, 5.0), (0.05, 5.0), (0.05, 5.0)]
        ),
        ('frequency', 'Constant'): (
            [5.0, 1.0, 0.3, 0.2],
            [(0.1, 30.0), (0.1, 10.0), (0.05, 5.0), (0.05, 5.0)]
        ),
    }
    
    key = (component, model_type)
    if key not in defaults:
        raise ValueError(f'No default parameters for {key}, Please provide initial_guess and bounds.')
    
    return defaults[key]
