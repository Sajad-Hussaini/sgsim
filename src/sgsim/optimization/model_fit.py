import numpy as np
from scipy.optimize import minimize
from ..core.stochastic_model import StochasticModel
from ..motion.ground_motion import GroundMotion
from ..motion import signal

def fit(model: StochasticModel, motion: GroundMotion, component: str, fit_range: tuple = (0.01, 0.99),
        initial_guess=None, bounds=None):
    """
    Fit stochastic model parameters to match a target motion.

    Parameters
    ----------
    component : str
        Component to fit ('modulating', 'frequency', or 'fas').
    model : StochasticModel
        The stochastic model to fit.
    motion : GroundMotion
        The target ground motion.
    fit_range : tuple, optional
        Tuple specifying the fractional energy range (start, end) over which to fit the time series characteristics (e.g., zc_ac).
    initial_guess : array-like, optional
        Initial parameter values. If None, uses defaults.
    bounds : list of tuples, optional
        Parameter bounds as [(min1, max1), (min2, max2), ...]. If None, uses defaults.

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

    result = minimize(objective_func, initial_guess, bounds=bounds, method='L-BFGS-B', jac="3-point")

    if result.success:
        objective_func(result.x)

    return model, result

def get_objective_function(component: str, model: StochasticModel, motion: GroundMotion, fit_range: tuple):
    """Create objective function for the specified component."""
    energy_slicer = signal_tools.slice_energy(motion.ce, fit_range)
    
    if component == 'modulating':
        target_ce = motion.ce
        target_max = target_ce.max()
        modulating_type = type(model.modulating).__name__
        
        def objective(params):
            model_ce = update_modulating(params, model, motion, modulating_type)
            return np.sum(np.square((model_ce - target_ce) / target_max))

    elif component == 'frequency':
        # Cache model type names
        upper_freq_type = type(model.upper_frequency).__name__
        lower_freq_type = type(model.lower_frequency).__name__
        
        # Pre-slice all target arrays
        target_zc_ac = motion.zc_ac[energy_slicer]
        target_zc_vel = motion.zc_vel[energy_slicer]
        target_zc_disp = motion.zc_disp[energy_slicer]
        target_pmnm_vel = motion.pmnm_vel[energy_slicer]
        target_pmnm_disp = motion.pmnm_disp[energy_slicer]
        target_fas = motion.fas
        
        # Pre-compute scale factor
        scale = target_zc_ac.max() / target_fas.max() if target_fas.max() > 0 else 1.0
        
        # Pre-concatenate target vector
        target = np.concatenate((target_zc_ac, target_zc_vel, target_zc_disp,
                                 target_pmnm_vel, target_pmnm_disp, target_fas * scale))
        target_max = target.max()
        
        # Pre-compute parameter slicing info
        fitables = [model.upper_frequency, model.lower_frequency, model.upper_damping, model.lower_damping]
        param_counts = [len(f.params) for f in fitables]
        param_slices = np.cumsum([0] + param_counts)
        
        def objective(params):
            model_output = update_frequency(params, model, motion, scale, energy_slicer,
                                           fitables, param_slices, upper_freq_type, lower_freq_type)
            return np.sum(np.square((model_output - target) / target_max))

    elif component == 'fas':
        fas_max = motion.fas.max() if motion.fas.max() > 0 else 1.0
        fas_vel_max = motion.fas_vel.max() if motion.fas_vel.max() > 0 else 1.0
        fas_disp_max = motion.fas_disp.max() if motion.fas_disp.max() > 0 else 1.0
        
        # Pre-compute normalized target once
        target = np.concatenate((motion.fas / fas_max,
                                 motion.fas_vel / fas_vel_max,
                                 motion.fas_disp / fas_disp_max))

        # Cache model type names
        upper_freq_type = type(model.upper_frequency).__name__
        lower_freq_type = type(model.lower_frequency).__name__
        
        # Pre-compute parameter slicing info
        fitables = [model.upper_frequency, model.lower_frequency, model.upper_damping, model.lower_damping]
        param_counts = [len(f.params) for f in fitables]
        param_slices = np.cumsum([0] + param_counts)
        
        def objective(params):
            model_output = update_fas(params, model, motion, fitables, param_slices, 
                                     upper_freq_type, lower_freq_type,
                                     fas_max, fas_vel_max, fas_disp_max)
            return np.sum(np.square(model_output - target))
            
    else:
        raise ValueError(f"Unknown component: {component}")
    
    return objective

def update_modulating(params, model: StochasticModel, motion: GroundMotion, modulating_type: str):
    """Update modulating function and return model cumulative energy."""
    et, tn = motion.ce.max(), motion.t.max()
    
    if modulating_type == 'BetaDual':
        p1, c1, dp2, c2, a1 = params
        model_params = (p1, c1, p1 + dp2, c2, a1, et, tn)
    elif modulating_type in ('BetaSingle', 'BetaBasic'):
        p1, c1 = params
        model_params = (p1, c1, et, tn)
    else:
        model_params = params
    
    model.modulating(motion.t, *model_params)

    return model.ce

def update_frequency(params, model: StochasticModel, motion: GroundMotion, scale: float, 
                    energy_slicer, fitables, param_slices, upper_freq_type: str, lower_freq_type: str):
    """Update damping functions and return statistics."""
    fitable_params = [params[param_slices[i]:param_slices[i+1]] for i in range(len(fitables))]

    if upper_freq_type in ("Linear", "Exponential") and lower_freq_type in ("Linear", "Exponential"):
        fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0], fitable_params[0][1] * fitable_params[1][1]]

    if upper_freq_type == "Constant" and lower_freq_type == "Constant":
        fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0]]
    
    if upper_freq_type in ("Linear", "Exponential") and lower_freq_type == "Constant":
        fitable_params[1] = [min(fitable_params[0][0], fitable_params[0][1]) * fitable_params[1][0]]

    for freq_model, model_params in zip(fitables, fitable_params):
        freq_model(motion.t, *model_params)

    return np.concatenate((model.zc_ac[energy_slicer],
                           model.zc_vel[energy_slicer],
                           model.zc_disp[energy_slicer],
                           model.pmnm_vel[energy_slicer],
                           model.pmnm_disp[energy_slicer],
                           model.fas * scale))

def update_fas(params, model: StochasticModel, motion: GroundMotion, 
              fitables, param_slices, upper_freq_type: str, lower_freq_type: str,
              fas_max: float, fas_vel_max: float, fas_disp_max: float):
    """Update damping functions and return statistics."""
    fitable_params = [params[param_slices[i]:param_slices[i+1]] for i in range(len(fitables))]

    if upper_freq_type in ("Linear", "Exponential") and lower_freq_type in ("Linear", "Exponential"):
        fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0], fitable_params[0][1] * fitable_params[1][1]]

    if upper_freq_type == "Constant" and lower_freq_type == "Constant":
        fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0]]
    
    if upper_freq_type in ("Linear", "Exponential") and lower_freq_type == "Constant":
        fitable_params[1] = [min(fitable_params[0][0], fitable_params[0][1]) * fitable_params[1][0]]

    for freq_model, model_params in zip(fitables, fitable_params):
        freq_model(motion.t, *model_params)
    model._stats
    # Normalize using pre-computed denominators
    return np.concatenate((model.fas / fas_max,
                           model.fas_vel / fas_vel_max,
                           model.fas_disp / fas_disp_max))

def get_default_parameters(component: str, model: StochasticModel):
    """Get default initial guess and bounds for parameters."""

    mod_defaults = {
        ('modulating', 'BetaDual'): ([0.1, 20.0, 0.2, 10.0, 0.6],
                                     [(0.01, 0.5), (1.0, 1000.0), (0.0, 0.5), (1.0, 1000.0), (0.0, 0.95)]),
        ('modulating', 'BetaSingle'): ([0.1, 20.0],
                                       [(0.01, 0.95), (1.0, 1000.0)]),
        ('modulating', 'BetaBasic'): ([0.1, 20.0],
                                      [(0.01, 0.95), (1.0, 1000.0)]),}

    freq_damping_defaults = {
        ('upper_frequency', 'Linear'): ([3.0, 2.0], [(0.5, 40.0), (0.5, 40.0)]),
        ('upper_frequency', 'Exponential'): ([3.0, 2.0], [(0.5, 40.0), (0.5, 40.0)]),
        ('upper_frequency', 'Constant'): ([5.0], [(0.5, 40.0)]),

        ('lower_frequency', 'Linear'): ([0.2, 0.5], [(0.01, 0.99), (0.01, 0.99)]),
        ('lower_frequency', 'Exponential'): ([0.2, 0.5], [(0.01, 0.99), (0.01, 0.99)]),
        ('lower_frequency', 'Constant'): ([0.2], [(0.01, 0.99)]),

        ('upper_damping', 'Linear'): ([0.1, 0.3], [(0.1, 0.99), (0.1, 0.99)]),
        ('upper_damping', 'Exponential'): ([0.1, 0.3], [(0.1, 0.99), (0.1, 0.99)]),
        ('upper_damping', 'Constant'): ([0.3], [(0.1, 0.99)]),
        
        ('lower_damping', 'Linear'): ([0.1, 0.2], [(0.1, 0.99), (0.1, 0.99)]),
        ('lower_damping', 'Exponential'): ([0.1, 0.2], [(0.1, 0.99), (0.1, 0.99)]),
        ('lower_damping', 'Constant'): ([0.2], [(0.1, 0.99)]),}

    if component == 'modulating':
        model_type = type(model.modulating).__name__
        key = (component, model_type)
        if key not in mod_defaults:
            raise ValueError(f'No default parameters for {key}. Please provide initial_guess and bounds.')
        return mod_defaults[key]

    elif component in ('frequency', 'fas'):
        initial_guess = []
        bounds = []

        roles_and_models = [
            ('upper_frequency', model.upper_frequency),
            ('lower_frequency', model.lower_frequency),
            ('upper_damping', model.upper_damping),
            ('lower_damping', model.lower_damping)]

        for role, model_obj in roles_and_models:
            model_type = type(model_obj).__name__
            key = (role, model_type)

            if key not in freq_damping_defaults:
                raise ValueError(f'No default parameters for {key}. Please provide initial_guess and bounds.')
            
            guess, bnds = freq_damping_defaults[key]
            initial_guess.extend(guess)
            bounds.extend(bnds)
        
        return initial_guess, bounds

    else:
        raise ValueError(f"Unknown component: {component}")
