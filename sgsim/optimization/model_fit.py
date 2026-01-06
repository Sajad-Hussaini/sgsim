import numpy as np
from scipy.optimize import minimize
from ..core.stochastic_model import StochasticModel
from ..motion.ground_motion import GroundMotion
from ..motion import signal_tools

def fit(model: StochasticModel, motion: GroundMotion, component: str, fit_range: tuple = (0.01, 0.99),
        initial_guess=None, bounds=None,
        damping_penalty_threshold: float = 0.8, damping_penalty_weight: float = 2.0):
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
        If None, the full range is used.
    initial_guess : array-like, optional
        Initial parameter values. If None, uses defaults.
    bounds : list of tuples, optional
        Parameter bounds as [(min1, max1), (min2, max2), ...]. If None, uses defaults.
    damping_penalty_threshold : float, optional
        Threshold for damping regularization penalty (default: 0.60).
        Encourages physically realistic damping values (< threshold) by penalizing high damping.
        Only applied when component='frequency' or 'fas'.
    damping_penalty_weight : float, optional
        Weight for damping regularization penalty (0.0 = no penalty, 0.5-1.0 = moderate).
        Encourages physically realistic damping values (< 0.60) by penalizing high damping.
        Only applied when component='frequency' or 'fas'.

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

    objective_func = get_objective_function(component, model, motion, fit_range, damping_penalty_threshold, damping_penalty_weight)

    result = minimize(objective_func, initial_guess, bounds=bounds, method='L-BFGS-B', jac="3-point")

    if result.success:
        objective_func(result.x)

def get_objective_function(component: str, model: StochasticModel, motion: GroundMotion, 
                          fit_range: tuple, damping_penalty_threshold: float, damping_penalty_weight: float):
    """Create objective function for the specified component."""
    if component == 'modulating':
        def objective(params):
            model_ce = update_modulating(params, model, motion)
            target_ce = motion.ce
            return np.sum(np.square((model_ce - target_ce) / target_ce.max()))

    elif component == 'frequency':
        motion.energy_slicer = signal_tools.slice_energy(motion.ce, fit_range)
        scale = motion.zc_ac[motion.energy_slicer].max() / motion.fas.max() if motion.fas.max() > 0 else 1.0
        
        # Get damping parameter indices for penalty calculation
        damping_indices = _get_damping_param_indices(model)
        
        def objective(params):
            model_output = update_frequency(params, model, motion, scale)
            target = np.concatenate((motion.zc_ac[motion.energy_slicer],
                                     motion.zc_vel[motion.energy_slicer],
                                     motion.zc_disp[motion.energy_slicer],
                                     motion.pmnm_vel[motion.energy_slicer],
                                     motion.pmnm_disp[motion.energy_slicer],
                                     motion.fas * scale))
            
            # Main fit error (normalized sum of squared residuals)
            fit_error = np.sum(np.square((model_output - target) / target.max()))
            
            # Damping regularization penalty (soft constraint toward realistic values)
            damping_penalty = _compute_damping_penalty(params, damping_indices, 
                                                       threshold=damping_penalty_threshold, weight=damping_penalty_weight)
            
            return fit_error + damping_penalty

    elif component == 'fas':
        # Get damping parameter indices for penalty calculation
        damping_indices = _get_damping_param_indices(model)
        
        def objective(params):
            model_output = update_fas(params, model, motion)
            target = motion.fas
            
            # Main fit error
            fit_error = np.sum(np.square((model_output - target) / target.max()))
            
            # Damping regularization penalty
            damping_penalty = _compute_damping_penalty(params, damping_indices, 
                                                       threshold=damping_penalty_threshold, weight=damping_penalty_weight)
            
            return fit_error + damping_penalty
    else:
        raise ValueError(f"Unknown component: {component}")
    
    return objective

def _get_damping_param_indices(model: StochasticModel):
    """
    Get the indices of damping parameters in the full parameter vector.
    
    Returns
    -------
    dict
        Dictionary with 'upper' and 'lower' keys, each containing list of parameter indices.
    """
    fitables = [model.upper_frequency, model.lower_frequency, model.upper_damping, model.lower_damping]
    param_counts = [len(f.params) for f in fitables]
    param_slices = np.cumsum([0] + param_counts)
    
    # Upper damping is at index 2, lower damping at index 3
    upper_damping_indices = list(range(param_slices[2], param_slices[3]))
    lower_damping_indices = list(range(param_slices[3], param_slices[4]))
    
    return {'upper': upper_damping_indices, 'lower': lower_damping_indices}

def _compute_damping_penalty(params, damping_indices, threshold, weight):
    """
    Compute regularization penalty for high damping values.
    
    Penalizes damping values above the threshold using a quadratic penalty.
    This encourages the optimizer to adjust frequencies before pushing damping to extremes.
    
    Parameters
    ----------
    params : array-like
        Full parameter vector being optimized.
    damping_indices : dict
        Dictionary with 'upper' and 'lower' keys containing parameter indices.
    threshold : float, optional
        Damping value above which penalty is applied (default: 0.30).
    weight : float, optional
        Penalty weight (default: 0.05). Higher values more strongly discourage high damping.
    
    Returns
    -------
    float
        Penalty value (0 if all damping values <= threshold).
    """
    if weight == 0.0:
        return 0.0
    
    penalty = 0.0
    
    # Penalize upper damping parameters
    for idx in damping_indices['upper']:
        excess = max(0.0, params[idx] - threshold)
        penalty += excess ** 2
    
    # Penalize lower damping parameters
    for idx in damping_indices['lower']:
        excess = max(0.0, params[idx] - threshold)
        penalty += excess ** 2
    
    return weight * penalty

def update_modulating(params, model: StochasticModel, motion: GroundMotion):
    """Update modulating function and return model cumulative energy."""
    modulating_type = type(model.modulating).__name__
    et, tn = motion.ce.max(), motion.t.max()
    
    if modulating_type == 'BetaDual':
        p1, c1, dp2, c2, a1 = params
        model_params = (p1, c1, p1 + dp2, c2, a1, et, tn)
    elif modulating_type in ('BetaSingle', 'BetaBasic'):
        p1, c1 = params
        model_params = (p1, c1, et, tn)
    else:
        model_params = params  # For other modulating types
    
    model.modulating(motion.t, *model_params)

    return model.ce

def update_frequency(params, model: StochasticModel, motion: GroundMotion, scale: float):
    """Update damping functions and return statistics."""
    fitables = [model.upper_frequency, model.lower_frequency, model.upper_damping, model.lower_damping]
    param_counts = [len(f.params) for f in fitables]
    param_slices = np.cumsum([0] + param_counts)
    fitable_params = [params[param_slices[i]:param_slices[i+1]] for i in range(len(fitables))]

    if (type(model.upper_frequency).__name__ in ("Linear", "Exponential") and
        type(model.lower_frequency).__name__ in ("Linear", "Exponential")):
        fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0], fitable_params[0][1] * fitable_params[1][1]]

    if (type(model.upper_frequency).__name__ in ("Constant",) and
        type(model.lower_frequency).__name__ in ("Constant",)):
        fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0]]
    
    if (type(model.upper_frequency).__name__ in ("Linear", "Exponential") and
        type(model.lower_frequency).__name__ in ("Constant",)):
        fitable_params[1] = [min(fitable_params[0][0], fitable_params[0][1]) * fitable_params[1][0]]

    for freq_model, model_params in zip(fitables, fitable_params):
        freq_model(motion.t, *model_params)

    return np.concatenate((model.zc_ac[motion.energy_slicer],
                           model.zc_vel[motion.energy_slicer],
                           model.zc_disp[motion.energy_slicer],
                           model.pmnm_vel[motion.energy_slicer],
                           model.pmnm_disp[motion.energy_slicer],
                           model.fas * scale))

def update_fas(params, model: StochasticModel, motion: GroundMotion):
    """Update damping functions and return statistics."""
    fitables = [model.upper_frequency, model.lower_frequency, model.upper_damping, model.lower_damping]
    param_counts = [len(f.params) for f in fitables]
    param_slices = np.cumsum([0] + param_counts)
    fitable_params = [params[param_slices[i]:param_slices[i+1]] for i in range(len(fitables))]

    if (type(model.upper_frequency).__name__ in ("Linear", "Exponential") and
        type(model.lower_frequency).__name__ in ("Linear", "Exponential")):
        fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0], fitable_params[0][1] * fitable_params[1][1]]

    if (type(model.upper_frequency).__name__ in ("Constant",) and
        type(model.lower_frequency).__name__ in ("Constant",)):
        fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0]]
    
    if (type(model.upper_frequency).__name__ in ("Linear", "Exponential") and
        type(model.lower_frequency).__name__ in ("Constant",)):
        fitable_params[1] = [min(fitable_params[0][0], fitable_params[0][1]) * fitable_params[1][0]]

    for freq_model, model_params in zip(fitables, fitable_params):
        freq_model(motion.t, *model_params)
    model._stats
    return model.fas

def get_default_parameters(component: str, model: StochasticModel):
    """Get default initial guess and bounds for parameters."""

    mod_defaults = {
        ('modulating', 'BetaDual'): (
            [0.1, 20.0, 0.2, 10.0, 0.6],
            [(0.01, 0.5), (1.0, 1000.0), (0.0, 0.5), (1.0, 1000.0), (0.0, 0.95)]
        ),
        ('modulating', 'BetaSingle'): (
            [0.1, 20.0],
            [(0.01, 0.95), (1.0, 1000.0)]
        ),
        ('modulating', 'BetaBasic'): (
            [0.1, 20.0],
            [(0.01, 0.95), (1.0, 1000.0)]
        ),
        }

    freq_damping_defaults = {
        # --- Upper Frequency ---
        ('upper_frequency', 'Linear'): ([3.0, 2.0], [(0.5, 40.0), (0.5, 40.0)]),
        ('upper_frequency', 'Exponential'): ([3.0, 2.0], [(0.5, 40.0), (0.5, 40.0)]),
        ('upper_frequency', 'Constant'): ([5.0], [(0.5, 40.0)]),

        # --- Lower Frequency ---
        ('lower_frequency', 'Linear'): ([0.2, 0.5], [(0.01, 0.99), (0.01, 0.99)]),
        ('lower_frequency', 'Exponential'): ([0.2, 0.5], [(0.01, 0.99), (0.01, 0.99)]),
        ('lower_frequency', 'Constant'): ([0.2], [(0.01, 0.99)]),

        # --- Upper Damping ---
        ('upper_damping', 'Linear'): ([0.1, 0.3], [(0.15, 0.95), (0.15, 0.95)]),
        ('upper_damping', 'Exponential'): ([0.1, 0.3], [(0.15, 0.95), (0.15, 0.95)]),
        ('upper_damping', 'Constant'): ([0.3], [(0.15, 0.95)]),
        
        # --- Lower Damping ---
        ('lower_damping', 'Linear'): ([0.1, 0.2], [(0.15, 0.95), (0.15, 0.95)]),
        ('lower_damping', 'Exponential'): ([0.1, 0.2], [(0.15, 0.95), (0.15, 0.95)]),
        ('lower_damping', 'Constant'): ([0.2], [(0.15, 0.95)]),
        }

    if component == 'modulating':
        model_type = type(model.modulating).__name__
        key = (component, model_type)
        if key not in mod_defaults:
            raise ValueError(f'No default parameters for {key}. Please provide initial_guess and bounds.')
        return mod_defaults[key]

    elif component in ('frequency', 'fas'):
        initial_guess = []
        bounds = []
        
        # Define the components and their roles in order
        roles_and_models = [
            ('upper_frequency', model.upper_frequency),
            ('lower_frequency', model.lower_frequency),
            ('upper_damping', model.upper_damping),
            ('lower_damping', model.lower_damping)]
        # Dynamically build the lists
        for role, model_obj in roles_and_models:
            model_type = type(model_obj).__name__
            key = (role, model_type)

            if key not in freq_damping_defaults:
                raise ValueError(f'No default parameters for {key}. Please provide initial_guess and bounds.')
            
            # Extend the lists with the defaults for this specific subcomponent
            guess, bnds = freq_damping_defaults[key]
            initial_guess.extend(guess)
            bounds.extend(bnds)
        
        return initial_guess, bounds

    else:
        raise ValueError(f"Unknown component: {component}")
