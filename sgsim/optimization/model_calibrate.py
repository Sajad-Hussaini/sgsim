import numpy as np
from scipy.optimize import curve_fit

def calibrate(func: str, model, motion, initial_guess=None, lower_bounds=None, upper_bounds=None):
    """
    Fit the stochastic model to a target motion.

    Parameters
    ----------
    func : str
        Calibration function type ('modulating', 'frequency', or 'damping').
    model : object
        The stochastic model to calibrate.
    motion : object
        The target motion data.
    initial_guess : array-like, optional
        Initial guess for the parameters.
    lower_bounds : array-like, optional
        Lower bounds for the parameters.
    upper_bounds : array-like, optional
        Upper bounds for the parameters.

    Returns
    -------
    model : object
        The calibrated model.
    """
    init_guess, lw_bounds, up_bounds = initialize_bounds(func, model, initial_guess, lower_bounds, upper_bounds)
    xdata, ydata, obj_func, sigmas = prepare_data(func, model, motion)
    curve_fit(obj_func, xdata, ydata, p0=init_guess, bounds=(lw_bounds, up_bounds), sigma=sigmas)
    return model

def initialize_bounds(func, model, init_guess, lw_bounds, up_bounds):
    """
    Initialize parameter bounds for calibration.

    Parameters
    ----------
    func : str
        Calibration function type.
    model : object
        The stochastic model.
    init_guess : array-like or None
        Initial guess for the parameters.
    lw_bounds : array-like or None
        Lower bounds for the parameters.
    up_bounds : array-like or None
        Upper bounds for the parameters.

    Returns
    -------
    init_guess : array-like
        Initial guess for the parameters.
    lw_bounds : array-like
        Lower bounds for the parameters.
    up_bounds : array-like
        Upper bounds for the parameters.
    """
    if None in (init_guess, lw_bounds, up_bounds):
        default_guess, default_lower, default_upper = get_default_bounds(func, model)
        init_guess = init_guess or default_guess
        lw_bounds = lw_bounds or default_lower
        up_bounds = up_bounds or default_upper
    return init_guess, lw_bounds, up_bounds

def get_default_bounds(func: str, model):
    """
    Get default parameter bounds for a given calibration function and model.

    Parameters
    ----------
    func : str
        Calibration function type.
    model : object
        The stochastic model.

    Returns
    -------
    tuple
        (initial_guess, lower_bounds, upper_bounds) for the parameters.
    """
    # Tuple-based lookup: (calibration function, model function name)
    frequency_common = ((5.0, 4.0, 1.0, 1.0), (0.0, 0.0, 0.1, 0.1), (25.0, 25.0, 10.0, 10.0))
    damping_common = ((0.5, 0.2, 0.1, 0.5), (0.1, 0.1, 0.1, 0.1), (5.0, 5.0, 5.0, 5.0))
    unified_bounds_config = {
        ('modulating', 'beta_dual'): ((0.1, 20.0, 0.2, 10.0, 0.6), (0.01, 1.0, 0.0, 1.0, 0.0), (0.7, 200.0, 0.8, 200.0, 0.95)),
        ('modulating', 'beta_single'): ((0.1, 20.0), (0.01, 1.0), (0.8, 200.0)),
        ('frequency', 'linear'): frequency_common,
        ('frequency', 'exponential'): frequency_common,
        ('frequency', 'rayleigh'): ((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), (50.0, 50.0, 50.0, 50.0)),
        ('damping', 'linear'): damping_common,
        ('damping', 'exponential'): damping_common,
        ('damping', 'rayleigh'): ((1.0, 1.0, 1.0, 1.0), (0.0, 0.0, 0.0, 0.0), (50.0, 50.0, 50.0, 50.0)),}

    if func == 'modulating':
        model_func_name = model.mdl_func.__name__
    elif func == 'frequency':
        model_func_name = model.wu_func.__name__
    elif func == 'damping':
        model_func_name = model.zu_func.__name__

    key = (func, model_func_name)
    if key not in unified_bounds_config:
        raise ValueError(f'Unknown configuration for {key}.')
    return unified_bounds_config[key]

def prepare_data(func, model, motion):
    """
    Prepare data and objective function for calibration.

    Parameters
    ----------
    func : str
        Calibration function type.
    model : object
        The stochastic model.
    motion : object
        The target motion data.

    Returns
    -------
    xdata : ndarray
        Input data for curve fitting.
    ydata : ndarray
        Target data for curve fitting.
    obj_func : callable
        Objective function for curve fitting.
    sigmas : ndarray or None
        Sigma values for weighting, if applicable.
    """
    if func == 'modulating':
        return prepare_modulating_data(model, motion)
    elif func == 'frequency':
        return prepare_frequency_data(model, motion)
    elif func == 'damping':
        return prepare_damping_data(model, motion)
    else:
        raise ValueError('Unknown Calibration Function.')

def prepare_modulating_data(model, motion):
    """
    Prepare data for modulating function calibration.

    Parameters
    ----------
    model : object
        The stochastic model.
    motion : object
        The target motion data.

    Returns
    -------
    xdata : ndarray
        Time array.
    ydata : ndarray
        Cumulative energy array.
    obj_func : callable
        Objective function for curve fitting.
    sigmas : None
        No sigma weighting used.
    """
    ydata = motion.ce
    xdata = motion.t
    obj_func = lambda _, *params: obj_mdl(params, model=model, motion=motion)
    return xdata, ydata, obj_func, None

def prepare_frequency_data(model, motion):
    """
    Prepare data for frequency function calibration.

    Parameters
    ----------
    model : object
        The stochastic model.
    motion : object
        The target motion data.

    Returns
    -------
    xdata : ndarray
        Time array (tiled).
    ydata : ndarray
        Concatenated zero crossing arrays.
    obj_func : callable
        Objective function for curve fitting.
    sigmas : ndarray
        Sigma values for weighting.
    """
    mdl_norm = 1 / ((model.mdl / np.max(model.mdl)) + 1e-2)
    ydata = np.concatenate((motion.mzc_ac, motion.mzc_disp))
    xdata = np.tile(motion.t, 2)
    obj_func = lambda _, *params: obj_freq(params, model=model)
    sigmas = np.tile(mdl_norm, 2)
    return xdata, ydata, obj_func, sigmas

def prepare_damping_data(model, motion):
    """
    Prepare data for damping function calibration.

    Parameters
    ----------
    model : object
        The stochastic model.
    motion : object
        The target motion data.

    Returns
    -------
    xdata : ndarray
        Time array (tiled).
    ydata : ndarray
        Concatenated zero crossing and extrema arrays.
    obj_func : callable
        Objective function for curve fitting.
    sigmas : ndarray
        Sigma values for weighting.
    """
    mdl_norm = 1 / ((model.mdl / np.max(model.mdl)) + 1e-2)
    ydata = np.concatenate((motion.mzc_ac, motion.mzc_vel, motion.mzc_disp, motion.pmnm_vel, motion.pmnm_disp))
    xdata = np.tile(motion.t, 5)
    obj_func = lambda _, *params: obj_damping(params, model=model)
    sigmas = np.tile(mdl_norm, 5)
    return xdata, ydata, obj_func, sigmas

def obj_mdl(params, model, motion):
    """
    The modulating objective function.

    Unique solution constraint: For 'beta_dual', p1 < p2 is enforced by p2 = p1 + dp2.

    Parameters
    ----------
    params : tuple
        Parameters for the modulating function.
    model : object
        The stochastic model.
    motion : object
        The target motion data.

    Returns
    -------
    ndarray
        Model cumulative energy array.
    """
    mdl_func = model.mdl_func.__name__
    et, tn = motion.ce[-1], motion.t[-1]
    if mdl_func == 'beta_dual':
        p1, c1, dp2, c2, a1 = params
        params = (p1, c1, p1 + dp2, c2, a1, et, tn)
    elif mdl_func == 'beta_single':
        p1, c1 = params
        params = (p1, c1, et, tn)
    model.mdl = params
    return model.ce

def obj_freq(params, model):
    """
    Frequency objective function in units of Hz.

    Physically, wu > wl so wu = wl + dwu.
    TODO: wu and wl must be the same form (i.e., linear, exponential, etc.).

    Parameters
    ----------
    params : tuple
        Parameters for the frequency functions.
    model : object
        The stochastic model.

    Returns
    -------
    ndarray
        Concatenated cumulative sum arrays for wu and wl.
    """
    half_param = len(params) // 2
    dwu_param, wl_param = params[:half_param], params[half_param:]
    wu_param = np.add(wl_param, dwu_param)
    model.wu = wu_param
    model.wl = wl_param
    wu_array = np.cumsum(model.wu / (2 * np.pi)) * model.dt
    wl_array = np.cumsum(model.wl / (2 * np.pi)) * model.dt
    return np.concatenate((wu_array, wl_array))

def obj_damping(params, model):
    """
    The damping objective function.

    TODO: zu and zl must be the same form (i.e., linear, exponential, etc.).

    Parameters
    ----------
    params : tuple
        Parameters for the damping functions.
    model : object
        The stochastic model.

    Returns
    -------
    ndarray
        Concatenated arrays for zero crossing and extrema statistics.
    """
    half_param = len(params) // 2
    zu_param = params[:half_param]
    zl_param = params[half_param:]
    model.zu = zu_param
    model.zl = zl_param
    return np.concatenate((model.mzc_ac, model.mzc_vel, model.mzc_disp, model.pmnm_vel, model.pmnm_disp))