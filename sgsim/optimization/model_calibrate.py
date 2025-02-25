import numpy as np
from scipy.optimize import curve_fit

def calibrate(func: str, model, motion, initial_guess=None, lower_bounds=None, upper_bounds=None):
    """ Fit the stochastic model to a target motion """
    init_guess, lw_bounds, up_bounds = initialize_bounds(func, model, initial_guess, lower_bounds, upper_bounds)
    xdata, ydata, obj_func = prepare_data(func, model, motion)
    curve_fit(obj_func, xdata, ydata, p0=init_guess, bounds=(lw_bounds, up_bounds))
    return model

def initialize_bounds(func, model, init_guess, lw_bounds, up_bounds):
    if None in (init_guess, lw_bounds, up_bounds):
        default_guess, default_lower, default_upper = get_default_bounds(func, model)
        init_guess = init_guess or default_guess
        lw_bounds = lw_bounds or default_lower
        up_bounds = up_bounds or default_upper
    return init_guess, lw_bounds, up_bounds

def get_default_bounds(func: str, model):
    """
    A simple unified bounds configuration
    """
    # Tuple-based lookup: (calibration function, model function name)
    frequency_common = ((5.0, 4.0, 1.0, 1.0), (0.0, 0.0, 0.1, 0.1), (25.0, 25.0, 10.0, 10.0))
    damping_common = ((0.5, 0.2, 0.1, 0.5), (0.1, 0.1, 0.1, 0.1), (5.0, 5.0, 5.0, 5.0))
    unified_bounds_config = {
        ('modulating', 'beta_dual'): ((0.1, 20.0, 0.2, 10.0, 0.6), (0.01, 1.0, 0.0, 1.0, 0.0), (0.7, 200.0, 0.8, 200.0, 0.95)),
        ('modulating', 'beta_single'): ((0.1, 20.0), (0.01, 1.0), (0.8, 200.0)),
        ('frequency', 'linear'): frequency_common,
        ('frequency', 'exponential'): frequency_common,
        ('damping', 'linear'): damping_common,
        ('damping', 'exponential'): damping_common}
    
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
    if func == 'modulating':
        return prepare_modulating_data(model, motion)
    elif func == 'frequency':
        return prepare_frequency_data(model, motion)
    elif func == 'damping':
        return prepare_damping_data(model, motion)
    else:
        raise ValueError('Unknown Calibration Function.')

def prepare_modulating_data(model, motion):
    ydata = motion.ce
    xdata = motion.t
    obj_func = lambda _, *params: obj_mdl(params, model=model, motion=motion)
    return xdata, ydata, obj_func

def prepare_frequency_data(model, motion):
    ydata = np.concatenate((motion.mzc_ac * model.mdl, motion.mzc_disp * model.mdl))
    xdata = np.tile(motion.t, 2)
    obj_func = lambda _, *params: obj_freq(params, model=model)
    return xdata, ydata, obj_func

def prepare_damping_data(model, motion):
    ydata = np.concatenate((motion.mzc_ac * model.mdl, motion.mzc_vel * model.mdl, motion.mzc_disp * model.mdl, motion.pmnm_vel * model.mdl, motion.pmnm_disp * model.mdl))
    xdata = np.concatenate((motion.t, motion.t, motion.t, motion.t, motion.t))
    obj_func = lambda _, *params: obj_damping(params, model=model)
    return xdata, ydata, obj_func

def obj_mdl(params, model, motion):
    """
    The modulating objective function
    Unique solution constraint 1: p1 < p2 -> p2 = p1+dp2 for beta_dual
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
    Frequency objective function in unit of Hz
    Physically wu > wl so wu = wl + dwu
    # TODO: wu and wl must be the same form (i.e., linear, exponential, etc.)
    """
    half_param = len(params) // 2
    dwu_param, wl_param = params[:half_param], params[half_param:]
    wu_param = np.add(wl_param, dwu_param)
    model.wu = wu_param
    model.wl = wl_param
    wu_array = np.cumsum(model.wu / (2 * np.pi)) * model.dt
    wl_array = np.cumsum(model.wl / (2 * np.pi)) * model.dt
    return np.concatenate((wu_array * model.mdl, wl_array * model.mdl))

def obj_damping(params, model):
    """
    The damping objective function
    # TODO: zu and zl must be the same form (i.e., linear, exponential, etc.)
    """
    half_param = len(params) // 2
    zu_param = params[:half_param]
    zl_param = params[half_param:]
    model.zu = zu_param
    model.zl = zl_param
    return np.concatenate((model.mzc_ac * model.mdl, model.mzc_vel * model.mdl, model.mzc_disp * model.mdl, model.pmnm_vel * model.mdl, model.pmnm_disp * model.mdl))