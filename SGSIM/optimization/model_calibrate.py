import numpy as np
from scipy.optimize import curve_fit
from ..motion.signal_processing import moving_average

def calibrate(func: str, model, motion, initial_guess=None, lower_bounds=None, upper_bounds=None):
    """ Fit the stochastic model to a target motion """
    initial_guess, lower_bounds, upper_bounds = initialize_bounds(func, model, initial_guess, lower_bounds, upper_bounds)
    xdata, ydata, obj_func, uncertainty = prepare_data(func, model, motion)
    curve_fit(obj_func, xdata, ydata, p0=initial_guess, bounds=(lower_bounds, upper_bounds), sigma=uncertainty, maxfev=10000)
    return model

def prepare_data(func, model, motion):
    scale = np.max(model.mdl) * 0.01 if hasattr(model, 'mdl') else None
    if func == 'modulating':
        return prepare_modulating_data(motion, model)
    elif func == 'freq':
        return prepare_freq_data(motion, model, scale)
    elif func == 'damping':
        return prepare_damping_data(motion, model, scale)
    elif func == 'damping pmnm':
        return prepare_damping_pmnm_data(motion, model, scale)
    elif func == 'all':
        return prepare_all_data(motion, model)
    else:
        raise ValueError('Unknown Calibration Function.')

def prepare_modulating_data(motion, model):
    ydata = motion.ce
    xdata = motion.t
    obj_func = lambda t, *params: obj_mdl(t, *params, motion=motion, model=model)
    return xdata, ydata, obj_func, None

def prepare_freq_data(motion, model, scale):
    ydata = np.concatenate((motion.mzc_ac, motion.mzc_disp))
    xdata = np.concatenate((motion.t, motion.t))
    obj_func = lambda t, *params: obj_freq(t, *params, model=model)
    uncertainty = 1 / (np.concatenate((model.mdl, model.mdl)) + scale)
    return xdata, ydata, obj_func, uncertainty

def prepare_damping_data(motion, model, scale):
    ydata = np.concatenate((motion.mzc_ac, motion.mzc_vel, motion.mzc_disp))
    xdata = np.concatenate((motion.t, motion.t, motion.t))
    obj_func = lambda t, *params: obj_damping(t, *params, model=model)
    uncertainty = 1 / (np.concatenate((model.mdl, model.mdl, model.mdl)) + scale)
    return xdata, ydata, obj_func, uncertainty

def prepare_damping_pmnm_data(motion, model, scale):
    ydata = np.concatenate((motion.pmnm_vel, motion.pmnm_disp))
    xdata = np.concatenate((motion.t, motion.t))
    obj_func = lambda t, *params: obj_damping_pmnm(t, *params, model=model)
    uncertainty = 1 / (np.concatenate((model.mdl, model.mdl)) + scale)
    return xdata, ydata, obj_func, uncertainty

def prepare_all_data(motion, model):
    ydata = np.concatenate((motion.mzc_ac, motion.mzc_vel, motion.mzc_disp, moving_average(motion.fas[model.freq_mask])))
    xdata = np.concatenate((motion.t, motion.t, motion.t, motion.freq[model.freq_mask]))
    obj_func = lambda t, *params: obj_all(t, *params, model=model)
    return xdata, ydata, obj_func, None

def initialize_bounds(func, model, initial_guess, lower_bounds, upper_bounds):
    if None in (initial_guess, lower_bounds, upper_bounds):
        default_guess, default_lower, default_upper = get_default_bounds(func, model)
        initial_guess = initial_guess or default_guess
        lower_bounds = lower_bounds or default_lower
        upper_bounds = upper_bounds or default_upper
    return initial_guess, lower_bounds, upper_bounds

def get_default_bounds(func: str, model):
    """
    Generate default initial guess, lower bounds, and upper bounds
    # TODO for now filter parameters must be the same form (i.e., linear, exponential, etc.)
    """
    bounds_config = {
        'modulating': {
            'beta_dual': ([0.1, 10.0, 0.2, 10.0, 0.6], [0.01, 0.0, 0.0, 0.0, 0.0], [0.7, 200.0, 0.8, 200.0, 0.95]),
            'beta_single': ([0.1, 10.0], [0.01, 0.0], [0.8, 200.0]),
            'gamma': ([1, 1, 1], [0, 0.0, 0.0], [200, 200, 200]),
            'housner': ([1, 1, 1, 1, 2], [0, 0, 0, 0.1, 0.2], [200, 200, 200, 50, 200])},
        'freq': {
            'linear': ([5.0, 5.0, 1.0, 1.0], [0.0, 0.0, 0.1, 0.1], [50.0, 50.0, 10.0, 10.0]),
            'exponential': ([5.0, 5.0, 1.0, 1.0], [0.0, 0.0, 0.1, 0.1], [50.0, 50.0, 10.0, 10.0])},
        'damping': {
            'linear': ([0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1], [10.0, 10.0, 10.0, 10.0]),
            'exponential': ([0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1], [10.0, 10.0, 10.0, 10.0])},
        'damping pmnm': {
            'linear': ([0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1], [10.0, 10.0, 10.0, 10.0]),
            'exponential': ([0.5, 0.5, 0.5, 0.5], [0.1, 0.1, 0.1, 0.1], [10.0, 10.0, 10.0, 10.0])},
        'all': {
            'linear': ([5.0, 5.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [50.0, 50.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0]),
            'exponential': ([5.0, 5.0, 1.0, 1.0, 0.5, 0.5, 0.5, 0.5], [0.0, 0.0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1], [50.0, 50.0, 10.0, 10.0, 10.0, 10.0, 10.0, 10.0])}
        }
    if func not in bounds_config:
        raise ValueError('Unknown Calibration Function.')

    func_config = bounds_config[func]
    if func == 'modulating':
        mdl_func = model.mdl_func.__name__
        if mdl_func not in func_config:
            raise ValueError('Unknown Model Function for {}.'.format(func))
        return func_config[mdl_func]

    elif func in ['freq', 'damping', 'damping pmnm', 'all']:
        func_name = model.wu_func.__name__ if func in ['freq', 'all'] else model.zu_func.__name__
        if func_name not in func_config:
            raise ValueError('Unknown Model Function for {}.'.format(func))
        return func_config[func_name]

def obj_mdl(t, *params, motion, model):
    """
    The modulating objective function
    Unique solution constraint 1: p1 < p2 -> p2 = p1+dp2 for beta_dual
    """
    mdl_func = model.mdl_func.__name__
    Et, tn = motion.ce[-1], motion.t[-1]
    if mdl_func == 'beta_dual':
        p1, c1, dp2, c2, a1 = params
        params = (p1, c1, p1 + dp2, c2, a1, Et, tn)
    elif mdl_func == 'beta_single':
        p1, c1 = params
        params = (p1, c1, Et, tn)
    model.get_mdl(*params)
    return model.get_ce()

def obj_freq(t, *params, model):
    """
    Frequency objective function in unit of Hz
    Physical relation that wu > wl so wu = wl + dwu
    # TODO For now wu and wl must be the same form (i.e., linear, exponential, etc.)
    """
    half_param = len(params) // 2
    dwu, wl = params[:half_param], params[half_param:]
    wu = np.add(wl, dwu)
    model.get_wu(*wu)
    model.get_wl(*wl)
    wu_array = np.cumsum(model.wu / (2 * np.pi)) * model.dt
    wl_array = np.cumsum(model.wl / (2 * np.pi)) * model.dt
    return np.concatenate((wu_array, wl_array))

def obj_damping(t, *params, model):
    """
    The damping objective function using all mzc
    # TODO For now zu and zl must be the same form (i.e., linear, exponential, etc.)
    """
    half_param = len(params) // 2
    model.get_zu(*params[:half_param])
    model.get_zl(*params[half_param:])
    model.get_stats()
    return np.concatenate((model.get_mzc_ac(), model.get_mzc_vel(), model.get_mzc_disp()))

def obj_damping_pmnm(t, *params, model):
    """
    The damping objective function using pmnm
    # TODO For now zu and zl must be the same form (i.e., linear, exponential, etc.)
    """
    half_param = len(params) // 2
    model.get_zu(*params[:half_param])
    model.get_zl(*params[half_param:])
    model.get_stats()
    return np.concatenate((model.get_pmnm_vel(), model.get_pmnm_disp()))

def obj_all(t, *params, model):
    """
    The damping objective function using all mzc.
    Physical relation that wu > wl so wu = wl + dwu
    # TODO For now wu, wl, zu, and zl must be the same form (i.e., linear, exponential, etc.)
    """
    quarter_param = len(params) // 4
    dwu = params[:quarter_param]
    wl = params[quarter_param:2*quarter_param]
    wu = np.add(wl, dwu)
    model.get_wu(*wu)
    model.get_wl(*wl)
    zu = params[2*quarter_param:3*quarter_param]
    zl = params[3*quarter_param:]
    model.get_zu(*zu)
    model.get_zl(*zl)
    model.get_stats()
    return np.concatenate((model.get_mzc_ac(), model.get_mzc_vel(), model.get_mzc_disp(), moving_average(model.get_fas()[model.freq_mask])))
