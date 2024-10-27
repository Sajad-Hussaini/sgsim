import numpy as np
from scipy.optimize import curve_fit

def model_fit(fit_func: str, model, target_motion,
              initial_guess: tuple[float, ...] = None,
              lower_bounds: tuple[float, ...] = None,
              upper_bounds: tuple[float, ...] = None):
    """
    Fit the stochastic model core to a target motion
    including the modulating, damping ratio, and frequency functions.
    fit_func:
        modulating: modulating function
        freq: upper and lower frequencies
        damping : upper and lower damping ratios using all mzc
        damping pmnm: upper and lower damping ratios using pmnm vel disp
    """
    # Default assignment if not provided
    if initial_guess is None or lower_bounds is None or upper_bounds is None:
        default_guess, default_lower, default_upper = get_default_bounds(fit_func)
        initial_guess = initial_guess if initial_guess is not None else default_guess
        lower_bounds = lower_bounds if lower_bounds is not None else default_lower
        upper_bounds = upper_bounds if upper_bounds is not None else default_upper

    scale = 1
    if fit_func == 'modulating':
        ydata = target_motion.ce
        xdata = target_motion.t
        obj_func = lambda t, *params: obj_mdl(t, *params, target_motion=target_motion, model=model)
        uncertainty = None

    elif fit_func == 'freq':
        ydata = np.concatenate((target_motion.mzc_ac, target_motion.mzc_disp))
        xdata = np.concatenate((target_motion.t, target_motion.t))
        obj_func = lambda t, *params: obj_freq(t, *params, model=model)
        uncertainty = 1 / (np.concatenate((model.mdl, model.mdl))+scale)

    elif fit_func == 'damping':
        ydata = np.concatenate((target_motion.mzc_ac, target_motion.mzc_vel, target_motion.mzc_disp))
        xdata = np.concatenate((target_motion.t, target_motion.t, target_motion.t))
        obj_func = lambda t, *params: obj_damping(t, *params, model=model)
        uncertainty = 1 / (np.concatenate((model.mdl, model.mdl, model.mdl))+scale)

    elif fit_func == 'damping pmnm':
        ydata = np.concatenate((target_motion.pmnm_vel, target_motion.pmnm_disp))
        xdata = np.concatenate((target_motion.t, target_motion.t))
        obj_func = lambda t, *params: obj_damping_pmnm(t, *params, model=model)
        uncertainty = 1 / (np.concatenate((model.mdl, model.mdl))+scale)

    else:
        raise ValueError('Unknown Fit Function.')
    curve_fit(obj_func, xdata, ydata, p0=initial_guess,
              bounds=(lower_bounds, upper_bounds),
              sigma = uncertainty, maxfev=10000)

def get_default_bounds(fit_func: str):
    """
    Helper function to generate initial guess, lower bounds, and upper bounds
    based on the fitting function.
    """
    if fit_func == 'modulating':
        initial_guess = [0.1, 10.0, 0.1, 10.0, 0.3]
        lower_bounds = [0.01, 0.0, 0.0, 0.0, 0.0]
        upper_bounds = [0.75, 200.0, 0.75, 200.0, 0.475]

    elif fit_func == 'freq':
        igwu, igwl = [7.0] * 2, [1.0] * 2
        lbwu, lbwl = [0.75] * 2, [0.1] * 2
        ubwu, ubwl = [30.0] * 2, [10.0] * 2
        initial_guess = [*igwu, *igwl]
        lower_bounds = [*lbwu, *lbwl]
        upper_bounds = [*ubwu, *ubwl]

    elif fit_func == 'damping' or fit_func == 'damping pmnm':
        igzu, igzl = [0.5] * 2, [0.5] * 2
        lbzu, lbzl = [0.1] * 2, [0.1] * 2
        ubzu, ubzl = [10.0] * 2, [10.0] * 2
        initial_guess = [*igzu, *igzl]
        lower_bounds = [*lbzu, *lbzl]
        upper_bounds = [*ubzu, *ubzl]

    else:
        raise ValueError('Unknown Fit Function.')
    return initial_guess, lower_bounds, upper_bounds

def obj_mdl(t: np.array, *params: tuple[float, ...], target_motion, model) -> np.array:
    """
    The modulating objective function
    """
    Et = target_motion.ce[-1]
    tn = target_motion.t[-1]
    all_params = (*params, Et, tn)

    model.get_mdl(*all_params)
    return model.get_ce()

def obj_freq(t: np.array, *params: tuple[float, ...], model) -> np.array:
    """
    Frequency objective function in unit of Hz.
    """
    half_param = len(params) // 2
    model.get_wu(*params[:half_param])
    model.get_wl(*params[half_param:])
    fitted_wu = np.cumsum(model.wu / (2 * np.pi)) * model.dt
    fitted_wl = np.cumsum(model.wl / (2 * np.pi)) * model.dt
    return np.concatenate((fitted_wu, fitted_wl))

def obj_damping(t: np.array, *params: tuple[float, ...], model) -> np.array:
    """
    The damping objective function using all mzc.
    """
    half_param = len(params) // 2
    model.get_zu(*params[:half_param])
    model.get_zl(*params[half_param:])
    model.get_stats()
    return np.concatenate((model.get_mzc_ac(), model.get_mzc_vel(), model.get_mzc_disp()))

def obj_damping_pmnm(t: np.array, *params: tuple[float, ...], model) -> np.array:
    """
    The damping objective function using pmnm.
    """
    half_param = len(params) // 2
    model.get_zu(*params[:half_param])
    model.get_zl(*params[half_param:])
    model.get_stats()
    return np.concatenate((model.get_pmnm_vel(), model.get_pmnm_disp()))