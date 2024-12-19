import numpy as np
from scipy.optimize import curve_fit
from ..motion.signal_processing import moving_average as mav

def model_fit(fit_func: str, model, target_motion,
              initial_guess: tuple[float, ...] = None,
              lower_bounds: tuple[float, ...] = None,
              upper_bounds: tuple[float, ...] = None):
    """
    Fit the stochastic model to a target motion
    including the modulating, damping ratio, and frequency functions.
    fit_func:
        modulating: optimizing modulating function
        freq: optimizing upper and lower dominant frequencies
        damping : optimizing upper and lower damping ratios using all mzc
        damping pmnm: optimizing upper and lower damping ratios using pmnm (vel, disp)
        all : optimizing freqs and damping ratios together at once
    Return:
        Fitted stocahstic model
    """
    # Default assignment if not provided
    if initial_guess is None or lower_bounds is None or upper_bounds is None:
        default_guess, default_lower, default_upper = get_default_bounds(fit_func, model)
        initial_guess = initial_guess if initial_guess is not None else default_guess
        lower_bounds = lower_bounds if lower_bounds is not None else default_lower
        upper_bounds = upper_bounds if upper_bounds is not None else default_upper

    scale = np.max(model.mdl)*0.01 if fit_func != 'modulating' else None
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

    elif fit_func == 'all':
        ydata = np.concatenate((target_motion.mzc_ac, target_motion.mzc_vel, target_motion.mzc_disp, mav(target_motion.fas[target_motion.slicer_freq])))
        xdata = np.concatenate((target_motion.t, target_motion.t, target_motion.t, target_motion.freq[target_motion.slicer_freq]))
        obj_func = lambda t, *params: obj_all(t, *params, target_motion=target_motion, model=model)
        uncertainty = None
    else:
        raise ValueError('Unknown Fit Function.')
    curve_fit(obj_func, xdata, ydata, p0=initial_guess, bounds=(lower_bounds, upper_bounds), sigma = uncertainty, maxfev=10000)
    return model

def get_default_bounds(fit_func: str, model):
    """
    Helper function to generate initial guess, lower bounds, and upper bounds
    based on the fitting function and its form.
    # TODO for now filter parameters must be the same form (i.e., linear, exponential, etc.)
    """
    if fit_func == 'modulating':
        mdl_func = model.mdl_func.__name__
        if mdl_func == 'beta_multi':
            initial_guess = [0.1, 10.0, 0.2, 10.0, 0.6]
            lower_bounds = [0.01, 0.0, 0.0, 0.0, 0.0]
            upper_bounds = [0.7, 200.0, 0.8, 200.0, 0.95]
        elif mdl_func == 'beta_single':
            initial_guess = [0.1, 10.0]
            lower_bounds = [0.01, 0.0]
            upper_bounds = [0.8, 200.0]
        elif mdl_func == 'gamma':
            initial_guess = [1, 1, 1]
            lower_bounds = [0, 0.0, 0.0]
            upper_bounds = [200, 200, 200]
        elif mdl_func == 'housner_pw':
            initial_guess = [1, 1, 1, 1, 2]
            lower_bounds = [0, 0, 0, 0.1, 0.2]
            upper_bounds = [200, 200, 200, 50, 200]

    elif fit_func == 'freq':
        wu_func = model.wu_func.__name__
        if wu_func in ['linear', 'exponential']:
            igwu, igwl = [5.0] * 2, [1.0] * 2
            lbwu, lbwl = [0.0] * 2, [0.1] * 2
            ubwu, ubwl = [50.0] * 2, [10.0] * 2
        initial_guess = [*igwu, *igwl]
        lower_bounds = [*lbwu, *lbwl]
        upper_bounds = [*ubwu, *ubwl]

    elif fit_func == 'damping' or fit_func == 'damping pmnm':
        zu_func = model.zu_func.__name__
        if zu_func in ['linear', 'exponential']:
            igzu, igzl = [0.5] * 2, [0.5] * 2
            lbzu, lbzl = [0.1] * 2, [0.1] * 2
            ubzu, ubzl = [10.0] * 2, [10.0] * 2
        initial_guess = [*igzu, *igzl]
        lower_bounds = [*lbzu, *lbzl]
        upper_bounds = [*ubzu, *ubzl]

    elif fit_func == 'all':
        wu_func = model.wu_func.__name__
        if wu_func in ['linear', 'exponential']:
            igwu, igwl = [5.0] * 2, [1.0] * 2
            lbwu, lbwl = [0.0] * 2, [0.1] * 2
            ubwu, ubwl = [50.0] * 2, [10.0] * 2
            igzu, igzl = [0.5] * 2, [0.5] * 2
            lbzu, lbzl = [0.1] * 2, [0.1] * 2
            ubzu, ubzl = [10.0] * 2, [10.0] * 2
        initial_guess = [*igwu, *igwl, *igzu, *igzl]
        lower_bounds = [*lbwu, *lbwl, *lbzu, *lbzl]
        upper_bounds = [*ubwu, *ubwl, *ubzu, *ubzl]
    else:
        raise ValueError('Unknown Fit Function.')
    return initial_guess, lower_bounds, upper_bounds

def obj_mdl(t: np.array, *params: tuple[float, ...], target_motion, model) -> np.array:
    """
    The modulating objective function
    Unique solution constraint 1: p1 < p2 -> p2 = p1+dp2 for beta_multi
    """
    mdl_func = model.mdl_func.__name__
    if mdl_func == 'beta_multi':
        Et = target_motion.ce[-1]
        tn = target_motion.t[-1]
        p1, c1, dp2, c2, a1 = params
        p2 = p1 + dp2
        all_params = (p1, c1, p2, c2, a1, Et, tn)
    elif mdl_func == 'beta_single':
        Et = target_motion.ce[-1]
        tn = target_motion.t[-1]
        p1, c1 = params
        all_params = (p1, c1, Et, tn)
    elif mdl_func == 'gamma':
        all_params = params
    elif mdl_func == 'housner_pw':
        all_params = params
    model.get_mdl(*all_params)
    return model.get_ce()

def obj_freq(t: np.array, *params: tuple[float, ...], model) -> np.array:
    """
    Frequency objective function in unit of Hz.
    Physical relation that wu > wl so wu = wl + dwu
    # TODO For now wu and wl must be the same form (i.e., linear, exponential, etc.)
    """
    half_param = len(params) // 2
    dwu = params[:half_param]
    wl = params[half_param:]
    wu = [a + b for a, b in zip(wl, dwu)]
    model.get_wu(*wu)
    model.get_wl(*wl)
    fitted_wu = np.cumsum(model.wu / (2 * np.pi)) * model.dt
    fitted_wl = np.cumsum(model.wl / (2 * np.pi)) * model.dt
    return np.concatenate((fitted_wu, fitted_wl))

def obj_damping(t: np.array, *params: tuple[float, ...], model) -> np.array:
    """
    The damping objective function using all mzc.
    # TODO For now zu and zl must be the same form (i.e., linear, exponential, etc.)
    """
    half_param = len(params) // 2
    model.get_zu(*params[:half_param])
    model.get_zl(*params[half_param:])
    model.get_stats()
    return np.concatenate((model.get_mzc_ac(), model.get_mzc_vel(), model.get_mzc_disp()))

def obj_damping_pmnm(t: np.array, *params: tuple[float, ...], model) -> np.array:
    """
    The damping objective function using pmnm.
    # TODO For now zu and zl must be the same form (i.e., linear, exponential, etc.)
    """
    half_param = len(params) // 2
    model.get_zu(*params[:half_param])
    model.get_zl(*params[half_param:])
    model.get_stats()
    return np.concatenate((model.get_pmnm_vel(), model.get_pmnm_disp()))

def obj_all(t: np.array, *params: tuple[float, ...], target_motion, model) -> np.array:
    """
    The damping objective function using all mzc.
    physical relation that wu > wl so wu = wl + dwu
    # TODO For now wu, wl, zu, and zl must be the same form (i.e., linear, exponential, etc.)
    """
    quarter_param = len(params) // 4
    dwu = params[:quarter_param]
    wl = params[quarter_param:2*quarter_param]
    wu = [a + b for a, b in zip(wl, dwu)]
    model.get_wu(*wu)
    model.get_wl(*wl)
    zu = params[2*quarter_param:3*quarter_param]
    zl = params[3*quarter_param:]
    model.get_zu(*zu)
    model.get_zl(*zl)
    model.get_stats()
    return np.concatenate((model.get_mzc_ac(), model.get_mzc_vel(), model.get_mzc_disp(), mav(model.get_fas()[target_motion.slicer_freq])))