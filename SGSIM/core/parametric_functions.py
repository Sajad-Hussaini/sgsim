import numpy as np
from scipy.special import beta

def linear(t, params):
    """ Linear function using (first, last) values """
    pf, pl = params
    result = (pf - (pf - pl) * (t / t[-1]))
    param_dict = {"pf": pf, "pl": pl}
    return result, param_dict

def constant(t, params):
    """ A constant-valued function """
    result = np.full(len(t), params[0])
    param_dict = {"pc": params[0]}
    return result, param_dict

def bilinear(t, params):
    """ Bilinear function using (first, max, last, tmax) values """
    pf, pm, pl, tmax = params
    result = np.piecewise(t, [t <= tmax, t > tmax],
                          [lambda t_val: pf - (pf - pm) * t_val / tmax,
                           lambda t_val: pm - (pm - pl) * (t_val - tmax) / (t[-1] - tmax)])
    param_dict = {"pf": pf, "pm": pm, "pl": pl, "tmax": tmax}
    return result, param_dict

def exponential(t, params):
    """ Exponential function using (first, last) values """
    pf, pl = params
    result = pf * np.exp(np.log(pl / pf) * (t / t[-1]))
    param_dict = {"pf": pf, "pl": pl}
    return result, param_dict

def beta_basic(t, params):
    """ Beta function """
    p1, c1, Et, tn = params
    mdl = ((t ** (c1 * p1) * (tn - t) ** (c1 * (1 - p1))) /
           (beta(1 + c1 * p1, 1 + c1 * (1 - p1)) * tn ** (1 + c1)))
    result = np.sqrt(Et * mdl)
    param_dict = {"p1": p1, "c1": c1, "Et": Et, "tn": tn}
    return result, param_dict

def beta_single(t, params):
    """ Beta with single strong phase """
    p1, c1, Et, tn = params
    mdl1 = 0.05 * (6 * (t[1:-1] * (tn - t[1:-1])) / (tn ** 3))
    mdl2 = 0.95 * np.exp(
        (c1 * p1) * np.log(t[1:-1]) + (c1 * (1 - p1)) * np.log(tn - t[1:-1]) -
        np.log(beta(1 + c1 * p1, 1 + c1 * (1 - p1))) - (1 + c1) * np.log(tn))
    multi_mdl = np.zeros_like(t)
    multi_mdl[1:-1] = mdl1 + mdl2
    result = np.sqrt(Et * multi_mdl)
    param_dict = {"p1": p1, "c1": c1, "Et": Et, "tn": tn}
    return result, param_dict

def beta_dual(t, params):
    """ Beta with dual strong phases """
    p1, c1, p2, c2, a1, Et, tn = params
    mdl1 = 0.05 * (6 * (t[1:-1] * (tn - t[1:-1])) / (tn ** 3))
    mdl2 = a1 * np.exp(
        (c1 * p1) * np.log(t[1:-1]) + (c1 * (1 - p1)) * np.log(tn - t[1:-1]) -
        np.log(beta(1 + c1 * p1, 1 + c1 * (1 - p1))) - (1 + c1) * np.log(tn))
    mdl3 = (0.95 - a1) * np.exp(
        (c2 * p2) * np.log(t[1:-1]) + (c2 * (1 - p2)) * np.log(tn - t[1:-1]) -
        np.log(beta(1 + c2 * p2, 1 + c2 * (1 - p2))) - (1 + c2) * np.log(tn))
    multi_mdl = np.zeros_like(t)
    multi_mdl[1:-1] = mdl1 + mdl2 + mdl3
    result = np.sqrt(Et * multi_mdl)
    param_dict = {"p1": p1, "c1": c1, "p2": p2, "c2": c2, "a1": a1, "Et": Et, "tn": tn}
    # Original formula
    # mdl1 = 0.05 * (6 * (t * (tn - t)) / (tn ** 3))
    # mdl2 = a1 * ((t ** (c1 * p1) * (tn - t) ** (c1 * (1 - p1))) /
    #               (beta(1 + c1 * p1, 1 + c1 * (1 - p1)) * tn ** (1 + c1)))
    # mdl3 = (1 - 0.05 - a1) * ((t ** (c2 * p2) * (tn - t) ** (c2 * (1 - p2))) /
    #                           (beta(1 + c2 * p2, 1 + c2 * (1 - p2)) * tn ** (1 + c2)))
    # multi_mdl = mdl1 + mdl2 + mdl3
    return result, param_dict

def gamma(t, params):
    """ Gamma function """
    p0, p1, p2 = params
    result = p0 * t ** p1 * np.exp(-p2 * t)
    param_dict = {"p0": p0, "p1": p1, "p2": p2}
    return result, param_dict

def housner(t, params):
    """ Housner-jenning piece-wise function """
    p0, p1, p2, t1, t2 = params
    result = np.piecewise(t, [(t >= 0) & (t < t1), (t >= t1) & (t <= t2), t > t2],
        [lambda t_val: p0 * (t_val / t1) ** 2, p0, lambda t_val: p0 * np.exp(-p1 * ((t_val - t2) ** p2))])
    param_dict = {"p0": p0, "p1": p1, "p2": p2, "t1": t1, "t2": t2}
    return result, param_dict
