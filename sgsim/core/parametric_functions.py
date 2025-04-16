import numpy as np
from scipy.special import beta

def linear(t, p0, pn):
    return p0 - (p0 - pn) * (t / t[-1])

def constant(t, p0):
    return np.full(len(t), p0)

def bilinear(t, p0, p_mid, pn, t_mid):
    return np.piecewise(t, [t <= t_mid, t > t_mid],
                          [lambda t_val: p0 - (p0 - p_mid) * t_val / t_mid,
                           lambda t_val: p_mid - (p_mid - pn) * (t_val - t_mid) / (t[-1] - t_mid)])

def exponential(t, p0, pn):
    return p0 * np.exp(np.log(pn / p0) * (t / t[-1]))

def exponential_peak(t, tp, p0, pn):
    return 0.1 + p0 * np.exp(-pn * np.abs(t / t[-1] - tp))

def beta_basic(t, p1, c1, et, tn):
    mdl = ((t ** (c1 * p1) * (tn - t) ** (c1 * (1 - p1))) /
           (beta(1 + c1 * p1, 1 + c1 * (1 - p1)) * tn ** (1 + c1)))
    return np.sqrt(et * mdl)

def beta_single(t, p1, c1, et, tn):
    mdl1 = 0.05 * (6 * (t[1:-1] * (tn - t[1:-1])) / (tn ** 3))
    mdl2 = 0.95 * np.exp(
        (c1 * p1) * np.log(t[1:-1]) + (c1 * (1 - p1)) * np.log(tn - t[1:-1]) -
        np.log(beta(1 + c1 * p1, 1 + c1 * (1 - p1))) - (1 + c1) * np.log(tn))
    multi_mdl = np.zeros_like(t)
    multi_mdl[1:-1] = mdl1 + mdl2
    return np.sqrt(et * multi_mdl)

def beta_dual(t, p1, c1, p2, c2, a1, et, tn):
    """ Original formula:
    mdl1 = 0.05 * (6 * (t * (tn - t)) / (tn ** 3))
    mdl2 = a1 * ((t ** (c1 * p1) * (tn - t) ** (c1 * (1 - p1))) /
                  (beta(1 + c1 * p1, 1 + c1 * (1 - p1)) * tn ** (1 + c1)))
    mdl3 = (1 - 0.05 - a1) * ((t ** (c2 * p2) * (tn - t) ** (c2 * (1 - p2))) /
                              (beta(1 + c2 * p2, 1 + c2 * (1 - p2)) * tn ** (1 + c2)))
    multi_mdl = mdl1 + mdl2 + mdl3 """""
    mdl1 = 0.05 * (6 * (t[1:-1] * (tn - t[1:-1])) / (tn ** 3))
    mdl2 = a1 * np.exp(
        (c1 * p1) * np.log(t[1:-1]) + (c1 * (1 - p1)) * np.log(tn - t[1:-1]) -
        np.log(beta(1 + c1 * p1, 1 + c1 * (1 - p1))) - (1 + c1) * np.log(tn))
    mdl3 = (0.95 - a1) * np.exp(
        (c2 * p2) * np.log(t[1:-1]) + (c2 * (1 - p2)) * np.log(tn - t[1:-1]) -
        np.log(beta(1 + c2 * p2, 1 + c2 * (1 - p2))) - (1 + c2) * np.log(tn))
    multi_mdl = np.zeros_like(t)
    multi_mdl[1:-1] = mdl1 + mdl2 + mdl3
    return np.sqrt(et * multi_mdl)

def gamma(t, p0, p1, p2):
    return p0 * t ** p1 * np.exp(-p2 * t)

def housner(t, p0, p1, p2, t1, t2):
    return np.piecewise(t, [(t >= 0) & (t < t1), (t >= t1) & (t <= t2), t > t2],
                        [lambda t_val: p0 * (t_val / t1) ** 2, p0, lambda t_val: p0 * np.exp(-p1 * ((t_val - t2) ** p2))])
