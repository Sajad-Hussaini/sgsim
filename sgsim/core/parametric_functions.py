import numpy as np
from scipy.special import beta

def linear(t, start, end):
    return start + (end - start) * (t / np.max(t))

def linear_slope(t, peak, rate, t_peak):
    return peak - rate * (t - t_peak)

def constant(t, value):
    return np.full_like(t, value)

def bilinear(t, start, mid, end, t_mid):
    return np.piecewise(t, [t <= t_mid, t > t_mid],
                        [lambda t_val: start - (start - mid) * t_val / t_mid,
                         lambda t_val: mid - (mid - end) * (t_val - t_mid) / (t[-1] - t_mid)])

def exponential(t, start, end):
    return start * np.exp(np.log(end / start) * (t / t[-1]))

def beta_basic(t, peak, concentration, energy, duration):
    mdl = ((t ** (concentration * peak) * (duration - t) ** (concentration * (1 - peak))) /
           (beta(1 + concentration * peak, 1 + concentration * (1 - peak)) * duration ** (1 + concentration)))
    return np.sqrt(energy * mdl)

def beta_single(t, peak, concentration, energy, duration):
    mdl1 = 0.05 * (6 * (t[1:-1] * (duration - t[1:-1])) / (duration ** 3))
    mdl2 = 0.95 * np.exp(
        (concentration * peak) * np.log(t[1:-1]) + (concentration * (1 - peak)) * np.log(duration - t[1:-1]) -
        np.log(beta(1 + concentration * peak, 1 + concentration * (1 - peak))) - (1 + concentration) * np.log(duration))
    multi_mdl = np.zeros_like(t)
    multi_mdl[1:-1] = mdl1 + mdl2
    return np.sqrt(energy * multi_mdl)

def beta_dual(t, peak, concentration, peak_2, concentration_2, energy_ratio, energy, duration):
    # Original formula:
    # mdl1 = 0.05 * (6 * (t * (duration - t)) / (duration ** 3))
    # mdl2 = energy_ratio * ((t ** (concentration * peak) * (duration - t) ** (concentration * (1 - peak))) / (beta(1 + concentration * peak, 1 + concentration * (1 - peak)) * duration ** (1 + concentration)))
    # mdl3 = (1 - 0.05 - energy_ratio) * ((t ** (concentration_2 * peak_2) * (duration - t) ** (concentration_2 * (1 - peak_2))) / (beta(1 + concentration_2 * peak_2, 1 + concentration_2 * (1 - peak_2)) * duration ** (1 + concentration_2)))
    # multi_mdl = mdl1 + mdl2 + mdl3
    mdl1 = 0.05 * (6 * (t[1:-1] * (duration - t[1:-1])) / (duration ** 3))
    mdl2 = energy_ratio * np.exp(
        (concentration * peak) * np.log(t[1:-1]) + (concentration * (1 - peak)) * np.log(duration - t[1:-1]) -
        np.log(beta(1 + concentration * peak, 1 + concentration * (1 - peak))) - (1 + concentration) * np.log(duration))
    mdl3 = (0.95 - energy_ratio) * np.exp(
        (concentration_2 * peak_2) * np.log(t[1:-1]) + (concentration_2 * (1 - peak_2)) * np.log(duration - t[1:-1]) -
        np.log(beta(1 + concentration_2 * peak_2, 1 + concentration_2 * (1 - peak_2))) - (1 + concentration_2) * np.log(duration))
    multi_mdl = np.zeros_like(t)
    multi_mdl[1:-1] = mdl1 + mdl2 + mdl3
    return np.sqrt(energy * multi_mdl)

def gamma(t, a, b, c):
    return a * t ** b * np.exp(-c * t)

def housner(t, a, b, c, t1, t2):
    return np.piecewise(t, [(t >= 0) & (t < t1), (t >= t1) & (t <= t2), t > t2],
                        [lambda t_val: a * (t_val / t1) ** 2, a, lambda t_val: a * np.exp(-b * ((t_val - t2) ** c))])
