" This modules contain functions to calculate errors and goodness of fit scores"

import numpy as np
from scipy.special import erfc

def find_error(metric_r, metric_s) -> float:
    """
    relative error between input real and simulated metrics
    including time series arrays like zero crossing, FAS, cumulative energy

    The error is based on the ratio of the area between the curves
    """
    return np.sum(np.abs(metric_r - metric_s)) / np.sum(metric_r)

def normalized_residual(metric_r, metric_s):
    " Normalize residual of input real and simulated metrics "

    metric_r, metric_s = np.broadcast_arrays(np.asarray(metric_r), np.asarray(metric_s))
    mask = (metric_r != 0) | (metric_s != 0)
    metric_r = metric_r[mask]
    metric_s = metric_s[mask]
    return 2 * np.abs(metric_r - metric_s) / (metric_r + metric_s)

def goodness_of_fit(metric_r, metric_s):
    """
    Goodness of fit score between input real and simulated metrics

    Parameters
    ----------
    metric_r : TYPE
        Real metric.
    metric_s : TYPE
        simulated metric.

    Returns
    -------
    TYPE
        GOF score / mean GOF score if inputs are arrays.

    """
    return 100 * np.mean(erfc(normalized_residual(metric_r, metric_s)), axis=-1)