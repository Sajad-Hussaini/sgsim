import numpy as np
from scipy.special import erfc

def relative_error(a, b):
    """
    Error between input metrics relative to b metric
    The error is based on the ratio of the area between the curves
    """
    return np.sum(np.abs(a - b)) / np.sum(b)

def normalized_residual(a, b):
    " Normalize residual of the input metrics "
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    mask = (b != 0) | (a != 0)
    b = b[mask]
    a = a[mask]
    return 2 * np.abs(b - a) / (b + a)

def goodness_of_fit(a, b):
    """ Goodness of fit score between two input metrics """
    return 100 * np.mean(erfc(normalized_residual(a, b)), axis=-1)
