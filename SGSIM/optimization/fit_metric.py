" This modules contain functions to calculate error and goodness of fit"
import numpy as np
from scipy.special import erfc

def find_error(im_r, im_s) -> float:
    """
    relative error between input real and simulated intensity measure
    including time series arrays like zero crossing, FAS, cumulative energy

    The error is based on the ratio of the area between the curves
    """
    return np.sum(np.abs(im_r - im_s)) / np.sum(im_r)

def normalized_residual(im_r, im_s):
    " Normalize residual of input intensity measure "
    return 2 * np.abs(im_r - im_s) / (im_r + im_s)

def goodness_of_fit(im_r, im_s):
    """
    Goodness of fit score between input intensity measures

    Parameters
    ----------
    im_r : TYPE
        Real intensity measure.
    im_s : TYPE
        Simulated intensity measure.

    Returns
    -------
    TYPE
        GOF score / mean GOF score if inputs are arrays.

    """
    return 100 * np.mean(erfc(normalized_residual(im_r, im_s)))