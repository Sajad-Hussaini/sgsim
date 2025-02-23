import numpy as np
from scipy.special import erfc

def find_error(metric_s, metric_r):
    """
    relative error between input real and simulated metrics
    TODO: The error is based on the ratio of the area between the curves
    """
    return np.sum(np.abs(metric_r - metric_s)) / np.sum(metric_r)

def normalized_residual(metric_s, metric_r):
    " Normalize residual of input real and simulated metrics "
    metric_r, metric_s = np.broadcast_arrays(np.asarray(metric_r), np.asarray(metric_s))
    mask = (metric_r != 0) | (metric_s != 0)
    metric_r = metric_r[mask]
    metric_s = metric_s[mask]
    return 2 * np.abs(metric_r - metric_s) / (metric_r + metric_s)

def goodness_of_fit(metric_s, metric_r):
    """ Goodness of fit score between input real and simulated metrics """
    return 100 * np.mean(erfc(normalized_residual(metric_r, metric_s)), axis=-1)

def get_gof(sim_motion, target_motion, metrics: list):
    """ Goodness of fit scores for input real/sim motion models using metrics list """
    gof = {}
    for metric in metrics:
        target_metric = getattr(target_motion, metric, None)
        sim_metric = getattr(sim_motion, metric, None)

        if target_metric is None or sim_metric is None:
            raise AttributeError(f"Metric '{metric}' not found in target_motion or sim_motion.")

        gof[metric] = goodness_of_fit(target_metric, sim_metric)
    return gof
