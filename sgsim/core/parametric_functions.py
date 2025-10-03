import numpy as np
from scipy.special import beta

# class ParametricFunction:
#     """
#     Base class for parametric functions with fixed and free parameters.
#     """
#     _param_names = []

#     def __init__(self, **known_params):
#         """
#         Initialize the function and set any known (fixed) parameters.

#         Raises
#         ------
#         ValueError
#             _description_
#         """
#         self.known_params = {}
#         for key, value in known_params.items():
#             if key not in self._param_names:
#                 raise ValueError(f"'{key}' is not a valid parameter name for {self.__class__.__name__}.")
#             self.known_params[key] = value

#     @property
#     def free_param_names(self):
#         """Returns a list of parameters that are not fixed."""
#         return [p for p in self._param_names if p not in self.known_params]

#     def __call__(self, t, *free_param_values):
#         """
#         Evaluate the parametric function at given t with provided free parameter values.

#         Parameters
#         ----------
#         t : array-like
#             Input values at which to evaluate the function.
#         *free_param_values : list
#             Values for the free (unfixed) parameters.

#         Returns
#         -------
#         result : array-like
#             The evaluated function values.

#         Raises
#         ------
#         ValueError
#             If the number of free parameter values does not match the expected number.
#         """
#         params = self.known_params.copy()
#         if len(self.free_param_names) != len(free_param_values):
#             raise ValueError(f"Expected {len(self.free_param_names)} free parameters, but got {len(free_param_values)}.")
            
#         params.update(zip(self.free_param_names, free_param_values))

#         return self._func(t, **params)

#     def _func(self, t, **kwargs):
#         """
#         Abstract method to be implemented by subclasses.

#         Parameters
#         ----------
#         t : array-like
#             Input values at which to evaluate the function.
#         **kwargs : dict
#             Parameter values for the function.

#         Raises
#         ------
#         NotImplementedError
#             If the subclass does not implement this method.
#         """
#         raise NotImplementedError("Subclasses must implement the _func method.")

#     def __repr__(self):
#         """
#         String representation of the ParametricFunction instance.

#         Returns
#         -------
#         str
#             A string showing the class name, fixed parameters, and free parameters.
#         """
#         fixed = ", ".join([f"{k}={v}" for k, v in self.known_params.items()])
#         free = ", ".join(self.free_param_names)
#         return f"{self.__class__.__name__}(Fixed: [{fixed if fixed else 'None'}], Free: [{free}])"

# class Linear(ParametricFunction):
#     _param_names = ['a', 'b']

#     def _func(self, t, a, b):
#         return a - (a - b) * (t / t[-1])

def linear(t, p0, pn):
    return p0 - (p0 - pn) * (t / t[-1])

def linear_peak_rate(t, p_peak, p_rate, t_peak):
    return p_peak - p_rate * (t - t_peak)

def constant(t, p0):
    return np.full(len(t), p0)

def bilinear(t, p0, p_mid, pn, t_mid):
    return np.piecewise(t, [t <= t_mid, t > t_mid],
                          [lambda t_val: p0 - (p0 - p_mid) * t_val / t_mid,
                           lambda t_val: p_mid - (p_mid - pn) * (t_val - t_mid) / (t[-1] - t_mid)])

def exponential(t, p0, pn):
    return p0 * np.exp(np.log(pn / p0) * (t / t[-1]))

def rayleigh(t, p0, pn):
    return (p0 * (t+0.1) + pn / (t+0.1)) / 2

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
    # Original formula:
    # mdl1 = 0.05 * (6 * (t * (tn - t)) / (tn ** 3))
    # mdl2 = a1 * ((t ** (c1 * p1) * (tn - t) ** (c1 * (1 - p1))) / (beta(1 + c1 * p1, 1 + c1 * (1 - p1)) * tn ** (1 + c1)))
    # mdl3 = (1 - 0.05 - a1) * ((t ** (c2 * p2) * (tn - t) ** (c2 * (1 - p2))) / (beta(1 + c2 * p2, 1 + c2 * (1 - p2)) * tn ** (1 + c2)))
    # multi_mdl = mdl1 + mdl2 + mdl3
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
