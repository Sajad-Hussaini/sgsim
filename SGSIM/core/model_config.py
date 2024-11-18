import numpy as np
from scipy.special import beta
from ..motion import signal_props as sps

class ModelConfig:
    """
    This class allows to
        flexible assignment of parametric functions using string names (i.e. linear).
        config time, frequency arrays using npts, dt (from target motion or user)
        config functions and parameters (if already have parameters)
        use a variable number of parameters for parametric functions
    """
    def __init__(self, npts: int, dt: float, mdl_type: str = 'beta_multi',
                 wu_type: str = 'linear', zu_type: str = 'linear',
                 wl_type: str = 'linear', zl_type: str = 'linear'):
        self.available_func = {
            'linear': self.linear,
            'constant': self.constant,
            'bilinear': self.bilinear,
            'exponential': self.exponential,
            'beta_basic': self.beta_basic,
            'beta_single': self.beta_single,
            'beta_multi': self.beta_multi,
            'gamma': self.gamma,
            'housner_pw': self.housner_pw}

        self.get_time_freq(npts, dt)
        self.mdl_func = self.available_func.get(mdl_type)
        self.wu_func = self.available_func.get(wu_type)
        self.zu_func = self.available_func.get(zu_type)
        self.wl_func = self.available_func.get(wl_type)
        self.zl_func = self.available_func.get(zl_type)

    def get_time_freq(self, npts, dt):
        self.dt = dt
        self.npts = npts
        self.t = np.linspace(0, (npts - 1) * dt, npts)
        npts_sim = int(2 ** np.ceil(np.log2(2 * npts)))
        # Nyq freq to avoid aliasing in simulations
        self.freq_sim = sps.get_freq(dt, npts_sim)
        # Nyq freq for fitting
        self.freq = sps.get_freq(dt, npts)
        self.slicer_freq = sps.get_freq_slice(self.freq)
        return self

    def get_mdl(self, *params):
        self.mdl_param = params
        self.mdl = self.mdl_func(self.t, *params)
        return self

    def get_wu(self, *params):
        self.wu_param = params
        self.wu = self.wu_func(self.t, *params)  * 2 * np.pi  # in angular freq
        return self

    def get_wl(self, *params):
        self.wl_param = params
        self.wl = self.wl_func(self.t, *params)  * 2 * np.pi  # in angular freq
        return self

    def get_zu(self, *params):
        self.zu_param = params
        self.zu = self.zu_func(self.t, *params)
        return self

    def get_zl(self, *params):
        self.zl_param = params
        self.zl = self.zl_func(self.t, *params)
        return self

    def linear(self, t, *params: tuple[float, ...]) -> np.array:
        p0, p1 = params
        return (p0 - (p0 - p1) * (t / t[-1]))

    def constant(self, t, *params: tuple[float, ...]) -> np.array:
        return np.full(len(t), params[0])

    def bilinear(self, t, *params: tuple[float, ...]) -> np.array:
        p0, p1, p2 = params
        tmid = t[len(t) // 2]
        return np.piecewise(t, [t <= tmid, t > tmid],
                            [lambda t_val: p0 - (p0 - p1) * t_val / tmid,
                             lambda t_val: p1 - (p1 - p2) * (t_val - tmid) / (t[-1] - tmid)])

    def exponential(self, t, *params: tuple[float, ...]) -> np.array:
        p0, p1 = params
        return p0 * np.exp(np.log(p1 / p0) * (t / t[-1]))

    def beta_basic(self, t, *params: tuple[float, ...]) -> np.array:
        p1, c1, Et, tn = params
        mdl = ((t ** (c1 * p1) * (tn - t) ** (c1 * (1 - p1))) /
               (beta(1 + c1 * p1, 1 + c1 * (1 - p1)) * tn ** (1 + c1)))
        return np.sqrt(Et * mdl)

    def beta_single(self, t, *params: tuple[float, ...]) -> np.array:
        p1, c1, Et, tn = params
        mdl1 = 0.05 * (6 * (t[1:-1] * (tn - t[1:-1])) / (tn ** 3))
        mdl2 = 0.95 * np.exp(
            (c1 * p1) * np.log(t[1:-1]) + (c1 * (1 - p1)) * np.log(tn - t[1:-1]) -
            np.log(beta(1 + c1 * p1, 1 + c1 * (1 - p1))) - (1 + c1) * np.log(tn))
        multi_mdl = np.concatenate((np.array([0]), (mdl1 + mdl2), np.array([0])))
        return np.sqrt(Et * multi_mdl)

    def beta_multi(self, t, *params: tuple[float, ...]) -> np.array:
        p1, c1, p2, c2, a1, Et, tn = params
        # Log-space computation to avoid overflow over t[1:-1] with zeros at ends
        mdl1 = 0.05 * (6 * (t[1:-1] * (tn - t[1:-1])) / (tn ** 3))
        mdl2 = a1 * np.exp(
            (c1 * p1) * np.log(t[1:-1]) + (c1 * (1 - p1)) * np.log(tn - t[1:-1]) -
            np.log(beta(1 + c1 * p1, 1 + c1 * (1 - p1))) - (1 + c1) * np.log(tn))
        mdl3 = (0.95 - a1) * np.exp(
            (c2 * p2) * np.log(t[1:-1]) + (c2 * (1 - p2)) * np.log(tn - t[1:-1]) -
            np.log(beta(1 + c2 * p2, 1 + c2 * (1 - p2))) - (1 + c2) * np.log(tn))
        multi_mdl = np.concatenate((np.array([0]), (mdl1 + mdl2 + mdl3), np.array([0])))

        # Original formula
        # mdl1 = 0.05 * (6 * (t * (tn - t)) / (tn ** 3))
        # mdl2 = a1 * ((t ** (c1 * p1) * (tn - t) ** (c1 * (1 - p1))) /
        #               (beta(1 + c1 * p1, 1 + c1 * (1 - p1)) * tn ** (1 + c1)))
        # mdl3 = (1 - 0.05 - a1) * ((t ** (c2 * p2) * (tn - t) ** (c2 * (1 - p2))) /
        #                           (beta(1 + c2 * p2, 1 + c2 * (1 - p2)) * tn ** (1 + c2)))
        # multi_mdl = mdl1 + mdl2 + mdl3
        return np.sqrt(Et * multi_mdl)

    def gamma(self, t, *params: tuple[float, ...]) -> np.array:
        p0, p1, p2 = params
        return p0 * t ** p1 * np.exp(-p2 * t)

    def housner_pw(self, t, *params: tuple[float, ...]) -> np.array:
        p0, p1, p2, t1, t2 = params
        return np.piecewise(t, [
            (t >= 0) & (t < t1),
            (t >= t1) & (t <= t2),
            t > t2],
            [lambda t_val: p0 * (t_val / t1) ** 2,
             p0,
             lambda t_val: p0 * np.exp(-p1 * ((t_val - t2) ** p2))])
