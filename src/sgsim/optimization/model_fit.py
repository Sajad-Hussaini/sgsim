import numpy as np
from scipy.optimize import minimize
from ..core.model import StochasticModel
from ..motion.ground_motion import GroundMotion
from ..motion import signal
from ..core.functions import ParametricFunction

class ModelFitter:
    def __init__(self, modulating: ParametricFunction,
                 upper_frequency: ParametricFunction, upper_damping: ParametricFunction,
                 lower_frequency: ParametricFunction, lower_damping: ParametricFunction,
                 ground_motion: GroundMotion):
        self.q = modulating
        self.wu = upper_frequency
        self.zu = upper_damping
        self.wl = lower_frequency
        self.zl = lower_damping
        self.gm = ground_motion

    def fit (self, fit_range: tuple = (0.01, 0.99), initial_guess: dict = None, bounds: dict = None):
        if initial_guess is None or bounds is None:
            default_guess, default_bounds = self._default_parameters()
            gs = initial_guess or default_guess
            bs = bounds or default_bounds

        results = []
        schemes = ['modulating', 'frequency']
        for scheme in schemes:
            objective_func = self._objective_function(scheme, fit_range)
            results.append(minimize(objective_func, gs, bounds=bs, method='L-BFGS-B', jac="3-point"))

        return results

    def _objective_function(self, scheme: str, fit_range: tuple):
        """Create objective function for the specified scheme."""
        slicer = signal.slice_energy(self.gm.ce, fit_range)
        
        if scheme == 'modulating':
            target_ce = self.gm.ce
            modulating_type = type(self.q).__name__
            
            def objective(params):
                model_ce = self.update_modulating(params, modulating_type)
                return np.mean(np.square(model_ce - target_ce))

        elif scheme == 'frequency':
            wu_type = type(self.wu).__name__
            wl_type = type(self.wl).__name__
            
            target_zc_ac = self.gm.zc_ac[slicer]
            target_zc_vel = self.gm.zc_vel[slicer]
            target_zc_disp = self.gm.zc_disp[slicer]
            target_pmnm_vel = self.gm.pmnm_vel[slicer]
            target_pmnm_disp = self.gm.pmnm_disp[slicer]
            target_fas = self.gm.fas
            
            fitables = [self.wu, self.wl, self.zu, self.zl]
            param_counts = [f.n_params for f in fitables]
            param_indices = np.cumsum([0] + param_counts)
            param_slices = [slice(start, end) for start, end in zip(param_indices[:-1], param_indices[1:])]
            
            def objective(params):
                m_zc_ac, m_zc_vel, m_zc_disp, m_pmnm_vel, m_pmnm_disp, m_fas = self.update_frequency(params, slicer, fitables, param_slices, wu_type, wl_type)
                return np.sum([np.mean(np.square(m_zc_ac - target_zc_ac)) / np.var(target_zc_ac),
                       np.mean(np.square(m_zc_vel - target_zc_vel)) / np.var(target_zc_vel),
                       np.mean(np.square(m_zc_disp - target_zc_disp)) / np.var(target_zc_disp),
                       np.mean(np.square(m_pmnm_vel - target_pmnm_vel)) / np.var(target_pmnm_vel),
                       np.mean(np.square(m_pmnm_disp - target_pmnm_disp)) / np.var(target_pmnm_disp),
                       np.mean(np.square(m_fas - target_fas)) / np.var(target_fas)])
        else:
            raise ValueError(f"Unknown scheme: {scheme}")
        
        return objective

    def update_modulating(self, params, modulating_type: str):
        """Update modulating function and return model cumulative energy."""
        et, tn = self.gm.ce.max(), self.gm.t.max()
        
        if modulating_type == 'BetaDual':
            p1, c1, dp2, c2, a1 = params
            model_params = (p1, c1, p1 + dp2, c2, a1, et, tn)
        elif modulating_type in ('BetaSingle', 'BetaBasic'):
            p1, c1 = params
            model_params = (p1, c1, et, tn)
        else:
            model_params = params
        
        self.q(self.gm.t, *model_params)
        model = StochasticModel(self.gm.npts, self.gm.dt, self.q, self.wu, self.zu, self.wl, self.zl)

        return model.ce

    def update_frequency(self, params, slicer, fitables, param_slices, wu_type: str, wl_type: str):
        """Update damping functions and return statistics."""
        fitable_params = [params[s] for s in param_slices]

        if wu_type in ("Linear", "Exponential") and wl_type in ("Linear", "Exponential"):
            fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0], fitable_params[0][1] * fitable_params[1][1]]

        if wu_type == "Constant" and wl_type == "Constant":
            fitable_params[1] = [fitable_params[0][0] * fitable_params[1][0]]
        
        if wu_type in ("Linear", "Exponential") and wl_type == "Constant":
            fitable_params[1] = [min(fitable_params[0][0], fitable_params[0][1]) * fitable_params[1][0]]

        for fn, pms in zip(fitables, fitable_params):
            fn(self.gm.t, *pms)

        model = StochasticModel(self.gm.npts, self.gm.dt, self.q, self.wu, self.zu, self.wl, self.zl)

        return (model.zc_ac[slicer], model.zc_vel[slicer], model.zc_disp[slicer], model.pmnm_vel[slicer], model.pmnm_disp[slicer], model.fas)

    def _default_parameters(self):
        """Get default initial guess and bounds for parameters."""
        all_defaults = {
            ('modulating', 'BetaDual'): ([0.1, 20.0, 0.2, 10.0, 0.6], [(0.01, 0.5), (2.0, 100.0), (0.0, 0.5), (2.0, 100.0), (0.0, 0.5)]),
            ('modulating', 'BetaSingle'): ([0.1, 20.0], [(0.01, 0.5), (2.0, 100.0)]),
            ('modulating', 'BetaBasic'): ([0.1, 20.0], [(0.01, 0.5), (2.0, 100.0)]),
            ('upper_frequency', 'Linear'): ([3.0, 2.0], [(0.5, 40.0), (0.5, 40.0)]),
            ('upper_frequency', 'Exponential'): ([3.0, 2.0], [(0.5, 40.0), (0.5, 40.0)]),
            ('upper_frequency', 'Constant'): ([5.0], [(0.5, 40.0)]),
            ('lower_frequency', 'Linear'): ([0.2, 0.5], [(0.01, 0.99), (0.01, 0.99)]),
            ('lower_frequency', 'Exponential'): ([0.2, 0.5], [(0.01, 0.99), (0.01, 0.99)]),
            ('lower_frequency', 'Constant'): ([0.2], [(0.01, 0.99)]),
            ('upper_damping', 'Linear'): ([0.1, 0.3], [(0.1, 0.99), (0.1, 0.99)]),
            ('upper_damping', 'Exponential'): ([0.1, 0.3], [(0.1, 0.99), (0.1, 0.99)]),
            ('upper_damping', 'Constant'): ([0.3], [(0.1, 0.99)]),
            ('lower_damping', 'Linear'): ([0.1, 0.2], [(0.1, 0.99), (0.1, 0.99)]),
            ('lower_damping', 'Exponential'): ([0.1, 0.2], [(0.1, 0.99), (0.1, 0.99)]),
            ('lower_damping', 'Constant'): ([0.2], [(0.1, 0.99)]),}

        initial_guess = []
        bounds = []

        context_and_type = [
            ('modulating', type(self.q).__name__),
            ('upper_frequency', type(self.wu).__name__),
            ('lower_frequency', type(self.wl).__name__),
            ('upper_damping', type(self.zu).__name__),
            ('lower_damping', type(self.zl).__name__)]

        for key in context_and_type:
            if key not in all_defaults:
                raise ValueError(f'No default parameters for {key}. Please provide initial_guess and bounds.')
            
            gs, bs = all_defaults[key]
            initial_guess.extend(gs)
            bounds.extend(bs)
        
        return initial_guess, bounds
