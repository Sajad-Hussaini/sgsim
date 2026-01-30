import numpy as np
from scipy.optimize import minimize
from ..core.model import StochasticModel
from ..motion.ground_motion import GroundMotion
from ..motion import signal
from ..core.functions import ParametricFunction

class ModelInverter:
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

    def fit(self, criteria: str = 'full', fit_range: tuple = (0.01, 0.99), initial_guess: dict = None, bounds: dict = None):
        if initial_guess is None or bounds is None:
            default_guess, default_bounds = self._default_parameters()
            gs = initial_guess or default_guess
            bs = bounds or default_bounds
        
        gs[criteria] = (gs['upper_frequency'] + gs['upper_damping'] + gs['lower_frequency'] + gs['lower_damping'])
        bs[criteria] = (bs['upper_frequency'] + bs['upper_damping'] + bs['lower_frequency'] + bs['lower_damping'])

        self.results = {}
        objective_q = self._objective_modulating(fit_range)
        opt_q = minimize(objective_q, gs['modulating'], bounds=bs['modulating'], method='L-BFGS-B', jac="3-point").x

        modulating_type = type(self.q).__name__
        # For BetaSingle, BetaBasic, BetaDual: append et and tn to params
        if modulating_type in ('BetaDual', 'BetaSingle', 'BetaBasic'):
            et, tn = self.gm.ce.max(), self.gm.t.max()
            opt_q = np.append(opt_q, [et, tn])

        self.results['modulating'] = {'type': modulating_type, 'params': opt_q.tolist()}

        objective_fn = self._objective_function(criteria, fit_range)
        opt_fn = minimize(objective_fn, gs[criteria], bounds=bs[criteria], method='L-BFGS-B', jac="3-point").x

        offset = 0
        for context, func in [('upper_frequency', self.wu), ('upper_damping', self.zu), ('lower_frequency', self.wl), ('lower_damping', self.zl)]:
            n_params = func.n_params
            self.results[context] = {'type': type(func).__name__, 'params': opt_fn[offset:offset+n_params].tolist()}
            offset += n_params

        return self.results

    def _objective_modulating(self, fit_range: tuple):
        """Create objective function for the specified scheme."""
        slicer = signal.slice_energy(self.gm.ce, fit_range)
        target_ce = self.gm.ce
        modulating_type = type(self.q).__name__
        et, tn = self.gm.ce.max(), self.gm.t.max()
        
        def objective(params):
            model_ce = self.update_modulating(params, modulating_type, et, tn)
            return np.mean(np.square(model_ce - target_ce))

        return objective

    def _objective_function(self, criteria: str, fit_range: tuple):
        """Create objective function for the specified scheme."""
        slicer = signal.slice_energy(self.gm.ce, fit_range)

        if criteria == 'full':
            wu_type = type(self.wu).__name__
            wl_type = type(self.wl).__name__
            
            target_zc_ac = self.gm.zc_ac[slicer]
            target_zc_vel = self.gm.zc_vel[slicer]
            target_zc_disp = self.gm.zc_disp[slicer]
            target_pmnm_vel = self.gm.pmnm_vel[slicer]
            target_pmnm_disp = self.gm.pmnm_disp[slicer]
            target_fas = self.gm.fas
            
            offset = 0
            param_slices = []
            for f in [self.wu, self.zu, self.wl, self.zl]:
                end = offset + f.n_params
                param_slices.append(slice(offset, end))
                offset = end

            q_array = self.q(self.gm.t, *self.results['modulating']['params'])
            
            def objective(params):
                m_zc_ac, m_zc_vel, m_zc_disp, m_pmnm_vel, m_pmnm_disp, m_fas = self.update_frequency(params, slicer, param_slices, wu_type, wl_type, q_array)
                return np.sum([np.mean(np.square(m_zc_ac - target_zc_ac)) / np.var(target_zc_ac),
                       np.mean(np.square(m_zc_vel - target_zc_vel)) / np.var(target_zc_vel),
                       np.mean(np.square(m_zc_disp - target_zc_disp)) / np.var(target_zc_disp),
                       np.mean(np.square(m_pmnm_vel - target_pmnm_vel)) / np.var(target_pmnm_vel),
                       np.mean(np.square(m_pmnm_disp - target_pmnm_disp)) / np.var(target_pmnm_disp),
                       np.mean(np.square(m_fas - target_fas)) / np.var(target_fas)])

        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        return objective

    def update_modulating(self, params, modulating_type: str, et: float, tn: float):
        """Update modulating function and return model cumulative energy."""
        if modulating_type == 'BetaDual':
            p1, c1, dp2, c2, a1 = params
            model_params = (p1, c1, p1 + dp2, c2, a1, et, tn)
        elif modulating_type in ('BetaSingle', 'BetaBasic'):
            p1, c1 = params
            model_params = (p1, c1, et, tn)
        else:
            model_params = params
        
        q_array = self.q(self.gm.t, *model_params)
        # dummy values for other parameters
        model = StochasticModel(self.gm.npts, self.gm.dt, q_array, q_array, q_array, q_array, q_array)

        return model.ce

    def update_frequency(self, params, slicer, param_slices, wu_type: str, wl_type: str, q_array: np.ndarray):
        """Update damping functions and return statistics."""
        fitable_params = [params[s] for s in param_slices]

        if wu_type in ("Linear", "Exponential") and wl_type in ("Linear", "Exponential"):
            fitable_params[2] = [fitable_params[0][0] * fitable_params[2][0], fitable_params[0][1] * fitable_params[2][1]]

        if wu_type == "Constant" and wl_type == "Constant":
            fitable_params[2] = [fitable_params[0][0] * fitable_params[2][0]]
        
        if wu_type in ("Linear", "Exponential") and wl_type == "Constant":
            fitable_params[2] = [min(fitable_params[0][0], fitable_params[0][1]) * fitable_params[2][0]]

        fn_arrays = []
        for fn, pms in zip([self.wu, self.zu, self.wl, self.zl], fitable_params):
            fn_arrays.append(fn(self.gm.t, *pms))

        model = StochasticModel(self.gm.npts, self.gm.dt, q_array, fn_arrays[0], fn_arrays[1], fn_arrays[2], fn_arrays[3])

        return (model.zc_ac[slicer], model.zc_vel[slicer], model.zc_disp[slicer], model.pmnm_vel[slicer], model.pmnm_disp[slicer], model.fas)

    def _default_parameters(self):
        """Get default initial guess and bounds for parameters."""
        all_defaults = {('modulating', 'BetaDual'): ([0.1, 20.0, 0.2, 10.0, 0.6], [(0.01, 0.5), (2.0, 100.0), (0.0, 0.5), (2.0, 100.0), (0.0, 0.5)]),
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
                        ('lower_damping', 'Constant'): ([0.2], [(0.1, 0.99)])}

        initial_guess = {}
        bounds = {}

        context_and_type = [('modulating', type(self.q).__name__),
                            ('upper_frequency', type(self.wu).__name__),
                            ('upper_damping', type(self.zu).__name__),
                            ('lower_frequency', type(self.wl).__name__),
                            ('lower_damping', type(self.zl).__name__)]

        for key in context_and_type:
            if key not in all_defaults:
                raise ValueError(f'No default parameters for {key}. Please provide initial_guess and bounds.')
            
            gs, bs = all_defaults[key]
            initial_guess[key[0]] = gs
            bounds[key[0]] = bs
        
        return initial_guess, bounds
