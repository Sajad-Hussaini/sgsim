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

        self._wu_type = type(self.wu).__name__
        self._wl_type = type(self.wl).__name__
        self._q_type = type(self.q).__name__

    def fit(self, criteria: str = 'full', fit_range: tuple = (0.01, 0.99), initial_guess: dict = None, bounds: dict = None):
        if initial_guess is None or bounds is None:
            default_guess, default_bounds = self._default_parameters()
            gs = initial_guess or default_guess
            bs = bounds or default_bounds

        self.results = {}
        # ========== Fit modulating function ==========
        objective_q = self._objective_modulating(fit_range)
        opt_q = minimize(objective_q, gs['modulating'], bounds=bs['modulating'], method='L-BFGS-B', jac="3-point").x

        # For BetaSingle, BetaBasic, BetaDual: append et and tn to params
        if self._q_type in ('BetaDual', 'BetaSingle', 'BetaBasic'):
            opt_q = np.append(opt_q, [self.gm.ce.max(), self.gm.t.max()])

        self.results['modulating'] = {'type': self._q_type, 'params': dict(zip(self.q.param_names, opt_q))}

        # ========== Fit frequency and damping functions ==========
        gs_fn = gs['upper_frequency'] + gs['upper_damping'] + gs['lower_frequency'] + gs['lower_damping']
        bs_fn = bs['upper_frequency'] + bs['upper_damping'] + bs['lower_frequency'] + bs['lower_damping']
        objective_fn = self._objective_function(criteria, fit_range)
        opt_fn = minimize(objective_fn, gs_fn, bounds=bs_fn, method='L-BFGS-B', jac="3-point").x

        raw_groups = self._slice_parameters(opt_fn)
        resolved_groups = self._resolve_parameters(raw_groups)
        funcs = [self.wu, self.zu, self.wl, self.zl]
        names = ['upper_frequency', 'upper_damping', 'lower_frequency', 'lower_damping']
        
        for name, func, vals in zip(names, funcs, resolved_groups):
            self.results[name] = {'type': type(func).__name__, 'params': dict(zip(func.param_names, vals))}

        return StochasticModel.load_from(self.results, self.gm.npts, self.gm.dt)
    
    def _slice_parameters(self, flat_params):
        """Helper to slice flat parameter array into [wu, zu, wl, zl] arrays."""
        groups = []
        offset = 0
        for func in [self.wu, self.zu, self.wl, self.zl]:
            groups.append(flat_params[offset : offset + func.n_params])
            offset += func.n_params
        return groups

    def _resolve_parameters(self, groups):
        """
        Convert optimization parameters (ratios) to physical parameters (Hz).
        Input: list of [p_wu, p_zu, p_wl, p_zl]
        Output: list of same structure but with corrected wl parameters.
        """
        p_wu, p_zu, p_wl, p_zl = groups
        
        # Copy to avoid side-effects
        final_wl = p_wl.copy()

        # The Ratio Logic
        is_wu_dynamic = self._wu_type in ("Linear", "Exponential")
        is_wl_dynamic = self._wl_type in ("Linear", "Exponential")
        is_wu_const = self._wu_type == "Constant"
        is_wl_const = self._wl_type == "Constant"

        if is_wu_dynamic and is_wl_dynamic:
            final_wl[0] = p_wu[0] * p_wl[0]
            final_wl[1] = p_wu[1] * p_wl[1]
        elif is_wu_const and is_wl_const:
            final_wl[0] = p_wu[0] * p_wl[0]
        elif is_wu_dynamic and is_wl_const:
            final_wl[0] = min(p_wu[0], p_wu[1]) * p_wl[0]
            
        return [p_wu, p_zu, final_wl, p_zl]

    def _objective_modulating(self, fit_range: tuple):
        """Create objective function for the specified scheme."""
        target_ce = self.gm.ce
        et, tn = self.gm.ce.max(), self.gm.t.max()
        
        def objective(params):
            if self._q_type == 'BetaDual':
                p = (params[0], params[1], params[0] + params[2], params[3], params[4], et, tn)
            elif self._q_type in ('BetaSingle', 'BetaBasic'):
                p = (params[0], params[1], et, tn)
            else:
                p = params
            
            q_array = self.q.compute(self.gm.t, *p)
            model_ce = signal.ce(self.gm.dt, q_array)
            return np.mean(np.square(model_ce - target_ce)) / np.var(target_ce)

        return objective

    def _objective_function(self, criteria: str, fit_range: tuple):
        """Create objective function for the specified scheme."""
        slicer = signal.slice_energy(self.gm.ce, fit_range)

        if criteria == 'full':
            targets = [self.gm.zc_ac[slicer], self.gm.zc_vel[slicer], self.gm.zc_disp[slicer],
                       self.gm.pmnm_vel[slicer], self.gm.pmnm_disp[slicer], self.gm.fas]
            variances = [np.var(t) for t in targets]

            q_params = self.results['modulating']['params']
            q_array = self.q.compute(self.gm.t, **q_params)
            
            def objective(params):
                raw_groups = self._slice_parameters(params)
                resolved_groups = self._resolve_parameters(raw_groups)
                wu_arr = self.wu.compute(self.gm.t, *resolved_groups[0])
                zu_arr = self.zu.compute(self.gm.t, *resolved_groups[1])
                wl_arr = self.wl.compute(self.gm.t, *resolved_groups[2])
                zl_arr = self.zl.compute(self.gm.t, *resolved_groups[3])

                model = StochasticModel(self.gm.npts, self.gm.dt, q_array, wu_arr, zu_arr, wl_arr, zl_arr)

                preds = [model.zc_ac[slicer], model.zc_vel[slicer], model.zc_disp[slicer],
                         model.pmnm_vel[slicer], model.pmnm_disp[slicer], model.fas]
                
                error = 0.0
                for pred, target, var in zip(preds, targets, variances):
                    error += np.mean(np.square(pred - target)) / var
                
                return error

        else:
            raise ValueError(f"Unknown criteria: {criteria}")
        
        return objective

    def _default_parameters(self):
        """Get default initial guess and bounds for parameters."""
        all_defaults = {('modulating', 'BetaDual'): ([0.1, 20.0, 0.2, 10.0, 0.6], [(0.01, 0.5), (2.0, 1000.0), (0.0, 0.5), (2.0, 1000.0), (0.0, 0.5)]),
                        ('modulating', 'BetaSingle'): ([0.1, 20.0], [(0.01, 0.5), (2.0, 1000.0)]),
                        ('modulating', 'BetaBasic'): ([0.1, 20.0], [(0.01, 0.5), (2.0, 1000.0)]),

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
