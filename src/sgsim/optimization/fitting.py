import numpy as np
from scipy.optimize import minimize
from ..core.model import StochasticModel
from ..motion.ground_motion import GroundMotion
from ..motion import signal
from ..core.functions import ParametricFunction

class ModulatingInverter:
    """Inverter for identifying time-domain modulating parameters."""
    def __init__(self, ground_motion: GroundMotion, modulating: ParametricFunction):
        self.gm = ground_motion
        self.q = modulating
        self._q_type = type(self.q).__name__

    def fit(self, fit_range: tuple = (0.01, 0.99), initial_guess: dict = None, bounds: dict = None):
        default_guess, default_bounds = self._default_parameters()
        
        gs = initial_guess.get('modulating', default_guess['modulating']) if initial_guess else default_guess['modulating']
        bs = bounds.get('modulating', default_bounds['modulating']) if bounds else default_bounds['modulating']

        objective_q, et, tn, t_shift = self._objective_modulating(fit_range)
        opt_q = minimize(objective_q, gs, bounds=bs, method='L-BFGS-B', jac="3-point").x

        # For specific Beta functions, append deterministic edge parameters
        if self._q_type in ('BetaDual', 'BetaSingle', 'BetaPeakConcentration', 'BetaCentroidSpread'):
            if self._q_type == 'BetaCentroidSpread':
                opt_q[1] = opt_q[0] * opt_q[1]  # Convert (spread/centroid) ratio back to absolute spread
            opt_q = np.append(opt_q, [et, tn])

        return {'type': self._q_type, 'time_shift': t_shift, 'params': dict(zip(self.q.param_names, opt_q))}

    def _objective_modulating(self, fit_range: tuple):
        slicer = signal.slice_energy(self.gm.ce, fit_range)
        t_shift = self.gm.t[slicer][0]
        
        target_ce = self.gm.ce[slicer] - self.gm.ce[slicer][0]
        t_sliced = self.gm.t[slicer] - t_shift
        
        et = target_ce.max()
        tn = t_sliced.max()
        
        def objective(params):
            if self._q_type == 'BetaDual':
                p = (params[0], params[1], params[0] + params[2], params[3], params[4], et, tn)
            elif self._q_type == 'BetaCentroidSpread':
                centroid = params[0]
                ratio = params[1]
                spread = centroid * ratio
                p = (centroid, spread, et, tn)
            elif self._q_type in ('BetaSingle', 'BetaPeakConcentration'):
                p = (params[0], params[1], et, tn)
            else:
                p = params
            
            q_array = self.q.compute(t_sliced, *p)
            model_ce = signal.ce(self.gm.dt, q_array)
            error = model_ce - target_ce
            t_var = np.var(target_ce)
            return np.mean(np.square(error)) / t_var if t_var > np.finfo(float).eps else np.mean(np.square(error))

        return objective, et, tn, t_shift

    def _default_parameters(self):
        all_defaults = {
            'BetaDual': ([0.1, 10.0, 0.2, 10.0, 0.6], [(0.01, 0.5), (2.0, 2000.0), (0.0, 0.5), (2.0, 2000.0), (0.0, 0.5)]),
            'BetaSingle': ([0.1, 10.0], [(0.01, 0.9), (0.1, 2000.0)]),
            'BetaPeakConcentration': ([0.1, 10.0], [(0.01, 0.9), (0.1, 2000.0)]),
            'BetaCentroidSpread': ([0.5, 0.2], [(0.1, 0.9), (0.01, 0.5)])
        }
        if self._q_type not in all_defaults:
            raise ValueError(f'No default parameters for {self._q_type}. Please provide initial_guess and bounds.')
        return {'modulating': all_defaults[self._q_type][0]}, {'modulating': all_defaults[self._q_type][1]}


class FrequencyInverter:
    """Inverter for identifying time-dependent frequency and damping parameters. Uses |gm.ac| to separately fit frequency/damping from modulating function."""
    def __init__(self, ground_motion: GroundMotion, 
                 upper_frequency: ParametricFunction, upper_damping: ParametricFunction,
                 lower_frequency: ParametricFunction, lower_damping: ParametricFunction):
        self.gm = ground_motion
        self.wu = upper_frequency
        self.zu = upper_damping
        self.wl = lower_frequency
        self.zl = lower_damping

        self._wu_type = type(self.wu).__name__
        self._zu_type = type(self.zu).__name__
        self._wl_type = type(self.wl).__name__
        self._zl_type = type(self.zl).__name__

    def fit(self, mode: str = 'stepwise', fit_range: tuple = (0.01, 0.99), initial_guess: dict = None, bounds: dict = None):
             
        default_guess, default_bounds = self._default_parameters()
        gs = {**default_guess, **(initial_guess or {})}
        bs = {**default_bounds, **(bounds or {})}
        
        results = {}
        
        if mode == 'joint':
            gs_fn = gs['upper_frequency'] + gs['upper_damping'] + gs['lower_frequency'] + gs['lower_damping']
            bs_fn = bs['upper_frequency'] + bs['upper_damping'] + bs['lower_frequency'] + bs['lower_damping']
            objective_fn = self._objective_frequency(fit_range)
            opt_fn = minimize(objective_fn, gs_fn, bounds=bs_fn, method='L-BFGS-B', jac="3-point").x

            raw_groups = self._slice_parameters(opt_fn)
            resolved_groups = self._resolve_parameters(raw_groups)
            funcs = [self.wu, self.zu, self.wl, self.zl]
            names = ['upper_frequency', 'upper_damping', 'lower_frequency', 'lower_damping']
            
            for name, func, vals in zip(names, funcs, resolved_groups):
                results[name] = {'type': type(func).__name__, 'params': dict(zip(func.param_names, vals))}
            return results

        elif mode == 'stepwise':
            # Step 1: Fit frequency from zero-crossings
            gs_w = gs['upper_frequency'] + gs['lower_frequency']
            bs_w = bs['upper_frequency'] + bs['lower_frequency']
            objective_w = self._objective_frequency_zc(fit_range)
            opt_w = minimize(objective_w, gs_w, bounds=bs_w, method='L-BFGS-B', jac="3-point").x

            p_wu = opt_w[:self.wu.n_params]
            p_wl = opt_w[self.wu.n_params:]
            resolved_groups = self._resolve_parameters([p_wu, np.zeros(self.zu.n_params, dtype=float), 
                                                        p_wl, np.zeros(self.zl.n_params, dtype=float)])
            wu_params = resolved_groups[0]
            wl_params = resolved_groups[2]
            results['upper_frequency'] = {'type': self._wu_type, 'params': dict(zip(self.wu.param_names, wu_params))}
            results['lower_frequency'] = {'type': self._wl_type, 'params': dict(zip(self.wl.param_names, wl_params))}

            # Step 2: Fit damping with fixed frequencies
            gs_d = gs['upper_damping'] + gs['lower_damping']
            bs_d = bs['upper_damping'] + bs['lower_damping']
            objective_d = self._objective_damping(fit_range, wu_params, wl_params)
            opt_d = minimize(objective_d, gs_d, bounds=bs_d, method='L-BFGS-B', jac="3-point").x

            p_zu = opt_d[:self.zu.n_params]
            p_zl = opt_d[self.zu.n_params:]
            results['upper_damping'] = {'type': self._zu_type, 'params': dict(zip(self.zu.param_names, p_zu))}
            results['lower_damping'] = {'type': self._zl_type, 'params': dict(zip(self.zl.param_names, p_zl))}
            return results
            
        raise ValueError(f"Unknown mode: {mode}")

    def _slice_parameters(self, flat_params):
        groups = []
        offset = 0
        for func in [self.wu, self.zu, self.wl, self.zl]:
            groups.append(flat_params[offset : offset + func.n_params])
            offset += func.n_params
        return groups

    def _resolve_parameters(self, groups):
        p_wu, p_zu, p_wl, p_zl = groups
        final_wl = p_wl.copy()
        
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

    def _cumulative_frequency(self, freq_hz: np.ndarray):
        out = np.empty_like(freq_hz, dtype=np.float64)
        out[0] = 0.0
        out[1:] = np.cumsum((freq_hz[:-1] + freq_hz[1:]) * 0.5) * self.gm.dt
        return out

    def _objective_frequency(self, fit_range: tuple):
        slicer = signal.slice_energy(self.gm.ce, fit_range)
        
        # Heuristic FAS range: focus on 0.1 Hz up to 90% of the Nyquist frequency
        nyquist = 0.5 / self.gm.dt
        f_slicer = (self.gm.freq >= 0.1) & (self.gm.freq <= 0.9 * nyquist)
        
        targets = [self.gm.zc_ac[slicer] - self.gm.zc_ac[slicer][0],
                   self.gm.zc_vel[slicer] - self.gm.zc_vel[slicer][0],
                   self.gm.zc_disp[slicer] - self.gm.zc_disp[slicer][0],
                   self.gm.pmnm_vel[slicer] - self.gm.pmnm_vel[slicer][0],
                   self.gm.pmnm_disp[slicer] - self.gm.pmnm_disp[slicer][0],
                   self.gm.fas[f_slicer]]
        variances = [np.var(t) for t in targets]
        
        # DECOUPLED FROM FITTED MODULATING FUNCTION
        q_array = np.abs(self.gm.ac)
        
        def objective(params):
            raw_groups = self._slice_parameters(params)
            resolved_groups = self._resolve_parameters(raw_groups)
            wu_arr = self.wu.compute(self.gm.t, *resolved_groups[0])
            zu_arr = self.zu.compute(self.gm.t, *resolved_groups[1])
            wl_arr = self.wl.compute(self.gm.t, *resolved_groups[2])
            zl_arr = self.zl.compute(self.gm.t, *resolved_groups[3])

            model = StochasticModel(self.gm.npts, self.gm.dt, q_array, wu_arr, zu_arr, wl_arr, zl_arr)

            preds = [model.zc_ac[slicer] - model.zc_ac[slicer][0],
                     model.zc_vel[slicer] - model.zc_vel[slicer][0],
                     model.zc_disp[slicer] - model.zc_disp[slicer][0],
                     model.pmnm_vel[slicer] - model.pmnm_vel[slicer][0],
                     model.pmnm_disp[slicer] - model.pmnm_disp[slicer][0],
                     model.fas[f_slicer]]
            
            error = 0.0
            for pred, target, var in zip(preds, targets, variances):
                if var > np.finfo(float).eps:
                    error += np.mean(np.square(pred - target)) / var
            return error
        return objective

    def _objective_frequency_zc(self, fit_range: tuple):
        slicer = signal.slice_energy(self.gm.ce, fit_range)
        targets = [self.gm.zc_ac[slicer] - self.gm.zc_ac[slicer][0],
                   self.gm.zc_disp[slicer] - self.gm.zc_disp[slicer][0]]
        variances = [np.var(t) for t in targets]

        def objective(params):
            p_wu = params[:self.wu.n_params]
            p_wl = params[self.wu.n_params:]
            resolved = self._resolve_parameters([p_wu, np.zeros(self.zu.n_params, dtype=float), 
                                                 p_wl, np.zeros(self.zl.n_params, dtype=float)])
            wu_arr = self.wu.compute(self.gm.t, *resolved[0])
            wl_arr = self.wl.compute(self.gm.t, *resolved[2])
            zc_ac = self._cumulative_frequency(wu_arr)
            zc_disp = self._cumulative_frequency(wl_arr)
            preds = [zc_ac[slicer] - zc_ac[slicer][0], zc_disp[slicer] - zc_disp[slicer][0]]
            
            error = 0.0
            for pred, target, var in zip(preds, targets, variances):
                if var > np.finfo(float).eps:
                    error += np.mean(np.square(pred - target)) / var
            return error
        return objective

    def _objective_damping(self, fit_range: tuple, wu_params: np.ndarray, wl_params: np.ndarray):
        slicer = signal.slice_energy(self.gm.ce, fit_range)

        # Heuristic FAS range: focus on 0.1 Hz up to 90% of the Nyquist frequency
        nyquist = 0.5 / self.gm.dt
        f_slicer = (self.gm.freq >= 0.1) & (self.gm.freq <= 0.9 * nyquist)

        targets = [self.gm.zc_ac[slicer] - self.gm.zc_ac[slicer][0],
                   self.gm.zc_vel[slicer] - self.gm.zc_vel[slicer][0],
                   self.gm.zc_disp[slicer] - self.gm.zc_disp[slicer][0],
                   self.gm.pmnm_vel[slicer] - self.gm.pmnm_vel[slicer][0],
                   self.gm.pmnm_disp[slicer] - self.gm.pmnm_disp[slicer][0],
                   self.gm.fas[f_slicer]]
        variances = [np.var(t) for t in targets]

        # DECOUPLED FROM FITTED MODULATING FUNCTION
        q_array = np.abs(self.gm.ac)
        wu_arr = self.wu.compute(self.gm.t, *wu_params)
        wl_arr = self.wl.compute(self.gm.t, *wl_params)

        def objective(params):
            p_zu = params[:self.zu.n_params]
            p_zl = params[self.zu.n_params:]
            zu_arr = self.zu.compute(self.gm.t, *p_zu)
            zl_arr = self.zl.compute(self.gm.t, *p_zl)

            model = StochasticModel(self.gm.npts, self.gm.dt, q_array, wu_arr, zu_arr, wl_arr, zl_arr)

            preds = [model.zc_ac[slicer] - model.zc_ac[slicer][0],
                     model.zc_vel[slicer] - model.zc_vel[slicer][0],
                     model.zc_disp[slicer] - model.zc_disp[slicer][0],
                     model.pmnm_vel[slicer] - model.pmnm_vel[slicer][0],
                     model.pmnm_disp[slicer] - model.pmnm_disp[slicer][0],
                     model.fas[f_slicer]]

            error = 0.0
            for pred, target, var in zip(preds, targets, variances):
                if var > np.finfo(float).eps:
                    error += np.mean(np.square(pred - target)) / var
            return error
        return objective

    def _default_parameters(self):
        all_defaults = {('upper_frequency', 'Linear'): ([3.0, 2.0], [(0.1, 40.0), (0.1, 40.0)]),
                        ('upper_frequency', 'Exponential'): ([3.0, 2.0], [(0.1, 40.0), (0.1, 40.0)]),
                        ('upper_frequency', 'Constant'): ([5.0], [(0.1, 40.0)]),
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

        context_and_type = [('upper_frequency', type(self.wu).__name__),
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


class ModelInverter:
    """
    Combined Inverter to fit both Modulating and Frequency parameters.
    Delegates to ModulatingInverter and FrequencyInverter.
    """
    def __init__(self, ground_motion: GroundMotion, modulating: ParametricFunction,
                 upper_frequency: ParametricFunction, upper_damping: ParametricFunction,
                 lower_frequency: ParametricFunction, lower_damping: ParametricFunction):
        self.gm = ground_motion
        self.modulating = modulating
        self.upper_frequency = upper_frequency
        self.upper_damping = upper_damping
        self.lower_frequency = lower_frequency
        self.lower_damping = lower_damping
        
        self.modulating_fitter = ModulatingInverter(self.gm, self.modulating)
        self.frequency_fitter = FrequencyInverter(self.gm, self.upper_frequency, self.upper_damping, self.lower_frequency, self.lower_damping)

    def fit(self, mode: str = 'stepwise', fit_range: tuple = (0.01, 0.99), initial_guess: dict = None, bounds: dict = None):
        
        self.results = {}
        
        # 1. Fit modulating function
        res_q = self.modulating_fitter.fit(fit_range, initial_guess, bounds)
        self.results['modulating'] = res_q

        # 2. Fit sequence tracking parameters (decoupled from modulating via gm.ac)
        res_f = self.frequency_fitter.fit(mode, fit_range, initial_guess, bounds)
        self.results.update(res_f)
        
        return StochasticModel.load_from(self.results, self.gm.npts, self.gm.dt)

