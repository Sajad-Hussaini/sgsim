import numpy as np
from scipy.optimize import minimize

from ..core.functions import ParametricFunction
from ..core.model import StochasticModel
from ..motion import signal
from ..motion.ground_motion import GroundMotion


FREQUENCY_GROUP_ORDER = ('upper_frequency', 'upper_damping', 'lower_frequency', 'lower_damping')
FULL_FREQUENCY_TARGETS = ('zc_ac', 'zc_vel', 'zc_disp', 'pmnm_vel', 'pmnm_disp', 'fas')
DYNAMIC_FUNCTION_TYPES = ('Linear', 'Exponential')
MODULATING_TYPES_WITH_DURATION = (
    'BetaDual',
    'BetaSingle',
    'BetaPeakConcentration',
    'BetaCentroidSpread',
)


class ModelInverter:
    def __init__(self, ground_motion: GroundMotion, modulating: ParametricFunction,
                 upper_frequency: ParametricFunction, upper_damping: ParametricFunction,
                 lower_frequency: ParametricFunction, lower_damping: ParametricFunction):
        self.gm = ground_motion
        self.q = modulating
        self.wu = upper_frequency
        self.zu = upper_damping
        self.wl = lower_frequency
        self.zl = lower_damping

        self._wu_type = type(self.wu).__name__
        self._wl_type = type(self.wl).__name__
        self._q_type = type(self.q).__name__
        self._frequency_funcs = {
            'upper_frequency': self.wu,
            'upper_damping': self.zu,
            'lower_frequency': self.wl,
            'lower_damping': self.zl,
        }

    def fit(self, criteria: str = 'full', fit_range: tuple = (0.01, 0.99), initial_guess: dict = None, bounds: dict = None):
        """Fit the modulating function first, then the frequency functions."""
        if criteria not in ('modulating', 'full', 'sequential'):
            raise ValueError(f"Unknown criteria: {criteria}")

        default_guess, default_bounds = self._default_parameters()
        guess = {**default_guess, **(initial_guess or {})}
        bound_map = {**default_bounds, **(bounds or {})}

        self.results = {}
        q_array = self._fit_modulating(guess, bound_map)

        if criteria == 'modulating':
            return None

        if criteria == 'full':
            frequency_groups = self._fit_full_frequency(fit_range, q_array, guess, bound_map)
        else:
            frequency_groups = self._fit_sequential_frequency(fit_range, q_array, guess, bound_map)

        self._store_frequency_results(frequency_groups)
        return StochasticModel.load_from(self.results, self.gm.npts, self.gm.dt)

    def _fit_modulating(self, guess: dict, bound_map: dict):
        objective = self._objective_modulating()
        solution = minimize(objective, guess['modulating'], bounds=bound_map['modulating'], method='L-BFGS-B', jac='3-point').x

        if self._q_type in MODULATING_TYPES_WITH_DURATION:
            solution = np.append(solution, [self.gm.ce.max(), self.gm.t.max()])

        params = dict(zip(self.q.param_names, solution))
        self.results['modulating'] = {'type': self._q_type, 'params': params}
        return self.q.compute(self.gm.t, **params)

    def _fit_full_frequency(self, fit_range, q_array, guess, bound_map):
        groups = self._initial_frequency_groups(guess)
        return self._fit_frequency_stage(groups, FREQUENCY_GROUP_ORDER, FULL_FREQUENCY_TARGETS, fit_range, q_array, bound_map)

    def _fit_sequential_frequency(self, fit_range, q_array, guess, bound_map):
        groups = self._initial_frequency_groups(guess)
        groups = self._fit_frequency_stage(groups, ('upper_frequency',), ('zc_ac',), fit_range, q_array, bound_map)
        groups = self._fit_frequency_stage(groups, ('lower_frequency',), ('zc_disp',), fit_range, q_array, bound_map)
        return self._fit_frequency_stage(groups, ('upper_damping', 'lower_damping'), FULL_FREQUENCY_TARGETS, fit_range, q_array, bound_map)

    def _initial_frequency_groups(self, guess):
        return {name: np.asarray(guess[name], dtype=float) for name in FREQUENCY_GROUP_ORDER}

    def _update_frequency_groups(self, groups, free_names, params):
        updated = {name: values.copy() for name, values in groups.items()}
        offset = 0
        for name in free_names:
            n_params = self._frequency_funcs[name].n_params
            updated[name] = np.asarray(params[offset:offset + n_params], dtype=float)
            offset += n_params
        return updated

    def _resolve_parameters(self, groups):
        upper_frequency = np.asarray(groups['upper_frequency'], dtype=float).copy()
        upper_damping = np.asarray(groups['upper_damping'], dtype=float).copy()
        lower_frequency = np.asarray(groups['lower_frequency'], dtype=float).copy()
        lower_damping = np.asarray(groups['lower_damping'], dtype=float).copy()

        upper_is_dynamic = self._wu_type in DYNAMIC_FUNCTION_TYPES
        lower_is_dynamic = self._wl_type in DYNAMIC_FUNCTION_TYPES
        upper_is_const = self._wu_type == 'Constant'
        lower_is_const = self._wl_type == 'Constant'

        if upper_is_dynamic and lower_is_dynamic:
            lower_frequency[0] = upper_frequency[0] * lower_frequency[0]
            lower_frequency[1] = upper_frequency[1] * lower_frequency[1]
        elif upper_is_const and lower_is_const:
            lower_frequency[0] = upper_frequency[0] * lower_frequency[0]
        elif upper_is_dynamic and lower_is_const:
            lower_frequency[0] = min(upper_frequency[0], upper_frequency[1]) * lower_frequency[0]

        return {
            'upper_frequency': upper_frequency,
            'upper_damping': upper_damping,
            'lower_frequency': lower_frequency,
            'lower_damping': lower_damping,
        }

    def _build_frequency_model(self, groups, q_array):
        wu_arr = self._frequency_funcs['upper_frequency'].compute(self.gm.t, *groups['upper_frequency'])
        zu_arr = self._frequency_funcs['upper_damping'].compute(self.gm.t, *groups['upper_damping'])
        wl_arr = self._frequency_funcs['lower_frequency'].compute(self.gm.t, *groups['lower_frequency'])
        zl_arr = self._frequency_funcs['lower_damping'].compute(self.gm.t, *groups['lower_damping'])
        return StochasticModel(self.gm.npts, self.gm.dt, q_array, wu_arr, zu_arr, wl_arr, zl_arr)

    def _target_data(self, target_names, fit_range):
        slicer = signal.slice_energy(self.gm.ce, fit_range)
        target_data = []
        for name in target_names:
            target = self.gm.fas if name == 'fas' else getattr(self.gm, name)[slicer]
            target_data.append((name, target, np.mean(np.square(target))))
        return slicer, target_data

    @staticmethod
    def _nmse(prediction, target, target_energy=None):
        energy = np.mean(np.square(target)) if target_energy is None else target_energy
        if energy <= np.finfo(float).eps:
            return 0.0
        return np.mean(np.square(prediction - target)) / energy

    def _frequency_error(self, model, target_data, slicer):
        error = 0.0
        for metric_name, target, target_energy in target_data:
            prediction = model.fas if metric_name == 'fas' else getattr(model, metric_name)[slicer]
            error += self._nmse(prediction, target, target_energy)
        return error

    def _make_frequency_objective(self, free_names, target_names, fit_range, q_array, base_groups):
        slicer, target_data = self._target_data(target_names, fit_range)

        def objective(params):
            candidate_groups = self._update_frequency_groups(base_groups, free_names, params)
            resolved_groups = self._resolve_parameters(candidate_groups)
            model = self._build_frequency_model(resolved_groups, q_array)
            return self._frequency_error(model, target_data, slicer)

        return objective

    def _fit_frequency_stage(self, groups, free_names, target_names, fit_range, q_array, bound_map):
        objective = self._make_frequency_objective(free_names, target_names, fit_range, q_array, groups)
        x0 = np.concatenate([groups[name] for name in free_names])
        stage_bounds = [bound for name in free_names for bound in bound_map[name]]
        solution = minimize(objective, x0, bounds=stage_bounds, method='L-BFGS-B', jac='3-point').x
        return self._update_frequency_groups(groups, free_names, solution)

    def _store_frequency_results(self, groups):
        resolved_groups = self._resolve_parameters(groups)
        for name in FREQUENCY_GROUP_ORDER:
            function = self._frequency_funcs[name]
            params = resolved_groups[name]
            self.results[name] = {'type': type(function).__name__, 'params': dict(zip(function.param_names, params))}

    def _objective_modulating(self):
        target_ce = self.gm.ce
        et, tn = self.gm.ce.max(), self.gm.t.max()

        def objective(params):
            if self._q_type == 'BetaDual':
                p = (params[0], params[1], params[0] + params[2], params[3], params[4], et, tn)
            elif self._q_type in MODULATING_TYPES_WITH_DURATION:
                p = (params[0], params[1], et, tn)
            else:
                p = params

            q_array = self.q.compute(self.gm.t, *p)
            model_ce = signal.ce(self.gm.dt, q_array)
            return self._nmse(model_ce, target_ce)

        return objective

    def _default_parameters(self):
        defaults = {
            ('modulating', 'BetaDual'): ([0.1, 10.0, 0.2, 10.0, 0.6], [(0.01, 0.5), (2.0, 2000.0), (0.0, 0.5), (2.0, 2000.0), (0.0, 0.5)]),
            ('modulating', 'BetaSingle'): ([0.1, 10.0], [(0.01, 0.9), (0.1, 2000.0)]),
            ('modulating', 'BetaPeakConcentration'): ([0.1, 10.0], [(0.01, 0.9), (0.1, 2000.0)]),
            ('modulating', 'BetaCentroidSpread'): ([0.2, 0.1], [(0.01, 0.9), (0.01, 0.49)]),
            ('upper_frequency', 'Linear'): ([3.0, 2.0], [(0.1, 40.0), (0.1, 40.0)]),
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
            ('lower_damping', 'Constant'): ([0.2], [(0.1, 0.99)]),
        }

        initial_guess = {}
        bounds = {}

        function_types = {
            'modulating': type(self.q).__name__,
            'upper_frequency': type(self.wu).__name__,
            'upper_damping': type(self.zu).__name__,
            'lower_frequency': type(self.wl).__name__,
            'lower_damping': type(self.zl).__name__,
        }

        for name, function_type in function_types.items():
            key = (name, function_type)
            if key not in defaults:
                raise ValueError(f'No default parameters for {key}. Please provide initial_guess and bounds.')

            guess, bound = defaults[key]
            initial_guess[name] = guess
            bounds[name] = bound

        return initial_guess, bounds
