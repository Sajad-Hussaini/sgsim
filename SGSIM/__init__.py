"""
SGSIM package initialization.
"""

import importlib

__all__ = ['StochasticModel', 'Motion', 'calibrate', 'ModelPlot']

module_map = {
    'StochasticModel': 'core.stochastic_model',
    'Motion': 'motion.motion_model',
    'calibrate': 'optimization.model_calibrate',
    'ModelPlot': 'visualization.model_plot',
    'signal_analysis': 'motion.signal_analysis',
    }

def __getattr__(name):
    # Dynamically import and return the requested module attribute
    if name in module_map:
        module = importlib.import_module(f'.{module_map[name]}', package=__name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
