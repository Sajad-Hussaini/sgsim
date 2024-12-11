import time
import importlib.resources
from SGSIM import TargetMotion, SimMotion, model_fit, StochasticModel, ModelPlot
# Step 1: Determine record file_path and source and initialize a target motion

# Use importlib.resources to get the file path relative to the SGSIM package
file_name = 'examples/real_records/N1.AT2'
# file_name = 'examples/real_records/IV6.AT2'  # or provide a path to your file
file_path = importlib.resources.files('SGSIM').joinpath(file_name)
source = 'nga' # for NGA database or 'esm' for ESM database accelerograms

target_range= (0.001, 0.999)
bandpass = (0.1, 25.0) # if a bandpass filtering is required
bandpass = None # if a bandpass filtering is not required
target_motion = TargetMotion(file_path, source).set_target_range(target_range, bandpass)
# %% Step 2: Initialize the stochastic model and define number of data points (npts), time step (dt), and functional forms
model = StochasticModel(npts = target_motion.npts, dt = target_motion.dt,
                        mdl_type = 'beta_multi',
                        wu_type = 'linear', zu_type = 'linear',
                        wl_type = 'linear', zl_type = 'linear')

# %% Step 3: Fit the stochastic model to the target motion in the order
start = time.perf_counter()
for fit_type in ['modulating', 'freq', 'damping']:  # approach 1
# for fit_type in ['modulating', 'all']:  # an alternative approach 2
    model_fit(fit_type, model, target_motion)
end = time.perf_counter()
print(f'Optimization done in {end - start:.1f}s.')

# %% Step 4: Simulate time series using the stocahstic model
model.simulate(nsim=18)  # nsim number of direct simulation of ac, vel, disp
model.get_properties()  # get all model properties (i.e., FAS, CE, zero crossings, local extrema)
# to access simulation use lower case attributes (e.g., model.ac, model.vel
# model.disp, model.fas, model.ce, model.mzc_ac, model.mzc_vel, model.mzc_disp, etc.)
model.print_parameters()
# %% Step 5: Initialize simulated motions to get their properties
sim_motion = SimMotion(model).get_properties()

# %% Step 6: Plot results using instance of ModelPlot
mp = ModelPlot(sim_motion, target_motion)
mp.plot_motion('Acceleration (g)', 0, -1)  # indices to plot 0 first and last -1 simulations
mp.plot_motion('Velocity (cm/s)', 0, -1)
mp.plot_motion('Displacement (cm)', 0, -1)
mp.plot_ce()
mp.plot_fas()
mp.plot_spectra()
mp.error_feature('mzc')
mp.error_feature('mle')
mp.error_feature('pmnm')
