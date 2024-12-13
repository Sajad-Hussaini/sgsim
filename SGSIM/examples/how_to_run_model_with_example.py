"""
Site-Based Stochastic Ground Motion Simulation (SGSIM)
Quick exmaple tutorial:
    Sayed Mohammad Sajad Hussaini
    For any inquiry please contanct: hussaini.smsajad@gmail.com
    Ph.D. Candidate at University of Minho, Portugal
    Institute for Sustainability and Innovation in Structural Engineering
    Stand4Heritage Project
"""
import time
import importlib.resources
from SGSIM import TargetMotion, SimMotion, model_fit, StochasticModel, ModelPlot

# Step 1: Determine record file_path and source and initialize a target motion

# Use importlib.resources to get the exmaple records from the SGSIM package
file_name = 'examples/real_records/N1.AT2'  # exmaple1
# file_name = 'examples/real_records/IV6.AT2'  # exmaple2
file_path = importlib.resources.files('SGSIM').joinpath(file_name)
# Alternatively, use your target file-path as the file_name
source = 'nga' # Use 'nga' for NGA database and 'esm' for ESM database accelerograms
target_range= (0.001, 0.999)  # Specify a target range (% of total energy)
bandpass = (0.1, 25.0) # if a bandpass filtering is required
bandpass = None        # if a bandpass filtering is not required
target_motion = TargetMotion(file_path, source).set_target_range(target_range, bandpass)
# %% Step 2: Initialize the stochastic model

# Need to define number of data points (npts), time step (dt), and functional forms for parameters
# mdl: modulating function, wu: upper dominant, and wl: lower dominant frequencies
# zu: upper bandwidth, and zl: lower bandwidth parameteres
# for mdl options are: 'beta_multi' for (two strong phase motions), 'beta_single', 'gamma'
# for filter parameters all should be the same currently and can choose 'linear', 'exponential'
model = StochasticModel(npts = target_motion.npts, dt = target_motion.dt,
                        mdl_type = 'beta_single',
                        wu_type = 'linear', zu_type = 'linear',
                        wl_type = 'linear', zl_type = 'linear')
# %% Step 3: Fit the stochastic model to the target motion in the order

# initial guess and bounds can be provided for the parameters otherwise it uses the defaults
start = time.perf_counter()
for fit_type in ['modulating', 'freq', 'damping']:  # approach 1
# for fit_type in ['modulating', 'all']:  # an alternative is approach 2
    model_fit(fit_type, model, target_motion, initial_guess=None, lower_bounds=None, upper_bounds=None)
end = time.perf_counter()
print(f'Model calibration done in {end - start:.1f}s.')
# %% Step 4: Simulate time series using the stocahstic model

model.simulate(nsim=18)  # nsim number of direct simulation of ac, vel, disp
model.get_properties()  # get all model properties (i.e., FAS, CE, zero crossings, local extrema)
# to access simulation use lower case attributes (e.g., model.ac, model.vel
# model.disp, model.fas, model.ce, model.mzc_ac, model.mzc_vel, model.mzc_disp, etc.)
model.print_parameters()
# %% Step 5: Initialize simulated motions to get the properties of each simulation (not the model)

# it provides a 2D array [nsim, arrays] for each parameter
sim_motion = SimMotion(model).get_properties()
# Similarly, to access simulations access like sim_motion.ac , sim_motion.fas, sim_motion.sa, sim_motion.sv, etc.
# In case of necessity to save properties use below by specifying filename and path
sim_motion.save_spectra(filename=r"Desktop\simulated_spectra.csv")
sim_motion.save_motions(filename=r"Desktop\simulated_motions.csv")
sim_motion.save_fas(filename=r"Desktop\simulated_fas.csv")
sim_motion.save_peak_motions(filename=r"Desktop\simulated_PG_parameters.csv")
sim_motion.save_characteristics(filename=r"Desktop\simulated_characteristics.csv")
# %% Step 6: Plot results using instance of ModelPlot

mp = ModelPlot(sim_motion, target_motion)
mp.plot_motion('Acceleration (g)', 0, -1)  # indices to plot first 0 and last -1 simulated motion
mp.plot_motion('Velocity (cm/s)', 0, -1)
mp.plot_motion('Displacement (cm)', 0, -1)
mp.plot_ce()
mp.plot_fas()
mp.plot_spectra()
mp.error_feature('mzc')
mp.error_feature('mle')
mp.error_feature('pmnm')

