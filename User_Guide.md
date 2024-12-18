# SGSIM User Guide

This is a quick tutorial to help you get started with SGSIM.

## Step-by-Step Tutorial

### Step 1: Import Libraries and Initialize Target Motion
```python
import time
from SGSIM import TargetMotion, SimMotion, model_fit, StochasticModel, ModelPlot

# Specify the file path for a target accelerogram
file_name = 'examples/real_records/N1.AT2'

# Specify source: 'nga' for NGA and 'esm' for ESM ground motion database
source = 'nga'

target_range= (0.001, 0.999)  # Specify a target range (% of total energy)

bandpass = (0.1, 25.0) # if a bandpass filtering is required (in Hz)
#bandpass = None        # if a bandpass filtering is not required

target_motion = TargetMotion(file_path, source).set_target_range(target_range, bandpass)
```
### Step 2: Initialize the Stochastic Model
```python
# Define the number of data points (npts), time step (dt) (can use the target motion as below)
# Also define parameters functional forms
# mdl, wu, zu, wl, zl are respectively modulating function, upper dominant frequency, upper bandwidth parameter, lower dominant frequency, lower bandwidth parameter

# mdl function options: 'beta_multi' for (two strong phase motions), 'beta_single', 'gamma'
# filter parameters options (all should be the same currently): 'linear', 'exponential'

model = StochasticModel(npts = target_motion.npts, dt = target_motion.dt,
                        mdl_type = 'beta_single',
                        wu_type = 'linear', zu_type = 'linear',
                        wl_type = 'linear', zl_type = 'linear')
```
### Step 3: Fit the Stochastic Model to the Target Motion
```python
# initial guess and bounds can be provided for the parameters otherwise it uses the defaults

start = time.perf_counter()

# approach 1
for fit_type in ['modulating', 'freq', 'damping']:
    model_fit(fit_type, model, target_motion, initial_guess=None, lower_bounds=None, upper_bounds=None)

# an alternative approach 2
# for fit_type in ['modulating', 'all']:
    # model_fit(fit_type, model, target_motion, initial_guess=None, lower_bounds=None, upper_bounds=None)

end = time.perf_counter()
print(f'Model calibration done in {end - start:.1f}s.')
```
### Step 4: Simulate Time Series Using the Stochastic Model
```python
# nsim number of direct simulation of ac, vel, disp
model.simulate(nsim=18)

# get all model properties (i.e., FAS, CE, zero crossings, local extrema)
model.get_properties()

# to access simulation use lower case attributes (e.g., model.ac, model.vel
# model.disp, model.fas, model.ce, model.mzc_ac, model.mzc_vel, model.mzc_disp, etc.)

# Print fitted parameters
model.print_parameters()
```
### Step 5: Initialize Simulated Motions and Save Results
```python
# it provides a 2D array [nsim, arrays] for each parameter
sim_motion = SimMotion(model).get_properties()

# Similarly, to access simulations access like sim_motion.ac , sim_motion.fas, sim_motion.sa, sim_motion.sv, etc.

# In case of necessity to save properties use below by specifying filename and path
sim_motion.save_spectra(filename=r"Desktop\simulated_spectra.csv")
sim_motion.save_motions(filename=r"Desktop\simulated_motions.csv")
sim_motion.save_fas(filename=r"Desktop\simulated_fas.csv")
sim_motion.save_peak_motions(filename=r"Desktop\simulated_PG_parameters.csv")
sim_motion.save_characteristics(filename=r"Desktop\simulated_characteristics.csv")
```
### Step 6: Plot Results Using ModelPlot
```python
mp = ModelPlot(sim_motion, target_motion)

# indices to plot first 0 and last -1 simulated motion
mp.plot_motion('Acceleration (g)', 0, -1)
mp.plot_motion('Velocity (cm/s)', 0, -1)
mp.plot_motion('Displacement (cm)', 0, -1)
mp.plot_ce()
mp.plot_fas()
mp.plot_spectra()
mp.error_feature('mzc')
mp.error_feature('mle')
mp.error_feature('pmnm')
```