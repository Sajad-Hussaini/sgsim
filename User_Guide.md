# SGSIM User Guide

This is a quick tutorial to help you get started with SGSIM.

## Step-by-Step Tutorial

### Step 1: Import Libraries and Initialize Target Real Motion
```python
import time
from SGSIM import StochasticModel, Motion, calibrate, ModelPlot

# Specify a file as the target real accelerogram
file = 'path/to-file/Northeidge.AT2'

source = 'nga'                # Specify source: 'nga' for NGA and 'esm' for ESM databases


target_range= (0.001, 0.999)  # Specify input target range (% of total energy)

bandpass = (0.1, 25.0)        # if a bandpass filtering is required (in Hz)
#bandpass = None              # if a bandpass filtering is not required

real_motion = Motion.from_file(file_path, source).set_target_range(target_range, bandpass)
```
### Step 2: Initialize the Stochastic Model
```python
# Define the number of data points (npts), time step (dt) (can use the target motion as below)
# Also define parameters functional forms
# mdl, wu, zu, wl, zl are respectively modulating function, upper dominant frequency, upper bandwidth parameter, lower dominant frequency, lower bandwidth parameter

# for mdl options are: 'beta_dual' for (two strong phase motions), 'beta_single', 'gamma' (for one strong phase)
# for filter parameters all should be the same currently and can choose 'linear', 'exponential'

model = StochasticModel(npts = real_motion.npts, dt = real_motion.dt,
                        mdl_type = 'beta_single',
                        wu_type = 'linear', zu_type = 'linear',
                        wl_type = 'linear', zl_type = 'linear')
```
### Step 3: Calibrate the Stochastic Model Using the Real Motion
```python
# initial guess and bounds can be provided for the parameters otherwise it uses the defaults
start = time.perf_counter()

# approach 1
for func in ['modulating', 'freq', 'damping']:
    calibrate(func, model, real_motion, initial_guess=None, lower_bounds=None, upper_bounds=None)
    
# an alternative approach 2
# for func in ['modulating', 'all']:
    # model_fit(func, model, real_motion, initial_guess=None, lower_bounds=None, upper_bounds=None)
    
end = time.perf_counter()
print(f'Model calibration done in {end - start:.1f}s.')
```
### Step 4: Simulate Ground Motions Using the Stochastic Model
```python
# nsim number of direct simulation (i.e. [nsim, npts]) of ac, vel, disp
model.simulate(n=18)

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
sim_motion = Motion.from_model(model).get_properties()

# Similarly, to access simulations use sim_motion.ac , sim_motion.fas, sim_motion.sa, sim_motion.sv, etc.

# In case of necessity to save properties use below by specifying filename and path
sim_motion.save_spectra(filename="path/simulated_spectra.csv")

sim_motion.save_motions(filename="path/simulated_motions.csv")

sim_motion.save_fas(filename="path/simulated_fas.csv")

sim_motion.save_peak_motions(filename="path/simulated_PG_parameters.csv")

sim_motion.save_characteristics(filename="path/simulated_characteristics.csv")
```
### Step 6: Plot Results Using ModelPlot
```python
mp = ModelPlot(model, sim_motion, real_motion)

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