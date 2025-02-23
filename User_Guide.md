# SGSIM User Guide

This is a quick tutorial to help you get started with SGSIM.

## Step-by-Step Tutorial

### Step 1: Import Libraries and Initialize a Real Target Motion
```python
import time
from sgsim import StochasticModel, Motion, calibrate, ModelPlot, functions, tools

# Path to the accelerogram file (e.g., Northridge.AT2) and source (e.g., 'NGA', 'ESM')
file_path = r'C:\path\to\Northridge.AT2'  
source = 'nga'
real_motion = Motion.from_file(file_path, source)
# Specify a target range (% of total energy) for the motion to avoid zero tails
real_motion.set_range(option='energy', range_slice=(0.001, 0.999))
# Pereform filtering if required usng bandpass freqs in Hz
#real_motion.filter(bandpass_freqs=(0.1, 25.0))
```
### Step 2: Initialize the Stochastic Model
```python
# Need to define number of data points (npts), time step (dt), and function reference for parameters:
# modulating options: 'beta_dual' for (two strong phase motions), 'beta_single', 'gamma' (for one strong phase)
# filter parameters should be the same and options are: 'linear', 'exponential'
model = StochasticModel(npts = real_motion.npts, dt = real_motion.dt, modulating = functions.beta_dual,
                        upper_frequency = functions.linear, upper_damping = functions.linear,
                        lower_frequency = functions.linear, lower_damping = functions.linear)
```
### Step 3: Calibrate the Stochastic Model Using the Real Motion
```python
start = time.perf_counter()
# Alternative schemes for calibration (change True to False to swtich between schemes)
# Initial guess and bounds are set to Defaults if not provided
scheme = ['modulating', 'freq', 'damping'] if True else ['modulating', 'all']
for func in scheme:
    calibrate(func, model, real_motion, initial_guess=None, lower_bounds=None, upper_bounds=None)
end = time.perf_counter()
print(f'\nModel calibration done in {end - start:.1f}s.')
```
### Step 4: Simulate Ground Motions Using the Stochastic Model
```python
# To print calibrated model parameters use below
model.show_parameters()
# To save the model parameters as plain text use below by specifying filename and path
model.save_parameters(filename=r"C:\path\to\model_params.txt")
# number of direct simulation (i.e. [n, npts]) of ac, vel, disp
# to access simulation use lower case attributes (e.g., model.ac, model.vel, model.disp, model.fas, etc.)
model.simulate(n=25)
```
### Step 5: Initialize Simulated Motions and Save Results
```python
# an instance of simulated motions to to obtain various properties of each simulation
# direct access to each simulation properties as sim_motion.ac , sim_motion.fas, sim_motion.sa, sim_motion.sv, etc.
sim_motion = Motion.from_model(model)

# In case of necessity to save properties as csv file use below by specifying filename and path
sim_motion.save_simulations(filename = r"C:\path\to\simulated_spectra.csv", x_var = "tp", y_vars = ["sa", "sv", "sd"])
sim_motion.save_simulations(filename = r"C:\path\to\simulated_motions.csv", x_var = "t", y_vars = ["ac", "vel", "disp"])
```
### Step 6: Plot Results Using ModelPlot
```python
mp = ModelPlot(model, sim_motion, real_motion)
# Possibility to use a dict for plot config such as dpi, figsize
config = {'figure.dpi':300}

mp.plot_motion('Acceleration (g)', 0, -1, config=config)
mp.plot_motion('Velocity (cm/s)', 0, -1, config=config)
mp.plot_motion('Displacement (cm)', 0, -1, config=config)
mp.plot_ce(config=config)
mp.plot_fas(config=config)
mp.plot_spectra(spectrum='sa', config=config)
mp.error_feature(feature='mzc', config=config)
mp.error_feature(feature='mle', config=config)
mp.error_feature(feature='pmnm', config=config)
mp.error_ce(config=config)
```