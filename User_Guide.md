# SGSIM User Guide

This is a quick tutorial to help you get started with SGSIM.

## Step-by-Step Tutorial

### Step 1: Import Libraries and Initialize a Real Target Motion
```python
import time
from SGSIM import StochasticModel, Motion, calibrate, ModelPlot

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
# Need to define number of data points (npts), time step (dt), and functional forms for parameters:
# mdl: modulating function
# wu, zu: upper dominant frequency and damping ratio
# wl, zl: lower dominant frequency and damping ratio

# mdl func options: 'beta_dual' for (two strong phase motions), 'beta_single', 'gamma' (for one strong phase)
# filter parameters should be the same and options are: 'linear', 'exponential'
model = StochasticModel(npts = real_motion.npts, dt = real_motion.dt,
                        mdl_func = 'beta_single',
                        wu_func = 'linear', zu_func = 'linear',
                        wl_func = 'linear', zl_func = 'linear')
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
model.parameters()
# To save the model parameters as hdf5 use below by specifying filename and path
model.save_parameters(filename=r"C:\Users\Sajad\OneDrive - Universidade do Minho\Python-Scripts\examples\params.h5")
# number of direct simulation (i.e. [n, npts]) of ac, vel, disp
# to access simulation use lower case attributes (e.g., model.ac, model.vel, model.disp, model.fas, etc.)
model.simulate(n=25)
```
### Step 5: Initialize Simulated Motions and Save Results
```python
# an instance of simulated motions to to obtain various properties of each simulation
# direct access to each simulation properties as sim_motion.ac , sim_motion.fas, sim_motion.sa, sim_motion.sv, etc.
sim_motion = Motion.from_model(model)

# In case of necessity to save properties use below by specifying filename and path
sim_motion.save_spectra(filename=r"C:\path\to\simulated_spectra.csv")
sim_motion.save_motions(filename=r"C:\path\to\simulated_motions.csv")
sim_motion.save_fas(filename=r"C:\path\to\simulated_fas.csv")
sim_motion.save_peak_motions(filename=r"C:\path\to\simulated_PG_parameters.csv")
sim_motion.save_characteristics(filename=r"C:\path\to\simulated_characteristics.csv")

# Alternatively, save related groups of properties in a single hdf5 file (often more efficient))
sim_motion.save_simulations(filename=r"C:\path\to\simulations.h5", option=('spectra', 'motions', 'peak_motions', 'characteristics'))
```
### Step 6: Plot Results Using ModelPlot
```python
mp = ModelPlot(model, sim_motion, real_motion)
# Possibility to use **kwargs such as dpi, figsize
dpi = 150
# indices to plot: first 0 and last -1 simulated motion
mp.plot_motion('Acceleration (g)', 0, -1, dpi=dpi)
mp.plot_motion('Velocity (cm/s)', 0, -1, dpi=dpi)
mp.plot_motion('Displacement (cm)', 0, -1, dpi=dpi)
mp.plot_ce(dpi=dpi)
mp.plot_fas(dpi=dpi)
mp.plot_spectra(dpi=dpi)
mp.error_feature('mzc', dpi=dpi)
mp.error_feature('mle', dpi=dpi)
mp.error_feature('pmnm', dpi=dpi)
```