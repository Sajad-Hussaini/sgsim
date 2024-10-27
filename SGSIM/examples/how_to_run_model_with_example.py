import time
from SGSIM import TargetMotion, SimMotion, model_fit, StochasticModel, ModelPlot, file_selector


file_path = file_selector.open_files()[0]
# file_path = r'SGSIM/examples/real_records/N1.AT2'
source = 'nga' # for NGA database or 'esm' for ESM database accelerograms
# Step 1: Initialize the target motion for a range based on percentage of total energy
target_range= (0.001, 0.999)
target_motion = TargetMotion(file_path, source).set_target_range(target_range)

# Step 2: Initialize the stochastic model by setting time array and type of model parameters
model = StochasticModel(
    npts = target_motion.npts, dt = target_motion.dt,
    mdl_type = 'beta_multi', wu_type = 'linear', zu_type = 'linear',
    wl_type = 'linear', zl_type = 'linear')

# Step 3: Fit the stochastic model to the target motion in the order
start = time.perf_counter()

model_fit('modulating', model, target_motion)  # 1
model_fit('freq', model, target_motion)        # 2
model_fit('damping', model, target_motion)     # 3

end = time.perf_counter()
print(f'Optimization done in {end - start:.1f}s.')

# Step 4: Simulate time series using the stocahstic model
ac, vel, disp = model.simulate()  # 1 direct simulation of ac, vel, disp
model.get_properties()            # get all model properties (i.e., FAS, CE, zero crossings, local extrema)
model.multi_simulate(10)          # 10 simulations of ac, vel, disp

# Step 5: Get simulation properties for multi simulated motion
sim_motion = SimMotion(model).get_properties()

# Step 6: Plot results using instance of ModelPlot
mp = ModelPlot(sim_motion, target_motion, model)
mp.plot_motion('Acceleration (g)', 1, 2)  # 1 and 2 refer to first and second simualation done above
mp.plot_motion('Velocity (cm/s)', 1, 2)
mp.plot_motion('Displacement (cm)', 1, 2)
mp.plot_ce()
mp.plot_fas()
mp.plot_spectra()
mp.error_feature('mzc')
mp.error_feature('mle')
mp.error_feature('pmnm')
