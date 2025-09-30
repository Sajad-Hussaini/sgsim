.. _example_basic_simulation:

Basic Simulation Example
==========================

Creating a Ground Motion instance from a record file
-----------------------------------------------------

.. code-block:: python

   from sgsim import GroundMotion

   # Path to your accelerogram file
   file_path = 'path/to/your/motion.AT2'
   source = 'nga'

   # Create a GroundMotion instance by loading a record file
   ground_motion = GroundMotion.from_file(file_path, source)

   # If necessary, trim to 0.1% - 99.9% cumulative energy
   ground_motion.trim(option='energy', value=(0.001, 0.999))

   # If necessary, apply a bandpass filter (e.g., 0.1–25 Hz)
   ground_motion.filter(bandpass_freqs=(0.1, 25.0))


Creating a Stochastic Model instance
--------------------------------------

.. code-block:: python

   from sgsim import StochasticModel, functions

   # Create a StochasticModel instance by specifying functional forms and basic parameters npts and dt
   model = StochasticModel(npts=ground_motion.npts, dt=ground_motion.dt, 
                           modulating=functions.beta_dual,
                           upper_frequency=functions.linear, upper_damping=functions.linear,
                           lower_frequency=functions.linear, lower_damping=functions.linear)

Calibrating the Stochastic Model based on Target Ground Motion
---------------------------------------------------------------

.. code-block:: python

   from sgsim import calibrate

   funcs = ['modulating', 'frequency', 'damping']
   for func in funcs:
      calibrate(func, model, ground_motion, initial_guess=None, lower_bounds=None, upper_bounds=None)

   model.summary()

Simulating Ground Motions and Creating Ground Motion Instances for Analysis
-----------------------------------------------------------------------------

.. code-block:: python

   ac, vel, disp = model.simulate(n=10)
   print(f"number of acceleration time series : {ac.shape[0]}")
   print(f"number of samples in each time series : {ac.shape[1]}")

   npts, dt = model.npts, model.dt
   simulated_motion = GroundMotion(npts, dt, ac, vel, disp)

Visualizing the Results and Comparing Set of Simulations with Real Record
----------------------------------------------------------------------------

.. code-block:: python

   from sgsim import ModelPlot

   mp = ModelPlot(model, simulated_motion, ground_motion)

   # index first and last simulated motion to visualize (0, -1)
   mp.plot_motions(0, -1)
   # additional plotting functions
   mp.plot_ac_ce()
   mp.plot_ce()
   mp.plot_fas()
   mp.plot_spectra(spectrum='sa')
   mp.plot_feature(feature='mzc')
   mp.plot_feature(feature='pmnm')
