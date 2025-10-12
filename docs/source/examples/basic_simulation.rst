.. _example_basic_simulation:

Quick Start: Basic Simulation Example
======================================

Creating a Ground Motion instance from a record file
-----------------------------------------------------

.. code-block:: python

   from sgsim import GroundMotion

   # Path to your accelerogram file
   file = 'path/to/your/motion.AT2'
   source = 'nga'

   # Create a GroundMotion instance by loading a record file
   ground_motion = GroundMotion.load_from(source=source, file=file)

   # If necessary, trim to 0.1% - 99.9% cumulative energy
   # If necessary, apply a bandpass filter (e.g., 0.1–25 Hz)
   ground_motion.trim("energy", (0.001, 0.999)).filter((0.1, 25.0))


Creating a Stochastic Model instance
--------------------------------------

.. code-block:: python

   from sgsim import StochasticModel, functions

   # Create a StochasticModel instance by specifying functional forms and basic parameters npts and dt   
   model = StochasticModel(npts=ground_motion.npts, dt=ground_motion.dt,
                        modulating=functions.BetaSingle(),
                        upper_frequency=functions.Linear(), upper_damping=functions.Linear(),
                        lower_frequency=functions.Linear(), lower_damping=functions.Linear())


Fitting the Stochastic Model to a Target Ground Motion
---------------------------------------------------------------

.. code-block:: python

   for func in ['modulating', 'frequency']:
    model.fit(func, ground_motion)

   model.summary()

Simulating Ground Motions
-----------------------------------------------------------------------------

.. code-block:: python

   simulated_motion = model.simulate(n=10)
   print(f"number of acceleration time series : {simulated_motion.ac.shape[0]}")
   print(f"number of samples in each time series : {simulated_motion.ac.shape[1]}")


Visualizing the Results
----------------------------------------------------------------------------

.. code-block:: python

   from sgsim import ModelPlot

   # if necessary create period tp arrays for spectral plots
   # e.g., from 0.05s to 10s with increment of 0.05s
   sm.tp = (0.05, 10.05, 0.05)
   gm.tp = (0.05, 10.05, 0.05)

   # create a ModelPlot instance
   mp = ModelPlot(model, simulated_motion, ground_motion)

   # index first and last simulated motion to visualize (0, -1)
   mp.plot_motions(0, -1)
   # additional plotting functions
   mp.plot_ac_ce()
   mp.plot_ce()
   mp.plot_fas()
   # additional plotting functions
   mp.plot_spectra(spectrum='sa')
   mp.plot_feature(feature='mzc')
   mp.plot_feature(feature='pmnm')
