.. _example_basic_simulation:

Quick Start: Basic Simulation Example
======================================

Loading a Ground Motion from a record file
-----------------------------------------------------

.. code-block:: python

   from sgsim import GroundMotion

   # Path to your accelerogram file
   file = 'path/to/your/motion.AT2'
   source = 'nga'

   # Create a GroundMotion (gm) instance by loading a record file
   gm = GroundMotion.load_from(source=source, file=file)

   # If necessary, trim to 0.1% - 99.9% cumulative energy
   # If necessary, apply a bandpass filter (e.g., 0.1–25 Hz)
   gm.trim("energy", (0.001, 0.999)).filter((0.1, 25.0))


Creating and Fitting a Stochastic Model to the Target Ground Motion
-------------------------------------------------------------------------------------------------

.. code-block:: python

   from sgsim import StochasticModel, functions

   # Specify model's array data and functional forms
   model = StochasticModel(modulating=functions.BetaSingle(),
                        upper_frequency=functions.Linear(), upper_damping=functions.Linear(),
                        lower_frequency=functions.Linear(), lower_damping=functions.Linear())


   model.fit(gm)  # Default fitting procedure

   model.summary()  # Summary of fitted model


Simulating Ground Motions and Visualizing the Results
-----------------------------------------------------------------------------

.. code-block:: python

   from sgsim import ModelPlot

   # 10 simulated ground motions (sm) based on the fitted model
   sm = model.simulate(n=10)

   sm.available_IMs()  # List available IMs

   # Create a ModelPlot for visualizations
   mp = ModelPlot(model, sm, gm)

   gm.tp = sm.tp = (0.1, 10.1, 0.1)  # Set periods for response spectra or set to an array value

   # Index first and last simulated motion to visualize (0, -1)
   mp.plot_motions(0, -1)
   mp.plot_fas()
   mp.plot_ac_ce()
   mp.plot_spectra(spectrum='sa')
   mp.plot_feature(feature='mzc')
   mp.plot_feature(feature='pmnm')
