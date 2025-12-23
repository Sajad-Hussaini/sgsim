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
   # If necessary, apply a bandpass filter (e.g., 0.1–25 Hz) or baseline correction
   gm = gm.trim_by_energy((0.001, 0.999)).butterworth_filter((0.01, 100.0))


Creating and Fitting a Stochastic Model to the Target Ground Motion
-------------------------------------------------------------------------------------------------

.. code-block:: python

   from sgsim import StochasticModel, Functions

   # Specify model's array data and functional forms
   model = StochasticModel(modulating=Functions.BetaSingle(),
                        upper_frequency=Functions.Linear(), upper_damping=Functions.Linear(),
                        lower_frequency=Functions.Linear(), lower_damping=Functions.Linear())


   model.fit(gm)  # Default fitting procedure

   model.summary()  # Summary of fitted model


Simulating Ground Motions and Visualizing the Results
-----------------------------------------------------------------------------

.. code-block:: python

   from sgsim import ModelPlot

   # 10 simulated ground motions (sm) based on the fitted model
   sm = model.simulate(n=10)

   sm.list_IMs()  # List available IMs (more IM will be added gradually or by user)

   # Create a ModelPlot for visualizations
   mp = ModelPlot(model, sm, gm)

   gm.tp = sm.tp = np.arange(0.1, 10.1, 0.1)

   # Index first and last simulated motion to visualize (0, -1)
   mp.plot_motions(0, -1)
   mp.plot_fas()
   mp.plot_ac_ce()
   mp.plot_spectra(gm.tp, spectrum='sa')

   # the slope of below curves should match than the number of cumualtive counts, however these characteristics are more prone to noise so perfit fit is not recommended
   mp.plot_feature(feature='mzc')
   mp.plot_feature(feature='pmnm')


   # To export selected IMs of simulated motions to CSV files
   sm.to_csv('output_ground_motion.csv', ims=['pga', 'pgv', 'sa', 'fas'], periods=gm.tp)  # Or any other period range
