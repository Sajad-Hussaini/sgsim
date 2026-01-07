.. _example_basic_simulation:

Quick Start: Basic Simulation Example
======================================

This guide demonstrates how to fit a stochastic model to a recorded ground motion and generate synthetic simulations.

Step 1: Import Libraries
------------------------

Import necessary `sgsim` classes along with NumPy.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sgsim import GroundMotion, StochasticModel, Functions, ModelPlot

Step 2: Load the Target Ground Motion
-------------------------------------

Load an existing accelerogram record to serve as the target for the simulation.

.. code-block:: python

   # Path to your accelerogram file
   file_path = 'path/to/your/motion.AT2'

   # Load the target ground motion (gm)
   gm = GroundMotion.load_from(source='nga', file=file_path)
   # If necessary trim and filter to ensure clean data for fitting
   gm = gm.trim_by_energy((0.001, 0.999)).butterworth_filter((0.01, 100.0))

Step 3: Define and Fit the Stochastic Model
-------------------------------------------

Define the functional forms for the model parameters (amplitude, frequency, and damping evolution) and fit them to the target motion.

.. code-block:: python

   # Initialize specific functions for model components:
   # 1. Modulating function (Envelope) -> Beta Function
   # 2. Frequency/Damping evolution -> Linear Functions
   model = StochasticModel(
       modulating=Functions.BetaSingle(),
       upper_frequency=Functions.Linear(), 
       upper_damping=Functions.Linear(),
       lower_frequency=Functions.Linear(), 
       lower_damping=Functions.Linear())

   # Fit the model to the target ground motion
   model.fit(gm)

   # Print a summary of the fitted parameters
   model.summary()

Step 4: Generate Simulations
----------------------------

Generate synthetic ground motions based on the fitted model parameters.

.. code-block:: python

   # Generate 10 simulated ground motions (sm)
   sm = model.simulate(n=10)

   # View simple scalar stats
   sm.list_IMs()

Step 5: Visualize the Results
-----------------------------

Use the ``ModelPlot`` class to compare the simulations against the target.

.. code-block:: python

   # Initialize the plotter with the Model, Simulations, and Target
   mp = ModelPlot(model, sm, gm)

   # Define the period range for spectral plots (0.1s to 10s)
   periods = np.arange(0.1, 10.1, 0.1)

   # 1. Plot Time-Series Comparison (Index 0 and Last simulation)
   mp.plot_motions(0, -1)

   # 2. Plot Fourier Amplitude Spectra
   mp.plot_fas()

   # 3. Plot Cumulative Energy
   mp.plot_ac_ce()

   # 4. Plot Response Spectra (SA)
   mp.plot_spectra(periods, spectrum='sa')

   # 5. Plot Physical Features (Zero-Crossings and Maxima)
   # Note: Ideally, slopes should match. However, these features are noise-sensitive, 
   # so a perfect fit is not strictly required.
   mp.plot_feature(feature='mzc')  # Mean Zero-Crossing rate
   mp.plot_feature(feature='pmnm') # Peaks per positive minima

Step 6: Export Simulations
--------------------------

Export the Intensity Measures (IMs) and spectral data of the synthetics to CSV.

.. code-block:: python

   # Export formatted results
   sm.to_csv('output_simulation.csv', ims=['pga', 'pgv', 'sa', 'fas'], periods=periods)
