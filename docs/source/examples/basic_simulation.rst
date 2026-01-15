.. _example_basic_simulation:

Quick Start: Basic Simulation Example
======================================

This guide demonstrates how to fit a stochastic model to a recorded ground motion and generate synthetic simulations.

Step 1: Import Libraries
------------------------

Import necessary `sgsim` classes along with NumPy.

.. code-block:: python

   # %% import libraries
   import numpy as np
   import matplotlib.pyplot as plt
   from sgsim import GroundMotion, StochasticModel
   from sgsim.Function

Step 2: Load the Target Ground Motion
-------------------------------------

Load an existing accelerogram record to serve as the target for the simulation.

.. code-block:: python

   # %% Prepare the target ground motion
   # Change the path to the file
   gm = GroundMotion.load_from(source='nga', file='RSN123_Example.AT2')
   # If necessary , preprocess the ground motion (e.g., trimming, baseline correction, tapering and filtering)
   gm = gm.trim_by_energy((0.001, 0.999)).taper(0.05).butterworth_filter((0.05, 100.0))

Step 3: Define and Fit the Stochastic Model
-------------------------------------------

Define the functional forms for the model parameters (amplitude, frequency, and damping evolution) and fit them to the target motion.

.. code-block:: python

   # %% Define and fit the stochastic model
   # 1. Modulating function (Envelope) -> Beta Function
   # 2. Frequency/Damping evolution -> Linear Functions
   model = StochasticModel(
       modulating=Functions.BetaSingle(),
       upper_frequency=Functions.Linear(), 
       upper_damping=Functions.Linear(),
       lower_frequency=Functions.Linear(), 
       lower_damping=Functions.Linear())

   # Fit the model to the target ground motion with default settings
   model.fit(gm)

   # Alternatively, fit specific components with custom initial guesses and bounds
   #model.fit(gm, ['modulating'])  # default is used if no bounds/initial_guess provided
   #model.fit(gm, component=["frequency"],
   #      initial_guess=[8.0, 0.1, np.sqrt(0.5), np.sqrt(0.5)],
   #      bounds=[(0.1, 40.0), (0.01, 0.99), (np.sqrt(0.5), np.sqrt(0.5)), (np.sqrt(0.5), np.sqrt(0.5))])

   # Print a summary of the fitted parameters
   model.summary()

Step 4: Generate Simulations
----------------------------

Generate synthetic ground motions based on the fitted model parameters.

.. code-block:: python

   # %% Generate simulated ground motions object
   # Generate 10 simulated ground motions (sm)
   sm = model.simulate(n=10)

   # View available IMs
   sm.list_IMs()

Step 5: Visualize the Results
-----------------------------

Use the ``ModelPlot`` class to compare the simulations against the target.

.. code-block:: python

   # %% Visualize results using default ModelPlot but it is recommended to use custom plots
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

   # %% Export to CSV including specific IMs and arbitrary spectral ordinates
   sm.to_csv('output_simulation.csv', ims=['pga', 'pgv', 'cav', 'sa'], periods=periods)   

   # %% Save time and acceleration arrays to a text\csv file using numpy
   output_path = "output_gm.csv"

   data = np.column_stack((sm.t, sm.ac.T))
   np.savetxt(output_path, data, fmt="%.6e", delimiter=',', header="use your header here")
