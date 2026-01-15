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
   from sgsim import GroundMotion, StochasticModel, Functions

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

You can use standard libraries like ``matplotlib`` to visualize the model, simulations, and the target.

.. tip::
   See :ref:`example_basic_groundmotion` for plotting response spectra and other intensity measures.


.. code-block:: python

   # %% Visualize Results using Matplotlib
   
   # 1. Compare Time Series (Target vs. First Simulation)
   fig, ax = plt.subplots(2, 1, sharex=True, figsize=(10, 6))
   
   # Plot Target
   ax[0].plot(gm.t, gm.ac, color='black', lw=0.8, label='Target')
   # Optional: Overlay Model Envelope
   ax[0].plot(model.t, model.modulating.value, color='red', linestyle='--', lw=1.5, label='Model Envelope')
   ax[0].plot(model.t, -model.modulating.value, color='red', linestyle='--', lw=1.5)
   ax[0].set_ylabel('Acceleration (g)')
   ax[0].legend(loc='upper right')
   ax[0].set_title('Target Ground Motion')
   
   # Plot First Simulation
   # sm.ac is an array of shape (n_simulations, n_points)
   ax[1].plot(sm.t, sm.ac[0], color='tab:blue', lw=0.8, label='Simulation #1')
   ax[1].set_ylabel('Acceleration (g)')
   ax[1].set_xlabel('Time (s)')
   ax[1].legend(loc='upper right')
   ax[1].set_title('Synthetic Ground Motion')
   
   plt.tight_layout()
   plt.show()

   # 2. Compare Fourier Amplitude Spectra (FAS)
   plt.figure(figsize=(6, 4))
   # Plot individual simulations (light gray background)
   # sm.fas is an array of shape (n_simulations, n_frequencies)
   plt.loglog(sm.freq, sm.fas.T, color='gray', alpha=0.3, lw=0.5)
   
   # Plot Simulation Mean
   plt.loglog(sm.freq, np.mean(sm.fas, axis=0), color='red', linestyle='--', label='Simulation Mean')
   
   # Plot Target
   plt.loglog(gm.freq, gm.fas, color='black', label='Target')
   
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('Fourier Amplitude')
   plt.legend()
   plt.grid(True, which="both", ls="-", alpha=0.2)
   plt.show()


Step 6: Export Simulations
--------------------------
.. tip::
   See :ref:`example_basic_groundmotion` for saving and plotting results.


