.. _example_basic_simulation:

Quick Start: Basic Simulation Example
======================================

This example demonstrates how to calibrate a stochastic model to a recorded ground motion.
We utilize the ``ModelInverter`` class to obtain a fitted ``StochasticModel``, which is then used to generate synthetic ground motion simulations.

Step 1: Import Libraries and Load the Target Ground Motion
------------------------------------------------------------

Import necessary `sgsim` classes along with necessary libraries.
Load an existing accelerogram record to serve as the target for the simulation.

.. code-block:: python

   # %% import libraries
   import numpy as np
   import matplotlib.pyplot as plt
   import time
   from sgsim import GroundMotion, ModelInverter, Functions

   # %% Prepare the target ground motion
   # Change the path to the file
   gm = GroundMotion.load_from(source='nga', file='RSN123_Example.AT2')
   # If necessary, preprocess the ground motion (e.g., trimming, baseline correction, tapering and filtering), otherwise skip this step
   gm = gm.trim_by_energy((0.001, 0.999)).taper(0.05).butterworth_filter((0.05, 100.0))
   # An alterantive to triming gm for just model inversion is to use fit_range in the inverter.fit() method to specify the time range for fitting (e.g., fit_range=(0.01, 0.99) for 1% to 99% cumulative energy)

Step 2: Perform Model Inversion
--------------------------------

Use the ``ModelInverter`` class to fit a stochastic model to the target ground motion.
Use ``Functions`` to define the functional forms for the model parameters (amplitude, frequency, and damping evolution with time).

.. code-block:: python

   # %% Specify the stochastic model functional forms
   q = Functions.BetaCentroidSpread()  # Modulating function
   wu = Functions.Constant()           # Upper frequency
   zu = Functions.Constant()           # Upper damping
   wl = Functions.Constant()           # Lower frequency
   zl = Functions.Constant()           # Lower damping

   inverter = ModelInverter(ground_motion=gm, modulating=q,
                            upper_frequency=wu, upper_damping=zu,
                            lower_frequency=wl, lower_damping=zl)

   # Optional: Provide initial guesses and fix parameters using tight bounds
   # Here we fix upper_damping to 0.707 and lower_damping to 1.0 by making min/max bounds equal
   ig = {
       "upper_frequency": [10.0],
       "upper_damping": [0.707],
       "lower_frequency": [0.3],
       "lower_damping": [1.0],
   }
   
   bs = {
       "upper_frequency": [(0.1, 40.0)],
       "upper_damping": [(0.707, 0.707)], # Fixed parameter
       "lower_frequency": [(0.01, 0.99)],
       "lower_damping": [(1.0, 1.0)],     # Fixed parameter
   }

   # Fit to the target ground motion and measure execution time
   start = time.time()
   model = inverter.fit(initial_guess=ig, bounds=bs, fit_range=(0.01, 0.99), mode='stepwise')  # stepwide mode is faster and lead to unique solution, while joint mode is more robust but slower and may lead to multiple local minima. Adjust fit_range as needed to focus on specific time windows of the ground motion.
   end = time.time()

   print(f"Execution time: {end - start:.2f} seconds")

   # Print a summary of the fitted parameters
   model.summary()

Step 3: Generate Simulations
-----------------------------

Generate synthetic ground motions using the fitted stochastic model parameters.
The resulting simulations are returned as a ``GroundMotion`` object.

.. seealso::
   For more details on working with ground motions, refer to :ref:`example_basic_groundmotion`.

.. code-block:: python

   # %% Generate 20 simulated ground motions
   sm = model.simulate(n=20)

   # %% View available IMs
   sm.list_IMs()

Step 4: Visualize the Results
-----------------------------

You can use standard libraries like ``matplotlib`` to visualize the model, simulations, and the target.

.. tip::
   See :ref:`example_basic_groundmotion` for plotting response spectra and other intensity measures.


.. code-block:: python

   # 1. Compare Acceleration Time Series
   fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'hspace': 0.15})

   # Target Ground Motion
   axes[0].plot(gm.t, gm.ac, color='black', lw=1, label='Target')
   axes[0].plot(model.t, model.q, color='red', ls='--', lw=1.2, label='Model Envelope')
   axes[0].plot(model.t, -model.q, color='red', ls='--', lw=1.2)
   axes[0].set_ylabel('Acceleration (g)')
   axes[0].legend(frameon=False)
   axes[0].set_title('Target vs Simulation - Acceleration')

   # First Simulation
   axes[1].plot(sm.t, sm.ac[0], color='tab:blue', lw=1, label='Simulation #1')
   axes[1].set_ylabel('Acceleration (g)')
   axes[1].set_xlabel('Time (s)')
   axes[1].legend(frameon=False)

   plt.show()

   # 2. Compare Displacement Time Series
   fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'hspace': 0.15})

   # Target Ground Motion
   axes[0].plot(gm.t, gm.disp, color='black', lw=1, label='Target')
   axes[0].set_ylabel('Displacement (cm)')
   axes[0].legend(frameon=False)
   axes[0].set_title('Target vs Simulation - Displacement')

   # First Simulation
   axes[1].plot(sm.t, sm.disp[0], color='tab:blue', lw=1, label='Simulation #1')
   axes[1].set_ylabel('Displacement (cm)')
   axes[1].set_xlabel('Time (s)')
   axes[1].legend(frameon=False)

   plt.show()

   # 3. Compare Fourier Amplitude Spectra (FAS)
   plt.figure(figsize=(7, 4))
   plt.loglog(sm.freq, sm.fas.T, color='gray', alpha=0.3, lw=0.7)  # 20 FAS simulations
   plt.loglog(sm.freq, np.mean(sm.fas, axis=0), color='red', ls='--', lw=2, label='Simulation Mean')
   plt.loglog(gm.freq, gm.fas, color='black', lw=1.5, label='Target')
   plt.loglog(model.freq / (2 * np.pi), model.fas, color='Blue', lw=1.5, label='Model')
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('Fourier Amplitude')
   plt.legend(frameon=False)
   plt.grid(True, which="both", ls=":", alpha=0.3)
   plt.title('Fourier Amplitude Spectra')
   plt.ylim(1e-4, 1e1)  # if necessary, adjust y-axis limits for better visualization
   plt.show()

   # 4. Compare Response Spectra (5% Damping)
   tp = np.arange(0.0, 10.0, 0.1)
   _, _, sa = gm.response_spectra(tp)
   _, _, sm_sa = sm.response_spectra(tp)

   plt.figure(figsize=(7, 4))
   plt.loglog(tp, sm_sa.T, color='gray', alpha=0.3, lw=0.7)
   plt.loglog(tp, np.mean(sm_sa, axis=0), color='red', ls='--', lw=2, label='Simulation Mean')
   plt.loglog(tp, sa, color='black', lw=1.5, label='Target')
   plt.xlabel('Period (s)')
   plt.ylabel('Spectral Acceleration (g)')
   plt.legend(frameon=False)
   plt.grid(True, which="both", ls=":", alpha=0.3)
   plt.title('Response Spectra (5% Damping)')
   plt.show()


Step 6: Export Simulations
---------------------------
.. tip::
   See :ref:`example_basic_groundmotion` for saving and plotting results.
   The simulated ground motions are ``GroundMotion`` objects and can be saved or processed similarly.


