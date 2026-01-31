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
   from sgsim import GroundMotion, ModelInverter, Functions

   # %% Prepare the target ground motion
   # Change the path to the file
   gm = GroundMotion.load_from(source='nga', file='RSN123_Example.AT2')
   # If necessary , preprocess the ground motion (e.g., trimming, baseline correction, tapering and filtering)
   gm = gm.trim_by_energy((0.001, 0.999)).taper(0.05).butterworth_filter((0.05, 100.0))

Step 2: Perform Model Inversion
--------------------------------

Use the ``ModelInverter`` class to fit a stochastic model to the target ground motion.
Use ``Functions`` to define the functional forms for the model parameters (amplitude, frequency, and damping evolution with time).

.. code-block:: python

   # %% Specify the stochastic model functional forms
   q = Functions.BetaBasic()  # Modulating function as BetaBasic
   wu = Functions.Linear()    # Upper frequency as Linear
   zu = Functions.Constant()  # Upper damping as Constant
   wl = Functions.Constant()  # Lower frequency as Constant
   zl = Functions.Constant()  # Lower damping as Constant
   inverter = ModelInverter(ground_motion=gm, modulating=q,
                           upper_frequency=wu, upper_damping=zu,
                           lower_frequency=wl, lower_damping=zl,)

   # Fit to the target ground motion with default settings and obtain the fitted model
   model= inverter.fit()

   # Print a summary of the fitted parameters
   model.summary()

Step 3: Generate Simulations
-----------------------------

Generate synthetic ground motions using the fitted stochastic model parameters.
The resulting simulations are returned as a ``GroundMotion`` object.

.. seealso::
   For more details on working with ground motions, refer to :ref:`example_basic_groundmotion`.

.. code-block:: python

   # %% Generate 10 simulated ground motions
   sm = model.simulate(n=10)

   # %% View available IMs
   sm.list_IMs()

Step 4: Visualize the Results
-----------------------------

You can use standard libraries like ``matplotlib`` to visualize the model, simulations, and the target.

.. tip::
   See :ref:`example_basic_groundmotion` for plotting response spectra and other intensity measures.


.. code-block:: python

   # 1. Compare Time Series (Target vs. First Simulation)
   fig, axes = plt.subplots(2, 1, sharex=True, figsize=(10, 6), gridspec_kw={'hspace': 0.15})

   # Target Ground Motion
   axes[0].plot(gm.t, gm.ac, color='black', lw=1, label='Target')
   axes[0].plot(model.t, model.q, color='red', ls='--', lw=1.2, label='Model Envelope')
   axes[0].plot(model.t, -model.q, color='red', ls='--', lw=1.2)
   axes[0].set_ylabel('Acceleration (g)')
   axes[0].legend(frameon=False)
   axes[0].set_title('Target Ground Motion')

   # First Simulation
   axes[1].plot(sm.t, sm.ac[0], color='tab:blue', lw=1, label='Simulation #1')
   axes[1].set_ylabel('Acceleration (g)')
   axes[1].set_xlabel('Time (s)')
   axes[1].legend(frameon=False)
   axes[1].set_title('Synthetic Ground Motion')

   plt.tight_layout()
   plt.show()

   # 2. Compare Fourier Amplitude Spectra (FAS)
   plt.figure(figsize=(7, 4))
   plt.loglog(sm.freq, sm.fas.T, color='gray', alpha=0.3, lw=0.7)  # 10 FAS simulations
   plt.loglog(sm.freq, np.mean(sm.fas, axis=0), color='red', ls='--', lw=2, label='Simulation Mean')
   plt.loglog(gm.freq, gm.fas, color='black', lw=1.5, label='Target')
   plt.loglog(model.freq / (2 * np.pi), model.fas, color='Blue', lw=1.5, label='Model')
   plt.xlabel('Frequency (Hz)')
   plt.ylabel('Fourier Amplitude')
   plt.legend(frameon=False)
   plt.grid(True, which="both", ls=":", alpha=0.3)
   plt.title('Fourier Amplitude Spectra')
   plt.tight_layout()
   plt.show()

   # 3. Compare Response Spectra (5% Damping)
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
   plt.tight_layout()
   plt.show()


Step 6: Export Simulations
---------------------------
.. tip::
   See :ref:`example_basic_groundmotion` for saving and plotting results.
   The simulated ground motions are ``GroundMotion`` objects and can be saved or processed similarly.


