.. _example_basic_groundmotion:

Quick Start: Basic Ground Motion Analysis Example
====================================================

This guide walks you through the basics of loading, processing, and analyzing ground motion data using ``sgsim``.

Step 1: Import Libraries
------------------------

First, import the ``GroundMotion`` class and ``matplotlib`` for plotting.

.. code-block:: python
   
   # %% Import necessary libraries
   import numpy as np
   import matplotlib.pyplot as plt
   from sgsim import GroundMotion

Step 2: Load a Ground Motion
----------------------------

You can create a ``GroundMotion`` instance in two primary ways: loading from a file or creating one from existing data arrays in memory.

Option A: Load from a Record File (e.g., PEER NGA, ESM, etc.)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # %% Loading from a file
   # Example A: Loading an NGA file (.AT2)
   # Change the path to the downloaded file
   gm = GroundMotion.load_from(source='nga', file='RSN123_Example.AT2')

   # Example B: Loading an ESM file (.ASC)
   # gm = GroundMotion.load_from(source='esm', file='IV.ABC.HNE.ASC')

Option B: Load from Memory (NumPy Arrays)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # %% Create dummy data for demonstration (time step and acceleration array)
   dt = 0.01  # Time step in seconds
   ac = np.random.normal(0, 0.1, 1000)  # Simulated acceleration array

   # Create instance from array
   gm = GroundMotion.load_from(source='array', dt=dt, ac=ac)

Step 3: Process the Signal
--------------------------

Once loaded, you can chain methods to process the ground motion.
Processing is often essential to remove noise and integration errors.

The code below performs the following standard workflow:

1.  **Trimming:** Isolates the significant shaking (5% to 95% of energy).
2.  **Baseline Correction:** Removes low-frequency drift.
3.  **Tapering:** Smooths the ends to zero to prevent spectral leakage.
4.  **Filtering:** Bandpass filter (0.1 Hz - 25 Hz) to remove noise or unwanted frequencies.

.. code-block:: python

   # %% Process the ground motion
   gm_trimmed = gm.trim_by_energy(energy_range=(0.05, 0.95))
   gm_corrected = gm_trimmed.baseline_correction(degree=1)
   gm_tapered = gm_corrected.taper(alpha=0.05)
   gm_processed = gm_tapered.butterworth_filter(bandpass_freqs=(0.1, 25.0), order=4)

   # Alternatively, chain all processing steps in one line
   # gm_processed = (gm.trim_by_energy(energy_range=(0.05, 0.95))
   #                  .baseline_correction(degree=1)
   #                  .taper(alpha=0.05)
   #                  .butterworth_filter(bandpass_freqs=(0.1, 25.0), order=4))  

Step 4: Access Intensity Measures (IMs)
---------------------------------------

You can easily access scalar Intensity Measures like PGA (Peak Ground Acceleration).

.. code-block:: python

   # %% List all available Intensity Measures
   gm_processed.list_IMs()

   # %% Access specific IMs
   print(f"PGA: {gm_processed.pga} g")
   print(f"PGV: {gm_processed.pgv} cm/s")

Step 5: Plot Time Series, Response Spectra, FAS
------------------------------------------------

You can compute spectral displacement (SD), velocity (SV), and acceleration (SA) for any range of periods.

.. code-block:: python

   # %% Plot time Series of processed ground motion
   fig, ax = plt.subplots(3, 1, sharex=True, figsize=(5, 4))
   # Acceleration
   ax[0].plot(gm_processed.t, gm_processed.ac, color='black')
   ax[0].set_ylabel('Acceleration (g)')
   ax[0].set_title('Processed Time Series')

   # Velocity
   ax[1].plot(gm_processed.t, gm_processed.vel, color='blue')
   ax[1].set_ylabel('Velocity (cm/s)')

   # Displacement
   ax[2].plot(gm_processed.t, gm_processed.disp, color='red')
   ax[2].set_ylabel('Displacement (cm)')
   ax[2].set_xlabel('Time (s)')

   plt.tight_layout()
   plt.show()

   # %% Calculate response spectra for a range of periods
   tp = np.arange(0.1, 10.1, 0.1)
   sd, sv, sa = gm_processed.response_spectra(periods=tp, damping=0.05)

   # %% Plot Spectral Acceleration (SA)
   plt.figure()
   plt.loglog(tp, sa)
   plt.title("Response Spectrum")
   plt.xlabel("Period (s)")
   plt.ylabel("Spectral Acceleration (g)")
   plt.show()

   # %% Plot Fourier Amplitude Spectrum (FAS)
   # Note: gm_processed.freq is frequency (Hz)
   plt.figure()
   plt.loglog(gm_processed.freq, gm_processed.fas)
   plt.title("Fourier Amplitude Spectrum (g.s)")
   plt.xlabel("Frequency (Hz)")
   plt.ylabel("Amplitude")
   plt.show()

Step 6: Export Results
----------------------

Finally, You can save any ground motion attributes (e.g., any IM, SA, FAS, etc.) to CSV file if spreadsheet analysis is needed.
You can also use numpy methods to simply save any attributes to txt or npy files.

.. code-block:: python

   # %% Export to CSV including specific IMs and arbitrary spectral ordinates
   gm_processed.to_csv('output_gm.csv', ims=['pga', 'pgv', 'cav', 'sa'], periods=tp)   

   # %% Save time and acceleration arrays to a text file using numpy
   output_path = "output_gm.txt"

   data = np.column_stack((gm_processed.t, gm_processed.ac))
   np.savetxt(output_path, data, fmt="%.6e", header="Time Acceleration")

