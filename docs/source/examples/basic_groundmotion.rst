.. _example_basic_groundmotion:

Quick Start: Basic Ground Motion Analysis Example
====================================================

This guide walks you through the basics of loading, processing, and analyzing ground motion data using ``sgsim``.

Step 1: Import Libraries
------------------------

First, import the ``GroundMotion`` class and ``matplotlib`` for plotting.

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sgsim import GroundMotion

Step 2: Load a Ground Motion
----------------------------

You can create a ``GroundMotion`` instance in two primary ways: loading from a file or creating one from existing data arrays in memory.

Option A: Load from a Record File (e.g., PEER NGA)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Path to your accelerogram file (e.g., PEER NGA .AT2 format)
   file_path = 'path/to/your/motion.AT2'

   # Create a GroundMotion instance
   gm = GroundMotion.load_from(source='nga', file=file_path)

Option B: Load from Memory (NumPy Arrays)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

   # Example dummy data (time step and acceleration array)
   dt = 0.01  # Time step in seconds
   ac = np.random.normal(0, 0.1, 1000)  # Simulated acceleration array

   # Create instance from array
   gm = GroundMotion.load_from(source='array', dt=dt, ac=ac)

Step 3: Process the Signal
--------------------------

Once loaded, you can chain methods to process the ground motion. For example, trim the record to significant duration (5-95% cumulative energy) and apply a bandpass filter.

.. code-block:: python

   # Trim to 0.1% - 99.9% cumulative energy and apply bandpass filter (0.01–100.0 Hz)
   gm = gm.trim_by_energy((0.001, 0.999)).butterworth_filter((0.01, 100.0))

Step 4: Access Intensity Measures (IMs)
---------------------------------------

You can easily access scalar Intensity Measures like PGA (Peak Ground Acceleration).

.. code-block:: python

   # List all available Intensity Measures
   gm.list_IMs()

   # Access specific values
   print(f"PGA: {gm.pga} g")
   print(f"PGV: {gm.pgv} cm/s")

Step 5: Compute and Plot Response Spectra
-----------------------------------------

Compute the spectral displacement (SD), velocity (SV), and acceleration (SA) for a range of periods.

.. code-block:: python

   # Define periods of interest
   periods = np.arange(0.1, 10.1, 0.1)

   # Calculate response spectra
   sd, sv, sa = gm.response_spectra(periods=periods)

   # Plot Spectral Acceleration (SA)
   plt.figure()
   plt.loglog(periods, sa)
   plt.title("Response Spectrum")
   plt.xlabel("Period (s)")
   plt.ylabel("Spectral Acceleration (g)")
   plt.show()

   # Plot Fourier Amplitude Spectrum (FAS)
   # Note: gm.freq is frequency (Hz)
   plt.figure()
   plt.loglog(gm.freq, gm.fas)
   plt.title("Fourier Amplitude Spectrum (g.s)")
   plt.xlabel("Frequency (Hz)")
   plt.ylabel("Amplitude")
   plt.show()

Step 6: Export Results
----------------------

Finally, save the processed ground motion IMs and spectra to a CSV file.

.. code-block:: python

   # Export to CSV including specific IMs and spectral ordinates
   gm.to_csv('output_ground_motion.csv', ims=['pga', 'pgv', 'sa', 'fas'], periods=tp)  # Or any other period range
