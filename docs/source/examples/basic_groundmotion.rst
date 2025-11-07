.. _example_basic_simulation:

Quick Start: Basic Ground Motion Analysis Example
====================================================

Loading a Ground Motion
-------------------------

.. code-block:: python

   from sgsim import GroundMotion

   # 1. from a record file
   # Path to your accelerogram file
   file = 'path/to/your/motion.AT2'
   source = 'nga'

   # Create a GroundMotion (gm) instance by loading a record file
   gm = GroundMotion.load_from(source=source, file=file)

   # 2. from array data assuming you have time step dt (float) and ac array (numpy array) 
   import numpy as np
   gm = GroundMotion.load_from(source='array', dt=dt, ac=ac)

   # If necessary, trim to 0.1% - 99.9% cumulative energy
   # If necessary, apply a bandpass filter (e.g., 0.1–25 Hz)
   gm.trim("energy", (0.001, 0.999)).filter((0.1, 25.0))

   # 3. Obtain avilable Intensity Measures (IMs)
   gm.available_IMs()  # List available IMs (more IM will be added gradually or by user)
   # Example of accessing some IMs:
   gm.pga

   gm.tp = (0.1, 10.1, 0.1)  # Set periods for response spectra or set to an array value
   gm.sa

   gm.fas
   # plotting FAS note that gm.freq is in rad/s
   import matplotlib.pyplot as plt
   plt.loglog(gm.freq / (2*np.pi), gm.fas)
