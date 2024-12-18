# SGSIM
<p align="justify">
<strong>SGSIM</strong> is a Python package for simulating target earthquake ground motions using a site-based stochastic model [1]. It derives model parameters that implicitly account for the earthquake and site characteristics of the target ground motion. Using these parameters, the package simulates ground motions for the specific earthquake scenario, accounting for their aleatoric variability. It also provides tools for visualizing the simulation results..
</p>

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)
- [Support the Project](#support-the-project)
- [References](#references)

## Features
- **Site-based Stochastic Modeling**: Configure and fit the stochastic model to ground motion data with customizable parameters.  
- **Simulation**: Simulate acceleration, velocity, and displacement time series without post-processing. Additional properties such as Fourier Amplitude Spectrum (FAS), Spectral Acceleration (SA), Spectral Velocity (SV), Spectral Displacement (SD), and more are available for rapid computation. The results can be saved in csv files.
- **Visualization**: Plot ground motion data, response and Fourier spectra, and other properties to verify and validate simulations.

## Installation
To install SGSIM via `pip`, run:
```bash
pip install SGSIM
```

Or install from source:
```bash
git clone https://github.com/Sajad-Hussaini/SGSIM.git
cd SGSIM
pip install .
```

## Usage
For a detailed guide on using SGSIM, refer to the `how_to_run_model_with_example.py` file. This example provides a step-by-step walkthrough for simulating a recorded accelerogram with SGSIM.  
To run the example, import the script as shown below (to view the module, go to its definition).  
`from SGSIM.examples import how_to_run_model_with_example`  
More examples and instructions will be available soon.  

## License
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See the LICENSE file for details.

## Contact
**S.M. Sajad Hussaini**  
Email: [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)

## Support This Python Package

Your contributions help maintain and improve this package:

- **PayPal Email:** [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)
- **Donate Here:** [paypal.me/sajadhussaini](https://www.paypal.com/paypalme/sajadhussaini)


## References
*Broadband stochastic simulation of earthquake ground motions with multiple strong phases.*  
*DOI: To be assigned upon publication*
