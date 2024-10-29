# SGSIM
<p align="justify">
<strong>SGSIM</strong> is a Python package for simulating a target earthquake ground motion time series. This package enables users to apply the site-based stochastic simulation model (as noted in the reference section) to a target ground motion time series and obtain model parameters that implicitly account for the earthquake and site characteristics. Given the model parameters, the package can simulate multiple realizations of the target motion to account for the aleatoric nature of the ground motion. It also provides tools for visualizing the simulation results.
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
- **Site-based Stochastic Modeling**: Configure and fit the stochastic model to ground motion data using customizable parameters.  
- **Time-Series Simulation**: Simulate acceleration, velocity, and displacement time series without post-processing. Additional properties such as Fourier Amplitude Spectrum (FAS), Spectral Acceleration (SA), Spectral Velocity (SV), Spectral Displacement (SD), and more are available for rapid computation.
- **Comprehensive Visualization**: Plot motion data, spectra, and other properties to verify and validate simulations.

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
To run the example, import the script as shown below (Use go to the definition to see the script). You can specify an accelerogram file path and format ('nga' for NGA or 'esm' for ESM source formats).  
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
