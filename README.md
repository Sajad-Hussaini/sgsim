# SGSIM
<p align="justify">
<strong>SGSIM</strong> is a Python package for simulating earthquake ground motions based on specific earthquake and site characteristics.
This package enables users to apply the site-based stochastic simulation model (as detailed in the reference section) to a target ground motion time series and provides tools for visualizing the simulation results. Designed for researchers, engineers, and others working with seismic data, SGSIM offers an efficient and user-friendly approach to generating ground motion time series at specific recording stations. It also allows for the generation of multiple realizations of a target record to account for variability, without relying on the selection and scaling of existing ground motions.
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

## Support the Project

If you find this project useful and would like to support my work, you can make a donation via PayPal:

- **PayPal Email:** [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)
- **Donate Here:** [paypal.me/sajadhussaini](https://www.paypal.com/paypalme/sajadhussaini)

Thank you for your support!


## References
*Broadband stochastic simulation of earthquake ground motions with multiple strong phases.*  
*DOI: To be assigned upon publication*
