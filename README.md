# SGSIM

**SGSIM** is a Python package for simulating, fitting, and visualizing stochastic ground motion models. It’s designed for researchers, engineers, and anyone working with seismic data who needs robust tools for generating and analyzing ground motion time-series.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Features
- **Stochastic Modeling**: Configure and fit models to ground motion data using customizable parameters.
- **Time-Series Simulation**: Simulate acceleration, velocity, and displacement time-series.
- **Comprehensive Visualization**: Plot motion data, spectra, cross-energy, and other critical properties.

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
For a detailed walkthrough of how to use SGSIM, check out the `how_to_run_model_with_example.py` file. This example illustrates step-by-step procedures for using the SGSIM package to simualate a recorded accelerogram.  
The user can access the script from below and specify an accelergoram file path and its format (an 'nga' for NGA or 'esm' for ESM source format).  
`from SGSIM.examples import how_to_run_model_with_example`  


## License
This project is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. See the LICENSE file for details.

## Contact
**S.M. Sajad Hussaini**  
Email: [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)

## References
Broadband stochastic simulation of earthquake ground motions with multiple strong phases. DOI: [10.5281/zenodo.13939496](https://doi.org/10.5281/zenodo.13939496)
