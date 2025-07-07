# SGSIM - Stochastic Ground Motion Simulation Model

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/merm.svg)](https://pypi.org/project/sgsim/)

**SGSIM** is a Python package for simulating earthquake ground motions using a site-compatible stochastic model [1]. It derives model parameters that implicitly account for the earthquake and site characteristics of the target ground motion. Using these parameters, the package simulates ground motions for the specific earthquake scenario, accounting for their aleatoric variability. It also provides tools for visualizing the simulation results.  

> 💡 **Tip**: To ensure compatibility with the user guide, it's recommended to use the latest **release** available on **GitHub**, **PyPI**, or **Zenodo**.

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [User Guide](#User-Guide)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Features

✅ **Site-based Stochastic Modeling**: Configure and fit the stochastic model to ground motion data with customizable parameters.  
✅ **Flexibility**: Various types of ground motions can be simulated rapidly.
✅ **Simulation**: Simulate ground motion time series, Fourier and response spectra, and other properties rapidly. The results can be saved in csv files.
✅ **Visualization**: Plot ground motion data, Fourier and response spectra, and other properties to verify and validate simulations.

## Installation

### Install from PyPI (Recommended)
```bash
pip install sgsim
```
### Install from Source
```bash
git clone https://github.com/Sajad-Hussaini/sgsim.git
cd sgsim
pip install .
```

## User Guide

For a step-by-step walkthrough on using **SGSIM**, refer to the [Quick Start with SGSIM](user_guide.ipynb).

> 📚 **Note**: The User Guide will be updated with more detailed instructions.

## License

SGSIM is released under the [MIT License](https://opensource.org/licenses/MIT).  
See the [LICENSE](LICENSE) file for the full text.

## Contact

For questions or assistance, please contact:

**S.M. Sajad Hussaini**  
📧 [hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)

> Please include "SGSIM" in the subject line for faster response.

### Support the Project

If you find this package useful, contributions to help maintain and improve it are always appreciated.

[![PayPal](https://img.shields.io/badge/PayPal-Donate-blue.svg)](https://www.paypal.com/paypalme/sajadhussaini)

## References

Please cite the following references for any formal study:  

**[1] Primary Reference**  
*BROADBAND STOCHASTIC SIMULATION OF EARTHQUAKE GROUND MOTIONS WITH MULTIPLE STRONG PHASES WITH AN APPLICATION TO THE 2023 KAHRAMANMARAS, TURKEY (TÜRKIYE), EARTHQUAKE*  
*DOI: https://doi.org/10.1177/87552930251331981 (journal of Earthquake Spectra)

**[2] SGSIM Package** 
*SGSIM: Stochastic Ground Motion Simulation Model*
*DOI: https://doi.org/10.5281/zenodo.14565922*
