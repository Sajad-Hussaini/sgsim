# SGSIM
<p align="justify">
<strong>SGSIM</strong> is a Python package for simulating target earthquake ground motions using a site-based stochastic model [1]. It derives model parameters that implicitly account for the earthquake and site characteristics of the target ground motion. Using these parameters, the package simulates ground motions for the specific earthquake scenario, accounting for their aleatoric variability. It also provides tools for visualizing the simulation results..
</p>

## Table of Contents
- [Features](#features)
- [Installation](#installation)
- [User Guide](#User-Guide)
- [License](#license)
- [Contact](#contact)
- [References](#references)

## Features
- **Site-based Stochastic Modeling**: Configure and fit the stochastic model to ground motion data with customizable parameters.  
- **Simulation**: Simulate ground motion time series, Fourier and response spectra, and other properties rapidly. The results can be saved in csv files.
- **Visualization**: Plot ground motion data, Fourier and response spectra, and other properties to verify and validate simulations.

## Installation
To install **SGSIM** from source:
```bash
git clone https://github.com/Sajad-Hussaini/SGSIM.git
cd SGSIM
pip install .
```
or install via `pip` (will be added), run:
```bash
pip install SGSIM
```

## User Guide
For a step-by-step walkthrough on using **SGSIM**, refer to the [Quick Start with SGSIM](user_guide.ipynb). The User Guide will be updated for more instructions.

## License
SGSIM is released under the [GNU AGPL 3.0](https://www.gnu.org/licenses/agpl-3.0.en.html).
See the [License](License) for the full text.

## Contact
S.M. Sajad Hussaini  
[hussaini.smsajad@gmail.com](mailto:hussaini.smsajad@gmail.com)

If you find this package useful, contributions to help maintain and improve it are always appreciated. You can donate via [PayPal](https://www.paypal.com/paypalme/sajadhussaini)

## References
Please cite the following references for any formal study:  

*[1] Broadband Stochastic Simulation of Earthquake Ground Motions with Multiple Strong Phases with An Application to the 2023 Kahramanmaraş Türkiye Earthquake*  
*DOI: To be assigned upon publication*

*[2] SGSIM Code*  
*DOI: https://doi.org/10.5281/zenodo.14565922*
