from functools import cached_property
import numpy as np
from . import signal_analysis
from . import signal_processing
from ..file_reading.record_reader import RecordReader
from ..core.domain_config import DomainConfig

class Motion(DomainConfig):
    """
    This class describes ground motions in terms of various proprties (e.g., spectra, peak motions, characteristics)
    """
    _CORE_ATTRS = frozenset({'npts', 'dt', 'ac', 'vel', 'disp'})

    def __init__(self, npts, dt, ac, vel, disp):
        """
        npts:  number of data points (array length)
        dt:    time step
        t:     time array
        ac:    acceleration array
        vel:   velocity array
        disp:  displacement array
        """
        super().__init__(npts, dt)
        self.ac = ac
        self.vel = vel
        self.disp = disp
    
    def clear_cache(self):
        """Clear cached properties, preserving core attributes."""
        core_values = {attr: getattr(self, attr) for attr in self._CORE_ATTRS}
        self.__dict__.clear()
        self.__dict__.update(core_values)

    def trim(self, option: str, value: tuple[float, float] | slice | int):
        """
        Trim the ground motion data using specified criteria.
        
        option (str): Trimming method:
            - 'energy': Trim based on cumulative energy range (e.g., (0.001, 0.999))
            - 'npts': Keep specified number of points from beginning
            - 'slice': Apply a custom slice directly to the motion arrays
        value: Parameters for the chosen option:
            - For 'energy': tuple[float, float] as (start_fraction, end_fraction)
            - For 'npts': int representing number of points to keep
            - For 'slice': slice object to apply directly
        
        Example:
            >>> motion.trim('energy', (0.05, 0.95))  # Keep middle 90% of energy
            >>> motion.trim('npts', 1000)            # Keep first 1000 points
            >>> motion.trim('slice', slice(100, 500)) # Custom slice
        """
        if option.lower() == 'energy':
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("Energy trimming requires a tuple of (start_fraction, end_fraction)")
            self.energy_slice = value
            slicer = self.energy_slice

        elif option.lower() == 'npts':
            if not isinstance(value, int) or value <= 0 or value > self.npts:
                raise ValueError("Number of points must be a positive integer less than the current number of points")
            slicer = slice(0, value)
        
        elif option.lower() == 'slice':
            if not isinstance(value, slice):
                raise ValueError("Slice option requires a Python slice object")
            slicer = value
        
        else:
            raise ValueError(f"Unsupported trim option: '{option}'. Use 'energy', 'npts', or 'slice'")
        self.ac = self.ac[slicer]
        self.vel = self.vel[slicer]
        self.disp = self.disp[slicer]
        self.npts = len(self.ac)
        self.clear_cache()
        return self
    
    def filter(self, bandpass_freqs: tuple[float, float]):
        """
        Perform a bandpass filtering using bandpass freqs as (lowcut, highcut) in Hz
        """
        self.ac = signal_processing.bandpass_filter(self.dt, self.ac, bandpass_freqs[0], bandpass_freqs[1])
        self.vel = signal_analysis.get_integral(self.dt, self.ac)
        self.disp = signal_analysis.get_integral(self.dt, self.vel)
        self.clear_cache()
        return self
    
    def resample(self, dt: float):
        """
        resample the motion data to a new time step.

        Args:
            dt (float): The new time step.
        """
        self.npts, self.dt, self.ac = signal_processing.resample(self.dt, dt, self.ac)
        self.vel = signal_analysis.get_integral(dt, self.ac)
        self.disp = signal_analysis.get_integral(dt, self.vel)
        self.clear_cache()
        return self

    def save_simulations(self, filename: str, x_var: str, y_vars: list[str]):
        """
        To save any related simulation data to a CSV file.

        Args:
            filename (str): Output file name.
            x_var (str): Independent variable (e.g., 'tp', 'freq', 't').
            y_vars list[str]: Dependent variables (e.g., ['sa', 'sv', 'sd']).
        """
        x_data = getattr(self, x_var.lower())
        y_data = [getattr(self, var.lower()).T for var in y_vars]
        data = np.column_stack((x_data, *y_data))
        n = y_data[0].shape[1] if y_data else 0
        header = x_var + "," + ",".join([f"{var}{i+1}" for var in y_vars for i in range(n)])
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
        return self
    
    @cached_property
    def fas(self):
        return signal_analysis.get_fas(self.npts, self.ac)

    @cached_property
    def fas_smooth(self):
        return signal_processing.moving_average(self.fas, 9)[..., self.freq_slice]

    @cached_property
    def ce(self):
        return signal_analysis.get_ce(self.dt, self.ac)
    
    @cached_property
    def mle_ac(self):
        return signal_analysis.get_mle(self.ac)

    @cached_property
    def mle_vel(self):
        return signal_analysis.get_mle(self.vel)

    @cached_property
    def mle_disp(self):
        return signal_analysis.get_mle(self.disp)

    @cached_property
    def mzc_ac(self):
        return signal_analysis.get_mzc(self.ac)

    @cached_property
    def mzc_vel(self):
        return signal_analysis.get_mzc(self.vel)

    @cached_property
    def mzc_disp(self):
        return signal_analysis.get_mzc(self.disp)

    @cached_property
    def pmnm_ac(self):
        return signal_analysis.get_pmnm(self.ac)

    @cached_property
    def pmnm_vel(self):
        return signal_analysis.get_pmnm(self.vel)

    @cached_property
    def pmnm_disp(self):
        return signal_analysis.get_pmnm(self.disp)

    @cached_property
    def spectra(self):
        return signal_analysis.get_spectra(self.dt, self.ac if self.ac.ndim == 2 else self.ac[None, :], period=self.tp, zeta=0.05)

    @property
    def sa(self):
        return self.spectra[2]
    
    @property
    def sv(self):
        return self.spectra[1]
    
    @property
    def sd(self):
        return self.spectra[0]

    @cached_property
    def pga(self):
        return signal_analysis.get_pgp(self.ac)

    @cached_property
    def pgv(self):
        return signal_analysis.get_pgp(self.vel)

    @cached_property
    def pgd(self):
        return signal_analysis.get_pgp(self.disp)

    @property
    def energy_slice(self):
        if not hasattr(self, '_energy_slice'):
            self._energy_slice = signal_analysis.slice_energy(self.dt, self.ac, (0.001, 0.999))  # Default range
        return self._energy_slice

    @energy_slice.setter
    def energy_slice(self, energy_range: tuple[float, float]):
        self._energy_slice = signal_analysis.slice_energy(self.dt, self.ac, energy_range)

    @classmethod
    def from_file(cls, file_path: str | tuple[str, str], source: str, **kwargs):
        """
        Construct a motion class from an accelergoram recording file.

        file_path: path to the file or the filename in a zip
        source:    source type (e.g., 'NGA')
        kwargs:    additional keyword arguments for RecordReader (e.g., 'skiprows')
        """
        record = RecordReader(file_path, source, **kwargs)
        return cls(npts=record.npts, dt=record.dt, ac=record.ac, vel=record.vel, disp=record.disp)

    @classmethod
    def from_model(cls, model):
        """ Construct a motion class from a calibrated stochastic model """
        return cls(npts=model.npts, dt=model.dt, ac=model.ac, vel=model.vel, disp=model.disp)