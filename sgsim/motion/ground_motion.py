from functools import cached_property
import numpy as np
from . import signal_tools
from ..file_reading.record_reader import RecordReader
from ..core.domain_config import DomainConfig

class GroundMotion(DomainConfig):
    """
    A class for handling ground motion data.

        Parameters
        ----------
        npts : int
            Number of data points (array length).
        dt : float
            Time step.
        ac : ndarray
            Acceleration array.
        vel : ndarray
            Velocity array.
        disp : ndarray
            Displacement array.
        """
    _CORE_ATTRS = DomainConfig._CORE_ATTRS | frozenset({'ac', 'vel', 'disp', 'tag'})

    def __init__(self, npts, dt, ac, vel, disp, tag=None):
        super().__init__(npts, dt)
        self.ac = ac
        self.vel = vel
        self.disp = disp
        self.tag = tag

    def trim(self, method: str, value: tuple[float, float] | int | slice):
        """
        Trim the ground motion data using the specified method.

        Parameters
        ----------
        method : {'energy', 'npts', 'slice'}
            Trimming method:
            - 'energy': Trim based on cumulative energy range (e.g., (0.001, 0.999)).
            - 'npts': Keep specified number of points from beginning.
            - 'slice': Apply a custom slice directly to the motion arrays.
        value : tuple of float, int, or slice
            Parameters for the chosen method:
            - For 'energy': tuple of (start_fraction, end_fraction).
            - For 'npts': int, number of points to keep.
            - For 'slice': slice object to apply directly.

        Returns
        -------
        self : GroundMotion
            The trimmed GroundMotion instance.

        Examples
        --------
        >>> motion.trim('energy', (0.05, 0.95))  # Keep middle 90% of energy
        >>> motion.trim('npts', 1000)            # Keep first 1000 points
        >>> motion.trim('slice', slice(100, 500)) # Custom slice
        """
        if method.lower() == 'energy':
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("Energy trimming requires a tuple of (start_fraction, end_fraction)")
            self.energy_slice = value
            slicer = self.energy_slice

        elif method.lower() == 'npts':
            if not isinstance(value, int) or value <= 0 or value > self.npts:
                raise ValueError("Number of points must be a positive integer less than the current number of points")
            slicer = slice(0, value)
        
        elif method.lower() == 'slice':
            if not isinstance(value, slice):
                raise ValueError("Slice method requires a Python slice object")
            slicer = value
        
        else:
            raise ValueError(f"Unsupported trim method: '{method}'. Use 'energy', 'npts', or 'slice'")
        self.ac = self.ac[slicer]
        self.vel = self.vel[slicer]
        self.disp = self.disp[slicer]
        self.npts = len(self.ac)  # auto clear cache
        return self
    
    def filter(self, bandpass_freqs: tuple[float, float]):
        """
        Perform a bandpass filter on the acceleration data.

        Parameters
        ----------
        bandpass_freqs : tuple of float
            Bandpass frequencies as (lowcut, highcut) in Hz.

        Returns
        -------
        self : GroundMotion
            The filtered GroundMotion instance.
        """
        self.ac = signal_tools.bandpass_filter(self.dt, self.ac, bandpass_freqs[0], bandpass_freqs[1])
        self.vel = signal_tools.get_integral(self.dt, self.ac)
        self.disp = signal_tools.get_integral(self.dt, self.vel)
        self.clear_cache()
        return self
    
    def resample(self, dt: float):
        """
        Resample the motion data to a new time step.

        Parameters
        ----------
        dt : float
            The new time step.

        Returns
        -------
        self : GroundMotion
            The resampled GroundMotion instance.
        """
        npts_new, dt_new, ac_new = signal_tools.resample(self.dt, dt, self.ac)
        self.ac = ac_new
        self.vel = signal_tools.get_integral(dt_new, self.ac)
        self.disp = signal_tools.get_integral(dt_new, self.vel)
        self.npts = npts_new  # auto clear cache
        self.dt = dt_new
        return self

    def save_simulations(self, filename: str, x_var: str, y_vars: list[str]):
        """
        Save simulation data to a CSV file.

        Parameters
        ----------
        filename : str
            Output file name.
        x_var : str
            Independent variable (e.g., 'tp', 'freq', 't').
        y_vars : list of str
            Dependent variables (e.g., ['sa', 'sv', 'sd']).

        Returns
        -------
        self : GroundMotion
            The GroundMotion instance (for chaining).
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
        """
        ndarray: Fourier amplitude spectrum of the acceleration.
        """
        return signal_tools.get_fas(self.npts, self.ac)

    @cached_property
    def fas_smooth(self):
        """
        ndarray: Smoothed Fourier amplitude spectrum.
        """
        return signal_tools.moving_average(self.fas, 9)[..., self.freq_slice]

    @cached_property
    def ce(self):
        """
        ndarray: Cumulative energy of the acceleration.
        """
        return signal_tools.get_ce(self.dt, self.ac)
    
    @cached_property
    def nce(self):
        """
        ndarray: Normalized cumulative energy of the acceleration.
        """
        return signal_tools.get_nce(self.dt, self.ac)
    
    @cached_property
    def mle_ac(self):
        """
        float: Mean log energy of the acceleration.
        """
        return signal_tools.get_mle(self.ac)

    @cached_property
    def mle_vel(self):
        """
        float: Mean log energy of the velocity.
        """
        return signal_tools.get_mle(self.vel)

    @cached_property
    def mle_disp(self):
        """
        float: Mean log energy of the displacement.
        """
        return signal_tools.get_mle(self.disp)

    @cached_property
    def mzc_ac(self):
        """
        float: Mean zero crossing rate of the acceleration.
        """
        return signal_tools.get_mzc(self.ac)

    @cached_property
    def mzc_vel(self):
        """
        float: Mean zero crossing rate of the velocity.
        """
        return signal_tools.get_mzc(self.vel)

    @cached_property
    def mzc_disp(self):
        """
        float: Mean zero crossing rate of the displacement.
        """
        return signal_tools.get_mzc(self.disp)

    @cached_property
    def pmnm_ac(self):
        """
        float: Peak mean normalized motion of the acceleration.
        """
        return signal_tools.get_pmnm(self.ac)

    @cached_property
    def pmnm_vel(self):
        """
        float: Peak mean normalized motion of the velocity.
        """
        return signal_tools.get_pmnm(self.vel)

    @cached_property
    def pmnm_disp(self):
        """
        float: Peak mean normalized motion of the displacement.
        """
        return signal_tools.get_pmnm(self.disp)

    @cached_property
    def spectra(self):
        """
        ndarray: Response spectra (displacement, velocity, acceleration).
        """
        return signal_tools.get_spectra(self.dt, self.ac if self.ac.ndim == 2 else self.ac[None, :], period=self.tp, zeta=0.05)

    @property
    def sa(self):
        """
        ndarray: Spectral acceleration.
        """
        return self.spectra[2]

    @property
    def sv(self):
        """
        ndarray: Spectral velocity.
        """
        return self.spectra[1]

    @property
    def sd(self):
        """
        ndarray: Spectral displacement.
        """
        return self.spectra[0]

    @cached_property
    def pga(self):
        """
        float: Peak ground acceleration.
        """
        return signal_tools.get_pgp(self.ac)

    @cached_property
    def pgv(self):
        """
        float: Peak ground velocity.
        """
        return signal_tools.get_pgp(self.vel)

    @cached_property
    def pgd(self):
        """
        float: Peak ground displacement.
        """
        return signal_tools.get_pgp(self.disp)

    @property
    def energy_slice(self):
        """
        slice: Slice object corresponding to the specified cumulative energy range.
        """
        if not hasattr(self, '_energy_slice'):
            self._energy_slice = signal_tools.slice_energy(self.ce, (0.001, 0.999))  # Default range
        return self._energy_slice

    @energy_slice.setter
    def energy_slice(self, energy_range: tuple[float, float]):
        """
        Set the energy slice range.

        Parameters
        ----------
        energy_range : tuple of float
            (start_fraction, end_fraction) for cumulative energy.
        """
        self._energy_slice = signal_tools.slice_energy(self.ce, energy_range)

    @classmethod
    def load_from(cls, source: str, tag=None, **kwargs):
        """
        Construct a GroundMotion instance from an accelerogram file/data.

        Parameters
        ----------
        source : str
            Source type (e.g., 'NGA', 'ESM', 'COL', 'RAW', 'COR', 'Array').
        tag : str, optional
            Optional tag for the GroundMotion instance.
        **kwargs
            Additional keyword arguments depending on the source type:
            - file : str
                Path to the file.
            - filename : str
                Filename inside the zip archive (for zip files).
            - zip_file : str
                Path to the zip file containing filename (for zip files).
            - dt : float
                Time step (required for 'Array' source).
            - ac : np.ndarray
                Acceleration data (required for 'Array' source).
            - skiprows : int
                Number of rows to skip at the beginning of the file (default is 1).
            - scale : float
                Scaling factor for the acceleration data (default is 1).

        Returns
        -------
        GroundMotion
            An instance of GroundMotion initialized from the file.
        """
        record = RecordReader(source, **kwargs)
        return cls(npts=record.npts, dt=record.dt, ac=record.ac, vel=record.vel, disp=record.disp, tag=tag)