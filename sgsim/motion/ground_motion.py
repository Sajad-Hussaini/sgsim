from functools import cached_property
import numpy as np
import csv
from . import signal_tools
from ..file_reading.record_reader import RecordReader
from ..core.domain_config import DomainConfig

class GroundMotion(DomainConfig):
    """
    Ground motion data container

    Parameters
    ----------
    npts : int
        Number of time points in the record.
    dt : float
        Time step interval in seconds.
    ac : ndarray
        Acceleration time series.
    vel : ndarray
        Velocity time series.
    disp : ndarray
        Displacement time series.
    tag : str, optional
        Identifier for the ground motion record.
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
        Trim ground motion time series.

        Parameters
        ----------
        method : {'energy', 'npts', 'slice'}
            Trimming approach. 'energy' trims by cumulative energy fraction,
            'npts' keeps first N points, 'slice' applies custom indexing.
        value : tuple of float, int, or slice
            Trim parameters. For 'energy': (start, end) fractions (e.g., 0.05, 0.95).
            For 'npts': number of points. For 'slice': slice object.

        Returns
        -------
        self
            Modified GroundMotion instance for method chaining.

        Examples
        --------
        >>> motion.trim('energy', (0.05, 0.95))
        >>> motion.trim('npts', 1000)
        >>> motion.trim('slice', slice(100, 500))
        """
        if method.lower() == 'energy':
            if not isinstance(value, tuple) or len(value) != 2:
                raise ValueError("Energy trimming requires a tuple of (start_fraction, end_fraction)")
            self.energy_slicer = value
            slicer = self.energy_slicer

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
        Apply bandpass filter to ground motion.

        Parameters
        ----------
        bandpass_freqs : tuple of float
            Lower and upper cutoff frequencies (Hz) as (f_low, f_high).

        Returns
        -------
        self
            Modified GroundMotion instance for method chaining.
        """
        self.ac = signal_tools.bandpass_filter(self.dt, self.ac, bandpass_freqs[0], bandpass_freqs[1])
        self.vel = signal_tools.get_integral(self.dt, self.ac)
        self.disp = signal_tools.get_integral(self.dt, self.vel)
        self.clear_cache()
        return self
    
    def resample(self, dt: float):
        """
        Resample ground motion to new time step.

        Parameters
        ----------
        dt : float
            Target time step in seconds.

        Returns
        -------
        self
            Modified GroundMotion instance for method chaining.
        """
        npts_new, dt_new, ac_new = signal_tools.resample(self.dt, dt, self.ac)
        self.ac = ac_new
        self.vel = signal_tools.get_integral(dt_new, self.ac)
        self.disp = signal_tools.get_integral(dt_new, self.vel)
        self.npts = npts_new  # auto clear cache
        self.dt = dt_new
        return self
    
    @property
    def fas(self):
        """
        Fourier amplitude spectrum of acceleration.

        Returns
        -------
        ndarray
            Fourier amplitude spectrum.
        """
        return signal_tools.get_fas(self.npts, self.ac)

    def smooth_fas(self, window: int = 9):
        """
        Smoothed Fourier amplitude spectrum.

        Parameters
        ----------
        window : int, optional
            Moving average window size. Default is 9.

        Returns
        -------
        ndarray
            Smoothed Fourier amplitude spectrum.
        """
        return signal_tools.moving_average(self.fas, window)

    @property
    def ce(self):
        """
        Cumulative energy of acceleration time series.

        Returns
        -------
        ndarray
            Cumulative energy array.
        """
        return signal_tools.get_ce(self.dt, self.ac)
    
    @property
    def mle_ac(self):
        """
        Mean local extrema of acceleration.

        Returns
        -------
        float
            Average of local peak values.
        """
        return signal_tools.get_mle(self.ac)

    @property
    def mle_vel(self):
        """
        Mean local extrema of velocity.

        Returns
        -------
        float
            Average of local peak values.
        """
        return signal_tools.get_mle(self.vel)

    @property
    def mle_disp(self):
        """
        Mean local extrema of displacement.

        Returns
        -------
        float
            Average of local peak values.
        """
        return signal_tools.get_mle(self.disp)

    @property
    def mzc_ac(self):
        """
        Mean zero-crossing of acceleration.

        Returns
        -------
        float
            Zero-crossings per unit time.
        """
        return signal_tools.get_mzc(self.ac)

    @property
    def mzc_vel(self):
        """
        Mean zero-crossing of velocity.

        Returns
        -------
        float
            Zero-crossings per unit time.
        """
        return signal_tools.get_mzc(self.vel)

    @property
    def mzc_disp(self):
        """
        Mean zero-crossing of displacement.

        Returns
        -------
        float
            Zero-crossings per unit time.
        """
        return signal_tools.get_mzc(self.disp)

    @property
    def pmnm_ac(self):
        """
        Positive-minima and negative-maxima of acceleration.

        Returns
        -------
        float
            Ratio of peak to mean absolute value.
        """
        return signal_tools.get_pmnm(self.ac)

    @property
    def pmnm_vel(self):
        """
        Positive-minima and negative-maxima of velocity.

        Returns
        -------
        float
            Ratio of peak to mean absolute value.
        """
        return signal_tools.get_pmnm(self.vel)

    @property
    def pmnm_disp(self):
        """
        Positive-minima and negative-maxima of displacement.

        Returns
        -------
        float
            Ratio of peak to mean absolute value.
        """
        return signal_tools.get_pmnm(self.disp)

    @cached_property
    def spectra(self):
        """
        Response spectra at 5% damping.

        Returns
        -------
        ndarray
            Array of shape (3, n_periods) with [Sd, Sv, Sa].
        """
        if not hasattr(self, 'tp'):
            raise AttributeError("Set 'tp' attribute (periods) before accessing spectra")
        return signal_tools.get_spectra(self.dt, self.ac if self.ac.ndim == 2 else self.ac[None, :], period=self.tp, zeta=0.05)

    @property
    def sa(self):
        """
        Spectral acceleration response.

        Returns
        -------
        ndarray
            Sa values at periods defined by tp.
        """
        return self.spectra[2]

    @property
    def sv(self):
        """
        Spectral velocity response.

        Returns
        -------
        ndarray
            Sv values at periods defined by tp.
        """
        return self.spectra[1]

    @property
    def sd(self):
        """
        Spectral displacement response.

        Returns
        -------
        ndarray
            Sd values at periods defined by tp.
        """
        return self.spectra[0]

    @property
    def pga(self):
        """
        Peak ground acceleration.

        Returns
        -------
        float
            Maximum absolute acceleration value.
        """
        return signal_tools.get_peak_param(self.ac)

    @property
    def pgv(self):
        """
        Peak ground velocity.

        Returns
        -------
        float
            Maximum absolute velocity value.
        """
        return signal_tools.get_peak_param(self.vel)

    @property
    def pgd(self):
        """
        Peak ground displacement.

        Returns
        -------
        float
            Maximum absolute displacement value.
        """
        return signal_tools.get_peak_param(self.disp)

    @property
    def energy_slicer(self):
        """
        Slice indices for cumulative energy range.

        Returns
        -------
        slice
            Index slice for energy-based trimming.
        """
        return self._energy_slicer

    @energy_slicer.setter
    def energy_slicer(self, energy_range: tuple[float, float]):
        """
        Set energy slice range.

        Parameters
        ----------
        energy_range : tuple of float
            Start and end fractions of cumulative energy (e.g., 0.05, 0.95).
        """
        self._energy_slicer = signal_tools.slice_energy(self.ce, energy_range)
    
    def to_csv(self, filename: str, features: list[str]):
        """
        Export selected features to CSV.

        Parameters
        ----------
        filename : str
            Output CSV file path.
        features : list of str
            List of feature names to export.
        """
        header = []
        row = []
        for feature in features:
            feature_l = feature.lower()
            attr = getattr(self, feature_l)

            # Spectral arrays (sa, sv, sd)
            if feature_l in ("sa", "sv", "sd"):
                if not hasattr(self, "tp"):
                    raise AttributeError("Set 'tp' attribute (periods) before accessing spectra.")
                for i, val in enumerate(attr.T):
                    header.append(f"{feature_l}_{self.tp[i]:.3f}")
                    row.append(val)
            # FAS (Fourier amplitude spectrum)
            elif feature_l == "fas":
                if not hasattr(self, "freq"):
                    raise AttributeError("Set 'freq' attribute (frequencies) before accessing spectra")
                for i, val in enumerate(attr.T):
                    header.append(f"fas_{self.freq[i] / (2*np.pi):.3f}")
                    row.append(val)
            else:
                header.append(feature_l)
                row.append(attr)

        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerow(row)

    @classmethod
    def load_from(cls, source: str, tag=None, **kwargs):
        """
        Load ground motion from file or array.

        Parameters
        ----------
        source : str
            Data source format: 'NGA', 'ESM', 'COL', 'RAW', 'COR', or 'Array'.
        tag : str, optional
            Record identifier.
        **kwargs
            Source-specific arguments:
            
            For file sources: file, filename, zip_file, skiprows, scale
            For 'Array' source: dt (float), ac (ndarray)

        Returns
        -------
        GroundMotion
            Loaded ground motion instance.

        Examples
        --------
        >>> gm = GroundMotion.load_from('NGA', file='record.AT2')
        >>> gm = GroundMotion.load_from('Array', dt=0.01, ac=acc_array)
        """
        record = RecordReader(source, **kwargs)
        return cls(npts=record.npts, dt=record.dt, ac=record.ac, vel=record.vel, disp=record.disp, tag=tag)