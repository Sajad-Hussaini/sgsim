from functools import cached_property
import numpy as np
import csv
from . import signal_tools
from ..file_reading.record_reader import RecordReader
from ..core.domain_config import DomainConfig
from ..optimization.fit_eval import relative_error, goodness_of_fit

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
        self.ac = ac.astype(np.float64, copy=False)
        self.vel = vel.astype(np.float64, copy=False)
        self.disp = disp.astype(np.float64, copy=False)
        self.tag = tag

    def trim(self, method: str, value: tuple[float, float] | int | slice):
        """
        Trim ground motion time series.

        Parameters
        ----------
        method : {'energy', 'npts', 'slice'}
            Trimming method to use.
        value : tuple of float, int, or slice
            Trim parameters. For 'energy': (start, end) fractions (e.g., 0.05, 0.95).
            For 'npts': number of points. For 'slice': slice object.

        Returns
        -------
        self
            Modified GroundMotion instance.
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
            Lower and upper cutoff frequencies in Hz.

        Returns
        -------
        self
            Modified GroundMotion instance.
        """
        self.ac = signal_tools.bandpass_filter(self.dt, self.ac, bandpass_freqs[0], bandpass_freqs[1])
        self.vel = signal_tools.get_integral(self.dt, self.ac)
        self.disp = signal_tools.get_integral(self.dt, self.vel)
        self._clear_cache()
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
            Modified GroundMotion instance.
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
            Moving average window size.

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
            Mean local extrema value.
        """
        return signal_tools.get_mle(self.ac)

    @property
    def mle_vel(self):
        """
        Mean local extrema of velocity.

        Returns
        -------
        float
            Mean local extrema value.
        """
        return signal_tools.get_mle(self.vel)

    @property
    def mle_disp(self):
        """
        Mean local extrema of displacement.

        Returns
        -------
        float
            Mean local extrema value.
        """
        return signal_tools.get_mle(self.disp)

    @property
    def mzc_ac(self):
        """
        Mean zero-crossing of acceleration.

        Returns
        -------
        float
            Zero-crossing.
        """
        return signal_tools.get_mzc(self.ac)

    @property
    def mzc_vel(self):
        """
        Mean zero-crossing of velocity.

        Returns
        -------
        float
            Zero-crossing.
        """
        return signal_tools.get_mzc(self.vel)

    @property
    def mzc_disp(self):
        """
        Mean zero-crossing of displacement.

        Returns
        -------
        float
            Zero-crossing.
        """
        return signal_tools.get_mzc(self.disp)

    @property
    def pmnm_ac(self):
        """
        Positive-minima and negative-maxima of acceleration.

        Returns
        -------
        float
            PMNM.
        """
        return signal_tools.get_pmnm(self.ac)

    @property
    def pmnm_vel(self):
        """
        Positive-minima and negative-maxima of velocity.

        Returns
        -------
        float
            PMNM.
        """
        return signal_tools.get_pmnm(self.vel)

    @property
    def pmnm_disp(self):
        """
        Positive-minima and negative-maxima of displacement.

        Returns
        -------
        float
            PMNM.
        """
        return signal_tools.get_pmnm(self.disp)

    @cached_property
    def spectra(self):
        """
        Response spectra at 5% damping.

        Returns
        -------
        ndarray
            Response spectra array with shape (3, n_periods).
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
            Spectral acceleration values.
        """
        return self.spectra[2]

    @property
    def sv(self):
        """
        Spectral velocity response.

        Returns
        -------
        ndarray
            Spectral velocity values.
        """
        return self.spectra[1]

    @property
    def sd(self):
        """
        Spectral displacement response.

        Returns
        -------
        ndarray
            Spectral displacement values.
        """
        return self.spectra[0]

    @property
    def pga(self):
        """
        Peak ground acceleration.

        Returns
        -------
        float
            Peak ground acceleration value.
        """
        return signal_tools.get_peak_param(self.ac)

    @property
    def pgv(self):
        """
        Peak ground velocity.

        Returns
        -------
        float
            Peak ground velocity value.
        """
        return signal_tools.get_peak_param(self.vel)

    @property
    def pgd(self):
        """
        Peak ground displacement.

        Returns
        -------
        float
            Peak ground displacement value.
        """
        return signal_tools.get_peak_param(self.disp)
    
    @property
    def cav(self):
        """
        Cumulative absolute velocity.

        Returns
        -------
        float
            CAV value.
        """
        return signal_tools.get_cav(self.dt, self.ac)
    
    @cached_property
    def spectrum_intensity(self):
        """
        spectrum intensity over period 0.1 to 2.5 seconds.

        Returns
        -------
        float
            VSI value.
        """
        vsi_tp = np.arange(0.1, 2.5, 0.05)
        sd, sv, sa = signal_tools.get_spectra(self.dt, self.ac if self.ac.ndim == 2 else self.ac[None, :], period=vsi_tp, zeta=0.05)
        dsi = signal_tools.get_integral(0.05, sd)[..., -1]
        vsi = signal_tools.get_integral(0.05, sv)[..., -1]
        asi = signal_tools.get_integral(0.05, sa)[..., -1]
        return dsi, vsi, asi
    
    @property
    def vsi(self):
        """
        Velocity spectrum intensity.

        Returns
        -------
        float
            VSI value.
        """
        
        return self.spectrum_intensity[1]
    
    @property
    def asi(self):
        """
        Acceleration spectrum intensity.

        Returns
        -------
        float
            ASI value.
        """
        
        return self.spectrum_intensity[2]
    
    @property
    def dsi(self):
        """
        Displacement spectrum intensity.

        Returns
        -------
        float
            DSI value.
        """
        
        return self.spectrum_intensity[0]

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
            Start and end fractions of cumulative energy.
        """
        self._energy_slicer = signal_tools.slice_energy(self.ce, energy_range)
    
    def to_csv(self, filename: str, ims: list[str]):
        """
        Export selected intensity measures (ims) to CSV.

        Parameters
        ----------
        filename : str
            Output CSV file path.
        ims : list of str
            List of ims to export.
        
        Notes
        -----
        For multi-record data (e.g., self.ac with shape (n_records, npts)),
        each record will be saved as a separate row in the CSV.
        """
        header = []
        rows = []
        
        # Determine number of records
        if self.ac.ndim == 1:
            n_records = 1
        else:
            n_records = self.ac.shape[0]
        
        # Build header
        for im in ims:
            im_l = im.lower()
            
            # Spectral arrays (sa, sv, sd)
            if im_l in ("sa", "sv", "sd"):
                if not hasattr(self, "tp"):
                    raise AttributeError("Set 'tp' attribute (periods) before accessing spectra.")
                for period in self.tp:
                    header.append(f"{im_l}_{period:.3f}")
            # FAS (Fourier amplitude spectrum)
            elif im_l == "fas":
                if not hasattr(self, "freq"):
                    raise AttributeError("Set 'freq' attribute (frequencies) before accessing FAS")
                for freq in self.freq:
                    header.append(f"fas_{freq / (2*np.pi):.3f}")
            # Scalar values
            else:
                header.append(im_l)
        
        # Build rows
        for i in range(n_records):
            row = []
            for im in ims:
                im_l = im.lower()
                attr = getattr(self, im_l)
                
                # Spectral arrays (sa, sv, sd)
                if im_l in ("sa", "sv", "sd"):
                    if attr.ndim == 1:
                        row.extend(attr)
                    else:
                        row.extend(attr[i])
                # FAS (Fourier amplitude spectrum)
                elif im_l == "fas":
                    if attr.ndim == 1:
                        row.extend(attr)
                    else:
                        row.extend(attr[i])
                # Scalar values or arrays that need indexing
                else:
                    if hasattr(attr, '__len__') and not isinstance(attr, str):
                        if len(attr) == n_records:
                            row.append(attr[i])
                        else:
                            # Single value for all records
                            row.append(attr)
                    else:
                        # Scalar value
                        row.append(attr)
            
            rows.append(row)
        
        # Write to CSV
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)
            writer.writerows(rows)
    
    def compare_with(self, component, ims: list[str], method: str, transform: callable = None):
        """
        Compare selected intensity measures (ims) between this GroundMotion instance and another component.

        This method computes similarity or error ims (e.g., goodness-of-fit, relative error)
        for specified attributes (such as 'sa', 'sv', 'fas', etc.) between the current ground motion
        and another GroundMotion or FittedModel instance.

        Parameters
        ----------
        component : GroundMotion or FittedModel
            The instance to compare against.
        ims : list of str
            Intensity measures (e.g., 'sa', 'sv', 'fas') to compare.
        method : {'gof', 're'}
            Comparison method: 'gof' for goodness-of-fit, 're' for relative error.
        transform : callable, optional
            Function to apply to both attribute values before comparison (e.g., np.log).

        Returns
        -------
        dict
            Dictionary mapping each IM name to its computed comparison value.

        Raises
        ------
        ValueError
            If an unsupported method is provided.

        """
        result = {}
        criterion_map = {'gof': goodness_of_fit, 're': relative_error}
        method = criterion_map.get(method.lower())
        if method is None:
            raise ValueError(f"Unknown method: {method}. Supported: {list(criterion_map.keys())}")
        for im in ims:
            self_attr = getattr(self, im)
            comp_attr = getattr(component, im)
            if transform is not None:
                self_attr = transform(self_attr)
                comp_attr = transform(comp_attr)
            result[im] = method(self_attr, comp_attr)
        return result

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
            Source-specific arguments.

        Returns
        -------
        GroundMotion
            Loaded ground motion instance.
        """
        record = RecordReader(source, **kwargs)
        return cls(npts=record.npts, dt=record.dt, ac=record.ac, vel=record.vel, disp=record.disp, tag=tag)

    @classmethod
    def list_IMs(cls):
        """
        List all available intensity measures (ims) and properties with descriptions.
        
        Returns
        -------
        dict
            Dictionary mapping im names to their descriptions.
            
        Examples
        --------
        >>> GroundMotion.list_IMs()
        >>> # or to get just the names:
        >>> list(GroundMotion.list_IMs().keys())
        
        Note
        ----
        Feel free to contact the developer (via Hussaini.smsajad@gmail.com) to add or include new ims.
        """
        ims = {
            # Peak parameters
            'pga': 'Peak Ground Acceleration',
            'pgv': 'Peak Ground Velocity',
            'pgd': 'Peak Ground Displacement',
            
            # Response spectra (requires tp attribute)
            'sa': 'Spectral Acceleration (requires tp)',
            'sv': 'Spectral Velocity (requires tp)',
            'sd': 'Spectral Displacement (requires tp)',
            
            # Intensity integrals
            'cav': 'Cumulative Absolute Velocity',
            'vsi': 'Velocity Spectrum Intensity (0.1-2.5s)',
            'asi': 'Acceleration Spectrum Intensity (0.1-2.5s)',
            'dsi': 'Displacement Spectrum Intensity (0.1-2.5s)',
            
            # Time series data
            'ac': 'Acceleration time series',
            'vel': 'Velocity time series',
            'disp': 'Displacement time series',
            
            # Frequency domain
            'fas': 'Fourier Amplitude Spectrum',
            'ce': 'Cumulative Energy',
            
            # Domain attributes
            't': 'Time array',
            'tp': 'Period array (for spectra)',
            'freq': 'Frequency array (for FAS)',
            
            # Statistical measures
            'mle_ac': 'Mean Local Extrema of Acceleration',
            'mle_vel': 'Mean Local Extrema of Velocity',
            'mle_disp': 'Mean Local Extrema of Displacement',
            'mzc_ac': 'Mean Zero Crossing of Acceleration',
            'mzc_vel': 'Mean Zero Crossing of Velocity',
            'mzc_disp': 'Mean Zero Crossing of Displacement',
            'pmnm_ac': 'Positive Min / Negative Max of Acceleration',
            'pmnm_vel': 'Positive Min / Negative Max of Velocity',
            'pmnm_disp': 'Positive Min / Negative Max of Displacement',
        }
        return ims

### ================================================================================

class GroundMotion3D:
    """
    Container for three-component ground motion data.
    """
    def __init__(self, gm1:GroundMotion, gm2:GroundMotion, gm3:GroundMotion):
        self.gm1 = gm1
        self.gm2 = gm2
        self.gm3 = gm3
        self.t = gm1.t
        self.dt = gm1.dt
        self.npts = gm1.npts
        self.freq = gm1.freq

    @property
    def ce(self):
        """
        Cumulative energy of the net three component of acceleration time series.

        Returns
        -------
        ndarray
            Combined cumulative energy array.
        """
        return self.gm1.ce + self.gm2.ce + self.gm3.ce
    
    @property
    def fas(self):
        """
        Fourier amplitude spectrum of the net three component of acceleration time series.

        Returns
        -------
        ndarray
            Combined Fourier amplitude spectrum.
        """
        return np.sqrt(self.gm1.fas ** 2 + self.gm2.fas ** 2 + self.gm3.fas ** 2)

### ================================================================================

