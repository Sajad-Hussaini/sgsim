from functools import cached_property
import numpy as np
import csv
from . import signal_tools
from ..file_reading.record_reader import RecordReader
from ..core.domain_config import DomainConfig
from ..optimization.fit_eval import relative_error, goodness_of_fit

class GroundMotion:
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
        self._npts = npts
        self._dt = dt
        self.ac = ac.astype(np.float64, copy=False)
        self.vel = vel.astype(np.float64, copy=False)
        self.disp = disp.astype(np.float64, copy=False)
        self.tag = tag
    
    @classmethod
    def load_from(cls, tag=None, **kwargs):
        """
        Load ground motion from file or array.

        Parameters
        ----------
        source : str
            Data source format: 'NGA', 'ESM', 'COL', 'RAW', 'COR' for file reading
                                'Array' for direct array input.
        tag : str, optional
            Record identifier.
        **kwargs
            Source-specific arguments.

        Returns
        -------
        GroundMotion
            Loaded ground motion instance.
        """
        record = RecordReader(**kwargs)
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
            'response_spectra': 'Acceleration, Velocity, Displacement Response Spectra (requires periods)',
            
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
            'freq': 'Frequency array (for FAS)',
            
            # Statistical measures
            'le_ac': 'Mean Local Extrema of Acceleration',
            'le_vel': 'Mean Local Extrema of Velocity',
            'le_disp': 'Mean Local Extrema of Displacement',
            'zc_ac': 'Mean Zero Crossing of Acceleration',
            'zc_vel': 'Mean Zero Crossing of Velocity',
            'zc_disp': 'Mean Zero Crossing of Displacement',
            'pmnm_ac': 'Positive Min / Negative Max of Acceleration',
            'pmnm_vel': 'Positive Min / Negative Max of Velocity',
            'pmnm_disp': 'Positive Min / Negative Max of Displacement',
            }
        return ims
    
    @property
    def npts(self):
        return self._npts

    @npts.setter
    def npts(self, value: int):
            self._npts = value
            self._clear_cache()

    @property
    def dt(self):
        return self._dt

    @dt.setter
    def dt(self, value: float):
            self._dt = value
            self._clear_cache()
    
    @cached_property
    def t(self):
        """
        ndarray: Time array for the configured number of points and time step.
        """
        return signal_tools.time(self.npts, self.dt)

    @cached_property
    def freq(self):
        """
        ndarray: Frequency array for the configured number of points and time step.
        """
        return signal_tools.frequency(self.npts, self.dt)

    def _clear_cache(self):
        """
        Clear cached properties, preserving core attributes.
        """
        core_values = {attr: getattr(self, attr, None) for attr in self._CORE_ATTRS}
        self.__dict__.clear()
        self.__dict__.update(core_values)
    
    def trim_by_index(self, start_index: int, end_index: int):
        """
        Trim ground motion by keeping the points between `start_index` and `end_index`.

        Returns
        -------
        GroundMotion
            New instance.
        """
        if start_index < 0 or end_index > self.npts:
            raise ValueError("start_index and end_index must be within current npts")
        return self.load_from(source="array", dt=self.dt, ac=self.ac[start_index:end_index], tag=self.tag)

    def trim_by_slice(self, slicer: slice):
        """
        Trim ground motion using a slice object.

        Returns
        -------
        GroundMotion
            New instance.
        """
        if not isinstance(slicer, slice):
            raise TypeError("Expected a slice object")
        return self.load_from(source="array", dt=self.dt, ac=self.ac[slicer], tag=self.tag)

    def trim_by_energy(self, energy_range: tuple[float, float]):
        """
        Trim ground motion to a specific cumulative energy range.

        Parameters
        ----------
        energy_range : tuple
            (start_fraction, end_fraction), e.g., (0.05, 0.95)

        Returns
        -------
        GroundMotion
            New instance.
        """
        slicer = signal_tools.slice_energy(self.ce, energy_range)
        return self.load_from(source="array", dt=self.dt, ac=self.ac[slicer], tag=self.tag)

    def trim_by_amplitude(self, threshold: float):
        """
        Trim ground motion based on acceleration amplitude threshold.

        Parameters
        ----------
        threshold : float
            Amplitude threshold.

        Returns
        -------
        GroundMotion
            New instance.
        """
        slicer = signal_tools.slice_amplitude(self.ac, threshold)
        return self.load_from(source="array", dt=self.dt, ac=self.ac[slicer], tag=self.tag)
    
    def taper(self, alpha: float = 0.05):
        """
        Apply tapering to the ground motion.

        Parameters
        ----------
        alpha : float, optional
            Shape parameter of the Tukey window, representing the fraction of the
            window inside the cosine tapered region.
            If zero, the Tukey window is equivalent to a rectangular window.
            If one, the Tukey window is equivalent to a Hann window.

        Returns
        -------
        GroundMotion
            New instance.
        """
        new_ac = signal_tools.taper(self.ac, alpha)
        return self.load_from(source="array", dt=self.dt, ac=new_ac, tag=self.tag)
    
    def butterworth_filter(self, bandpass_freqs: tuple[float, float], order: int = 4):
        """
        Apply butterworth filter.

        Parameters
        ----------
        bandpass_freqs : tuple
            (low_freq, high_freq) in Hz.
        order : int, optional
            Filter order (default 4).

        Returns
        -------
        GroundMotion
            New instance.
        """
        new_ac = signal_tools.butterworth_filter(self.dt, self.ac, *bandpass_freqs, order)
        return self.load_from(source="array", dt=self.dt, ac=new_ac, tag=self.tag)
    
    def baseline_correction(self, degree: int = 1):
        """
        Apply baseline correction.

        Parameters
        ----------
        degree : int, optional
            Degree of polynomial (default 1).

        Returns
        -------
        GroundMotion
            New instance.
        """
        new_ac = signal_tools.baseline_correction(self.ac, degree)
        return self.load_from(source="array", dt=self.dt, ac=new_ac, tag=self.tag)
    
    def resample(self, dt: float):
        """
        Resample to a new time step.

        Parameters
        ----------
        dt : float
            New time step.

        Returns
        -------
        GroundMotion
            New instance.
        """
        _, dt_new, ac_new = signal_tools.resample(self.dt, dt, self.ac)
        return self.load_from(source="array", dt=dt_new, ac=ac_new, tag=self.tag)
    
    @property
    def fas(self):
        """
        Fourier amplitude spectrum of acceleration.

        Returns
        -------
        ndarray
            Fourier amplitude spectrum.
        """
        return signal_tools.fas(self.dt, self.ac)
    
    @property
    def fps(self):
        """
        Fourier phase spectrum of acceleration.
        Returns unwrapped phase to ensure continuity (no jumps between -pi and pi).

        Returns
        -------
        ndarray
            Fourier phase spectrum.
        """
        return signal_tools.fps(self.ac)

    @property
    def ce(self):
        """
        Cumulative energy of acceleration time series.

        Returns
        -------
        ndarray
            Cumulative energy array.
        """
        return signal_tools.ce(self.dt, self.ac)
    
    @property
    def le_ac(self):
        """
        Mean local extrema of acceleration.

        Returns
        -------
        float
            Mean local extrema value.
        """
        return signal_tools.le(self.ac)

    @property
    def le_vel(self):
        """
        Mean local extrema of velocity.

        Returns
        -------
        float
            Mean local extrema value.
        """
        return signal_tools.le(self.vel)

    @property
    def le_disp(self):
        """
        Mean local extrema of displacement.

        Returns
        -------
        float
            Mean local extrema value.
        """
        return signal_tools.le(self.disp)

    @property
    def zc_ac(self):
        """
        Mean zero-crossing of acceleration.

        Returns
        -------
        float
            Zero-crossing.
        """
        return signal_tools.zc(self.ac)

    @property
    def zc_vel(self):
        """
        Mean zero-crossing of velocity.

        Returns
        -------
        float
            Zero-crossing.
        """
        return signal_tools.zc(self.vel)

    @property
    def zc_disp(self):
        """
        Mean zero-crossing of displacement.

        Returns
        -------
        float
            Zero-crossing.
        """
        return signal_tools.zc(self.disp)

    @property
    def pmnm_ac(self):
        """
        Positive-minima and negative-maxima of acceleration.

        Returns
        -------
        float
            PMNM.
        """
        return signal_tools.pmnm(self.ac)

    @property
    def pmnm_vel(self):
        """
        Positive-minima and negative-maxima of velocity.

        Returns
        -------
        float
            PMNM.
        """
        return signal_tools.pmnm(self.vel)

    @property
    def pmnm_disp(self):
        """
        Positive-minima and negative-maxima of displacement.

        Returns
        -------
        float
            PMNM.
        """
        return signal_tools.pmnm(self.disp)

    def response_spectra(self, periods: np.ndarray, damping: float = 0.05):
        """
        Calculate response spectra for given periods and damping.
        
        Parameters
        ----------
        periods : ndarray
            Array of periods in seconds.
        damping : float
            Damping ratio (default 0.05).
            
        Returns
        -------
        tuple
            (sd, sv, sa) arrays.
        """
        return signal_tools.response_spectra(self.dt, self.ac, period=periods, zeta=damping)

    @property
    def pga(self):
        """
        Peak ground acceleration.

        Returns
        -------
        float
            Peak ground acceleration value.
        """
        return signal_tools.peak_abs_value(self.ac)

    @property
    def pgv(self):
        """
        Peak ground velocity.

        Returns
        -------
        float
            Peak ground velocity value.
        """
        return signal_tools.peak_abs_value(self.vel)

    @property
    def pgd(self):
        """
        Peak ground displacement.

        Returns
        -------
        float
            Peak ground displacement value.
        """
        return signal_tools.peak_abs_value(self.disp)
    
    @property
    def cav(self):
        """
        Cumulative absolute velocity.

        Returns
        -------
        float
            CAV value.
        """
        return signal_tools.cav(self.dt, self.ac)
    
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
        sd, sv, sa = signal_tools.response_spectra(self.dt, self.ac, period=vsi_tp, zeta=0.05)
        dsi = np.sum(sd, axis=-1) * 0.05
        vsi = np.sum(sv, axis=-1) * 0.05
        asi = np.sum(sa, axis=-1) * 0.05
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
    
    def compute_intensity_measures(self, ims: list[str], periods: np.ndarray = None) -> dict:
        """
        Compute selected intensity measures.

        Parameters
        ----------
        ims : list[str]
            List of IM names to compute (e.g., ['pga', 'sa', 'cav']).
        periods : np.ndarray, optional
            Periods for spectral quantities (sa, sv, sd). Required if any spectral IM is requested.

        Returns
        -------
        dict
            Dictionary of computed IMs. Keys are column names (e.g., 'pga', 'sa_0.200').
            Values are either floats (single record) or arrays (multiple records).
        """
        periods = np.asarray(periods)
        results = {}
        
        # Determine if we have multiple records
        if self.ac.ndim == 1:
            n_records = 1
        else:
            n_records = self.ac.shape[0]

        # 1. Pre-compute spectra if requested (Batch Optimization)
        spectral_ims = [im for im in ims if im.lower() in ("sa", "sv", "sd")]
        spectral_data = {}
        
        if spectral_ims:
            if periods is None:
                raise ValueError("Periods must be provided to compute/export spectral quantities (sa, sv, sd).")
            # Compute once for all spectral types
            sd, sv, sa = self.response_spectra(periods)
            spectral_data['sd'] = sd
            spectral_data['sv'] = sv
            spectral_data['sa'] = sa
        
        # 2. Iterate and collect data
        for im in ims:
            im_l = im.lower()
            
            # Case A: Spectral IMs
            if im_l in spectral_data:
                data_matrix = spectral_data[im_l] # Shape: (periods,) or (records, periods)
                
                for idx, period in enumerate(periods):
                    key = f"{im_l}_{period:.3f}"
                    if n_records == 1:
                        results[key] = data_matrix[idx]
                    else:
                        results[key] = data_matrix[:, idx]

            # Case B: Fourier Amplitude Spectra (vector handling)
            elif im_l == "fas":
                 # FAS logic is slightly unique as it depends on freq, not period
                 # Usually FAS is exported as full spectrum, so logic might differ.
                 # For now, sticking to logic mirroring to_csv: export all freqs
                 freqs = self.freq
                 attr = self.fas
                 for idx, freq in enumerate(freqs):
                     key = f"fas_{freq:.3f}"
                     if n_records == 1:
                         results[key] = attr[idx]
                     else:
                         results[key] = attr[:, idx] # Check dim handling of fas property

            # Case C: Scalar IMs (PGA, PGV, etc.)
            else:
                attr = getattr(self, im_l)
                results[im_l] = attr

        return results

    def to_csv(self, filename: str, ims: list[str], periods: np.ndarray = None):
        """
        Export selected intensity measures (ims) to CSV.
        
        Parameters
        ----------
        filename : str
            Output path.
        ims : list[str]
            List of IM names.
        periods : np.ndarray, optional
            Periods for spectral quantities (sa, sv, sd). Required if any spectral IM is requested.
        """
        data = self.compute_intensity_measures(ims, periods)
        
        fieldnames = list(data.keys())
        
        # Determine number of rows to write
        # Check the length of the first value
        first_val = next(iter(data.values()))
        
        # Handle scalar (single record) vs list (multiple records)
        # If single record, values are floats (compute_intensity_measures returns scalar or array)
        if np.isscalar(first_val):
            n_rows = 1
        else:
            n_rows = len(first_val)
            
        rows = []
        for i in range(n_rows):
            row = {}
            for key, val in data.items():
                if n_rows == 1:
                     row[key] = val
                else:
                     row[key] = val[i]
            rows.append(row)
            
        with open(filename, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    
    def compare(self, other: "GroundMotion", ims: list[str], periods: np.ndarray = None, method: str = 'gof') -> dict:
        """
        Compare this ground motion with another using selected IMs.
        
        Parameters
        ----------
        other : GroundMotion
            Target ground motion or model.
        ims : list[str]
            List of IM names to compare.
        periods : np.ndarray, optional
            Periods for spectral quantities.
        method : str, optional
            Comparison method: 'gof' (Goodness of Fit) or 're' (Relative Error).
            
        Returns
        -------
        dict
            Dictionary of comparison scores.
        """
        criterion_map = {'gof': goodness_of_fit, 're': relative_error}
        if method.lower() not in criterion_map:
             raise ValueError(f"Unknown method: {method}. Supported: {list(criterion_map.keys())}")
        
        func = criterion_map[method.lower()]
        
        # 1. Compute IMs for both
        my_data = self.compute_intensity_measures(ims, periods)
        other_data = other.compute_intensity_measures(ims, periods)
        
        # 2. Compare matching keys
        scores = {}
        for key in my_data:
             if key in other_data:
                 scores[key] = func(my_data[key], other_data[key])
                 
        return scores

# =============================================================================

class GroundMotionMultiComponent:
    """
    Container for multi-component ground motion data (2 or 3 component).
    
    Parameters
    ----------
    *components : GroundMotion
        Variable number of GroundMotion objects (minimum 2).
    """
    def __init__(self, *components: GroundMotion):
        if len(components) < 2:
            raise ValueError("At least 2 components required for multi-component ground motion.")
        
        # Validate all components have same time parameters
        dt_ref = components[0].dt
        npts_ref = components[0].npts
        for i, gm in enumerate(components[1:], 1):
            if gm.dt != dt_ref or gm.npts != npts_ref:
                raise ValueError(f"Component {i} has mismatched dt or npts with component 0.")
        
        self.components = components
        self.n_components = len(components)
        self.t = components[0].t
        self.dt = components[0].dt
        self.npts = components[0].npts
        self.freq = components[0].freq
    
    @property
    def ac(self):
        """
        Magnitude of acceleration across all components (resultant acceleration).

        Returns
        -------
        ndarray
            Combined magnitude array.
        """
        return np.sqrt(np.sum([gm.ac ** 2 for gm in self.components], axis=0))
    
    @property
    def vel(self):
        """
        Magnitude of velocity across all components.

        Returns
        -------
        ndarray
            Combined magnitude array.
        """
        return np.sqrt(np.sum([gm.vel ** 2 for gm in self.components], axis=0))
    
    @property
    def disp(self):
        """
        Magnitude of displacement across all components.

        Returns
        -------
        ndarray
            Combined magnitude array.
        """
        return np.sqrt(np.sum([gm.disp ** 2 for gm in self.components], axis=0))

    @property
    def ce(self):
        """
        Cumulative energy of combined components.

        Returns
        -------
        ndarray
            Combined cumulative energy array.
        """
        return np.sum([gm.ce for gm in self.components], axis=0)
    
    @property
    def fas(self):
        """
        Fourier amplitude spectrum of combined components.

        Returns
        -------
        ndarray
            Combined Fourier amplitude spectrum.
        """
        return np.sqrt(np.sum([gm.fas ** 2 for gm in self.components], axis=0))
    
    @property
    def pga(self):
        """Peak ground acceleration (resultant)."""
        return signal_tools.peak_abs_value(self.ac)
    
    @property
    def pgv(self):
        """Peak ground velocity (resultant)."""
        return signal_tools.peak_abs_value(self.vel)
    
    @property
    def pgd(self):
        """Peak ground displacement (resultant)."""
        return signal_tools.peak_abs_value(self.disp)