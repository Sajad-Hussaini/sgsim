"""Ground motion container with signal processing and intensity measures."""
from functools import cached_property
import csv

import numpy as np

from . import signal
from ..io.record_reader import Record
from ..optimization.fit_eval import goodness_of_fit, relative_error

class GroundMotion:
    """
    Container for ground motion data and related operations.
    
    This class stores acceleration, velocity, and displacement time series
    along with derived intensity measures. All transformation methods return
    new instances rather than modifying in place.

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
        Identifier for the ground motion record (default is None).
    
    Notes
    -----
    Ground motion instances should be treated as immutable. Direct modification
    of npts, dt, ac, vel, or disp may lead to inconsistent cached properties.
    
    Examples
    --------
    Load from file:
    
    >>> gm = GroundMotion.load_from(source='NGA', file='record.at2')
    >>> gm.pga
    0.45
    
    Create from arrays:
    
    >>> import numpy as np
    >>> ac = np.random.randn(1000)
    >>> gm = GroundMotion.load_from(source='array', dt=0.01, ac=ac)
    >>> gm_trimmed = gm.trim_by_energy((0.05, 0.95))
    >>> gm_trimmed.npts
    900
    """

    def __init__(self, npts, dt, ac, vel, disp, tag=None):
        self.npts = npts
        self.dt = dt
        self.ac = ac
        self.vel = vel
        self.disp = disp
        self.tag = tag

    # Class methods ==================================================================

    _im_registry = {
        'ac': 'Acceleration time series',
        'vel': 'Velocity time series',
        'disp': 'Displacement time series',
        'response_spectra': 'Acceleration, Velocity, Displacement Response Spectra (requires periods)',
    }

    @classmethod
    def register_im(cls, name, description):
        """Decorator to register an intensity measure."""
        def decorator(func):
            cls._im_registry[name] = description
            return func
        return decorator

    @classmethod
    def load_from(cls, tag=None, **kwargs):
        """
        Load ground motion from file or array.
        
        Factory method for creating GroundMotion instances from various sources
        including seismic database formats (NGA, ESM) or direct array input.

        Parameters
        ----------
        source : str
            Data source format. Options:
            - 'NGA': PEER NGA database format
            - 'ESM': Engineering Strong Motion database
            - 'COL': Column-based text format
            - 'RAW': Raw binary format
            - 'COR': Corrected format
            - 'array': Direct NumPy array input
        tag : str, optional
            Record identifier for tracking (default is None).
        **kwargs : dict
            Source-specific arguments.

        Returns
        -------
        GroundMotion
            New ground motion instance.
        
        Raises
        ------
        ValueError
            If required parameters are missing for the specified source.
        FileNotFoundError
            If file does not exist for file-based sources.
        
        Notes
        -----
        For kwargs details, refer to Record documentation.

        See Also
        --------
        Record : Underlying file reading implementation.
        
        Examples
        --------
        Load from NGA file:
        
        >>> gm = GroundMotion.load_from(source='NGA', file='RSN123.at2', tag='RSN123')
        >>> gm.pga
        0.521
        
        Create from array:
        
        >>> import numpy as np
        >>> ac = np.sin(2 * np.pi * np.arange(1000) * 0.01)
        >>> gm = GroundMotion.load_from(source='array', dt=0.01, ac=ac, tag='synthetic')
        >>> gm.npts
        1000
        """
        record = Record(**kwargs)
        return cls(npts=record.npts, dt=record.dt, ac=record.ac, vel=record.vel, disp=record.disp, tag=tag)

    @classmethod
    def list_IMs(cls):
        """
        List all available intensity measures (IMs) and properties.
        
        Provides a comprehensive dictionary of all computable intensity measures
        with human-readable descriptions for documentation and user reference.
        
        Returns
        -------
        dict
            Dictionary mapping IM names (str) to descriptions (str).
            
        See Also
        --------
        compute_intensity_measures : Compute selected IMs.
        to_csv : Export IMs to CSV file.
        
        Notes
        -----
        Spectral quantities (sa, sv, sd) require a periods array parameter.
        Contact developer at Hussaini.smsajad@gmail.com to suggest new IMs.
        
        Examples
        --------
        List all available IMs:
        
        >>> ims_dict = GroundMotion.list_IMs()
        >>> for name, desc in ims_dict.items():
        ...     print(f"{name}: {desc}")
        
        Get just the names:
        
        >>> im_names = list(GroundMotion.list_IMs().keys())
        >>> 'pga' in im_names
        True
        """
        return cls._im_registry.copy()

    # Methods ========================================================
    
    def trim_by_index(self, start_index: int, end_index: int):
        """
        Trim ground motion by index range.
        
        Extracts a subset of the time series between specified indices,
        creating a new GroundMotion instance with reduced duration.

        Parameters
        ----------
        start_index : int
            Starting index (inclusive).
        end_index : int
            Ending index (exclusive).

        Returns
        -------
        GroundMotion
            New instance with trimmed time series.
        
        Raises
        ------
        ValueError
            If indices are out of bounds (start_index < 0 or end_index > npts).
        
        See Also
        --------
        trim_by_slice : Trim using Python slice notation.
        trim_by_energy : Trim based on cumulative energy range.
        
        Examples
        --------
        >>> gm = GroundMotion.load_from(source='array', dt=0.01, ac=np.random.randn(1000))
        >>> gm_trimmed = gm.trim_by_index(100, 900)
        >>> gm_trimmed.npts
        800
        """
        if start_index < 0 or end_index > self.npts:
            raise ValueError("start_index and end_index must be within current npts")
        return self.load_from(source="array", dt=self.dt, ac=self.ac[start_index:end_index], tag=self.tag)

    def trim_by_slice(self, slicer: slice):
        """
        Trim ground motion using Python slice object.
        
        Provides flexible slicing similar to NumPy array indexing with support
        for negative indices and step values.

        Parameters
        ----------
        slicer : slice
            Python slice object (e.g., slice(100, 500, 2)).

        Returns
        -------
        GroundMotion
            New instance with sliced time series.
        
        Raises
        ------
        TypeError
            If slicer is not a slice object.
        
        See Also
        --------
        trim_by_index : Trim with explicit start/end indices.
        
        Examples
        --------
        >>> gm = GroundMotion.load_from(source='array', dt=0.01, ac=np.random.randn(1000))
        >>> gm_trimmed = gm.trim_by_slice(slice(100, 900))
        >>> gm_trimmed.npts
        800
        
        Every other point:
        
        >>> gm_decimated = gm.trim_by_slice(slice(None, None, 2))
        >>> gm_decimated.npts
        500
        """
        if not isinstance(slicer, slice):
            raise TypeError("Expected a slice object")
        return self.load_from(source="array", dt=self.dt, ac=self.ac[slicer], tag=self.tag)

    def trim_by_energy(self, energy_range: tuple[float, float]):
        """
        Trim ground motion to retain specified cumulative energy range.
        
        Identifies time window containing the target energy range (e.g., 5%-95%)
        based on cumulative energy of acceleration. Useful for focusing on
        significant motion and removing weak pre/post-event portions.

        Parameters
        ----------
        energy_range : tuple of float
            (start_fraction, end_fraction) where fractions are in [0, 1].
            Example: (0.05, 0.95) retains central 90% of energy.

        Returns
        -------
        GroundMotion
            New instance trimmed to energy range.
        
        Raises
        ------
        ValueError
            If fractions are not in [0, 1] or start >= end.
        
        See Also
        --------
        ce : Cumulative energy property.
        trim_by_amplitude : Trim based on amplitude threshold.
        
        Notes
        -----
        This method is particularly useful for removing weak motion at the
        beginning and end of records, which can affect baseline correction
        and filtering operations.
        
        Examples
        --------
        Retain central 90% of energy:
        
        >>> gm = GroundMotion.load_from(source='NGA', file='record.at2')
        >>> gm_trimmed = gm.trim_by_energy((0.05, 0.95))
        >>> gm_trimmed.npts < gm.npts
        True
        """
        slicer = signal.slice_energy(self.ce, energy_range)
        return self.load_from(source="array", dt=self.dt, ac=self.ac[slicer], tag=self.tag)

    def trim_by_amplitude(self, threshold: float):
        """
        Trim ground motion based on acceleration amplitude threshold.
        
        Identifies the time window where acceleration exceeds the specified
        threshold, removing weak motion at start and end.

        Parameters
        ----------
        threshold : float
            Amplitude threshold in same units as acceleration (typically g).

        Returns
        -------
        GroundMotion
            New instance trimmed to significant motion window.
        
        See Also
        --------
        trim_by_energy : Alternative trimming based on energy content.
        pga : Peak ground acceleration.
        
        Notes
        -----
        Common practice is to use threshold = 0.05 * PGA for removing
        insignificant portions while preserving strong motion window.
        
        Examples
        --------
        >>> gm = GroundMotion.load_from(source='array', dt=0.01, ac=np.random.randn(1000))
        >>> threshold = 0.05 * gm.pga
        >>> gm_trimmed = gm.trim_by_amplitude(threshold)
        """
        slicer = signal.slice_amplitude(self.ac, threshold)
        return self.load_from(source="array", dt=self.dt, ac=self.ac[slicer], tag=self.tag)
    
    def taper(self, alpha: float = 0.05):
        """
        Apply Tukey window tapering to ground motion.
        
        Smoothly tapers the beginning and end of the time series to zero using
        a Tukey (tapered cosine) window. Reduces spectral leakage in frequency
        domain analysis and prevents edge effects in filtering.

        Parameters
        ----------
        alpha : float, optional
            Taper fraction in [0, 1]. Fraction of window inside cosine tapered region.
            - 0: Rectangular window (no tapering)
            - 1: Hann window (full taper)
            - 0.05: Default, tapers 5% at each end (default is 0.05).

        Returns
        -------
        GroundMotion
            New instance with tapered acceleration.
        
        See Also
        --------
        butterworth_filter : Frequency domain filtering.
        
        Notes
        -----
        Tapering is recommended before applying Fourier transforms or filters
        to minimize edge discontinuities.
        
        Examples
        --------
        Apply 5% taper (default):
        
        >>> gm = GroundMotion.load_from(source='array', dt=0.01, ac=np.random.randn(1000))
        >>> gm_tapered = gm.taper()
        
        Apply 10% taper:
        
        >>> gm_tapered = gm.taper(alpha=0.10)
        """
        new_ac = signal.taper(self.ac, alpha)
        return self.load_from(source="array", dt=self.dt, ac=new_ac, tag=self.tag)
    
    def butterworth_filter(self, bandpass_freqs: tuple[float, float], order: int = 4):
        """
        Apply Butterworth bandpass filter using second-order sections (SOS).
        
        Zero-phase Butterworth filter for frequency content selection without
        introducing phase distortion. Uses SOS format for improved numerical
        stability compared to transfer function representation.

        Parameters
        ----------
        bandpass_freqs : tuple of float
            (low_freq, high_freq) in Hz. Defines passband range.
        order : int, optional
            Filter order controlling steepness of rolloff (default is 4).
            Higher orders give sharper cutoffs but may introduce instability.

        Returns
        -------
        GroundMotion
            New instance with filtered acceleration.
        
        Raises
        ------
        ValueError
            If low_freq >= high_freq or frequencies exceed Nyquist limit.
        
        See Also
        --------
        taper : Recommended before filtering to reduce edge effects.
        fas : Fourier amplitude spectrum for frequency content inspection.
        
        Notes
        -----
        The filter is applied using scipy.signal.butter with second-order sections
        (SOS) format via sosfilt for improved numerical stability. This approach
        avoids numerical issues that can occur with high-order filters in transfer
        function format. Velocity and displacement are recomputed from filtered
        acceleration.
        
        High-frequency cutoff is automatically limited to 99% of Nyquist frequency
        to prevent aliasing issues.
        
        Examples
        --------
        Apply 0.1-25 Hz bandpass:
        
        >>> gm = GroundMotion.load_from(source='NGA', file='record.at2')
        >>> gm_filtered = gm.butterworth_filter((0.1, 25.0), order=4)
        
        High-pass filter above 0.5 Hz:
        
        >>> gm_highpass = gm.butterworth_filter((0.5, 50.0))
        """
        new_ac = signal.butterworth_filter(self.dt, self.ac, *bandpass_freqs, order)
        return self.load_from(source="array", dt=self.dt, ac=new_ac, tag=self.tag)
    
    def baseline_correction(self, degree: int = 1):
        """
        Apply polynomial baseline correction.
        
        Removes long-period drift by fitting and subtracting a polynomial trend
        from acceleration. Common preprocessing step for integrating to velocity
        and displacement.

        Parameters
        ----------
        degree : int, optional
            Polynomial degree for trend fitting (default is 1).
            - 0: Remove mean (DC offset)
            - 1: Remove linear trend
            - 2: Remove quadratic trend

        Returns
        -------
        GroundMotion
            New instance with corrected acceleration.
        
        See Also
        --------
        butterworth_filter : Alternative approach using high-pass filtering.
        
        Notes
        -----
        Baseline correction is essential when acceleration records show velocity
        or displacement drift after integration. Linear correction (degree=1) is
        most common in practice.
        
        Examples
        --------
        Remove linear trend:
        
        >>> gm = GroundMotion.load_from(source='array', dt=0.01, ac=data)
        >>> gm_corrected = gm.baseline_correction(degree=1)
        
        Remove mean only:
        
        >>> gm_demeaned = gm.baseline_correction(degree=0)
        """
        new_ac = signal.baseline_correction(self.ac, degree)
        return self.load_from(source="array", dt=self.dt, ac=new_ac, tag=self.tag)
    
    def resample(self, dt: float):
        """
        Resample to new time step using Fourier method.
        
        Changes time step by resampling in frequency domain, preserving
        frequency content up to new Nyquist frequency.

        Parameters
        ----------
        dt : float
            New time step in seconds.

        Returns
        -------
        GroundMotion
            New instance with resampled time step.
        
        Raises
        ------
        ValueError
            If dt <= 0.
        
        Notes
        -----
        Uses FFT-based resampling which preserves frequency content accurately
        but may introduce artifacts if new_dt is much larger than original.
        For upsampling (smaller dt), consider interpolation methods instead.
        
        Examples
        --------
        Downsample from 0.005s to 0.01s:
        
        >>> gm = GroundMotion.load_from(source='array', dt=0.005, ac=data)
        >>> gm_resampled = gm.resample(dt=0.01)
        >>> gm_resampled.dt
        0.01
        """
        _, dt_new, ac_new = signal.resample(self.dt, dt, self.ac)
        return self.load_from(source="array", dt=dt_new, ac=ac_new, tag=self.tag)
    
    def response_spectra(self, periods: np.ndarray, damping: float = 0.05):
        """
        Calculate response spectra for given periods and damping.
        
        Computes spectral displacement (Sd), velocity (Sv), and acceleration (Sa)
        for elastic single-degree-of-freedom oscillators.

        Parameters
        ----------
        periods : ndarray
            Array of natural periods in seconds.
        damping : float, optional
            Damping ratio (default is 0.05 for 5% damping).

        Returns
        -------
        sd : ndarray
            Spectral displacement in cm.
        sv : ndarray
            Spectral velocity in cm/s.
        sa : ndarray
            Spectral acceleration in g.
        
        See Also
        --------
        vsi : Velocity spectrum intensity.
        asi : Acceleration spectrum intensity.
        
        Notes
        -----
        Response spectra are computed using Nigam-Jennings method with
        numerical integration. Results represent peak absolute response of
        linear oscillators.
        
        Examples
        --------
        >>> gm = GroundMotion.load_from(source='NGA', file='record.at2')
        >>> periods = np.logspace(-2, 1, 50)  # 0.01 to 10 seconds
        >>> sd, sv, sa = gm.response_spectra(periods, damping=0.05)
        >>> sa_at_1s = sa[np.argmin(np.abs(periods - 1.0))]
        """
        return signal.response_spectra(self.dt, self.ac, period=periods, zeta=damping)
    
    def compute_intensity_measures(self, ims: list[str], periods: np.ndarray = None) -> dict:
        """
        Compute selected intensity measures.
        
        Batch computation of multiple IMs with optimized calculation of
        spectral quantities.

        Parameters
        ----------
        ims : list of str
            IM names to compute (e.g., ['pga', 'sa', 'cav']).
            Use list_IMs() for available options.
        periods : ndarray, optional
            Periods for spectral IMs (sa, sv, sd). Required if any spectral IM requested.

        Returns
        -------
        dict
            Dictionary with IM names as keys. Spectral IMs have keys like 'sa_0.200'.
            Values are floats (single record) or arrays (multiple records).
        
        Raises
        ------
        ValueError
            If periods not provided when spectral IMs requested.
        AttributeError
            If invalid IM name specified.
        
        See Also
        --------
        list_IMs : List all available IMs.
        to_csv : Export computed IMs to file.
        
        Examples
        --------
        >>> gm = GroundMotion.load_from(source='NGA', file='record.at2')
        >>> ims = gm.compute_intensity_measures(['pga', 'pgv', 'cav'])
        >>> ims['pga']
        0.521
        
        With spectral quantities:
        
        >>> periods = np.array([0.2, 0.5, 1.0])
        >>> ims = gm.compute_intensity_measures(['pga', 'sa'], periods=periods)
        >>> ims['sa_0.200']
        0.842
        """
        periods = np.asarray(periods) if periods is not None else None
        results = {}
        
        # Determine if we have multiple records
        if self.ac.ndim == 1:
            n_records = 1
        else:
            n_records = self.ac.shape[0]

        # Pre-compute spectra if requested (optimization)
        spectral_ims = [im for im in ims if im.lower() in ("sa", "sv", "sd")]
        spectral_data = {}
        
        if spectral_ims:
            if periods is None:
                raise ValueError("Periods must be provided to compute spectral quantities (sa, sv, sd).")
            # Compute once for all spectral types
            sd, sv, sa = self.response_spectra(periods)
            spectral_data['sd'] = sd
            spectral_data['sv'] = sv
            spectral_data['sa'] = sa
        
        # Iterate and collect data
        for im in ims:
            im_l = im.lower()
            
            # Case A: Spectral IMs
            if im_l in spectral_data:
                data_matrix = spectral_data[im_l]
                
                for idx, period in enumerate(periods):
                    key = f"{im_l}_{period:.3f}"
                    if n_records == 1:
                        results[key] = data_matrix[idx]
                    else:
                        results[key] = data_matrix[:, idx]

            # Case B: Fourier Amplitude Spectra
            elif im_l == "fas":
                 freqs = self.freq
                 attr = self.fas
                 for idx, freq in enumerate(freqs):
                     key = f"fas_{freq:.3f}"
                     if n_records == 1:
                         results[key] = attr[idx]
                     else:
                         results[key] = attr[:, idx]

            # Case C: Scalar IMs
            else:
                attr = getattr(self, im_l)
                results[im_l] = attr

        return results

    def to_csv(self, filename: str, ims: list[str], periods: np.ndarray = None):
        """
        Export intensity measures to CSV file.
        
        Writes computed IMs to comma-separated values format suitable for
        further analysis or database storage.

        Parameters
        ----------
        filename : str
            Output file path.
        ims : list of str
            IM names to export.
        periods : ndarray, optional
            Periods for spectral IMs.
        
        Raises
        ------
        ValueError
            If periods not provided when spectral IMs requested.
        IOError
            If file cannot be written.
        
        See Also
        --------
        compute_intensity_measures : Underlying computation method.
        
        Examples
        --------
        >>> gm = GroundMotion.load_from(source='NGA', file='record.at2')
        >>> periods = np.array([0.2, 0.5, 1.0, 2.0])
        >>> gm.to_csv('output.csv', ims=['pga', 'pgv', 'sa'], periods=periods)
        """
        data = self.compute_intensity_measures(ims, periods)
        
        fieldnames = list(data.keys())
        first_val = next(iter(data.values()))
        
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
        Compare with another ground motion using goodness-of-fit metrics.
        
        Quantifies similarity between this ground motion and a target/model
        using specified intensity measures.

        Parameters
        ----------
        other : GroundMotion
            Target ground motion for comparison.
        ims : list of str
            IM names to compare.
        periods : ndarray, optional
            Periods for spectral IMs.
        method : str, optional
            Comparison metric: 'gof' (goodness of fit) or 're' (relative error).
            Default is 'gof'.

        Returns
        -------
        dict
            Comparison scores for each IM. Lower is better for 'gof',
            closer to 0 is better for 're'.
        
        Raises
        ------
        ValueError
            If method is not recognized.
        
        See Also
        --------
        goodness_of_fit : GOF calculation method.
        relative_error : Relative error calculation.
        
        Notes
        -----
        Goodness-of-fit (GOF) metric from Anderson (2004) combines bias and
        variance into single score. Values < 1 indicate good agreement.
        
        Examples
        --------
        >>> target = GroundMotion.load_from(source='NGA', file='target.at2')
        >>> synthetic = GroundMotion.load_from(source='array', dt=0.01, ac=simulated)
        >>> scores = synthetic.compare(target, ims=['pga', 'pgv', 'sa'], 
        ...                            periods=np.array([0.2, 1.0, 2.0]))
        >>> scores['pga']
        0.15
        """
        criterion_map = {'gof': goodness_of_fit, 're': relative_error}
        if method.lower() not in criterion_map:
             raise ValueError(f"Unknown method: {method}. Supported: {list(criterion_map.keys())}")
        
        func = criterion_map[method.lower()]
        
        my_data = self.compute_intensity_measures(ims, periods)
        other_data = other.compute_intensity_measures(ims, periods)
        
        scores = {}
        for key in my_data:
             if key in other_data:
                 scores[key] = func(my_data[key], other_data[key])
                 
        return scores
    
    # Properties ========================================================

    @property
    @register_im('vsi', 'Velocity Spectrum Intensity (0.1-2.5s)')
    def vsi(self):
        """
        Velocity spectrum intensity (0.1-2.5s range).
        
        Integral of pseudo-velocity response spectrum over period range
        0.1-2.5 seconds with 5% damping. Correlates with damage potential.

        Returns
        -------
        float
            VSI value.
        
        See Also
        --------
        asi : Acceleration spectrum intensity.
        dsi : Displacement spectrum intensity.
        
        where PSV is pseudo-spectral velocity at 5% damping.
        """
        return self.spectrum_intensity[1]
    
    @property
    @register_im('asi', 'Acceleration Spectrum Intensity (0.1-2.5s)')
    def asi(self):
        """
        Acceleration spectrum intensity (0.1-2.5s range).
        
        Integral of acceleration response spectrum over period range
        0.1-2.5 seconds with 5% damping.

        Returns
        -------
        float
            ASI value.
        
        See Also
        --------
        vsi : Velocity spectrum intensity.
        dsi : Displacement spectrum intensity.
        """
        return self.spectrum_intensity[2]
    
    @property
    @register_im('dsi', 'Displacement Spectrum Intensity (0.1-2.5s)')
    def dsi(self):
        """
        Displacement spectrum intensity (0.1-2.5s range).
        
        Integral of displacement response spectrum over period range
        0.1-2.5 seconds with 5% damping.

        Returns
        -------
        float
            DSI value.
        
        See Also
        --------
        vsi : Velocity spectrum intensity.
        asi : Acceleration spectrum intensity.
        """
        return self.spectrum_intensity[0]
    
    @cached_property
    @register_im('t', 'Time array')
    def t(self):
        """
        Time array corresponding to recorded points.

        Returns
        -------
        ndarray
            Time values from 0 to (npts-1)*dt.
        
        """
        return signal.time(self.npts, self.dt)

    @cached_property
    @register_im('freq', 'Frequency array (for FAS)')
    def freq(self):
        """
        Frequency array for Fourier transform.

        Returns
        -------
        ndarray
            Frequencies from 0 to Nyquist frequency.
        
        See Also
        --------
        fas : Fourier amplitude spectrum.
        
        """
        return signal.frequency(self.npts, self.dt)
    
    @cached_property
    @register_im('fas', 'Fourier Amplitude Spectrum of acceleration')
    def fas(self):
        """
        Fourier amplitude spectrum of acceleration.

        Returns
        -------
        ndarray
            Amplitude values at frequencies given by freq property.
        
        See Also
        --------
        freq : Corresponding frequency array.
        fps : Fourier phase spectrum.
        fas_vel : FAS of velocity.
        
        Notes
        -----
        Single-sided spectrum (positive frequencies only).
        """
        return signal.fas(self.dt, self.ac)
    
    @cached_property
    @register_im('fas_vel', 'FAS of Velocity')
    def fas_vel(self):
        """
        Fourier amplitude spectrum of velocity.

        Returns
        -------
        ndarray
            Amplitude values at frequencies given by freq property.
        
        See Also
        --------
        fas : FAS of acceleration.
        fas_disp : FAS of displacement.
        """
        return signal.fas(self.dt, self.vel)

    @cached_property
    @register_im('fas_disp', 'FAS of Displacement')
    def fas_disp(self):
        """
        Fourier amplitude spectrum of displacement.

        Returns
        -------
        ndarray
            Amplitude values at frequencies given by freq property.
        
        See Also
        --------
        fas : FAS of acceleration.
        fas_vel : FAS of velocity.
        """
        return signal.fas(self.dt, self.disp)
    
    @cached_property
    @register_im('fps', 'Fourier Phase Spectrum of acceleration')
    def fps(self):
        """
        Fourier phase spectrum of acceleration (unwrapped).
        
        Phase angle of Fourier transform with unwrapping to ensure
        continuity across -π to π boundaries.

        Returns
        -------
        ndarray
            Unwrapped phase values in radians.
        
        See Also
        --------
        fas : Fourier amplitude spectrum.
        
        Notes
        -----
        Phase unwrapping removes 2π discontinuities using numpy.unwrap,
        essential for phase velocity analysis and signal reconstruction.
        """
        return signal.fps(self.ac)

    @cached_property
    @register_im('ce', 'Cumulative Energy')
    def ce(self):
        """
        Cumulative energy of acceleration time series.
        
        Running integral of squared acceleration, representing energy
        accumulation over time (Arias intensity concept).

        Returns
        -------
        ndarray
            Cumulative energy at each time point.
        
        See Also
        --------
        trim_by_energy : Trim based on energy range.
        cav : Cumulative absolute velocity.
        
        """
        return signal.ce(self.dt, self.ac)
    
    @cached_property
    @register_im('le_ac', 'Mean Local Extrema of Acceleration')
    def le_ac(self):
        """
        Mean local extrema of acceleration.
        
        Average number of peaks and valleys in acceleration time series.

        Returns
        -------
        ndarray
            Cumulative number of local extrema.
        
        See Also
        --------
        le_vel : Local extrema of velocity.
        le_disp : Local extrema of displacement.
        pga : Peak ground acceleration.
        """
        return signal.le(self.ac)

    @cached_property
    @register_im('le_vel', 'Mean Local Extrema of Velocity')
    def le_vel(self):
        """
        Mean local extrema of velocity.

        Returns
        -------
        ndarray
            Cumulative number of local extrema.
        
        See Also
        --------
        le_ac : Local extrema of acceleration.
        pgv : Peak ground velocity.
        """
        return signal.le(self.vel)

    @cached_property
    @register_im('le_disp', 'Mean Local Extrema of Displacement')
    def le_disp(self):
        """
        Mean local extrema of displacement.

        Returns
        -------
        ndarray
            Cumulative number of local extrema.
        
        See Also
        --------
        le_vel : Local extrema of velocity.
        pgd : Peak ground displacement.
        """
        return signal.le(self.disp)

    @cached_property
    @register_im('zc_ac', 'Zero Crossing of Acceleration')
    def zc_ac(self):
        """
        Zero-crossing of acceleration.
        
        Cumulative number of sign changes in acceleration time series, related to dominant frequency content.

        Returns
        -------
        ndarray
            Cumulative count of zero-crossing.
        
        See Also
        --------
        zc_vel : Zero-crossing of velocity.
        le_ac : Local extrema measure.
        """
        return signal.zc(self.ac)

    @cached_property
    @register_im('zc_vel', 'Zero Crossing of Velocity')
    def zc_vel(self):
        """
        Zero-crossing of velocity.

        Returns
        -------
        ndarray
            Cumulative count of zero-crossing.
        
        See Also
        --------
        zc_ac : Zero-crossing of acceleration.
        """
        return signal.zc(self.vel)

    @cached_property
    @register_im('zc_disp', 'Zero Crossing of Displacement')
    def zc_disp(self):
        """
        Zero-crossing of displacement.

        Returns
        -------
        ndarray
            Cumulative count of zero-crossing.
        
        See Also
        --------
        zc_vel : Zero-crossing of velocity.
        """
        return signal.zc(self.disp)

    @cached_property
    @register_im('pmnm_ac', 'Positive Min / Negative Max Ratio of Acceleration')
    def pmnm_ac(self):
        """
        Positive-minima to negative-maxima of acceleration.

        Returns
        -------
        ndarray
            Cumulative number of positive valley and negative peaks.
        """
        return signal.pmnm(self.ac)

    @cached_property
    @register_im('pmnm_vel', 'Positive Min / Negative Max Ratio of Velocity')
    def pmnm_vel(self):
        """
        Positive-minima to negative-maxima ratio of velocity.

        Returns
        -------
        ndarray
            Cumulative number of positive valley and negative peaks.
        """
        return signal.pmnm(self.vel)

    @cached_property
    @register_im('pmnm_disp', 'Positive Min / Negative Max Ratio of Displacement')
    def pmnm_disp(self):
        """
        Positive-minima to negative-maxima ratio of displacement.

        Returns
        -------
        ndarray
            Cumulative number of positive valley and negative peaks.
        """
        return signal.pmnm(self.disp)

    @cached_property
    @register_im('pga', 'Peak Ground Acceleration')
    def pga(self):
        """
        Peak ground acceleration.
        
        Maximum absolute acceleration value in record.

        Returns
        -------
        float
            PGA value.
        
        See Also
        --------
        pgv : Peak ground velocity.
        pgd : Peak ground displacement.
        
        Examples
        --------
        >>> gm.pga
        0.521
        """
        return signal.peak_abs_value(self.ac)

    @cached_property
    @register_im('pgv', 'Peak Ground Velocity')
    def pgv(self):
        """
        Peak ground velocity.
        
        Maximum absolute velocity value in record.

        Returns
        -------
        float
            PGV value.
        
        See Also
        --------
        pga : Peak ground acceleration.
        pgd : Peak ground displacement.
        """
        return signal.peak_abs_value(self.vel)

    @cached_property
    @register_im('pgd', 'Peak Ground Displacement')
    def pgd(self):
        """
        Peak ground displacement.
        
        Maximum absolute displacement value in record.

        Returns
        -------
        float
            PGD value.
        
        See Also
        --------
        pga : Peak ground acceleration.
        pgv : Peak ground velocity.
        """
        return signal.peak_abs_value(self.disp)
    
    @cached_property
    @register_im('cav', 'Cumulative Absolute Velocity')
    def cav(self):
        """
        Cumulative absolute velocity.
        
        Integral of absolute acceleration over time, damage potential indicator.

        Returns
        -------
        float
            CAV value.
        
        See Also
        --------
        ce : Cumulative energy.
        
        Notes
        -----
        Computed as:
        
        .. math:: CAV = \\int_0^T |a(t)| dt
        
        """
        return signal.cav(self.dt, self.ac)
    
    @cached_property
    def spectrum_intensity(self):
        """
        Spectrum intensities (Sd, Sv, Sa) over 0.1-2.5s period range.
        
        Internal property computing all three spectrum intensity types
        simultaneously with 5% damping.

        Returns
        -------
        tuple of float
            (dsi, vsi, asi) values.
        
        See Also
        --------
        dsi : Displacement spectrum intensity.
        vsi : Velocity spectrum intensity.
        asi : Acceleration spectrum intensity.
        
        Notes
        -----
        Uses 0.05s period increment for numerical integration.
        """
        vsi_tp = np.arange(0.1, 2.5, 0.05)
        sd, sv, sa = signal.response_spectra(self.dt, self.ac, period=vsi_tp, zeta=0.05)
        dsi = np.sum(sd, axis=-1) * 0.05
        vsi = np.sum(sv, axis=-1) * 0.05
        asi = np.sum(sa, axis=-1) * 0.05
        return dsi, vsi, asi


# =============================================================================

class GroundMotionMultiComponent:
    """
    Container for multi-component ground motion data.
    
    Handles 2D or 3D ground motion records by combining horizontal and/or
    vertical components. Provides resultant intensity measures computed from
    vector combination of components.
    
    Parameters
    ----------
    *components : GroundMotion
        Two or three GroundMotion instances (e.g., H1, H2, V components).
        All must have matching dt and npts.
    
    Attributes
    ----------
    components : tuple of GroundMotion
        Individual component records.
    n_components : int
        Number of components (2 or 3).
    t : ndarray
        Time array (from first component).
    dt : float
        Time step (from first component).
    npts : int
        Number of points (from first component).
    freq : ndarray
        Frequency array (from first component).
    
    Raises
    ------
    ValueError
        If less than 2 components provided or if components have mismatched
        time parameters.
    
    See Also
    --------
    GroundMotion : Single-component ground motion container.
    
    Examples
    --------
    Create 2-component horizontal ground motion:
    
    >>> gm_h1 = GroundMotion.load_from(source='NGA', file='H1.at2')
    >>> gm_h2 = GroundMotion.load_from(source='NGA', file='H2.at2')
    >>> gm_2d = GroundMotionMultiComponent(gm_h1, gm_h2)
    >>> gm_2d.pga  # Resultant PGA
    0.612
    
    Create 3-component ground motion:
    
    >>> gm_v = GroundMotion.load_from(source='NGA', file='V.at2')
    >>> gm_3d = GroundMotionMultiComponent(gm_h1, gm_h2, gm_v)
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
    
    @cached_property
    def ac(self):
        """
        Resultant acceleration magnitude across all components.
        
        Vector sum of component accelerations at each time point.

        Returns
        -------
        ndarray
            Resultant acceleration time series.
        
        Notes
        -----
        Computed as:
        
        .. math:: a_{res}(t) = \\sqrt{\\sum_i a_i^2(t)}
        """
        return np.sqrt(np.sum([gm.ac ** 2 for gm in self.components], axis=0))
    
    @cached_property
    def vel(self):
        """
        Resultant velocity magnitude across all components.

        Returns
        -------
        ndarray
            Resultant velocity time series.
        """
        return np.sqrt(np.sum([gm.vel ** 2 for gm in self.components], axis=0))
    
    @cached_property
    def disp(self):
        """
        Resultant displacement magnitude across all components.

        Returns
        -------
        ndarray
            Resultant displacement time series.
        """
        return np.sqrt(np.sum([gm.disp ** 2 for gm in self.components], axis=0))

    @cached_property
    def ce(self):
        """
        Total cumulative energy summed across components.

        Returns
        -------
        ndarray
            Combined cumulative energy array.
        """
        return np.sum([gm.ce for gm in self.components], axis=0)
    
    @cached_property
    def fas(self):
        """
        Resultant Fourier amplitude spectrum across components.

        Returns
        -------
        ndarray
            Combined FAS magnitude.

        Notes
        -----
        Computed as:
        
        ..math:: FAS_{res}(f) = \\sqrt{\\sum_i FAS_i^2(f)}            

        """
        return np.sqrt(np.sum([gm.fas ** 2 for gm in self.components], axis=0))
    
    @cached_property
    def fas_vel(self):
        """
        Resultant FAS of velocity across components.

        Returns
        -------
        ndarray
            Combined FAS magnitude.
        """
        return np.sqrt(np.sum([gm.fas_vel ** 2 for gm in self.components], axis=0))
    
    @cached_property
    def fas_disp(self):
        """
        Resultant FAS of displacement across components.

        Returns
        -------
        ndarray
            Combined FAS magnitude.
        """
        return np.sqrt(np.sum([gm.fas_disp ** 2 for gm in self.components], axis=0))
    
    @cached_property
    def pga(self):
        """
        Peak resultant ground acceleration.

        Returns
        -------
        float
            Resultant PGA.
        """
        return signal.peak_abs_value(self.ac)
    
    @cached_property
    def pgv(self):
        """
        Peak resultant ground velocity.

        Returns
        -------
        float
            Resultant PGV.
        """
        return signal.peak_abs_value(self.vel)
    
    @cached_property
    def pgd(self):
        """
        Peak resultant ground displacement.

        Returns
        -------
        float
            Resultant PGD.
        """
        return signal.peak_abs_value(self.disp)