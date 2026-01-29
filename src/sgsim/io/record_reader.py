"""Ground motion record reader supporting multiple file formats."""
import numpy as np

from . import reading_tools
from ..motion.signal import integrate, time


class Record:
    """
    Ground motion record from various file formats.

    Reads acceleration data from file and computes velocity and displacement.

    Parameters
    ----------
    source : str
        Data source format: 'NGA', 'ESM', 'COL', 'RAW', 'COR' for files,
        'Array' for direct array input.
    file : str, optional
        Path to the file (for file-based sources).
    filename : str, optional
        Filename inside zip archive.
    zip_file : str, optional
        Path to zip file containing filename.
    dt : float, optional
        Time step (required for 'Array' source).
    ac : ndarray, optional
        Acceleration data (required for 'Array' source).
    skiprows : int, optional
        Rows to skip at file start (default: 1).
    scale : float, optional
        Scaling factor for acceleration (default: 1.0).

    Attributes
    ----------
    ac : ndarray
        Acceleration time series.
    vel : ndarray
        Velocity time series.
    disp : ndarray
        Displacement time series.
    t : ndarray
        Time vector.
    dt : float
        Time step.
    npts : int
        Number of data points.

    Examples
    --------
    >>> record = Record(source='NGA', file='RSN1_HELENA.AT2')
    >>> record.ac  # acceleration array
    """

    def __init__(self, **kwargs):
        source = kwargs.get('source')
        if source is None:
            raise ValueError("'source' parameter is required.")
        self.source = source.lower()

        self.file = kwargs.get('file')
        self.filename = kwargs.get('filename')
        self.zip_file = kwargs.get('zip_file')

        self.dt = kwargs.get('dt')
        self.ac = kwargs.get('ac')

        self.skiprows = kwargs.get('skiprows', 1)
        self.scale = kwargs.get('scale', 1.0)

        self._read_file()

    def _read_file(self):
        """Read file content and parse using appropriate parser."""
        if self.filename and self.zip_file:
            self.contents = reading_tools.read_file_from_zip(self.filename, self.zip_file)
        elif self.file:
            self.contents = reading_tools.read_file(self.file)

        parser_method = getattr(self, f"_parser_{self.source}", None)
        if not callable(parser_method):
            raise ValueError(f"Unsupported source: {self.source}")
        parser_method()
        self._process_motion()

    def _process_motion(self):
        """Compute time, velocity, and displacement from acceleration."""
        self.ac = self.ac.astype(np.float64, copy=False) * self.scale
        self.npts = self.ac.shape[-1]
        self.t = time(self.npts, self.dt)
        self.vel = integrate(self.dt, self.ac)
        self.disp = integrate(self.dt, self.vel)

    def _parser_nga(self):
        """Parse NGA record file (.AT2)."""
        rec_info = self.contents[3].lower().split()
        rec_data = self.contents[4:-1]

        dt_idx = rec_info.index('dt=') + 1
        self.dt = round(float(rec_info[dt_idx].rstrip('sec,')), 3)
        self.ac = np.loadtxt(rec_data).flatten()

    def _parser_esm(self):
        """Parse ESM record file (.ASC)."""
        rec_data = self.contents[64:-1]
        self.dt = round(float(self.contents[28].split()[1]), 3)
        self.ac = np.loadtxt(rec_data).flatten()

    def _parser_col(self):
        """Parse double-column record file [t, ac]."""
        col_data = np.loadtxt(self.contents, skiprows=self.skiprows)
        self.dt = round(col_data[1, 0] - col_data[0, 0], 3)
        self.ac = col_data[:, 1]

    def _parser_array(self):
        """Parse data from numpy array."""
        if self.ac is None or self.dt is None:
            raise ValueError("'ac' and 'dt' must be provided for Array source.")

    def _parser_raw(self):
        """Parse RAW record file (.RAW)."""
        rec_info = self.contents[16].split()
        rec_data = self.contents[25:-2]
        self.dt = round(float(rec_info[rec_info.index('period:') + 1].rstrip('s,')), 3)
        self.ac = np.loadtxt(rec_data).flatten()

    def _parser_cor(self):
        """Parse COR record file (.COR)."""
        rec_info = self.contents[16].split()
        rec_data = self.contents[29:-1]
        endline = rec_data.index('-> corrected velocity time histories\n') - 2
        rec_data = rec_data[0:endline]
        self.dt = round(float(rec_info[rec_info.index('period:') + 1].rstrip('s,')), 3)
        self.ac = np.loadtxt(rec_data).flatten()
