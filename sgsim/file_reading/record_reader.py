import numpy as np
from . import reading_tools
from ..motion.signal_tools import get_time, get_integral

class RecordReader:
    """
    A class to read ground motion time series from various sources.

        Parameters
        ----------
        source : str
            Source type (e.g., 'NGA', 'ESM', 'COL', 'RAW', 'COR', 'Array').
        
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
    Attributes
    ----------
        ac : np.ndarray
            Acceleration time series.
        vel : np.ndarray
            Velocity time series.
        disp : np.ndarray
            Displacement time series.
        t : np.ndarray
            Time vector corresponding to the time series.
        dt : float
            Time step of the ground motion.
        npts : int
            Number of data points in the time series.
        """
    def __init__(self, source: str, **kwargs):
        self.source = source.lower()

        self.file = kwargs.get('file')
        self.filename = kwargs.get('filename')
        self.zip_file = kwargs.get('zip_file')

        self.dt = kwargs.get('dt')
        self.ac = kwargs.get('ac')

        self.skiprows = kwargs.get('skiprows', 1)
        self.scale = kwargs.get('scale', 1)

        self._read_file()

    def _read_file(self):
        """
        Read file content line by line and use the right parser to read data
        """
        if self.filename and self.zip_file:
            self.contents = reading_tools.read_file_from_zip(self.filename, self.zip_file)
        elif self.file:
            self.contents = reading_tools.read_file(self.file)

        parser_method = getattr(self, f"_parser_{self.source}", None)
        if not callable(parser_method):
            raise ValueError(f"Unsupported source: {self.source}")
        parser_method()
        self._process_motion()
        return self
    
    def _process_motion(self):
        """
        Calculates time, velocity, and displacement from acceleration.
        """
        if self.ac is None or self.dt is None:
            raise ValueError("Acceleration ('ac') and time step ('dt') must be set by the parser.")
        self.npts = len(self.ac)
        self.t = get_time(self.npts, self.dt)
        self.vel = get_integral(self.dt, self.ac)
        self.disp = get_integral(self.dt, self.vel)

    def _parser_nga(self):
        """
        Reading the NGA record file (.AT2)
        """
        recInfo = self.contents[3].split()
        recData = self.contents[4:-1]

        dt_key = 'dt=' if 'dt=' in recInfo else 'DT='
        self.dt = round(float(recInfo[recInfo.index(dt_key) + 1].rstrip('SEC,')), 3)
        self.ac = np.loadtxt(recData).flatten() * self.scale
        return self

    def _parser_esm(self):
        """
        Reading the ESM records (.ASC)
        """
        recData = self.contents[64:-1]
        self.dt = round(float(self.contents[28].split()[1]), 3)
        self.ac = np.loadtxt(recData).flatten() * self.scale
        return self

    def _parser_col(self):
        """
        Reading the double-column record file [t, ac]
        """
        col_data = np.loadtxt(self.contents, skiprows=self.skiprows)
        self.dt = round(col_data[1, 0] - col_data[0, 0], 3)
        self.ac = col_data[:, 1] * self.scale
        return self
    
    def _parser_array(self):
        """
        Reading data from a numpy array
        """
        if not self.ac and not self.dt:
            raise ValueError("For 'array' source, 'ac' and 'dt' must be provided in kwargs.")
        return self

    def _parser_raw(self):
        """
        Reading the RAW files (.RAW)
        """
        recInfo = self.contents[16].split()
        recData = self.contents[25:-2]
        self.dt = round(float(recInfo[recInfo.index('period:') + 1].rstrip('s,')), 3)
        self.ac = np.loadtxt(recData).flatten() * self.scale
        return self

    def _parser_cor(self):
        """
        Reading the COR files (.COR)
        """
        recInfo = self.contents[16].split()
        recData = self.contents[29:-1]
        endline = recData.index('-> corrected velocity time histories\n') - 2
        recData = recData[0:endline]
        self.dt = round(float(recInfo[recInfo.index('period:') + 1].rstrip('s,')), 3)
        self.ac = np.loadtxt(recData).flatten() * self.scale
        return self
