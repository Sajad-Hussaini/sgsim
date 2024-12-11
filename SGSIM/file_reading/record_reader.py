import numpy as np
from .read_tool import read_file, read_file_from_zip

class RecordReader:
    " A class to read records from different ground motion databases "
    def __init__(self, file_path: str | tuple[str, str], source: str, **kwargs):
        """
        Read records from different ground motion databases.


        Parameters
        ----------
        file_path : str | tuple[str, str]
            a signle file-path or a tuple as (filename, zip-path)
        source : str
            'nga' for PEER NGA format
            'esm' for ESM format
            'col' for two-column format [t, ac]
            'raw' for RAW format
            'cor' for COR format
        **kwargs : str
            'skiprows' (default 0) for col format

        Returns
        -------
        None.

        """
        self.file_path = file_path
        self.source = source.lower()
        self.skip_rows = kwargs.get('skiprows', 1)
        self._read_file()

    def _read_file(self):
        """
        Read file content line by line and use the right parser to read data.
        """
        if isinstance(self.file_path, tuple) and len(self.file_path) == 2:
            self.file_content = read_file_from_zip(*self.file_path)
        else:
            self.file_content = read_file(self.file_path)

        parser_method = getattr(self, f"parser_{self.source}", None)
        if not callable(parser_method):
            raise ValueError(f"Unsupported source: {self.source}")
        parser_method()
        return self

    def parser_nga(self):
        """
        Reading the NGA record file (.AT2)
        """
        recInfo = self.file_content[3].split()
        recData = self.file_content[4:-1]

        dt_key = 'dt=' if 'dt=' in recInfo else 'DT='
        self.dt = round(float(recInfo[recInfo.index(dt_key) + 1].rstrip('SEC,')), 3)
        self.ac = np.loadtxt(recData).flatten()
        self.npts = len(self.ac)
        self.t = np.arange(self.npts) * self.dt
        return self

    def parser_esm(self):
        """
        Reading the ESM records (.ASC)
        """
        recData = self.file_content[64:-1]
        self.dt = round(float(self.file_content[28].split()[1]), 3)
        self.ac = np.loadtxt(recData).flatten()
        self.npts = len(self.ac)
        self.t = np.arange(self.npts) * self.dt
        return self

    def parser_col(self):
        """
        Reading the double-column record file [t, ac]
        """
        col_data = np.loadtxt(self.file_content, skiprows=self.skip_rows)
        self.ac = col_data[:, 1]
        self.t = np.round(col_data[:, 0], 3)
        self.dt = round(self.t[3] - self.t[2], 3)
        self.npts = len(self.ac)
        return self

    def parser_raw(self):
        """
        Reading the RAW files (.RAW)
        """
        recInfo = self.file_content[16].split()
        recData = self.file_content[25:-2]
        self.dt = round(float(recInfo[recInfo.index('period:') + 1].rstrip('s,')), 3)
        self.ac = np.loadtxt(recData).flatten()
        self.npts = len(self.ac)
        self.t = np.arange(self.npts) * self.dt
        return self

    def parser_cor(self):
        """
        Reading the COR files (.COR)
        """
        recInfo = self.file_content[16].split()
        recData = self.file_content[29:-1]
        endline = recData.index('-> corrected velocity time histories\n') - 2
        recData = recData[0:endline]
        self.dt = round(float(recInfo[recInfo.index('period:') + 1].rstrip('s,')), 3)
        self.ac = np.loadtxt(recData).flatten()
        self.npts = len(self.ac)
        self.t = np.arange(self.npts) * self.dt
        return self
