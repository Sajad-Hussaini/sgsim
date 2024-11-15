import numpy as np
import zipfile
import io

class RecordReader:
    def __init__(self, file_path: str | tuple[str, str], source: str, **kwargs):
        """
        Read records from different ground motion databases.


        Parameters
        ----------
        file_path : str | tuple[str, str]
            a signle file path or a tuple as (filename, zip path)
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
        self.source = source
        self.skip_rows = kwargs.get('skiprows', 1)
        self.read_file()

    def read_file(self):
        """
        Read file content and determine the format to parse.
        """
        if isinstance(self.file_path, tuple) and len(self.file_path) == 2:
            self.read_zip_file(*self.file_path)
        else:
            self.read_content()
        reading_methods = {
            'nga': self.read_nga,
            'esm': self.read_esm,
            'col': self.read_col,
            'raw': self.read_raw,
            'cor': self.read_cor}
        reading_method = reading_methods.get(self.source)
        if not reading_method:
            raise ValueError(f'Unsupported source: {self.source}')
        reading_method()
        return self

    def read_zip_file(self, filename, zip_path):
        """
        Read the content of a file within a zip file.
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_file:
                if filename not in zip_file.namelist():
                    raise FileNotFoundError(f'{filename} is not in the zip file.')
                with zip_file.open(filename, 'r') as file:
                    with io.TextIOWrapper(file, encoding='utf-8') as text_file:
                        self.file_content = text_file.readlines()
            return self
        except Exception as e:
            raise IOError(f'Error reading zip file: {str(e)}')

    def read_content(self):
        """
        Read the lines of a file.
        """
        try:
            with open(self.file_path, 'r') as inputfile:
                self.file_content = inputfile.readlines()
            return self
        except Exception as e:
            raise IOError(f'Error reading the record file: {str(e)}')

    def read_nga(self):
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

    def read_col(self):
        """
        Reading the double-column record file [t, ac]
        """
        col_data = np.loadtxt(self.file_content, skiprows=self.skip_rows)
        self.ac = col_data[:, 1]
        self.t = np.round(col_data[:, 0], 3)
        self.dt = round(self.t[3] - self.t[2], 3)
        self.npts = len(self.ac)
        return self

    def read_raw(self):
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

    def read_cor(self):
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

    def read_esm(self):
        """
        Reading the ESM records (.ASC)
        """
        recData = self.file_content[64:-1]
        self.dt = round(float(self.file_content[28].split()[1]), 3)
        self.ac = np.loadtxt(recData).flatten()
        self.npts = len(self.ac)
        self.t = np.arange(self.npts) * self.dt
        return self
