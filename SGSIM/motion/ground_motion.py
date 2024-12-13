import numpy as np
import pandas as pd
from .signal_processing import bandpass_filter
from . import signal_props
from . import signal_processing
from ..file_reading.record_reader import RecordReader

class CoreMotion:
    def __init__(self, npts: int, dt: float, t, ac):
        """
        A class to provide functionality and description for ground motions.

        Parameters
        ----------
        npts : int
            length of time series.
        dt : float
            time step.
        t : TYPE
            time array.
        ac : TYPE
            acceleration array.

        Returns
        -------
        None.

        """
        self.npts = npts
        self.dt = dt
        self.t = t
        self.ac = ac

    def get_properties(self, period_range: tuple[float, float, float] = (0.04, 10.01, 0.01)):
        """ Calculate properties of the ground motion """
        self.tp = np.arange(*period_range)
        self.sd, self.sv, self.sa = signal_props.get_spectra(self.dt, self.ac if self.ac.ndim==2 else self.ac[None, :], period=self.tp, zeta=0.05)
        self.ce = signal_props.get_ce(self.dt, self.ac)

        self.mzc_ac = signal_props.get_mzc(self.ac)
        self.mzc_vel = signal_props.get_mzc(self.vel)
        self.mzc_disp = signal_props.get_mzc(self.disp)

        self.mle_ac = signal_props.get_mle(self.ac)
        self.mle_vel = signal_props.get_mle(self.vel)
        self.mle_disp = signal_props.get_mle(self.disp)

        self.pmnm_ac = signal_props.get_pmnm(self.ac)
        self.pmnm_vel = signal_props.get_pmnm(self.vel)
        self.pmnm_disp = signal_props.get_pmnm(self.disp)

        self.pga = signal_props.get_pga(self.ac)
        self.pgv = signal_props.get_pgv(self.dt, self.ac)
        self.pgd = signal_props.get_pgd(self.dt, self.ac)

        self.fas = signal_props.get_fas(self.npts, self.ac)
        self.freq = signal_props.get_freq(self.npts, self.dt)
        self.slicer_freq = signal_props.get_freq_slice(self.freq, (0.1, 25.0))
        self.fas_star = signal_processing.moving_average(self.fas, 9)[..., self.slicer_freq]
        return self

    def save_to_csv(self, row_data, col_data, filename, label="SA", index_col="Tp"):
        """
        # TODO :A general method to save any array-like data to a CSV file.

        Parameters
        ----------
        row_data : array-like
            The data to be saved in rows of csv.
        col_data : None or array-like
            The first row (columns) of csv.
        filename : str
            The path and name of the file to save the data.
        label : str
            A label to prepend the rows.
        index_col : str
            The label for the index column in the CSV file.

        Returns
        -------
        None
        """
        rows = []
        for i, d in enumerate(row_data):
            row_label = f"{label}_{i + 1}"
            rows.append([row_label] + list(d))
        columns = [index_col] + list[col_data]
        df = pd.DataFrame(rows, columns=columns)
        df.to_csv(filename, index=False)
        return self

    def save_spectra(self, filename: str):
        " To save response spectra and the periods to a csv file"

        data_arrays = [self.sa, self.sv, self.sd]
        property_labels = ["SA", "SV", "SD"]

        rows = []
        for label, data_array in zip(property_labels, data_arrays):
            for i, data in enumerate(data_array):
                row_label = f"{label}_{i + 1}"
                rows.append([row_label] + list(data))

        columns = ["Tp"] + list(self.tp)
        spectra_df = pd.DataFrame(rows, columns=columns)
        spectra_df.to_csv(filename, index=False)
        return self

    def save_fas(self, filename: str):
        " To save FAS and the frequencies to a csv file"

        data_arrays = [self.fas]
        property_labels = ["FAS"]

        rows = []
        for label, data_array in zip(property_labels, data_arrays):
            for i, data in enumerate(data_array):
                row_label = f"{label}_{i + 1}"
                rows.append([row_label] + list(data))

        columns = ["freq"] + list(self.freq)
        motion_df = pd.DataFrame(rows, columns=columns)
        motion_df.to_csv(filename, index=False)
        return self

    def save_motions(self, filename: str):
        " To save ground motion time series and the time to a csv file"

        data_arrays = [self.ac, self.vel, self.disp]
        property_labels = ["AC", "Vel", "Disp"]

        rows = []
        for label, data_array in zip(property_labels, data_arrays):
            for i, data in enumerate(data_array):
                row_label = f"{label}_{i + 1}"
                rows.append([row_label] + list(data))

        columns = ["t"] + list(self.t)
        motion_df = pd.DataFrame(rows, columns=columns)
        motion_df.to_csv(filename, index=False)
        return self

    def save_peak_motions(self, filename: str):
        " To save peak ground motion parameters to a csv file"

        data_arrays = [self.pga, self.pgv, self.pgd]
        property_labels = ["PGA", "PGV", "PGD"]

        rows = []
        for label, data_array in zip(property_labels, data_arrays):
            for i, data in enumerate(data_array):
                row_label = f"{label}_{i + 1}"
                rows.append([row_label, data])

        columns = ["Index", "Value"]
        motion_df = pd.DataFrame(rows, columns=columns)
        motion_df.to_csv(filename, index=False)
        return self

    def save_characteristics(self, filename: str):
        " To save ground motion characteristics (e.g., CE, MZC) and the time to a csv file"

        data_arrays = [self.ce, self.mzc_ac, self.mzc_vel, self.mzc_disp]
        property_labels = ["CE", "MZC_AC", "MZC_Vel", "MZC_Disp"]

        rows = []
        for label, data_array in zip(property_labels, data_arrays):
            for i, data in enumerate(data_array):
                row_label = f"{label}_{i + 1}"
                rows.append([row_label] + list(data))

        columns = ["t"] + list(self.t)
        characteristics_df = pd.DataFrame(rows, columns=columns)
        characteristics_df.to_csv(filename, index=False)
        return self

class TargetMotion(RecordReader, CoreMotion):
    """
    A class to describe a target ground motion from a file
    """
    def __init__(self, file_path: str | tuple[str, str], source: str, **kwargs):
        RecordReader.__init__(self, file_path, source, **kwargs)
        CoreMotion.__init__(self, self.npts, self.dt, self.t, self.ac)

    def set_target_range(self, target_range: tuple[float, float], bandpass_freqs: tuple[float, float]=None):
        """
        Define the target range of the ground motion
        based on a range of total energy percentages i.e., (0.001, 0.999)
        for bandpass filtering use bandpass freqs as (lowcut, highcut)
        otherwise pass None
        """
        self.slicer = signal_props.get_energy_slice(self.dt, self.ac, target_range)
        self.t = np.round(self.t[self.slicer] - self.t[self.slicer][0], 3)
        self.npts = len(self.t)
        if bandpass_freqs is not None:
            self.ac = self.ac[self.slicer]
            self.ac = bandpass_filter(self.dt, self.ac, bandpass_freqs[0], bandpass_freqs[1])
            self.disp = signal_props.get_disp(self.dt, self.ac)
            self.vel = signal_props.get_vel(self.dt, self.ac)
        else:
            self.disp = signal_props.get_disp(self.dt, self.ac)[self.slicer]
            self.vel = signal_props.get_vel(self.dt, self.ac)[self.slicer]
            # update ac after obtaining vel and disp for the target range
            self.ac = self.ac[self.slicer]

        self.get_properties()
        return self

class SimMotion(CoreMotion):
    """
    A class to describe the simulated ground motions from the stochatic model
    """
    def __init__(self, model):
        super().__init__(model.npts, model.dt, model.t, model.ac)
        self.model = model
        self.vel = model.vel
        self.disp = model.disp

    def update(self):
        """update internal attributes to reflect changes in the model."""
        self.ac = self.model.ac
        self.vel = self.model.vel
        self.disp = self.model.disp
        self.get_properties()
        return self