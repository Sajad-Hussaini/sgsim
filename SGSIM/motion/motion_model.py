import numpy as np
import pandas as pd
from . import signal_analysis
from . import signal_processing
from ..file_reading.record_reader import RecordReader

class Motion:
    """
    This Class allows to
        Provide functionality to describe ground motions
        using direct input data or from a file
    """
    def __init__(self, npts, dt, t, ac, vel=None, disp=None):
        """
        To read data from a file use from_file method otherwise
        provide input data directly

        npts:  number of data points (array length)
        dt:    time step
        t:     time array
        ac:    acceleration array
        vel:   velocity array
        disp:  displacement array
        """
        self.npts = npts
        self.dt = dt
        self.t = t
        self.ac = ac
        self.vel = vel
        self.disp = disp

    @classmethod
    def from_file(cls, file_path: str | tuple[str, str], source: str, **kwargs):
        """
        Read data from a file or a filename in a zip

        file_path: path to the file or a filename in a zip
        source:    source type (e.g., 'NGA')
        kwargs:    additional keyword arguments for RecordReader (e.g., 'skiprows')
        """
        record = RecordReader(file_path, source, **kwargs)
        return cls(npts=record.npts, dt=record.dt, t=record.t, ac=record.ac)

    @classmethod
    def from_model(cls, model):
        """ Read data from a stochastic model """
        return cls(npts=model.npts, dt=model.dt, t=model.t, ac=model.ac, vel=model.vel, disp=model.disp)

    def get_properties(self, period_range: tuple[float, float, float] = (0.04, 10.01, 0.01)):
        """ Calculate ground motion properties """
        self.tp = np.arange(*period_range)
        self.sd, self.sv, self.sa = signal_analysis.get_spectra(self.dt, self.ac if self.ac.ndim==2 else self.ac[None, :], period=self.tp, zeta=0.05)
        self.ce = signal_analysis.get_ce(self.dt, self.ac)

        self.mzc_ac = signal_analysis.get_mzc(self.ac)
        self.mzc_vel = signal_analysis.get_mzc(self.vel)
        self.mzc_disp = signal_analysis.get_mzc(self.disp)

        self.mle_ac = signal_analysis.get_mle(self.ac)
        self.mle_vel = signal_analysis.get_mle(self.vel)
        self.mle_disp = signal_analysis.get_mle(self.disp)

        self.pmnm_ac = signal_analysis.get_pmnm(self.ac)
        self.pmnm_vel = signal_analysis.get_pmnm(self.vel)
        self.pmnm_disp = signal_analysis.get_pmnm(self.disp)

        self.pga = signal_analysis.get_pga(self.ac)
        self.pgv = signal_analysis.get_pgv(self.dt, self.ac)
        self.pgd = signal_analysis.get_pgd(self.dt, self.ac)

        self.fas = signal_analysis.get_fas(self.npts, self.ac)
        self.freq = signal_analysis.get_freq(self.npts, self.dt)
        self.freq_mask = signal_analysis.get_freq_mask(self.freq, (0.1, 25.0))
        self.fas_star = signal_processing.moving_average(self.fas, 9)[..., self.freq_mask]
        return self

    def set_target_range(self, target_range: tuple[float, float], bandpass_freqs: tuple[float, float]=None):
        """
        Define a target range for the ground motion (often for a real target motion)
            using total energy percentages i.e., (0.001, 0.999)
        Define a bandpass filtering if required
            bandpass freqs as (lowcut, highcut) or None
        """
        self.energy_mask = signal_analysis.get_energy_mask(self.dt, self.ac, target_range)
        self.t = np.round(self.t[self.energy_mask] - self.t[self.energy_mask][0], 3)
        self.npts = len(self.t)
        if bandpass_freqs is not None:
            self.ac = self.ac[self.energy_mask]
            self.ac = signal_processing.bandpass_filter(self.dt, self.ac, bandpass_freqs[0], bandpass_freqs[1])
            self.disp = signal_analysis.get_disp(self.dt, self.ac)
            self.vel = signal_analysis.get_vel(self.dt, self.ac)
        else:
            self.disp = signal_analysis.get_disp(self.dt, self.ac)[self.energy_mask]
            self.vel = signal_analysis.get_vel(self.dt, self.ac)[self.energy_mask]
            # update ac after obtaining vel and disp for the target range
            self.ac = self.ac[self.energy_mask]
        self.get_properties()
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
