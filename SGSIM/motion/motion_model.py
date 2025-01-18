import numpy as np
import pandas as pd
from . import signal_analysis
from . import signal_processing
from ..file_reading.record_reader import RecordReader
import h5py

class Motion:
    """
    This Class allows to
        Provide functionality to describe ground motions
        using direct input data or from a file
    """
    def __init__(self, npts, dt, ac, vel=None, disp=None):
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
        self.ac = ac
        self.vel = vel
        self.disp = disp

    @property
    def t(self):
        if not hasattr(self, '_t'):
            self._t = signal_analysis.get_time(self.npts, self.dt)
        return self._t

    @t.setter
    def t(self, value):
        self._t = value

    @property
    def tp(self):
        if not hasattr(self, '_tp'):
            self.tp = (0.04, 10.04, 0.01)
        return self._tp

    @tp.setter
    def tp(self, period_range: tuple[float, float, float]):
        self._tp = np.arange(*period_range)

    @property
    def sd(self):
        if not hasattr(self, '_sd'):
            self._sd, self._sv, self._sa = signal_analysis.get_spectra(
                self.dt, self.ac if self.ac.ndim == 2 else self.ac[None, :], period=self.tp, zeta=0.05)
        return self._sd

    @property
    def sv(self):
        if not hasattr(self, '_sv'):
            self._sd, self._sv, self._sa = signal_analysis.get_spectra(
                self.dt, self.ac if self.ac.ndim == 2 else self.ac[None, :], period=self.tp, zeta=0.05)
        return self._sv

    @property
    def sa(self):
        if not hasattr(self, '_sa'):
            self._sd, self._sv, self._sa = signal_analysis.get_spectra(
                self.dt, self.ac if self.ac.ndim == 2 else self.ac[None, :], period=self.tp, zeta=0.05)
        return self._sa

    @property
    def ce(self):
        if not hasattr(self, '_ce'):
            self._ce = signal_analysis.get_ce(self.dt, self.ac)
        return self._ce

    @property
    def mzc_ac(self):
        if not hasattr(self, '_mzc_ac'):
            self._mzc_ac = signal_analysis.get_mzc(self.ac)
        return self._mzc_ac

    @property
    def mzc_vel(self):
        if not hasattr(self, '_mzc_vel'):
            self._mzc_vel = signal_analysis.get_mzc(self.vel)
        return self._mzc_vel

    @property
    def mzc_disp(self):
        if not hasattr(self, '_mzc_disp'):
            self._mzc_disp = signal_analysis.get_mzc(self.disp)
        return self._mzc_disp

    @property
    def mle_ac(self):
        if not hasattr(self, '_mle_ac'):
            self._mle_ac = signal_analysis.get_mle(self.ac)
        return self._mle_ac

    @property
    def mle_vel(self):
        if not hasattr(self, '_mle_vel'):
            self._mle_vel = signal_analysis.get_mle(self.vel)
        return self._mle_vel

    @property
    def mle_disp(self):
        if not hasattr(self, '_mle_disp'):
            self._mle_disp = signal_analysis.get_mle(self.disp)
        return self._mle_disp

    @property
    def pmnm_ac(self):
        if not hasattr(self, '_pmnm_ac'):
            self._pmnm_ac = signal_analysis.get_pmnm(self.ac)
        return self._pmnm_ac

    @property
    def pmnm_vel(self):
        if not hasattr(self, '_pmnm_vel'):
            self._pmnm_vel = signal_analysis.get_pmnm(self.vel)
        return self._pmnm_vel

    @property
    def pmnm_disp(self):
        if not hasattr(self, '_pmnm_disp'):
            self._pmnm_disp = signal_analysis.get_pmnm(self.disp)
        return self._pmnm_disp

    @property
    def pga(self):
        if not hasattr(self, '_pga'):
            self._pga = signal_analysis.get_pgp(self.ac)
        return self._pga

    @property
    def pgv(self):
        if not hasattr(self, '_pgv'):
            self._pgv = signal_analysis.get_pgp(self.vel)
        return self._pgv

    @property
    def pgd(self):
        if not hasattr(self, '_pgd'):
            self._pgd = signal_analysis.get_pgp(self.disp)
        return self._pgd

    @property
    def fas(self):
        if not hasattr(self, '_fas'):
            self._fas = signal_analysis.get_fas(self.npts, self.ac)
        return self._fas

    @property
    def freq(self):
        if not hasattr(self, '_freq'):
            self._freq = signal_analysis.get_freq(self.npts, self.dt)
        return self._freq

    @property
    def freq_mask(self):
        if not hasattr(self, '_freq_mask'):
            self.freq_mask = (0.1, 25.0)
        return self._freq_mask

    @freq_mask.setter
    def freq_mask(self, target_range: tuple[float, float]):
        self._freq_mask = signal_analysis.get_freq_mask(self.freq, target_range)

    @property
    def fas_star(self):
        if not hasattr(self, '_fas_star'):
            self._fas_star = signal_processing.moving_average(self.fas, 9)[..., self.freq_mask]
        return self._fas_star

    @property
    def energy_mask(self):
        if not hasattr(self, '_energy_mask'):
            raise ValueError("The range based on energy has not been set (e.g., 0.001, 0.999).")
        return self._energy_mask

    @energy_mask.setter
    def energy_mask(self, target_range: tuple[float, float]):
        self._energy_mask = signal_analysis.get_energy_mask(self.dt, self.ac, target_range)

    def set_target_range(self, target_range: tuple[float, float], bandpass_freqs: tuple[float, float]=None):
        """
        Define a target range for the ground motion (often for a real target motion)
            using total energy percentages i.e., (0.001, 0.999)
        Define a bandpass filtering if required
            bandpass freqs as (lowcut, highcut) or None
        """
        self.energy_mask = target_range
        self.t = np.round(self.t[self.energy_mask] - self.t[self.energy_mask][0], 3)
        self.npts = len(self.t)
        if bandpass_freqs is not None:
            # Cut and filter ac, then obtain vel and disp
            self.ac = self.ac[self.energy_mask]
            self.ac = signal_processing.bandpass_filter(self.dt, self.ac, bandpass_freqs[0], bandpass_freqs[1])
            self.vel = signal_analysis.get_integral(self.dt, self.ac)
            self.disp = signal_analysis.get_integral(self.dt, self.vel)
        else:
            # Obtain vel and disp from full ac, then cut them by mask
            self.vel = signal_analysis.get_integral(self.dt, self.ac)
            self.disp = signal_analysis.get_integral(self.dt, self.vel)
            # Now update ac, vel, disp
            self.ac = self.ac[self.energy_mask]
            self.vel = self.vel[self.energy_mask]
            self.disp = self.disp[self.energy_mask]
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

    def save_simfile(self, filename: str):
        """
        Save simulation data to an HDF5 file with improved structure.

        filename: The name of the HDF5 file to save the data to.
        """
        with h5py.File(filename, 'w') as hdf:
            # Save spectra
            spectra_group = hdf.create_group('spectra')
            spectra_group.create_dataset('tp', data=self.tp)
            spectra_group.create_dataset('sa', data=self.sa)
            spectra_group.create_dataset('sv', data=self.sv)
            spectra_group.create_dataset('sd', data=self.sd)
            spectra_group.create_dataset('freq', data=self.freq)
            spectra_group.create_dataset('fas', data=self.fas)
            spectra_group['tp'].attrs['units'] = 's'
            spectra_group['sa'].attrs['units'] = 'g'
            spectra_group['sv'].attrs['units'] = 'cm/s'
            spectra_group['sd'].attrs['units'] = 'cm'
            spectra_group['freq'].attrs['units'] = 'Hz'
            spectra_group['fas'].attrs['units'] = 'g/sqrt(Hz)'
            spectra_group.attrs['description'] = 'Fourier Amplitude and Response Spectra using 5% damping ratio'

            # Save ground motion time series
            motions_group = hdf.create_group('motions')
            motions_group.create_dataset('dt', data=self.dt)
            motions_group.create_dataset('npts', data=self.npts)
            motions_group.create_dataset('t', data=self.t)
            motions_group.create_dataset('ac', data=self.ac)
            motions_group.create_dataset('vel', data=self.vel)
            motions_group.create_dataset('disp', data=self.disp)
            motions_group['t'].attrs['units'] = 's'
            motions_group['ac'].attrs['units'] = 'g'
            motions_group['vel'].attrs['units'] = 'cm/s'
            motions_group['disp'].attrs['units'] = 'cm'
            motions_group.attrs['description'] = 'Time series of ground motions'

            # Save peak motions
            peak_motions_group = hdf.create_group('peak_motions')
            peak_motions_group.create_dataset('pga', data=self.pga)
            peak_motions_group.create_dataset('pgv', data=self.pgv)
            peak_motions_group.create_dataset('pgd', data=self.pgd)
            peak_motions_group['pga'].attrs['units'] = 'g'
            peak_motions_group['pgv'].attrs['units'] = 'cm/s'
            peak_motions_group['pgd'].attrs['units'] = 'cm'
            peak_motions_group.attrs['description'] = 'Peak ground motion parameters'

            # Save characteristics
            characteristics_group = hdf.create_group('characteristics')
            characteristics_group.create_dataset('ce', data=self.ce)
            characteristics_group.create_dataset('mzc_ac', data=self.mzc_ac)
            characteristics_group.create_dataset('mzc_vel', data=self.mzc_vel)
            characteristics_group.create_dataset('mzc_disp', data=self.mzc_disp)
            characteristics_group.create_dataset('pmnm_vel', data=self.pmnm_vel)
            characteristics_group.create_dataset('pmnm_disp', data=self.pmnm_disp)
            characteristics_group['ce'].attrs['units'] = 'g^2.s'
            characteristics_group.attrs['description'] = 'Ground motion characteristics'
        return self

    @classmethod
    def from_simfile(cls, filename: str):
        """
        Load simulation data from an HDF5 file and create a Motion instance.

        filename: The name of the HDF5 file to load the data from.
        """
        with h5py.File(filename, 'r') as hdf:
            # Load spectra
            tp = hdf['spectra/tp'][:]
            sa = hdf['spectra/sa'][:]
            sv = hdf['spectra/sv'][:]
            sd = hdf['spectra/sd'][:]
            freq = hdf['spectra/freq'][:]
            fas = hdf['spectra/fas'][:]

            # Load motions
            t = hdf['motions/t'][:]
            ac = hdf['motions/ac'][:]
            vel = hdf['motions/vel'][:]
            disp = hdf['motions/disp'][:]

            # Load peak motions
            pga = hdf['peak_motions/pga'][:]
            pgv = hdf['peak_motions/pgv'][:]
            pgd = hdf['peak_motions/pgd'][:]

            # Load characteristics
            ce = hdf['characteristics/ce'][:]
            mzc_ac = hdf['characteristics/mzc_ac'][:]
            mzc_vel = hdf['characteristics/mzc_vel'][:]
            mzc_disp = hdf['characteristics/mzc_disp'][:]
            pmnm_vel = hdf['characteristics/pmnm_vel'][:]
            pmnm_disp = hdf['characteristics/pmnm_disp'][:]

        # Create a new Motion instance with the loaded data so that you can use in ModelPlot
        motion_instance = cls(npts=len(t), dt=np.round(t[1] - t[0], 3), ac=ac, vel=vel, disp=disp)
        motion_instance._tp = tp
        motion_instance._sa = sa
        motion_instance._sv = sv
        motion_instance._sd = sd
        motion_instance._freq = freq
        motion_instance._fas = fas
        motion_instance._pga = pga
        motion_instance._pgv = pgv
        motion_instance._pgd = pgd
        motion_instance._ce = ce
        motion_instance._mzc_ac = mzc_ac
        motion_instance._mzc_vel = mzc_vel
        motion_instance._mzc_disp = mzc_disp
        motion_instance._pmnm_vel = pmnm_vel
        motion_instance._pmnm_disp = pmnm_disp
        return motion_instance

    @classmethod
    def from_file(cls, file_path: str | tuple[str, str], source: str, **kwargs):
        """
        Read data from a file or a filename in a zip

        file_path: path to the file or a filename in a zip
        source:    source type (e.g., 'NGA')
        kwargs:    additional keyword arguments for RecordReader (e.g., 'skiprows')
        """
        record = RecordReader(file_path, source, **kwargs)
        return cls(npts=record.npts, dt=record.dt, ac=record.ac)

    @classmethod
    def from_model(cls, model):
        """ Read data from a stochastic model """
        return cls(npts=model.npts, dt=model.dt, ac=model.ac, vel=model.vel, disp=model.disp)
