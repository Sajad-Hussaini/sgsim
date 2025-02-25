import numpy as np
from . import signal_analysis
from . import signal_processing
from ..file_reading.record_reader import RecordReader
from ..core.domain_config import DomainConfig

class Motion(DomainConfig):
    """
    This class describes ground motions in terms of various proprties (e.g., spectra, peak motions, characteristics)
    """
    def __init__(self, npts, dt, ac, vel, disp):
        """
        npts:  number of data points (array length)
        dt:    time step
        t:     time array
        ac:    acceleration array
        vel:   velocity array
        disp:  displacement array
        """
        super().__init__(npts, dt)
        self.ac = ac
        self.vel = vel
        self.disp = disp

    @property
    def fas(self):
        if not hasattr(self, '_fas'):
            self._fas = signal_analysis.get_fas(self.npts, self.ac)
        return self._fas

    @property
    def fas_star(self):
        if not hasattr(self, '_fas_star'):
            self._fas_star = signal_processing.moving_average(self.fas, 9)[..., self.freq_mask]
        return self._fas_star

    @property
    def ce(self):
        if not hasattr(self, '_ce'):
            self._ce = signal_analysis.get_ce(self.dt, self.ac)
        return self._ce
    
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
    def energy_mask(self):
        if not hasattr(self, '_energy_mask'):
            raise ValueError("The range based on energy has not been set (e.g., 0.001, 0.999).")
        return self._energy_mask

    @energy_mask.setter
    def energy_mask(self, target_range: tuple[float, float]):
        self._energy_mask = signal_analysis.get_energy_mask(self.dt, self.ac, target_range)

    def set_range(self, option: str, range_slice: tuple[float, float]|np.ndarray):
        """
        Define a range for the ground motion using an option
        [option]: 'energy' which requires a tuple of total energy percentages i.e., (0.001, 0.999)
                'mask' which requires a numpy array of boolean values
                other options to be added later (i.e., amplitude)
        """
        if option.lower() == 'energy':
            self.energy_mask = range_slice
            mask = self.energy_mask
        elif option.lower() == 'mask':
            mask = range_slice
        else:
            raise ValueError("The option is not supported yet.")
        self.npts = mask.sum()
        self.ac = self.ac[mask]
        self.vel = self.vel[mask]
        self.disp = self.disp[mask]
        return self
    
    def filter(self, bandpass_freqs: tuple[float, float]):
        """
        Perform a bandpass filtering using bandpass freqs as (lowcut, highcut) in Hz
        """
        self.ac = signal_processing.bandpass_filter(self.dt, self.ac, bandpass_freqs[0], bandpass_freqs[1])
        self.vel = signal_analysis.get_integral(self.dt, self.ac)
        self.disp = signal_analysis.get_integral(self.dt, self.vel)
        return self
    
    def upsample(self, dt_new: float):
        """
        Upsample the motion data to a finer time step.

        Args:
            dt_new (float): The new, finer time step.
        """
        self.npts, self.dt, self.ac = signal_processing.upsample(self.dt, dt_new, self.ac)
        self.vel = signal_analysis.get_integral(dt_new, self.ac)
        self.disp = signal_analysis.get_integral(dt_new, self.vel)
        return self

    def save_simulations(self, filename: str, x_var: str, y_vars: list[str]):
        """
        To save any related simulation data to a CSV file.

        Args:
            filename (str): Output file name.
            x_var (str): Independent variable (e.g., 'tp', 'freq', 't').
            y_vars list[str]: Dependent variables (e.g., ['sa', 'sv', 'sd']).
        """
        x_data = getattr(self, x_var.lower())
        y_data = [getattr(self, var.lower()).T for var in y_vars]
        data = np.column_stack((x_data, *y_data))
        n = y_data[0].shape[1] if y_data else 0
        header = x_var + "," + ",".join([f"{var}{i+1}" for var in y_vars for i in range(n)])
        np.savetxt(filename, data, delimiter=",", header=header, comments="")
        return self

    @classmethod
    def from_file(cls, file_path: str | tuple[str, str], source: str, **kwargs):
        """
        Construct a motion class from an accelergoram recording file.

        file_path: path to the file or the filename in a zip
        source:    source type (e.g., 'NGA')
        kwargs:    additional keyword arguments for RecordReader (e.g., 'skiprows')
        """
        record = RecordReader(file_path, source, **kwargs)
        return cls(npts=record.npts, dt=record.dt, ac=record.ac, vel=record.vel, disp=record.disp)

    @classmethod
    def from_model(cls, model):
        """ Construct a motion class from a calibrated stochastic model """
        return cls(npts=model.npts, dt=model.dt, ac=model.ac, vel=model.vel, disp=model.disp)
