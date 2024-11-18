import numpy as np
from .signal_processing import bandpass_filter
from . import signal_props as sps
from ..file_reading.record_reader import RecordReader

class CoreMotion:
    def __init__(self, npts: int, dt: float, t, ac):
        """
        A class to provide functionality and description of ground motions.

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
        self.fas = sps.get_fas(self.npts, self.ac)
        self.sd, self.sv, self.sa = sps.get_spectra(self.dt, self.ac if self.ac.ndim==2 else self.ac[None, :], period=self.tp, zeta=0.05)
        self.ce = sps.get_ce(self.dt, self.ac)

        self.mzc_ac = sps.get_mzc(self.ac)
        self.mzc_vel = sps.get_mzc(self.vel)
        self.mzc_disp = sps.get_mzc(self.disp)

        self.mle_ac = sps.get_mle(self.ac)
        self.mle_vel = sps.get_mle(self.vel)
        self.mle_disp = sps.get_mle(self.disp)

        self.pmnm_ac = sps.get_pmnm(self.ac)
        self.pmnm_vel = sps.get_pmnm(self.vel)
        self.pmnm_disp = sps.get_pmnm(self.disp)

        self.freq = sps.get_freq(self.dt, self.npts)
        self.slicer_freq = sps.get_freq_slice(self.freq)
        return self

class TargetMotion(RecordReader, CoreMotion):
    """
    A class to describe a target ground motion
    """
    def __init__(self, file_path: str | tuple[str, str], source: str, **kwargs):
        RecordReader.__init__(self, file_path, source, **kwargs)
        CoreMotion.__init__(self, self.npts, self.dt, self.t, self.ac)

    # def set_target_range(self, target_range: tuple[float, float]):
    #     """
    #     Define the target range of the motion
    #     based on a range of total energy percentages i.e., (0.001, 0.999)
    #     """
    #     self.slicer = sps.get_energy_slice(self.dt, self.t, self.ac, target_range)
    #     self.t = np.round(self.t[self.slicer] - self.t[self.slicer][0], 3)
    #     self.npts = len(self.t)

    #     self.disp = sps.get_disp(self.dt, self.ac)[self.slicer]
    #     self.vel = sps.get_vel(self.dt, self.ac)[self.slicer]

    #     # update ac after obtaining vel and disp for the target range
    #     self.ac = self.ac[self.slicer]

    #     self.get_properties()
    #     return self
    def set_target_range(self, target_range: tuple[float, float]):
        """
        Define the target range of the ground motion
        based on a range of total energy percentages i.e., (0.001, 0.999)
        """
        self.slicer = sps.get_energy_slice(self.dt, self.ac, target_range)
        self.t = np.round(self.t[self.slicer] - self.t[self.slicer][0], 3)
        self.npts = len(self.t)

        self.ac = self.ac[self.slicer]
        self.ac = bandpass_filter(self.dt, self.ac)
        self.disp = sps.get_disp(self.dt, self.ac)
        self.vel = sps.get_vel(self.dt, self.ac)

        self.get_properties()
        return self

class SimMotion(CoreMotion):
    """
    A class to describe the simulated ground motions
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
        return self