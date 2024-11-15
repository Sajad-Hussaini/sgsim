import numpy as np
from scipy.fft import rfft, rfftfreq
import SGSIM.data_processing.motion_processor as mps
from SGSIM.file_reading.record_reader import RecordReader

class MotionCore:
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
        """ Calculate characteristics of the ground motion """
        self.tp = np.arange(*period_range)
        self.fas = np.abs(rfft(self.ac)) / np.sqrt(self.npts / 2)
        self.sd, self.sv, self.sa = mps.get_spectra(self.dt, self.ac if self.ac.ndim==2 else self.ac[None, :], period=self.tp, zeta=0.05)
        self.ce = mps.get_ce(self.dt, self.ac)

        self.mzc_ac = mps.find_mzc(self.ac)
        self.mzc_vel = mps.find_mzc(self.vel)
        self.mzc_disp = mps.find_mzc(self.disp)

        self.mle_ac = mps.find_mle(self.ac)
        self.mle_vel = mps.find_mle(self.vel)
        self.mle_disp = mps.find_mle(self.disp)

        self.pmnm_ac = mps.find_pmnm(self.ac)
        self.pmnm_vel = mps.find_pmnm(self.vel)
        self.pmnm_disp = mps.find_pmnm(self.disp)

        # Nyq freq for fitting
        self.freq = rfftfreq(self.npts, self.dt) * 2 * np.pi
        self.slicer_freq = (self.freq >= 0.1 * 2 * np.pi) & (self.freq <= 25 * 2 * np.pi)
        return self

class TargetMotion(RecordReader, MotionCore):
    """
    A class to describe a target ground motion
    """
    def __init__(self, file_path: str | tuple[str, str], source: str, **kwargs):
        RecordReader.__init__(self, file_path, source, **kwargs)
        MotionCore.__init__(self, self.npts, self.dt, self.t, self.ac)

    # def set_target_range(self, target_range: tuple[float, float]):
    #     """
    #     Define the target range of the motion
    #     based on a range of total energy percentages i.e., (0.001, 0.999)
    #     """
    #     self.slicer = mps.find_slice(self.dt, self.t, self.ac, target_range)
    #     self.t = np.round(self.t[self.slicer] - self.t[self.slicer][0], 3)
    #     self.npts = len(self.t)

    #     self.disp = mps.get_disp(self.dt, self.ac)[self.slicer]
    #     self.vel = mps.get_vel(self.dt, self.ac)[self.slicer]

    #     # update ac after obtaining vel and disp for the target range
    #     self.ac = self.ac[self.slicer]

    #     self.get_properties()
    #     return self
    def set_target_range(self, target_range: tuple[float, float]):
        """
        Define the target range of the ground motion
        based on a range of total energy percentages i.e., (0.001, 0.999)
        """
        self.slicer = mps.find_slice(self.dt, self.t, self.ac, target_range)
        self.t = np.round(self.t[self.slicer] - self.t[self.slicer][0], 3)
        self.npts = len(self.t)

        self.ac = self.ac[self.slicer]
        self.ac = mps.bandpass_filter(self.dt, self.ac)
        self.disp = mps.get_disp(self.dt, self.ac)
        self.vel = mps.get_vel(self.dt, self.ac)

        self.get_properties()
        return self

class SimMotion(MotionCore):
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