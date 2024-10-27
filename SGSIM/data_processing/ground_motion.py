import numpy as np
from scipy.fft import rfft, rfftfreq
import SGSIM.data_processing.motion_processor as mps
from SGSIM.file_reading.record_reader import RecordReader

class MotionCore:
    """
    A class to provide functionality and description of ground motions
    """
    def __init__(self, npts: float, dt: float, t, ac):
        self.npts = npts
        self.dt = dt
        self.t = t
        self.ac = ac
        self.vel = None
        self.disp = None

    def get_properties(self, period_range: tuple[float, float, float] = (0.04, 10.01, 0.01)):
        """ Calculate characteristics of the motion """
        self.tp = np.arange(*period_range)
        self.fas = np.abs(rfft(self.ac)) / np.sqrt(self.npts / 2)
        self.sd, self.sv, self.sa = mps.get_spectra(self.dt, self.ac, period_range=period_range)
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
    def __init__(self, file_path: str or tuple[str, str],
                 source: str, **kwargs):
        RecordReader.__init__(self, file_path, source, **kwargs)  # init RecordReader
        MotionCore.__init__(self, self.npts, self.dt, self.t, self.ac)

    def set_target_range(self, target_range: tuple[float, float]):
        """
        Define the target range of the motion
        based on a range of total energy percentages i.e., (0.001, 0.999)
        """
        slicer = mps.find_slice(self.dt, self.t, self.ac, target_range)
        self.t = np.round(self.t[slicer] - self.t[slicer][0], 3)
        self.npts = len(self.t)

        self.disp = mps.get_disp(self.dt, self.ac)[slicer]
        self.vel = mps.get_vel(self.dt, self.ac)[slicer]

        # update ac after obtaining vel and disp for the target range
        self.ac = self.ac[slicer]

        self.get_properties()
        return self

class SimMotion(MotionCore):
    """
    A class to describe the simulated ground motion/s
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