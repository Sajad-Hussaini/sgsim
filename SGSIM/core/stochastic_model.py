import numpy as np
from scipy.fft import irfft
from SGSIM.core import filter_freq as ff
from SGSIM.core.model_core import ModelCore

class StochasticModel(ModelCore):
    """
    This class allows to
        Simulate ground motion time series (acceleration, velocity, displacement)
        a single or multi simulation
    """
    def __init__(self, npts: int, dt: float, mdl_type: str = 'beta_multi',
                 wu_type: str = 'linear', zu_type: str = 'linear',
                 wl_type: str = 'linear', zl_type: str = 'linear'):
        super().__init__(npts, dt, mdl_type, wu_type, zu_type, wl_type, zl_type)
        self.seed = None

    def set_seed(self, number):
        """
        A fixed seed for White noise to avoid stochastic behavior
        None value produces variability
        """
        self.seed = number
        return self

    def simulate(self) -> tuple[np.array, np.array, np.array]:
        """
        Simulate ground-motions using fitted moddel parameters
        Based on the frequency representation
        returns: ac, vel, disp
        """
        white_noise = np.random.default_rng(seed=self.seed).standard_normal(self.npts)
        sim_Fourier = np.zeros(len(self.freq_sim), dtype='complex')
        for i in range(self.npts):
            # irfft and np.sum cancel out dt/dt
            # npts used in varx
            sim_Fourier += ((ff.get_frf(self.wu[i], self.zu[i], self.wl[i], self.zl[i], self.freq_sim) *
                             np.exp(-1j * self.freq_sim * self.t[i])) *
                            (white_noise[i] * self.mdl[i] / np.sqrt(self.variance[i] * 2 / self.npts)))

        sim_ac = irfft(sim_Fourier)[0:self.npts]  # to avoid aliasing
        sim_vel = np.zeros_like(sim_Fourier)
        sim_disp = np.zeros_like(sim_Fourier)

        # FT(w)/jw + pi*delta(w)*FT(0)  integration in freq domain
        sim_vel[1:] = sim_Fourier[1:] / (1j * self.freq_sim[1:])
        sim_vel = irfft(sim_vel)[0:self.npts]

        sim_disp[1:] = -sim_Fourier[1:] / (self.freq_sim[1:] ** 2)
        sim_disp = irfft(sim_disp)[0:self.npts]
        return sim_ac, sim_vel, sim_disp

    def multi_simulate(self, nsim):
        """
        Multiple simulation of ground-motions
        """
        sim_motions = np.array([self.simulate() for _ in range(nsim)])
        self.ac, self.vel, self.disp = sim_motions[:, 0], sim_motions[:, 1], sim_motions[:, 2]
        return self