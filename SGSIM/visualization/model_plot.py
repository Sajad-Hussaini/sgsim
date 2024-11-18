import numpy as np
import matplotlib.pyplot as plt
from . import plot_tool as pts
from ..optimization import fit_metric

class ModelPlot:
    """
    This class allows to
        plot various simulation results and comparison with a target motion
    """
    def __init__(self, simulated_motion, target_motion, model):
        self.sim = simulated_motion
        self.tm = target_motion
        self.model = model

    def plot_motion(self, motion_type, id1, id2):
        """
        Plot the specified type of motion of the record and simulations
        """
        if not hasattr(self.sim, 'ac'):
            raise ValueError("""No simulations available.""")
        if 'Acceleration' in motion_type:
            attr = 'ac'
        elif 'Velocity' in motion_type:
            attr = 'vel'
        elif 'Displacement' in motion_type:
            attr = 'disp'
        else:
            raise ValueError("Invalid motion type")
        rec = getattr(self.tm, attr)
        sim1 = getattr(self.sim, attr)[id1]
        sim2 = getattr(self.sim, attr)[id2]
        pts.plot_motion(self.sim.t, rec, sim1, sim2, ylabel=motion_type)
        return self

    def plot_ce(self):
        """
        Cumulative energy plot of the record and simulations
        """
        if not hasattr(self.sim, 'ce'):
            raise ValueError("""No simulations available.""")
        pts.plot_mean_std(self.sim.t, self.tm.ce, self.sim.ce)
        plt.legend(loc='lower right', frameon=False)
        plt.xlabel('Time (s)')
        plt.ylabel(r'Cumulative energy $\mathregular{(g^2.s)}$')
        # plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
        plt.show()
        return self

    def plot_fas(self, log_scale=True):
        """
        FAS plot of the record and simulations
        """
        if not hasattr(self.sim, 'fas'):
            raise ValueError("""No simulations available.""")
        pts.plot_mean_std(self.tm.freq / (2 * np.pi), self.tm.fas, self.sim.fas)
        plt.ylim(np.min(self.tm.fas[self.sim.slicer_freq]), 2 * np.max(self.tm.fas[self.sim.slicer_freq]))
        plt.xlim([0.1, 25])
        plt.xscale('log')
        if log_scale:
            plt.yscale('log')
            leg_loc = 'lower center'
        else:
            leg_loc = 'upper right'
        plt.legend(loc=leg_loc, frameon=False)
        plt.xlabel('Frequency (Hz)')
        plt.ylabel(r'Fourier amplitude spectrum $\mathregular{(g\sqrt{s})}$')
        plt.show()
        return self

    def plot_spectra(self, spectrum='sa', log_scale=True):
        """
        Plot the specified type of spectrum (sa, sv, or sd) of the record and simulations
        """
        labels = {'sa': 'acceleration (g)',
               'sv': r'velocity ($\mathregular{\frac{cm}{s}}$)',
               'sd': 'displacement (cm)'}
        if not hasattr(self.sim, spectrum):
            raise ValueError("""No simulations available.""")
        pts.plot_mean_std(self.sim.tp, getattr(self.tm, spectrum), getattr(self.sim, spectrum))
        plt.xscale('log')
        if log_scale:
            plt.yscale('log')
            leg_loc = 'lower center'
        else:
            leg_loc = 'upper right'
        plt.legend(loc=leg_loc, frameon=False)
        plt.xlabel('Period (s)')
        plt.ylabel(f'Spectral {labels.get(spectrum)}')
        plt.show()
        return self

    def error_ce(self):
        """
        Comparing the cumulative energy and energy distribution
        of the record, model, and simulations
        """
        pts.plot_ac_ce(self.tm, self.model)
        model_error = fit_metric.find_error(self.tm.ce, self.model.ce)
        print("{}".format(f"CE error: {model_error:.3f}"))
        return self

    def error_feature(self, feature='mzc', no_sim = True):
        """
        Comparing the indicated error of the record, model, and simulations
        mzc, mle, pmnm
        """
        if not hasattr(self.sim, 'ac'):
            raise ValueError("""No simulations available.""")

        temp_ac = getattr(self.sim, f"{feature}_ac")
        mean_ac = np.mean(temp_ac, axis=0)

        temp_vel = getattr(self.sim, f"{feature}_vel")
        mean_vel = np.mean(temp_vel, axis=0)

        temp_disp = getattr(self.sim, f"{feature}_disp")
        mean_disp = np.mean(temp_disp, axis=0)

        sim_error_ac = fit_metric.find_error(getattr(self.tm, f"{feature}_ac"), mean_ac)
        sim_error_vel = fit_metric.find_error(getattr(self.tm, f"{feature}_vel"), mean_vel)
        sim_error_disp = fit_metric.find_error(getattr(self.tm, f"{feature}_disp"), mean_disp)

        model_error_ac = fit_metric.find_error(getattr(self.tm, f"{feature}_ac"), getattr(self.model, f"{feature}_ac"))
        model_error_vel = fit_metric.find_error(getattr(self.tm, f"{feature}_vel"), getattr(self.model, f"{feature}_vel"))
        model_error_disp = fit_metric.find_error(getattr(self.tm, f"{feature}_disp"), getattr(self.model, f"{feature}_disp"))

        pts.plot_feature(self.tm, self.model, None, feature) if no_sim else pts.plot_feature(self.tm, self.model, self.sim, feature)
        print('\n')
        print(f"{feature} model error:  ac: {model_error_ac:<10.2f}  vel: {model_error_vel:<10.2f}  disp: {model_error_disp:<10.2f}")
        print(f"{feature} sim error:    ac: {sim_error_ac:<10.2f}  vel: {sim_error_vel:<10.2f}  disp: {sim_error_disp:<10.2f}")
        return self

