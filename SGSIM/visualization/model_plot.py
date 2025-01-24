import os
import numpy as np
import matplotlib.pyplot as plt
from . import plot_tool as pts
from ..optimization.fit_eval import find_error

class ModelPlot:
    """
    This class allows to
        plot various simulation results and comparison with a target motion
    """
    def __init__(self, model, simulated_motion, real_motion, custom_rc=None):
        self.model = model
        self.sim = simulated_motion
        self.real = real_motion
        if custom_rc is None:
            self.custom_rc = os.path.join(os.path.dirname(__file__), 'custom_rc.matplotlibrc')
        else:
            self.custom_rc = custom_rc

    def _set_default_figsize(self, config, default_size):
        if config is None:
            config = {}
        config.setdefault('figure.figsize', default_size)
        return config

    def plot_motion(self, motion_type, id1, id2, config=None):
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
        rec = getattr(self.real, attr)
        sim1 = getattr(self.sim, attr)[id1]
        sim2 = getattr(self.sim, attr)[id2]
        
        config = self._set_default_figsize(config, (14/2.54, 4/2.54))
        with plt.rc_context(rc=config, fname=self.custom_rc):
            pts.plot_motion(self.real.t, sim1, sim2, rec, ylabel=motion_type)
            plt.show()
        return self

    def plot_ce(self, config=None):
        """
        Cumulative energy plot of the record and simulations
        """
        if not hasattr(self.sim, 'ce'):
            raise ValueError("""No simulations available.""")
        config = self._set_default_figsize(config, (7/2.54, 6/2.54))
        with plt.rc_context(rc=config, fname=self.custom_rc):
            pts.plot_mean_std(self.real.t, self.sim.ce, self.real.ce)
            plt.legend(loc='lower right', frameon=False)
            plt.xlabel('Time (s)')
            plt.ylabel(r'Cumulative energy $\mathregular{(g^2.s)}$')
            # plt.ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
            plt.show()
        return self

    def plot_fas(self, log_scale=True, config=None):
        """
        FAS plot of the record and simulations
        """
        if not hasattr(self.sim, 'fas'):
            raise ValueError("""No simulations available.""")
        config = self._set_default_figsize(config, (7/2.54, 6/2.54))
        with plt.rc_context(rc=config, fname=self.custom_rc):
            pts.plot_mean_std(self.real.freq / (2 * np.pi), self.sim.fas, self.real.fas)
            plt.ylim(np.min(self.real.fas[self.real.freq_mask]), 2 * np.max(self.real.fas[self.real.freq_mask]))
            plt.xlim([0.1, 25.0])
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

    def plot_spectra(self, spectrum='sa', log_scale=True, config=None):
        """
        Plot the specified type of spectrum (sa, sv, or sd) of the record and simulations
        """
        labels = {'sa': 'acceleration (g)',
               'sv': r'velocity ($\mathregular{\frac{cm}{s}}$)',
               'sd': 'displacement (cm)'}
        if not hasattr(self.sim, spectrum):
            raise ValueError("""No simulations available.""")
        config = self._set_default_figsize(config, (7/2.54, 6/2.54))
        with plt.rc_context(rc=config, fname=self.custom_rc):
            pts.plot_mean_std(self.real.tp, getattr(self.sim, spectrum), getattr(self.real, spectrum))
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

    def error_ce(self, config=None):
        """
        Comparing the cumulative energy and energy distribution
        of the record, model, and simulations
        """
        config = self._set_default_figsize(config, (12/2.54, 5/2.54))
        with plt.rc_context(rc=config, fname=self.custom_rc):
            pts.plot_ac_ce(self.model, self.real)
            plt.show()
        model_error = find_error(self.real.ce, self.model.ce)
        print("{}".format(f"CE error: {model_error:.3f}"))
        return self

    def error_feature(self, feature='mzc', sim_plot=False, config=None):
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

        sim_error_ac = find_error(getattr(self.real, f"{feature}_ac"), mean_ac)
        sim_error_vel = find_error(getattr(self.real, f"{feature}_vel"), mean_vel)
        sim_error_disp = find_error(getattr(self.real, f"{feature}_disp"), mean_disp)

        model_error_ac = find_error(getattr(self.real, f"{feature}_ac"), getattr(self.model, f"{feature}_ac"))
        model_error_vel = find_error(getattr(self.real, f"{feature}_vel"), getattr(self.model, f"{feature}_vel"))
        model_error_disp = find_error(getattr(self.real, f"{feature}_disp"), getattr(self.model, f"{feature}_disp"))

        config = self._set_default_figsize(config, (10/2.54, 7/2.54))
        with plt.rc_context(rc=config, fname=self.custom_rc):
            pts.plot_feature(self.model, None, self.real, feature) if not sim_plot else pts.plot_feature(self.model, self.sim, self.real, feature)
            plt.show()
        print()
        if feature in ['pmnm', 'mle']:
            print(f"{feature} model error:   vel: {model_error_vel:<10.2f}     disp: {model_error_disp:<10.2f}")
            if sim_plot:
                print(f"{feature} sim error:     vel: {sim_error_vel:<10.2f}     disp: {sim_error_disp:<10.2f}")
        else:
            print(f"{feature} model error:  ac: {model_error_ac:<10.2f}     vel: {model_error_vel:<10.2f}     disp: {model_error_disp:<10.2f}")
            if sim_plot:
                print(f"{feature} sim error:    ac: {sim_error_ac:<10.2f}     vel: {sim_error_vel:<10.2f}     disp: {sim_error_disp:<10.2f}")
        return self
