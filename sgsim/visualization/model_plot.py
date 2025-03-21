import os
import numpy as np
import matplotlib.pyplot as plt
from ..optimization.fit_eval import find_error, goodness_of_fit

class ModelPlot:
    """
    This class allows to
        plot various simulation results and comparison with a target motion
    """
    def __init__(self, model, simulated_motion, real_motion):
        self.model = model
        self.sim = simulated_motion
        self.real = real_motion
        self.rcp = {
            'font.family': 'Times New Roman',
            'font.size': 9,

            'lines.linewidth': 0.5,

            'axes.titlesize': 'medium',
            'axes.linewidth': 0.2,

            'xtick.major.width': 0.2,
            'ytick.major.width': 0.2,
            'xtick.minor.width': 0.15,
            'ytick.minor.width': 0.15,

            'legend.framealpha': 1.0,
            'legend.frameon': False,

            'figure.dpi': 900,
            'figure.figsize': (3.937, 3.1496),  # 10 cm by 8 cm
            'figure.constrained_layout.use': True,

            'patch.linewidth': 0.5,
            }

    def plot_motions(self, id1, id2, config=None):
        """
        plot ground motion time histories of acceleration, velocity, and displacement in a 3 by 3 grid
        """
        if not hasattr(self.sim, 'ac'):
            raise ValueError("No simulations available.")

        motion_types = [
            (r"Acceleration (cm/$\mathregular{s^2}$)", "ac"),
            ("Velocity (cm/s)", "vel"),
            ("Displacement (cm)", "disp")
            ]

        config = {**self.rcp, 'figure.figsize': (15/2.54, 10/2.54), **(config or {})}
        with plt.rc_context(rc=config):
            fig, axes = plt.subplots(3, 3, sharex=True, sharey='row')
            for row_idx, (ylabel, attr) in enumerate(motion_types):
                rec = getattr(self.real, attr)
                sim1 = getattr(self.sim, attr)[id1]
                sim2 = getattr(self.sim, attr)[id2]

                axes[row_idx, 0].plot(self.real.t, rec, label='Real', color='tab:blue')
                axes[row_idx, 0].set_ylabel(ylabel)

                axes[row_idx, 1].plot(self.sim.t, sim1, label='Simulation', color='tab:orange')
                axes[row_idx, 2].plot(self.sim.t, sim2, label='Simulation', color='tab:orange')

                for ax in axes[row_idx]:
                    ax.axhline(y=0, color='k', linestyle='--', lw=0.1, zorder=0)
                    ax.set_xlabel('Time (s)') if row_idx == 2 else None
                    ax.minorticks_on()

                max_val = max(np.max(np.abs(rec)), np.max(np.abs(sim1)), np.max(np.abs(sim2)))
                axes[row_idx, 0].set_ylim([-1.05 * max_val, 1.05 * max_val])
                axes[row_idx, 0].yaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
            axes[0, 0].set_title('Real')
            axes[0, 1].set_title('Simulation')
            axes[0, 2].set_title('Simulation')
            fig.align_ylabels(axes)
            plt.show()
        return self

    def plot_ce(self, config=None):
        """
        Cumulative energy plot of the record and simulations
        """
        if not hasattr(self.sim, 'ce'):
            raise ValueError("""No simulations available.""")

        config = {**self.rcp, 'figure.figsize':  (8/2.54, 6/2.54), **(config or {})}
        with plt.rc_context(rc=config):
            self._plot_mean_std(self.real.t, self.sim.ce, self.real.ce)
            plt.legend(loc='lower right', frameon=False)
            plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
            plt.xlabel('Time (s)')
            plt.ylabel(r'Cumulative energy ($\mathregular{cm^2/s^3}$)')
            plt.show()
        return self

    def plot_fas(self, log_scale=True, config=None):
        """
        FAS plot of the record and simulations
        """
        if not hasattr(self.sim, 'fas'):
            raise ValueError("""No simulations available.""")
        config = {**self.rcp, 'figure.figsize':  (8/2.54, 6/2.54), **(config or {})}
        with plt.rc_context(rc=config):
            self._plot_mean_std(self.real.freq / (2 * np.pi), self.sim.fas, self.real.fas)
            plt.ylim(np.min(self.real.fas[self.real.freq_slice]), 2 * np.max(self.real.fas[self.real.freq_slice]))
            plt.xlim([0.1, 25.0])
            plt.xscale('log')
            if log_scale:
                plt.yscale('log')
                leg_loc = 'lower center'
            else:
                leg_loc = 'upper right'
            plt.legend(loc=leg_loc, frameon=False)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel(r'Fourier amplitude spectrum ($\mathregular{cm/s^2}$)')
            plt.show()
        return self

    def plot_spectra(self, spectrum='sa', log_scale=True, config=None):
        """
        Plot the specified type of spectrum (sa, sv, or sd) of the record and simulations
        """
        labels = {'sa': r'acceleration ($\mathregular{cm/s^2}$)',
               'sv': 'velocity (cm/s)',
               'sd': 'displacement (cm)'}
        if not hasattr(self.sim, spectrum):
            raise ValueError("""No simulations available.""")
        config = {**self.rcp, 'figure.figsize':  (8/2.54, 6/2.54), **(config or {})}
        with plt.rc_context(rc=config):
            self._plot_mean_std(self.real.tp, getattr(self.sim, spectrum), getattr(self.real, spectrum))
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

    def plot_ac_ce(self, config=None):
        """
        Comparing the cumulative energy and energy distribution
        of the record, model, and simulations
        """
        config = {**self.rcp, 'figure.figsize':  (12/2.54, 5/2.54), **(config or {})}
        with plt.rc_context(rc=config):
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
            axes[0].plot(self.real.t, self.real.ac, c='tab:blue')
            axes[0].plot(self.model.t, self.model.mdl, c='tab:orange', ls='--')
            axes[0].plot(self.model.t, -self.model.mdl, c='tab:orange', ls='--')
            axes[0].axhline(y=0, color='k', ls='--', lw=0.15)
            axes[0].set_ylabel(r'Acceleration ($\mathregular{cm/s^2}$)')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylim([-1.05 * max(abs(self.real.ac)), 1.05 * max(abs(self.real.ac))])
            axes[0].yaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
            axes[0].minorticks_on()

            axes[1].plot(self.real.t, self.real.ce, label= 'Target', c='tab:blue')
            axes[1].plot(self.model.t, self.model.ce, label= 'self.model', c='tab:orange', ls='--')
            axes[1].set_ylabel(r'Cumulative energy $\mathregular{(cm^2/s^3)}$')
            axes[1].set_xlabel('Time (s)')
            axes[1].yaxis.set_major_locator(plt.MaxNLocator(5))
            axes[1].legend(loc='lower right', frameon=False)
            axes[1].minorticks_on()
            plt.show()
        return self

    def plot_feature(self, feature='mzc', model_plot=True, sim_plot=False, config=None):
        """
        Comparing the indicated error of the record, model, and simulations
        mzc, mle, pmnm
        """
        if not hasattr(self.sim, 'ac'):
            raise ValueError("""No simulations available.""")

        config = {**self.rcp, 'figure.figsize':  (10/2.54, 7/2.54), **(config or {})}
        with plt.rc_context(rc=config):
            plt.plot(self.real.t, getattr(self.real, f"{feature}_ac"), label="Target acceleration",
            c='tab:blue', zorder=2) if feature == 'mzc' else None
            plt.plot(self.real.t, getattr(self.real, f"{feature}_vel"), label="Target velocity",
                    c='tab:orange', zorder=2)
            plt.plot(self.real.t, getattr(self.real, f"{feature}_disp"), label="Target displacement",
                    c='tab:green', zorder=2)

            if model_plot:
                plt.plot(self.model.t, getattr(self.model, f"{feature}_ac"),
                        label="Model acceleration", c='tab:cyan', zorder=3)  if feature == 'mzc' else None
                plt.plot(self.model.t, getattr(self.model, f"{feature}_vel"),
                        label="Model velocity", c='tab:pink', zorder=3)
                plt.plot(self.model.t, getattr(self.model, f"{feature}_disp"),
                        label="Model displacement", c='tab:olive', zorder=3)

            if sim_plot:
                plt.plot(self.sim.t, getattr(self.sim, f"{feature}_ac").T,
                        color='tab:gray', lw=0.15)  if feature == 'mzc' else None
                plt.plot(self.sim.t, getattr(self.sim, f"{feature}_vel")[:-1].T,
                        color='tab:gray', lw=0.15)
                plt.plot(self.sim.t, getattr(self.sim, f"{feature}_vel")[-1],
                        color='tab:gray', lw=0.15, label="self.simulations")
                plt.plot(self.sim.t, getattr(self.sim, f"{feature}_disp").T,
                        color='tab:gray', lw=0.15)

            plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
            plt.xlabel("Time (s)")
            plt.ylabel("Cumulative mean zero crossing" if feature == 'mzc'
                    else "Cumulative mean local extrema" if feature == 'mle'
                    else 'Cumulative mean positive-minima\nand negative-maxima')
            plt.show()
        return self
    
    @staticmethod
    def _plot_mean_std(t, sims, rec):
        """
        Plot the common part of ce_plot and fas_plot
        """
        mean_all = np.mean(sims, axis=0)
        std_all = np.std(sims, axis=0)
        plt.plot(t, rec.flatten(), c='tab:blue', label='Target', zorder=2)
        plt.plot(t, mean_all, c='tab:orange', label='Mean', zorder=4)
        plt.plot(t, mean_all - std_all, c='k', linestyle='-.', label=r'Mean $\mathregular{\pm \, \sigma}$', zorder=3)
        plt.plot(t, mean_all + std_all, c='k', linestyle='-.', zorder=3)
        plt.plot(t, sims[:-1].T, c='tab:gray', lw=0.15, zorder=1)
        plt.plot(t, sims[-1], c='tab:gray', lw=0.15, label="Simulations", zorder=1)
        plt.minorticks_on()
    
    def _compute_metrics(self, metric_func):
        """Compute metrics for model and simulation parameters."""
        model_params = ['ce', 'fas', 'mzc_ac', 'mzc_vel', 'mzc_disp', 'pmnm_vel', 'pmnm_disp']
        sim_params = ['ce', 'fas', 'sa', 'sv', 'sd', 'mzc_ac', 'mzc_vel', 'mzc_disp', 'pmnm_vel', 'pmnm_disp']
        
        return (
            {param: metric_func(getattr(self.real, param), getattr(self.model, param)) 
             for param in model_params},
            {param: metric_func(getattr(self.real, param), getattr(self.sim, param)) 
             for param in sim_params}
             )

    @property
    def errors(self):
        return self._compute_metrics(find_error)

    @property
    def goodness_of_fits(self):
        return self._compute_metrics(goodness_of_fit)

    def show_metrics(self, metric_type, save_path=None):
        """Print metrics with optional saving."""
        if metric_type.lower() not in ['errors', 'gof']:
            raise ValueError("metric_type must be 'errors' or 'gof'")
        
        name = "Errors" if metric_type.lower() == 'errors' else "Goodness of Fit"
        model_metrics, sim_metrics = self.errors if metric_type.lower() == 'errors' else self.goodness_of_fits
        
        output = [
            f"{name} for Model:",
            *[f"  {p}: {v}" for p, v in model_metrics.items()],
            f"\n{name} for Simulations:",
            *[f"  {p}: {v}" for p, v in sim_metrics.items()]
            ]
        print("\n".join(output))
        
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w') as f:
                f.write("\n".join(output))
            print(f"Saving {name}: Done.")