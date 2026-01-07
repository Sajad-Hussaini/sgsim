import numpy as np
import matplotlib.pyplot as plt
from ..motion import signal_tools
from .style import style

class ModelPlot:
    """
    Visualization and comparison of simulation results with a target ground motion.

    Provides methods to plot time histories, spectra, cumulative energy, and feature metrics
    for both simulated and real ground motions, as well as the underlying model.
    """
    def __init__(self, model, simulated_motion, real_motion):
        """
        Initialize ModelPlot with model, simulated, and real ground motion data.

        Parameters
        ----------
        model : object
            The model object containing fitted model.
        simulated_motion : object
            Simulated ground motion object (ensemble or single simulation).
        real_motion : object
            Real (target) ground motion object.
        """
        self.model = model
        self.sim = simulated_motion
        self.real = real_motion

    def plot_motions(self, id1, id2, config=None):
        """
        Plot ground motion time histories (acceleration, velocity, displacement) in a 3x3 grid.

        Parameters
        ----------
        id1 : int
            Index of the first simulation to plot.
        id2 : int
            Index of the second simulation to plot.
        config : dict or None, optional
            Plot style configuration.
        """
        if not hasattr(self.sim, 'ac'):
            raise ValueError("No simulations available.")

        motion_types = [(r"Acceleration (cm/$s^2$)", "ac"), ("Velocity (cm/s)", "vel"), ("Displacement (cm)", "disp")]

        with style(config):
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
                    ax.axhline(y=0, color='k', linestyle='--', lw=0.15, zorder=0)
                    ax.set_xlabel('Time (s)') if row_idx == 2 else None
                    ax.minorticks_on()

                max_val = max(np.max(np.abs(rec)), np.max(np.abs(sim1)), np.max(np.abs(sim2)))
                axes[row_idx, 0].set_ylim([-1.05 * max_val, 1.05 * max_val])
                axes[row_idx, 0].yaxis.set_major_locator(plt.MaxNLocator(min_n_ticks=5, nbins='auto', symmetric=True))
                axes[row_idx, 0].xaxis.set_major_locator(plt.MaxNLocator(min_n_ticks=4, nbins='auto'))
            axes[0, 0].set_title('Real')
            axes[0, 1].set_title('Simulation')
            axes[0, 2].set_title('Simulation')
            fig.align_ylabels(axes)
            plt.show()

    def plot_ce(self, config=None):
        """
        Plot cumulative energy (CE) of the record and simulations.

        Parameters
        ----------
        config : dict or None, optional
            Plot style configuration.
        """
        if not hasattr(self.sim, 'ce'):
            raise ValueError("""No cumulative energy available.""")
        with style(config):
            fig, ax = plt.subplots()
            self._plot_mean_std(self.real.t, self.sim.ce, self.real.ce, ax)
            ax.legend(loc='lower right')
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.set_xlabel('Time (s)')
            ax.set_ylabel(r'Cumulative energy ($cm^2/s^3$)')
            plt.show()

    def plot_fas(self, log_scale=True, plot_range=(0.1, 25.0), config=None):
        """
        Plot Fourier Amplitude Spectrum (FAS) of the record and simulations.

        Parameters
        ----------
        log_scale : bool, default=True
            Whether to use logarithmic scale for y-axis.
        config : dict or None, optional
            Plot style configuration.
        """
        if not hasattr(self.sim, 'fas'):
            raise ValueError("""No Fourier spectrum available.""")
        self.real.freq_slicer = signal_tools.slice_freq(self.real.freq, plot_range)
        with style(config):
            fig, ax = plt.subplots()
            self._plot_mean_std(self.real.freq, self.sim.fas, self.real.fas, ax)
            ax.set_ylim(np.min(self.real.fas[self.real.freq_slicer]), 2 * np.max(self.real.fas[self.real.freq_slicer]))
            ax.set_xlim([0.1, 25.0])
            ax.set_xscale('log')
            if log_scale:
                ax.set_yscale('log')
                leg_loc = 'lower center'
            else:
                leg_loc = 'upper right'
            ax.legend(loc=leg_loc)
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel(r'Fourier amplitude spectrum (cm/$s^2$)')
            plt.show()

    def plot_spectra(self, periods, spectrum='sa', log_scale=True, plot_range=(0.1, 25.5), config=None):
        """
        Plot the specified type of response spectrum (sa, sv, or sd) for record and simulations.

        Parameters
        ----------
        spectrum : {'sa', 'sv', 'sd'}, default='sa'
            Type of spectrum to plot: 'sa' (acceleration), 'sv' (velocity), or 'sd' (displacement).
        log_scale : bool, default=True
            Whether to use logarithmic scale for y-axis.
        config : dict or None, optional
            Plot style configuration.
        """
        labels = {'sa': r'acceleration (cm/$s^2$)', 'sv': 'velocity (cm/s)', 'sd': 'displacement (cm)'}
        with style(config):
            fig, ax = plt.subplots()
            sim_spectra = self.sim.response_spectra(periods)
            real_spectra = self.real.response_spectra(periods)
            if spectrum=="sd" :
                simspec = sim_spectra[0]
                realspec = real_spectra[0]
            elif spectrum=="sv" :
                simspec = sim_spectra[1]
                realspec = real_spectra[1]
            else :
                simspec = sim_spectra[2]
                realspec = real_spectra[2]
            self._plot_mean_std(periods, simspec, realspec, ax)
            ax.set_xscale('log')
            if log_scale:
                ax.set_yscale('log')
                leg_loc = 'lower center'
            else:
                leg_loc = 'upper right'
            ax.legend(loc=leg_loc)
            ax.set_xlabel('Period (s)')
            ax.set_ylabel(f'Spectral {labels.get(spectrum)}')
            plt.show()

    def plot_ac_ce(self, config=None):
        """
        Compare cumulative energy and energy distribution of the record and model.

        Parameters
        ----------
        config : dict or None, optional
            Plot style configuration.
        """
        with style(config):
            fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
            axes[0].plot(self.real.t, self.real.ac, c='tab:blue')
            axes[0].plot(self.model.t, self.model.modulating.values, c='tab:orange', ls='--')
            axes[0].plot(self.model.t, -self.model.modulating.values, c='tab:orange', ls='--')
            axes[0].axhline(y=0, color='k', ls='--', lw=0.1, zorder=0)
            axes[0].set_ylabel(r'Acceleration (cm/$s^2$)')
            axes[0].set_xlabel('Time (s)')
            axes[0].set_ylim([-1.05 * max(abs(self.real.ac)), 1.05 * max(abs(self.real.ac))])
            axes[0].yaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
            axes[0].minorticks_on()

            axes[1].plot(self.real.t, self.real.ce, label= 'Target', c='tab:blue')
            axes[1].plot(self.model.t, self.model.ce, label= 'Model', c='tab:orange', ls='--')
            axes[1].set_ylabel(r'Cumulative energy $(cm^2/s^3)$')
            axes[1].set_xlabel('Time (s)')
            axes[1].yaxis.set_major_locator(plt.MaxNLocator(5))
            axes[1].legend(loc='lower right')
            axes[1].minorticks_on()
            plt.show()

    def plot_feature(self, feature='zc', model_plot=True, sim_plot=False, config=None):
        """
        Compare a specific feature (error metric) of the record, model, and simulations.

        Parameters
        ----------
        feature : {'zc', 'le', 'pmnm'}, default='zc'
            Feature to plot: 'zc' (mean zero crossing), 'le' (mean local extrema),
            or 'pmnm' (positive-minima/negative-maxima).
        model_plot : bool, default=True
            Whether to plot the model feature.
        sim_plot : bool, default=False
            Whether to plot simulation features.
        config : dict or None, optional
            Plot style configuration.
        """
        if not hasattr(self.sim, 'ac'):
            raise ValueError("""No characteristics available.""")
        with style(config):
            plt.plot(self.real.t, getattr(self.real, f"{feature}_ac"), label="Target acceleration",
                     c='tab:blue', zorder=2) if feature == 'zc' else None
            plt.plot(self.real.t, getattr(self.real, f"{feature}_vel"), label="Target velocity",
                     c='tab:orange', zorder=2)
            plt.plot(self.real.t, getattr(self.real, f"{feature}_disp"), label="Target displacement",
                     c='tab:green', zorder=2)

            if model_plot:
                plt.plot(self.model.t, getattr(self.model, f"{feature}_ac"),
                        label="Model acceleration", c='tab:cyan', zorder=3)  if feature == 'zc' else None
                plt.plot(self.model.t, getattr(self.model, f"{feature}_vel"),
                        label="Model velocity", c='tab:pink', zorder=3)
                plt.plot(self.model.t, getattr(self.model, f"{feature}_disp"),
                        label="Model displacement", c='tab:olive', zorder=3)

            if sim_plot:
                plt.plot(self.sim.t, getattr(self.sim, f"{feature}_ac").T,
                        color='tab:gray', lw=0.1)  if feature == 'zc' else None
                plt.plot(self.sim.t, getattr(self.sim, f"{feature}_vel")[:-1].T,
                        color='tab:gray', lw=0.1)
                plt.plot(self.sim.t, getattr(self.sim, f"{feature}_vel")[-1],
                        color='tab:gray', lw=0.1, label="Simulations")
                plt.plot(self.sim.t, getattr(self.sim, f"{feature}_disp").T,
                        color='tab:gray', lw=0.1)

            plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
            plt.xlabel("Time (s)")
            plt.ylabel("Cumulative mean zero crossing" if feature == 'zc'
                    else "Cumulative mean local extrema" if feature == 'le'
                    else 'Cumulative mean positive-minima\nand negative-maxima')
            plt.show()
    
    @staticmethod
    def _plot_mean_std(t, sims, rec, ax=None):
        """
        Plot mean, standard deviation, and all simulations versus the target.

        Parameters
        ----------
        t : array-like
            Time or frequency axis.
        sims : array-like
            Simulated ensemble (2D: n_sim, n_points).
        rec : array-like
            Target (real) record (1D).
        ax : matplotlib.axes.Axes or None, optional
            Axis to plot on. If None, uses current axis.
        """
        if ax is None:
            ax = plt.gca()
        mean_all = np.mean(sims, axis=0)
        std_all = np.std(sims, axis=0)
        ax.plot(t, rec.T, c='tab:blue', label='Target', zorder=2)
        ax.plot(t, mean_all, c='tab:orange', label='Mean', zorder=4)
        ax.plot(t, mean_all - std_all, c='k', linestyle='-.', label=r'Mean $\pm \, \sigma$', zorder=3)
        ax.plot(t, mean_all + std_all, c='k', linestyle='-.', zorder=3)
        ax.plot(t, sims[:-1].T, c='tab:gray', lw=0.15, zorder=1)
        ax.plot(t, sims[-1], c='tab:gray', lw=0.15, label="Simulations", zorder=1)
        ax.minorticks_on()
