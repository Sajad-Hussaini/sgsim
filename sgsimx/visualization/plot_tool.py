import numpy as np
import matplotlib.pyplot as plt

def plot_ac_ce(model, target):
    """
    Comparing the cumulative energy and energy distribution
    of the record, model, and simulations
    """
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=False)
    axes[0].plot(target.t, target.ac, c='tab:blue')
    axes[0].plot(model.t, model.mdl, c='tab:red', ls='--')
    axes[0].plot(model.t, -model.mdl, c='tab:red', ls='--')
    axes[0].axhline(y=0, color='k', ls='--', lw=0.15)
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylim([-1.05 * max(abs(target.ac)), 1.05 * max(abs(target.ac))])
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
    axes[0].minorticks_on()

    axes[1].plot(target.t, target.ce, label= 'Target', c='tab:blue')
    axes[1].plot(model.t, model.ce, label= 'Model', c='tab:red', ls='--')
    axes[1].set_ylabel(r'Cumulative energy $\mathregular{(g^2.s)}$')
    axes[1].set_xlabel('Time (s)')
    axes[1].legend(loc='lower right', frameon=False)
    # axes[1].ticklabel_format(axis='y', style='sci', scilimits=(-3,3))
    axes[1].minorticks_on()

def plot_feature(model, sim, target, feature='mzc'):
    """
    Comparing the indicated error of the record, model, and simulations
    mzc, mle, pmnm
    """
    plt.plot(target.t, getattr(target, f"{feature}_ac"), label="Target acceleration",
            c='tab:blue', zorder=2) if feature == 'mzc' else None
    plt.plot(target.t, getattr(target, f"{feature}_vel"), label="Target velocity",
            c='tab:orange', linestyle='--', zorder=2)
    plt.plot(target.t, getattr(target, f"{feature}_disp"), label="Target displacement",
            c='tab:green', linestyle='-.', zorder=2)

    if model is not None:
        plt.plot(model.t, getattr(model, f"{feature}_ac"),
                label="Model acceleration", c='tab:red', linestyle=(0, (4, 1, 4, 1, 1, 1)),
                zorder=3)  if feature == 'mzc' else None
        plt.plot(model.t, getattr(model, f"{feature}_vel"),
                label="Model velocity", c='black', linestyle=(0, (6, 8)),
                zorder=3)
        plt.plot(model.t, getattr(model, f"{feature}_disp"),
                label="Model displacement", c='brown',
                linestyle=(0,(4,2,1,1,1,2)), zorder=3)

    if sim is not None:
        plt.plot(sim.t, getattr(sim, f"{feature}_ac").T,
                color='tab:gray', lw=0.15)  if feature == 'mzc' else None
        plt.plot(sim.t, getattr(sim, f"{feature}_vel")[:-1].T,
                color='tab:gray', lw=0.15)
        plt.plot(sim.t, getattr(sim, f"{feature}_vel")[-1],
                color='tab:gray', lw=0.15, label="Simulations")
        plt.plot(sim.t, getattr(sim, f"{feature}_disp").T,
                color='tab:gray', lw=0.15)

    plt.legend(loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False)
    plt.xlabel("Time (s)")
    plt.ylabel("Cumulative mean zero crossing" if feature == 'mzc'
            else "Cumulative mean local extrema" if feature == 'mle'
            else 'Cumulative mean positive-minima\nand negative-maxima')

def plot_motion(t, sim1, sim2, rec, ylabel='Acceleration (g)'):
    """
    3 time series multiple plot
    """
    fig, axes = plt.subplots(1, 3, sharex=True, sharey=True)
    axes[0].plot(t, rec, label='Target', color='tab:blue')
    axes[0].set_ylabel(f'{ylabel}')
    axes[0].yaxis.set_major_locator(plt.MaxNLocator(5, symmetric=True))
    axes[0].minorticks_on()
    # axes.flatten()[0].set_ylim([-1.05 * max(abs(rec)), 1.05 * max(abs(rec))])
    for idx, sim_data in enumerate([sim1, sim2], start=1):
        # sim_data = sim_data - np.linspace(0.0, sim_data[-1], len(sim_data))
        axes[idx].plot(t, sim_data, label='Simulation', color='tab:red')
    for ax in axes:
        ax.axhline(y=0, color='k', linestyle='--', lw=0.15)
        ax.set_xlabel('Time (s)')
        ax.legend(loc='lower right', frameon=False, handlelength=0)

def plot_mean_std(t, sims, rec):
    """
    Plot the common part of ce_plot and fas_plot
    """
    mean_all = np.mean(sims, axis=0)
    std_all = np.std(sims, axis=0)
    plt.plot(t, rec.flatten(), c='tab:blue', label='Target', zorder=2)
    plt.plot(t, mean_all, c='tab:red', linestyle='--', label='Mean', zorder=4)
    plt.plot(t, mean_all-std_all, c='k', linestyle='-.', label=r'Mean $\mathregular{\pm \, \sigma}$', zorder=3)
    plt.plot(t, mean_all+std_all, c='k', linestyle='-.', zorder=3)
    plt.plot(t, sims[:-1].T, c='tab:gray', lw=0.15, zorder=1)
    plt.plot(t, sims[-1], c='tab:gray', lw=0.15, label="Simulations", zorder=1)
    plt.minorticks_on()
