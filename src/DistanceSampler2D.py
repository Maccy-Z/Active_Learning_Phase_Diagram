# Code to plot the results. Run directly for automatic evaluation
from DistanceSampler import DistanceSampler
from gaussian_sampler import suggest_point
from utils import tri_pd, bin_pd, quad_pd, CustomScalarFormatter, to_real_scale
from config import Config

from matplotlib import pyplot as plt
from matplotlib import axes as Axes
import numpy as np
import time


class DistanceSampler2D(DistanceSampler):
    def __init__(self, init_phases, init_Xs, cfg: Config, init_probs=None, save_dir="../saves"):
        super().__init__(init_phases, init_Xs, cfg, init_probs, save_dir)
        assert cfg.N_dim == 2, "2D sampling and plotting only"

    # Load in axes for plotting
    def set_plots(self, axs, fig):
        self.p_obs_ax: Axes = axs[2]
        self.acq_ax: Axes = axs[1]
        self.pd_ax: Axes = axs[0]
        self.fig: plt.Figure = fig

        self.cbs = []

    def _clear_cb(self):
        for cb in self.cbs:
            cb.remove()
        self.cbs = []

    def _show_cb(self, im, ax):
        vmin, vmax = im.get_clim()
        ticks = np.linspace(vmin, vmax, 3)

        fmt = CustomScalarFormatter(useMathText=True)  # FormatStrFormatter('%.1g')
        fmt.set_scientific(True)
        fmt.set_powerlimits((-1, 1))

        cb = self.fig.colorbar(im, ax=ax, ticks=ticks, aspect=30)
        cb.ax.yaxis.set_major_formatter(fmt)
        cb.ax.tick_params(labelsize=10)
        self.cbs.append(cb)

    def plot(self, plot_holder):
        self._clear_cb()
        self.p_obs_ax.clear()
        self.acq_ax.clear()
        self.pd_ax.clear()

        extent = sum(self.cfg.extent, ())

        # Plot probability of error
        error_prob = plot_holder['error_prob']
        max_error_prob = np.amax(error_prob)
        self.p_obs_ax.set_title("Error prob (%)")
        im = self.p_obs_ax.imshow(error_prob, extent=extent,
                                  origin="lower", vmax=max_error_prob, vmin=0., aspect='auto')
        self._show_cb(im, self.p_obs_ax)

        # Plot acquisition function
        avg_dists = plot_holder['acq_fn']
        self.acq_ax.set_title(f'Acquisition fn')
        # self.acq_ax.set_yticks([-2, -1, 0, 1, 2])
        im = self.acq_ax.imshow(avg_dists, extent=extent,
                                origin="lower", aspect='auto')  # , vmin=sec_low)  # Phase diagram
        self._show_cb(im, self.acq_ax)

        # Plot current phase diagram and next sample point
        new_point = plot_holder['new_point']
        pd = plot_holder['pd']
        X_obs, phase_obs, _ = self.obs_holder.get_og_obs()
        xs_train, ys_train = X_obs[:, 0], X_obs[:, 1]
        self.pd_ax.set_title(f"PD and points")
        self.pd_ax.imshow(pd, extent=extent, origin="lower", aspect='auto')  # Phase diagram
        self.pd_ax.scatter(xs_train, ys_train, marker="x", s=30, c=phase_obs, cmap='bwr')  # Existing observations
        self.pd_ax.scatter(new_point[0], new_point[1], s=80, c='tab:orange')  # New observations

        # self.pd_ax.set_yticks([-2, -1, 0, 1, 2])
        # plt.colorbar()

    def single_obs(self):
        """
        Suggest a point and plot the results
        Note plotting here is done in real scale
        """
        new_point, prob_at_point, (pd_old, acq_fn, pd_probs) = suggest_point(self.pool, self.obs_holder, self.cfg)

        new_point = to_real_scale(new_point, self.cfg.extent)

        pd = pd_old.reshape([self.cfg.N_display for _ in range(2)]).T
        acq_fn = acq_fn.reshape([self.cfg.N_eval for _ in range(2)]).T
        error_prob = 1 - np.amax(pd_probs, axis=1)
        error_prob = error_prob.reshape([self.cfg.N_eval for _ in range(2)]).T

        plot_holder = {'new_point': new_point, 'pd': pd, 'acq_fn': acq_fn, 'error_prob': error_prob, 'pd_probs': pd_probs}
        self.plot(plot_holder)

        return new_point, prob_at_point


def main(save_dir):
    pd_fn = quad_pd

    print(save_dir)
    cfg = Config()

    # Init observations to start off
    X_init = [[0.0, -0.5], [0, 0.5]]
    phase_init = [pd_fn(X)[0] for X in X_init]
    prob_init = [pd_fn(X)[1] for X in X_init]

    distance_sampler = DistanceSampler2D(phase_init, X_init, cfg, init_probs=prob_init, save_dir=save_dir)

    # Create three subplots
    fig = plt.figure(figsize=(11, 3.3))
    axes1 = fig.add_subplot(131)
    axes2 = fig.add_subplot(132)
    axes3 = fig.add_subplot(133)

    distance_sampler.set_plots([axes1, axes2, axes3], fig)

    for i in range(51):
        print()
        print("Step:", i)
        st = time.time()
        new_point, prob = distance_sampler.single_obs()
        print(f'Time taken: {time.time() - st:.2f}s')
        print(f'{new_point =}, {prob = }')
        obs_phase, _ = pd_fn(new_point, train=True)

        distance_sampler.add_obs(obs_phase, new_point, prob=0.99)

        # print(distance_sampler.obs_holder.get_og_obs())
        if i % 3 == 0:
            fig.show()


if __name__ == "__main__":
    save_dir = "../saves"

    main(save_dir)
