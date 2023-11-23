# Code to plot the results. Run directly for automatic evaluation
from matplotlib import pyplot as plt
from matplotlib import axes as Axes
import numpy as np
import time

from gaussian_sampler_new import suggest_point, suggest_two_points
from utils import ObsHolder, new_save_folder, tri_pd, bin_pd, quad_pd, CustomScalarFormatter
from config import Config


class DistanceSampler:
    def __init__(self, init_phases, init_Xs, cfg: Config, save_dir="./saves"):
        save_path = new_save_folder(save_dir)
        print(f'{save_path = }')
        print()

        self.cfg = cfg
        self.obs_holder = ObsHolder(self.cfg, save_path=save_path)

        # Check if initial inputs are within allowed area
        points = np.array(init_Xs)
        if points.shape[1] != self.cfg.N_dim:
            raise ValueError(f"Number of dimensions must be {self.cfg.N_dim}, got {points.shape[1]}")
        bounds = np.array(self.cfg.extent)
        mins, maxs = bounds[:, 0], bounds[:, 1]

        all_in = np.logical_and(points >= mins, points <= maxs).all(axis=1)
        if not all_in.all():
            print("\033[31mWarning: Observations are outside search area\033[0m")

        # Check phases are allowed
        for phase, X in zip(init_phases, init_Xs, strict=True):
            self.obs_holder.make_obs(X, phase=phase)

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

    def plot(self, first_point, pd_old, avg_dists, pd_probs, sec_point=None):
        self._clear_cb()
        self.p_obs_ax.clear()
        self.acq_ax.clear()
        self.pd_ax.clear()

        # Reshape 1d array to 2d
        side_length = int(np.sqrt(len(pd_old)))
        pd_old = pd_old.reshape((side_length, side_length)).T
        side_length = int(np.sqrt(len(avg_dists)))
        avg_dists = avg_dists.reshape((side_length, side_length)).T
        max_probs = np.amax(pd_probs, axis=1)
        side_length = int(np.sqrt(len(max_probs)))
        max_probs = max_probs.reshape((side_length, side_length)).T

        extent = sum(self.cfg.extent, ())

        # Plot probability of observaion
        self.p_obs_ax.set_title("Error prob (%)")
        self.p_obs_ax.set_yticks([-2, -1, 0, 1, 2])
        im = self.p_obs_ax.imshow(1 - max_probs, extent=extent,
                                  origin="lower", vmax=1, vmin=0, aspect='auto')
        self._show_cb(im, self.p_obs_ax)

        # Plot acquisition function
        self.acq_ax.set_title(f'Acquisition fn')
        self.acq_ax.set_yticks([-2, -1, 0, 1, 2])
        im = self.acq_ax.imshow(avg_dists, extent=extent,
                                origin="lower", aspect='auto')  # , vmin=sec_low)  # Phase diagram

        self._show_cb(im, self.acq_ax)

        # Plot current phase diagram and next sample point
        X_obs, phase_obs = self.obs_holder.get_og_obs()
        xs_train, ys_train = X_obs[:, 0], X_obs[:, 1]
        self.pd_ax.set_title(f"PD and points")
        self.pd_ax.imshow(pd_old, extent=extent, origin="lower", aspect='auto')  # Phase diagram
        self.pd_ax.scatter(xs_train, ys_train, marker="x", s=30, c=phase_obs, cmap='bwr')  # Existing observations
        self.pd_ax.scatter(first_point[0], first_point[1], s=80, c='tab:orange')  # New observations
        if sec_point is not None:
            self.pd_ax.scatter(sec_point[0], sec_point[1], s=80, c='r')  # New observations

        self.pd_ax.set_yticks([-2, -1, 0, 1, 2])
        # plt.colorbar()

    def make_obs(self, phase: int, X: np.ndarray):
        self.obs_holder.make_obs(X, phase=phase)
        self.obs_holder.save()

        new_point, (pd_old, avg_dists, pd_probs), prob_at_point = suggest_point(self.obs_holder, self.cfg)
        self.plot(new_point, pd_old, avg_dists, pd_probs)

        return new_point, prob_at_point

    def single_obs(self):
        self.obs_holder.save()

        new_point, prob_at_point, (pd_old, avg_dists, pd_probs) = suggest_point(self.obs_holder, self.cfg)
        self.plot(new_point, pd_old, avg_dists, pd_probs)

        return new_point, prob_at_point

    def dual_obs(self):
        first_point, sec_point, (pd_old, avg_dists, pd_probs) = suggest_two_points(self.obs_holder, self.cfg)
        self.plot(first_point, pd_old, avg_dists, pd_probs, sec_point=sec_point)

        return first_point, sec_point


def main(save_dir):
    pd_fn = tri_pd

    print(save_dir)
    cfg = Config()

    # Init observations to start off
    # X_init, _, _ = make_grid(cfg.N_init, cfg.extent)
    X_init = [[0.0, 1.], [0, -1]]
    phase_init = [pd_fn(X) for X in X_init]

    distance_sampler = DistanceSampler(phase_init, X_init, cfg, save_dir=save_dir)

    fig = plt.figure(figsize=(11, 3.3))

    # Create three subplots
    axes1 = fig.add_subplot(131)
    axes2 = fig.add_subplot(132)
    axes3 = fig.add_subplot(133)

    distance_sampler.set_plots([axes1, axes2, axes3], fig)

    for i in range(51):
        print()
        print("Step:", i)
        st = time.time()
        new_points, prob = distance_sampler.single_obs()
        print(f'Time taken: {time.time() - st:.2f}s')
        print(f'{new_points =}, {prob = }')
        new_points = np.array(new_points).reshape(-1, 2)
        for p in new_points:
            obs_phase = pd_fn(p)
            distance_sampler.obs_holder.make_obs(p, obs_phase)

        # print(distance_sampler.obs_holder.get_og_obs())

        fig.show()
        # exit(5)


if __name__ == "__main__":
    save_dir = "./saves"

    main(save_dir)
