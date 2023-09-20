from matplotlib import pyplot as plt
from matplotlib import axes as Axes
from matplotlib.ticker import FormatStrFormatter
import numpy as np

from gaussian_sampler import suggest_point, suggest_two_points
from utils import ObsHolder, make_grid, new_save_folder, tri_pd, bin_pd, quad_pd
from config import Config


class DistanceSampler:
    def __init__(self, init_phases, init_Xs, cfg, save_dir="./saves"):
        save_path = new_save_folder(save_dir)
        print(f'{save_path = }')
        print()

        self.cfg = cfg
        self.obs_holder = ObsHolder(self.cfg, save_path=save_path)

        # Check if initial inputs are within allowed area
        vectors = np.array(init_Xs)
        xmin, xmax, ymin, ymax = self.cfg.extent
        inside_x = np.logical_and(vectors[:, 0] >= xmin, vectors[:, 0] <= xmax)
        inside_y = np.logical_and(vectors[:, 1] >= ymin, vectors[:, 1] <= ymax)
        in_area = np.all(inside_x & inside_y)
        if not in_area:
            print("\033[31mWarning: Observations are outside search area\033[0m")

        # Check phases are allowed

        for phase, X in zip(init_phases, init_Xs, strict=True):
            self.obs_holder.make_obs(X, phase=phase)

    # Load in axes for plotting
    def set_plots(self, axs, fig):
        self.p_obs_ax: Axes = axs[0]
        self.acq_ax: Axes = axs[1]
        self.pd_ax: Axes = axs[2]
        self.fig: plt.Figure = fig

        self.cbs = []

    def _clear_cb(self):
        for cb in self.cbs:
            cb.remove()
        self.cbs = []

    def _show_cb(self, im, ax):

        vmin, vmax = im.get_clim()
        ticks = np.linspace(vmin, vmax, 3)
        fmt = FormatStrFormatter('%.1g')
        cb = self.fig.colorbar(im, ax=ax, ticks=ticks, aspect=30)
        cb.ax.yaxis.set_major_formatter(fmt)
        cb.ax.tick_params(labelsize=8)

        self.cbs.append(cb)

    def plot(self, first_point, pd_old, avg_dists, pd_probs, sec_point=None):
        self._clear_cb()
        self.p_obs_ax.clear()
        self.acq_ax.clear()
        self.pd_ax.clear()

        # Reshape 1d array to 2d
        side_length = int(np.sqrt(len(pd_old)))
        pd_old = pd_old.reshape((side_length, side_length))
        side_length = int(np.sqrt(len(avg_dists)))
        avg_dists = avg_dists.reshape((side_length, side_length))
        max_probs = np.amax(pd_probs, axis=1)
        side_length = int(np.sqrt(len(max_probs)))
        max_probs = max_probs.reshape((side_length, side_length))

        # Plot probability of observaion
        self.p_obs_ax.set_title("P(obs)")
        im = self.p_obs_ax.imshow(1 - max_probs, extent=self.cfg.extent,
                                  origin="lower", vmax=1, vmin=0, aspect='auto')
        self._show_cb(im, self.p_obs_ax)

        # Plot acquisition function
        self.acq_ax.set_title(f'Acq. fn')
        im = self.acq_ax.imshow(avg_dists, extent=self.cfg.extent,
                                origin="lower", aspect='auto')  # , vmin=sec_low)  # Phase diagram

        self._show_cb(im, self.acq_ax)

        # Plot current phase diagram and next sample point
        X_obs, phase_obs = self.obs_holder.get_og_obs()
        xs_train, ys_train = X_obs[:, 0], X_obs[:, 1]
        self.pd_ax.set_title(f"PD and points")
        self.pd_ax.imshow(pd_old, extent=self.cfg.extent, origin="lower", aspect='auto')  # Phase diagram
        self.pd_ax.scatter(xs_train, ys_train, marker="x", s=30, c=phase_obs, cmap='bwr')  # Existing observations
        self.pd_ax.scatter(first_point[0], first_point[1], s=80, c='tab:orange')  # New observations
        if sec_point is not None:
            self.pd_ax.scatter(sec_point[0], sec_point[1], s=80, c='r')  # New observations
        # plt.colorbar()

    def make_obs(self, phase: int, X: np.ndarray):
        self.obs_holder.make_obs(X, phase=phase)
        self.obs_holder.save()

        new_point, (pd_old, avg_dists, pd_probs), prob_at_point = suggest_point(self.obs_holder, self.cfg)
        self.plot(new_point, pd_old, avg_dists, pd_probs)

        return new_point, prob_at_point

    def single_obs(self):
        self.obs_holder.save()

        new_point, (pd_old, avg_dists, pd_probs), prob_at_point = suggest_point(self.obs_holder, self.cfg)
        self.plot(new_point, pd_old, avg_dists, pd_probs)

        return new_point, prob_at_point

    def dual_obs(self):
        first_point, sec_point, (pd_old, avg_dists, pd_probs) = suggest_two_points(self.obs_holder, self.cfg)
        self.plot(first_point, pd_old, avg_dists, pd_probs, sec_point=sec_point)

        return first_point, sec_point


def main(save_dir):
    pd_fn = bin_pd

    print(save_dir)
    cfg = Config()

    # Init observations to start off
    # X_init, _, _ = make_grid(cfg.N_init, cfg.extent)
    X_init = [[0.0, 1.], [0, -1.25]]
    phase_init = [pd_fn(X) for X in X_init]

    distance_sampler = DistanceSampler(phase_init, X_init, cfg, save_dir=save_dir)

    fig = plt.figure(figsize=(11, 3.3))

    # Create three subplots
    axes1 = fig.add_subplot(131)
    axes2 = fig.add_subplot(132)
    axes3 = fig.add_subplot(133)

    distance_sampler.set_plots([axes1, axes2, axes3], fig)

    for i in range(cfg.steps):
        print()
        print("Step:", i)
        new_points, prob = distance_sampler.single_obs()
        print(f'{new_points =}, {prob = }')
        fig.show()

        new_points = np.array(new_points).reshape(-1, 2)
        for p in new_points:
            obs_phase = pd_fn(p)
            distance_sampler.obs_holder.make_obs(p, obs_phase)


if __name__ == "__main__":
    save_dir = "./saves"

    main(save_dir)
