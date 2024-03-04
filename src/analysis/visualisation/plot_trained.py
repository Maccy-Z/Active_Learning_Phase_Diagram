# Analyse trained GP models. Plot predictions at given time
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(parent_dir)
cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))
print(os.getcwd())

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.ticker as ticker

from src.DistanceSampler2D import DistanceSampler2D
from src.utils import ObsHolder
from src.gaussian_sampler import gen_pd, fit_gp
from src.utils import make_grid, Config


def plot_all():
    T = 150
    obs_holder = ObsHolder.load("./saves/14")
    obs_holder._obs_pos = obs_holder._obs_pos[:T]
    obs_holder.obs_phase = obs_holder.obs_phase[:T]
    cfg = obs_holder.cfg

    distance_sampler = DistanceSampler2D(init_phases=obs_holder.obs_phase, init_Xs=obs_holder._obs_pos, init_probs=None, cfg=cfg, save_dir="/tmp")

    # Create three subplots
    fig = plt.figure(figsize=(11, 3.3))
    axes1 = fig.add_subplot(131)
    axes2 = fig.add_subplot(132)
    axes3 = fig.add_subplot(133)

    distance_sampler.set_plots([axes1, axes2, axes3], fig)

    distance_sampler.single_obs()

    # def set_ticks(ax):
    #     ax.set_xticks(np.linspace(-2, 2, 3), labels=np.linspace(-2, 2, 3).astype(int), fontsize=11)
    #     ax.set_yticks(np.linspace(-2, 2, 3), labels=np.linspace(-2, 2, 3).astype(int), fontsize=11)
    #
    # set_ticks(axes1)
    # set_ticks(axes2)
    # set_ticks(axes3)

    plt.subplots_adjust(left=0.04, right=0.99, top=0.92, bottom=0.08, wspace=0.15, hspace=0.4)

    # fig.tight_layout()
    fig.show()


def plot_single(T, ax):
    obs_holder = ObsHolder.load("../saves/0")
    obs_holder._obs_pos = obs_holder._obs_pos[:T]
    obs_holder.obs_phase = obs_holder.obs_phase[:T]
    obs_holder.obs_prob = obs_holder.obs_prob[:T]
    cfg = obs_holder.cfg

    # Generate predictions
    model = fit_gp(obs_holder, cfg=cfg)
    X_display, _ = make_grid(cfg.N_display, cfg.unit_extent)
    probs_old = gen_pd(model, X_display, cfg=cfg)
    pd_old = np.argmax(probs_old, axis=1)
    pd = pd_old.reshape([cfg.N_display for _ in range(2)]).T

    # Plot observations
    X_obs, phase_obs, _ = obs_holder.get_og_obs()
    xs_train, ys_train = X_obs[:, 0], X_obs[:, 1]

    ax.scatter(xs_train, ys_train, marker="x", s=30, c=phase_obs, cmap='bwr')
    ax.imshow(pd, origin="lower", extent=sum(cfg.extent, ()), aspect='auto')

    # Set plot pretty format
    ax.set_box_aspect(1)
    ax.set_xticks(cfg.extent[0])
    ax.set_yticks(cfg.extent[1])
    ax.tick_params(labelsize=16)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.0f'))

def main():

    n_rows, n_cols = 2, 5
    plot_steps = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    assert n_rows * n_cols == len(plot_steps), "Number of subplots must match number of steps to plot."
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(15, 6))

    for T, ax in zip(plot_steps, axs.flatten()):
        print("T:", T)
        plot_single(T, ax)
        ax.set_title(f"T={T}", fontsize=15)

    plt.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.07, wspace=0.1, hspace=0.2)
    # plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
