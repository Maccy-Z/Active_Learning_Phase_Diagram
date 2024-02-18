import argparse
import edit_source_files
import GPy
import numpy as np
import time
import scipy
from matplotlib import pyplot as plt

from utils import ObsHolder, make_grid, new_save_folder
from config import Config
from Distance_Sampler_2D import fit_gp, acquisition

np.set_printoptions(precision=2)
N_phases = 3


def plot(obs_holder, max_probs, avg_dists, pd_old, new_point, save_path):
    i = len(obs_holder.obs_phase) + 1


    # Plot probability of observaion
    plt.subplot(1, 3, 1)
    plt.title("P(obs) ")
    plt.imshow(1 - max_probs.reshape(cfg.N_eval, cfg.N_eval), extent=(-2, 2, -2, 2),
               origin="lower", vmax=1, vmin=0)

    # Plot acquisition function
    plt.subplot(1, 3, 2)
    plt.title(f'Step {i}:  Acq. fn')

    sec_low = np.sort(np.unique(avg_dists))[1]
    try:
        plt.imshow(avg_dists.reshape(cfg.N_eval, cfg.N_eval), extent=(-2, 2, -2, 2),
                   origin="lower", vmin=sec_low)  # Phase diagram
    except ValueError:
        print(sec_low)
        pass

    # Plot current phase diagram and next sample point
    X_obs, Y_obs = obs_holder.get_obs()
    xs_train, ys_train = X_obs[:, 0], X_obs[:, 1]

    plt.subplot(1, 3, 3)
    plt.title(f"PD and points")

    plt.imshow(pd_old.reshape(cfg.N_display, cfg.N_display), extent=(-2, 2, -2, 2), origin="lower")  # Phase diagram
    plt.scatter(xs_train, ys_train, marker="x", s=30, c=Y_obs, cmap='bwr')  # Existing observations
    plt.scatter(new_point[0], new_point[1], s=80, c='tab:orange')  # New observations

    # plt.colorbar()

    plt.savefig(f'{save_path}/{i}.png', bbox_inches="tight")
    plt.show()


def main(X_init, phase_init, cfg, save_dir):
    obs_holder = ObsHolder(cfg)
    print()
    print(f'{save_dir = }')
    save_path = new_save_folder(save_dir)
    print(f'{save_path = }')
    print()

    # Init observations to start off
    for xs, p in zip(X_init, phase_init):
        obs_holder.make_obs(xs, phase=p)

    i = 0
    st = time.time()

    # Fit to existing observations
    models = fit_gp(obs_holder, cfg=cfg)

    # Find max_x A(x)
    new_Xs, _, _ = make_grid(cfg.N_eval, xmin=cfg.xmin, xmax=cfg.xmax)  # Points to test for aquisition

    pd_old, avg_dists, max_probs = acquisition(models, new_Xs, cfg)
    max_pos = np.argmax(avg_dists)
    new_point = new_Xs[max_pos]
    # obs_holder.make_obs(new_point)

    print()
    print()
    print(f'Iter: {i}, Time: {time.time() - st:.3g}s ')

    print()
    print("New point to sample from:", new_point)


    plot(obs_holder, max_probs, avg_dists, pd_old, new_point, save_path)

    obs_holder.save(save_path)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enter initial observations")
    parser.add_argument("X", help="X coordinate of observations, comma seperated")
    parser.add_argument("Y", help="Y coordinate of observations, comma seperated")
    parser.add_argument("obs", help="Observed phase")


    args = parser.parse_args()

    xs = np.array([float(x) for x in args.X.split(",")])
    ys = np.array([float(y) for y in args.Y.split(",")])
    obs = np.array([int(o) for o in args.obs.split(",")])
    assert len(xs) == len(ys) == len(obs), "Number of observations and positions must be equal"

    Xs = np.stack([xs, ys]).T
    obs = np.array(obs)

    # Length check

    print()
    print(f'{Xs = }')
    print(f'{obs = }')

    cfg = Config()
    main(Xs, obs, cfg, save_dir="../saves")


