# Evaluate error of GP model
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))

import pickle
import numpy as np
import math
from matplotlib import pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D

from src.utils import make_grid, ObsHolder, c_print
from src.config import Config
from src.gaussian_sampler import fit_gp, gen_pd


# Distance between true PD and prediction
def dist(obs_holder, *, true_pd, cfg, points, t):
    # Truncate observations up to time t, assuming first 2 observations are given.
    Xs, Ys, prob = obs_holder.get_og_obs()
    T = t + 2
    obs_holder._obs_pos = Xs[:T]
    obs_holder.obs_phase = Ys[:T]
    obs_holder.obs_prob = prob[:T]

    model = fit_gp(obs_holder, cfg=cfg)

    X_eval, _ = make_grid(points, cfg.unit_extent)
    probs = gen_pd(model, X_eval, cfg=cfg)

    pd_pred = np.argmax(probs, axis=1).reshape(*points)
    true_pd_grid = np.stack(true_pd).reshape(*points)

    diff = np.not_equal(pd_pred, true_pd_grid)
    diff_mean = np.mean(diff)

    if t == 240:
        plt_phase = pd_pred.flatten()
        plt_true = true_pd_grid.flatten()

        print(f'{plt_true.shape = }')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_eval[:, 0], X_eval[:, 1], X_eval[:, 2], c=plt_true)
        plt.show()

        print(f'{plt_phase.shape = }')
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X_eval[:, 0], X_eval[:, 1], X_eval[:, 2], c=plt_phase)
        plt.show()

    return diff_mean


def deduplicate(obs_holder: ObsHolder):
    c_print("Deduplicating observations", "green")
    Xs, phases, probs = obs_holder.get_og_obs()
    seen_Xs, seen_phase, seen_prob = [], [], []
    for X, phase, prob in zip(Xs, phases, probs):
        # Check if the current array is identical to any of the arrays seen so far
        if not any(np.array_equal(X, seen_arr) for seen_arr in seen_Xs):
            seen_Xs.append(X)
            seen_phase.append(phase)
            seen_prob.append(prob)
        else:
            print(f"Duplicate found: {X}, phase: {phase}")

    obs_holder._obs_pos = seen_Xs
    obs_holder.obs_phase = seen_phase
    obs_holder.obs_prob = seen_prob


def main():
    true_pd = np.array([[[1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                        [[0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                        [[0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                        [[0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                        [[0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2], [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2], [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2], [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2]],
                        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1, 1, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2], [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2], [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2],
                         [0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2], [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2], [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2],
                         [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2]],
                        [[0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2], [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2],
                         [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2], [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 2], [0, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2], [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2],
                         [0, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]],
                        [[0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2], [0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2], [0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2], [0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2],
                         [0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 2]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 2],
                         [0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
                         [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1]],
                        [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                         [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]]])

    c_print("Reducing true_pd to 11x11x10", color='magenta')
    true_pd = true_pd[:, :, 1:]
    print(true_pd.shape)

    # true_pd = np.zeros([11, 11])
    true_pd = true_pd.astype(int)

    eval_points = true_pd.shape
    c_print(f'Evaluation points = {eval_points}', color='yellow')
    c_print(f'Note: Plotting is done with x and y axis inverted at z=0', color='yellow')
    assert len(true_pd) != 0, ("Fill in the true phase diagram as a 2D numpy array, with the same number of points as eval_points. "
                               "It might need transposing / reflecting to orient properly. ")
    # plt.imshow(true_pd[10, :, :], origin='lower')
    # plt.show()
    f = sorted([int(s) for s in os.listdir("../saves")])
    save_name = f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'../saves/{save_name}')
    deduplicate(og_obs)

    with open(f'../saves/{save_name}/cfg.pkl', "rb") as f:
        cfg: Config = pickle.load(f)
    assert true_pd.ndim == cfg.N_dim, "Observation dimension must be the same as model dimension"
    print(cfg.N_dist, true_pd.shape)

    errors = []
    for t in range(len(og_obs.obs_phase)):
        obs_holder = copy.deepcopy(og_obs)
        error = dist(obs_holder, true_pd=true_pd, points=eval_points, t=t, cfg=cfg)
        errors.append(error)
        print(f'{t = }, {error = :.3g}')

    plt.plot(errors)
    plt.ylim([0, 0.5])
    plt.show()

    print()
    for s in [10, 20, 30, 40, 50]:
        print(f'{errors[s]}')

    print("Errors:")
    print("Copy me to error_plot.py")
    print(errors)


if __name__ == "__main__":
    main()
