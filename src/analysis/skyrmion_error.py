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

from src.utils import make_grid, ObsHolder, c_print, synth_3d_pd
from src.config import Config
from src.gaussian_sampler import fit_gp, gen_pd
from src.experiments.pd import skyrmion_pd_2D, skyrmion_pd_3D


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
    c_print(f'{len(seen_Xs)} unique observations', 'green')


# Distance between true PD and prediction
def dist(obs_holder, *, true_pd, cfg, points, t, n_dim):
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

    # if t == 500:
    #     if n_dim == 3:
    #         plt_phase = pd_pred.flatten()
    #         plt_true = true_pd_grid.flatten()
    #         print(f'{plt_true.shape = }')
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.scatter(X_eval[:, 0], X_eval[:, 1], X_eval[:, 2], c=plt_true)
    #         plt.show()
    #
    #         print(f'{plt_phase.shape = }')
    #         fig = plt.figure()
    #         ax = fig.add_subplot(111, projection='3d')
    #         ax.scatter(X_eval[:, 0], X_eval[:, 1], X_eval[:, 2], c=plt_phase)
    #         plt.show()
    #     elif n_dim == 2:
    #         plt.imshow(true_pd_grid, origin='lower')
    #         plt.show()
    #         plt.imshow(pd_pred, origin='lower')
    #         plt.show()

    return diff_mean


def main():
    # Get true pd
    pd_fn = skyrmion_pd_3D
    Extent = ((0, 1.), (0., 1.), (0.1, 1.))
    n_dim = 3
    grid = (11, 11, 10)

    points, _ = make_grid(grid, Extent)
    # points = to_real_scale(points, Extent)
    true_pd = [pd_fn(X) for X in points]
    true_pd = np.stack(true_pd).reshape(*grid)  # .T
    # true_pd = np.flip(true_pd, axis=1)

    eval_points = true_pd.shape
    c_print(f'Evaluation points = {eval_points}', color='yellow')
    assert len(true_pd) != 0, ("Fill in the true phase diagram as a 2D numpy array, with the same number of points as eval_points. "
                               "It might need transposing / reflecting to orient properly. ")

    # Load model
    #f = sorted([int(s) for s in os.listdir("../saves")])
    save_name = "10-95"  #f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'../saves/{save_name}')
    deduplicate(og_obs)

    with open(f'../saves/{save_name}/cfg.pkl', "rb") as f:
        cfg: Config = pickle.load(f)
    assert true_pd.ndim == cfg.N_dim, "Observation dimension must be the same as model dimension"
    print(cfg.N_dist, true_pd.shape)
    assert cfg.N_dist == true_pd.shape

    # Calculate errors
    errors = []
    # for t in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]:
    for t in range(len(og_obs.obs_phase)):
        obs_holder = copy.deepcopy(og_obs)
        error = dist(obs_holder, true_pd=true_pd, points=eval_points, t=t, cfg=cfg, n_dim=n_dim)
        errors.append(error)
        print(f'{t = }, {error = :.3g}')

    print()
    print("Errors:")
    print("Copy me to error_plot.py")
    print(errors)

    plt.plot(errors)
    plt.ylim([0, 0.5])
    plt.show()


if __name__ == "__main__":
    main()
