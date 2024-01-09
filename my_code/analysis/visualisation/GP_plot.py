# Evaluate error of GP model and plot predictions
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))

import pickle
import GPy
import numpy as np
import math
from matplotlib import pyplot as plt
import copy

from utils import make_grid, ObsHolder, tri_pd, bin_pd, quad_pd
from config import Config
from gaussian_sampler import fit_gp


# Sample phase diagrams from models.
def single_pds(models: list[GPy.core.GP], xs):
    """
    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: List of possible phase diagrams.
    """
    y_preds = []
    for phase_i, pd_model in enumerate(models):
        y_pred, _ = pd_model.predict(xs, include_likelihood=False)  # m.shape = [n**2, 1]
        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    sample_pds = np.argmax(y_preds, axis=0).T
    return None, None, sample_pds


# Distance between true PD and prediction
def dist(obs_holder, *, pd_fn, cfg, points, t, ax, true_t):
    # Observations up to time t
    Xs, Ys = obs_holder.get_og_obs()

    Xs = obs_holder._obs_pos = Xs[:true_t+1]
    Ys = obs_holder.obs_phase = Ys[:true_t+1]

    plot_Xs, X1, X2 = make_grid(points, cfg.extent)
    model_Xs, _, _ = make_grid(points, (0, 1, 0, 1))
    models = fit_gp(obs_holder, cfg=cfg)
    pds = single_pds(models, model_Xs)[2].reshape(points, points)

    true_pd = []
    for X in plot_Xs:
        true_phase = pd_fn(X, train=False)
        true_pd.append(true_phase)

    true_pd = np.stack(true_pd).reshape(points, points)

    diff = np.not_equal(pds, true_pd)
    diff_mean = np.mean(diff)

    ax.set_title(f'{t = }')
    ax.imshow(pds, origin="lower", extent=cfg.extent, aspect='auto')
    ax.scatter(Xs[:true_t+1, 0], Xs[:true_t+1, 1], marker="x", s=10, c=Ys[:true_t+1], cmap='bwr')  # Existing observations

    # Here set the ticks and labels
    xmin, xmax, ymin, ymax = cfg.extent
    ax.set_xticks(np.linspace(xmin, xmax, 2), labels=np.linspace(xmin, xmax, 2), fontsize=12)
    ax.set_yticks(np.linspace(ymin, ymax, 2), labels=np.linspace(ymin, ymax, 2), fontsize=11)
    return diff_mean


def deduplicate(array_list):
    seen = []  # List to store the distinct arrays seen so far
    count_dict = {}  # Dictionary to store the result

    count = 0
    for idx, arr in enumerate(array_list):
        # Check if the current array is identical to any of the arrays seen so far
        if not any(np.array_equal(arr, seen_arr) for seen_arr in seen):
            seen.append(arr)
            count += 1
        count_dict[idx] = count

    reversed_dict = {}
    for key, value in count_dict.items():
        if value not in reversed_dict:
            reversed_dict[value] = key
    return reversed_dict


def main():
    pd_fn = bin_pd
    f = sorted([int(s) for s in os.listdir("./saves")])

    # Select which save file to load
    save_name = f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'./saves/{save_name}')

    all_obs = og_obs._obs_pos
    obs_map = deduplicate(all_obs)
    print(obs_map)
    with open(f'./saves/{save_name}/cfg.pkl', "rb") as f:
        cfg = pickle.load(f)

    fig, axs = plt.subplots(2, 4, figsize=(10, 4))
    errors = []
    for i, t in enumerate(range(6, 86, 10)):
        print(t)
        true_t = obs_map[t]
        # print(t)

        # Select the current subplot to update
        row = i // 4  # Integer division to get the row index
        col = i % 4  # Modulo to get the column index
        ax = axs[row, col]

        obs_holder = copy.deepcopy(og_obs)

        error = dist(obs_holder, pd_fn=pd_fn, points=19, true_t=true_t, cfg=cfg, ax=ax, t=t)
        errors.append(error)
    plt.subplots_adjust(left=0.05, right=0.98, top=0.94, bottom=0.07, wspace=0.4, hspace=0.3)
    # plt.tight_layout()
    plt.show()

    print()
    print(errors)


if __name__ == "__main__":
    main()
