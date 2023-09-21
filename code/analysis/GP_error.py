# Evaluate error of GP model
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

from code.utils import make_grid, ObsHolder, tri_pd, bin_pd, quad_pd
from code.config import Config
from code.gaussian_sampler import fit_gp


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
        y_pred, _ = pd_model.predict(xs, include_likelihood=True)  # m.shape = [n**2, 1]

        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    sample_pds = np.argmax(y_preds, axis=0).T
    return None, None, sample_pds


# Distance between true PD and prediction
def dist(obs_holder, *, pd_fn, cfg, points, t, ax ):
    # Observations up to time t
    T = t + 2
    Xs, Ys = obs_holder.get_og_obs()

    obs_holder._obs_pos = Xs[:T]
    obs_holder.obs_phase = Ys[:T]

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

    if t % 10 == 0:
        ax.set_title(f'{t = }')
        ax.imshow(pds, origin="lower", extent=cfg.extent)
        ax.scatter(Xs[:T, 0], Xs[:T, 1], marker="x", s=10, c=Ys[:T], cmap='bwr')  # Existing observations

        ax.set_xticks(np.linspace(-2, 2, 3), labels = np.linspace(-2, 2, 3),fontsize=11)
        ax.set_yticks(np.linspace(-2, 2, 3), labels = np.linspace(-2, 2, 3),fontsize=11)
    return diff_mean


def main():
    pd_fn = quad_pd
    f = sorted([int(s) for s in os.listdir("./saves")])

    save_name = f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'./saves/{save_name}')

    with open(f'./saves/{save_name}/cfg.pkl', "rb") as f:
        cfg = pickle.load(f)

    fig, axs = plt.subplots(2, 5, figsize=(10, 4))
    errors = []
    for i, t in enumerate(range(10, len(og_obs.obs_phase) - 1, 10)):
        print(t)
        # Plot number
        row = i // 5  # Integer division to get the row index
        col = i % 5  # Modulo to get the column index
        # Select the current subplot to update
        print(i, row,col)
        ax = axs[row, col]

        obs_holder = copy.deepcopy(og_obs)
        error = dist(obs_holder, pd_fn=pd_fn, points=19, t=t, cfg=cfg, ax=ax)
        errors.append(error)
    plt.subplots_adjust(left=0.04, right=0.99, top=0.95, bottom=0.05, wspace=0.4, hspace=0.4)
    #plt.tight_layout()
    plt.show()

    # plt.plot(errors)
    # plt.show()
    #
    # print()
    # for s in [10, 20, 30, 40, 50]:
    #     print(f'{errors[s]}')

    print()
    print(errors)


if __name__ == "__main__":
    main()
