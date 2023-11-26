import copy
import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

import my_code.edit_source_files
import pickle
import GPy
import numpy as np
import math
from matplotlib import pyplot as plt
import scipy.stats as stats

from my_code.config import Config
from my_code.utils import make_grid, ObsHolder

cfg = Config()


# Sample phase diagrams from models.
def single_pds(models: list[GPy.core.GP], xs, sample=None):
    """
    @param models: List of models for each phase
    @param xs: Points to sample from
    @param sample: Use mean or sample from GP.
    @return: List of possible phase diagrams.
    """
    y_preds, y_vars = [], []
    for phase_i, pd_model in enumerate(models):
        # Base
        y_pred, var = pd_model.predict(xs, include_likelihood=False)
        # liklyhooded
        y_pred = pd_model.likelihood.predictive_mean(y_pred, var)

        y_preds.append(y_pred)
        y_vars.append(var)

    y_preds = np.stack(y_preds)
    y_vars = np.stack(y_vars)

    # sample_pds = np.argmax(y_preds, axis=0).T
    # print(y_preds.shape)
    sample_pds = y_preds.swapaxes(1, 2).reshape(-1, y_preds.shape[1])
    sample_vars = y_vars.swapaxes(1, 2).reshape(-1, y_preds.shape[1])
    return sample_pds, sample_vars


# Fit model to existing observations
def fit_gp(obs_holder, t) -> list[GPy.core.GP]:
    "Trains a model for each phase."
    X, Y = obs_holder.get_obs()

    var, r = obs_holder.get_kern_param()
    # var, r = 10, 2

    print(f'{var = }, {r = }')

    models = []
    for i in range(3):
        phase_i = (Y == i)   # Between -1 and 1
        kernel = GPy.kern.Matern52(input_dim=2, variance=var, lengthscale=r)
        model = GPy.models.GPClassification(X, phase_i.reshape(-1, 1), kernel)

        models.append(model)
    return models


# Plot a grid of samples
def plot_samples(obs_holder, points=25, t=None):
    # Observations up to time t
    if t is not None:
        T = t + obs_holder.cfg.N_init ** 2
        obs_holder.obs_pos = obs_holder.obs_pos[:T]
        obs_holder.obs_phase = obs_holder.obs_phase[:T]

    Xs = np.array(obs_holder.obs_pos)
    obs = np.array(obs_holder.obs_phase)

    plot_Xs, X1, X2 = make_grid(points, cfg.extent)

    models = fit_gp(obs_holder, t=t)
    pds, vars = single_pds(models, plot_Xs, sample=None)
    pds = np.concatenate([pds, np.sqrt(vars)], axis=0)

    n_img = len(pds)
    num_rows = math.isqrt(n_img)
    num_cols = math.ceil(n_img / num_rows)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(8, 8))

    # x, y = (1, -1.5)
    # point = np.array([[x, y]])
    # print(single_pds(models, point))

    for i, ax in enumerate(axes.flatten()):
        if i < n_img // 2:
            # ax.scatter(x, y, c="black")

            pd = pds[i]
            print(pd)
            ax.imshow(pd.reshape(points, points), extent=(-2, 2, -2, 2), origin="lower", vmin=0, vmax=1)  # Phase diagram
            ax.scatter(Xs[:, 0], Xs[:, 1], marker="x", s=40, c=obs, cmap='bwr')  # Existing observations
            ax.axis("off")
            ax.set_title(f'Phase {i}')
        elif i < n_img:
            pd = pds[i]

            print(pd)
            ax.imshow(pd.reshape(points, points), extent=(-2, 2, -2, 2), origin="lower", vmin=0, vmax=1.5)  # Phase diagram
            ax.scatter(Xs[:, 0], Xs[:, 1], marker="x", s=40, c=obs, cmap='bwr')  # Existing observations
            ax.axis("off")
            ax.set_title(f'STD {i - n_img // 2}')

        else:
            print("Removing")
            ax.set_axis_off()

    # plt.subplots_adjust(hspace=0.3)
    plt.tight_layout()

    # fig.colorbar(c)

    if t is not None:
        fig.suptitle(f"Samples of phase diagrams after {T} samples")
    else:
        fig.suptitle(f'Samples of phase diagrams')
    plt.show()


if __name__ == "__main__":
    save_name = "21"
    obs_h: ObsHolder = ObsHolder.load(f'./saves/{save_name}')
    obs_h.cfg.kern_r = 0.7
    obs_h.cfg.kern_var = 1

    for t in [50]:
        print()

        print("t =", t + 16)
        obs_holder = copy.deepcopy(obs_h)
        plot_samples(obs_holder, t=t, points=50)
#
# import GPy
# import numpy as np
#
# kernel = GPy.kern.Matern52(input_dim=1, variance=1., lengthscale=100)
#
# x1 = np.array([[i] for i in range(5)])
# x2 = np.array([[0] for i in range(5)])
#
# K = kernel.K(x1, X2=x2)
# K = K.T[0]
# print(K)
#
#