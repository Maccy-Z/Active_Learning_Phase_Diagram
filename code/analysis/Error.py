import sys
import os
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

cwd = os.getcwd()
os.chdir(os.path.dirname(cwd))

import code.edit_source_files
import pickle
import GPy
import numpy as np
import math
from matplotlib import pyplot as plt
import copy

from code.utils import make_grid, ObsHolder, tri_pd


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
    return sample_pds


# Fit model to existing observations
def fit_gp(obs_holder, t, *, cfg) -> list[GPy.core.GP]:
    "Trains a model for each phase."
    X, Y = obs_holder.get_obs()

    k_var, k_r = cfg.kern_var, cfg.kern_r
    var, r = obs_holder.get_kern_param(t=t)

    models = []
    for i in range(3):
        phase_i = (Y == i) * 2 - 1  # Between -1 and 1

        kernel = GPy.kern.Matern52(input_dim=2, variance=var, lengthscale=r)
        model = GPy.models.GPClassification(X, phase_i.reshape(-1, 1), kernel)
        #model = GPy.models.GPRegression(X, phase_i.reshape(-1, 1), kernel, noise_var=cfg.noise_var)

        models.append(model)
    return models


# Plot a grid of samples
def plot_samples(obs_holder, *, cfg, points=25, t=None):
    # Observations up to time t
    if t is not None:
        T = t + obs_holder.cfg.N_init ** 2
        obs_holder.obs_pos = obs_holder.obs_pos[:T]
        obs_holder.obs_phase = obs_holder.obs_phase[:T]

    plot_Xs, X1, X2 = make_grid(points, cfg.extent)

    models = fit_gp(obs_holder, cfg=cfg, t=t)
    pds = single_pds(models, plot_Xs).reshape(points, points)

    true_pd = []
    for X in plot_Xs:
        true_phase = tri_pd(X)
        true_pd.append(true_phase)

    true_pd = np.stack(true_pd).reshape(points, points)

    diff = np.not_equal(pds, true_pd)
    diff_mean = np.mean(diff)

    return diff_mean


def main():
    f = sorted([int(s) for s in os.listdir("./saves")])

    save_name = f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'./saves/{save_name}')

    with open(f'./saves/{save_name}/cfg.pkl', "rb") as f:
        cfg = pickle.load(f)

    errors = []
    for t in range(len(og_obs.obs_phase) - 16):
        obs_holder = copy.deepcopy(og_obs)
        error = plot_samples(obs_holder, points=50, t=t, cfg=cfg)
        errors.append(error)

    plt.plot(errors)
    plt.show()

    print()
    for s in [9, 19, 29, 39, 49]:
        print(f'{errors[s]}')


if __name__ == "__main__":
    main()
