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
        y_pred, _ = pd_model.predict(xs, include_likelihood=False)  # m.shape = [n**2, 1]

        y_preds.append(y_pred)

    y_preds = np.stack(y_preds)
    sample_pds = np.argmax(y_preds, axis=0).T
    return None, None, sample_pds


# Distance between true PD and prediction
def dist(obs_holder, *, true_pd, cfg, points, t):
    # Observations up to time t
    T = t + 2
    Xs, Ys = obs_holder.get_og_obs()

    obs_holder._obs_pos = Xs[:T]
    obs_holder.obs_phase = Ys[:T]

    plot_Xs, X1, X2 = make_grid(points, cfg.extent)
    model_Xs, _, _ = make_grid(points, (0, 1, 0, 1))
    models = fit_gp(obs_holder, cfg=cfg)
    pds = single_pds(models, model_Xs)[2].reshape(points, points)

    true_pd = np.stack(true_pd).reshape(points, points)

    diff = np.not_equal(pds, true_pd)
    diff_mean = np.mean(diff)

    return diff_mean


def main():
    true_pd = []
    eval_points = 21
    assert true_pd != [], ("Fill in the true phase diagram as a 2D numpy array, with the same number of points as eval_points. "
                           "It might need transposing / reflecting to orient properly. ")

    f = sorted([int(s) for s in os.listdir("./saves")])

    save_name = f[-1]
    print(f'{save_name = }')

    og_obs = ObsHolder.load(f'./saves/{save_name}')

    with open(f'./saves/{save_name}/cfg.pkl', "rb") as f:
        cfg = pickle.load(f)

    errors = []
    for t in range(len(og_obs.obs_phase)):
        obs_holder = copy.deepcopy(og_obs)
        error = dist(obs_holder, true_pd=true_pd, points=eval_points, t=t, cfg=cfg)
        errors.append(error)
    plt.plot(errors)
    plt.show()
    #
    # print()
    # for s in [10, 20, 30, 40, 50]:
    #     print(f'{errors[s]}')

    print()
    print(errors)


if __name__ == "__main__":
    main()
